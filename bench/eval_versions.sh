#!/usr/bin/env bash
# Runs ON the pod: general accuracy (lm-evaluation-harness) for precision versions of
# the swap plugin, one vLLM server per version.
#   VERSIONS="high int4" bash bench/eval_versions.sh            # full run
#   VERSIONS="high" LIMIT=20 bash bench/eval_versions.sh        # smoke test
# Tasks: OpenLLM-v1 style loglikelihood tasks (MMLU, ARC-Challenge, HellaSwag, Winogrande,
# TruthfulQA mc2) via /v1/completions, and IFEval via /v1/chat/completions (thinking off).
# Results: results/evals/<version>/ (lm_eval JSON + logs). STOP_WHEN_DONE=1 stops this pod
# at the end (RunPod API), keeping /workspace (a pod volume) and its results.
set -uo pipefail
cd /workspace/vpa
export HF_HOME=/workspace/hf VLLM_DISABLE_COMPILE_CACHE=1 HF_DATASETS_TRUST_REMOTE_CODE=1
MODEL="${MODEL:-Qwen/Qwen3-4B}"
VERSIONS="${VERSIONS:-high fp8 int4 int4a8}"
ALL=high,fp8,int4,int4a8
OUT="${OUT:-results/evals}"; mkdir -p "$OUT"
LIMIT="${LIMIT:-}"
LL_TASKS="${LL_TASKS-mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2}"
GEN_TASKS="${GEN_TASKS-ifeval}"
GEN_VERSIONS="${GEN_VERSIONS:-$VERSIONS}"   # versions that run GEN_TASKS
CODE_TASKS="${CODE_TASKS:-}"                 # e.g. humaneval_plus,mbpp_plus (runs the generated code)
export HF_ALLOW_CODE_EVAL=1
LIM=(); [ -n "$LIMIT" ] && LIM=(--limit "$LIMIT")
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/progress.log"; }

wait_ready() {
  for _ in $(seq 1 180); do curl -sf localhost:8000/v1/models >/dev/null && return 0; sleep 5; done
  return 1
}

switch_to() {  # POST the version, then poll until every layer holds it
  local v=$1
  curl -s -X POST localhost:8000/v1/precision -H 'Content-Type: application/json' \
       -d "{\"version\": \"$v\"}" >/dev/null
  for _ in $(seq 1 120); do
    s=$(curl -s localhost:8000/v1/precision)
    if python3 - "$v" "$s" <<'EOF'
import json, sys
v, s = sys.argv[1], json.loads(sys.argv[2])
act = {x["name"]: x.get("active_layers") for x in s.get("versions", [])}
sys.exit(0 if (not s.get("migrating") and act.get(v) == s.get("layers")) else 1)
EOF
    then echo "$s" > "$OUT/$v/precision_state.json"; return 0; fi
    sleep 2
  done
  return 1
}

for v in $VERSIONS; do
  mkdir -p "$OUT/$v"
  log "== $v: start server"
  VLLM_PLUGINS=precision_autoscaler,precision_autoscaler_api VPA_FORMAT=swap VPA_VERSIONS=$ALL \
    VPA_MODE=high vllm serve "$MODEL" --quantization precision_autoscaler \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000 --seed 0 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    > "$OUT/$v/server.log" 2>&1 &
  SERVER=$!
  if ! wait_ready; then log "server for $v did not start"; tail -30 "$OUT/$v/server.log"; kill $SERVER; continue; fi
  if [ "$v" != high ] && ! switch_to "$v"; then log "switch to $v failed"; kill $SERVER; continue; fi
  [ "$v" = high ] && switch_to high
  log "$v active: $(cat "$OUT/$v/precision_state.json" | cut -c1-160)"

  if [ -n "$LL_TASKS" ]; then
  log "$v: loglikelihood tasks ($LL_TASKS)"
  lm_eval --model local-completions \
    --model_args "model=$MODEL,base_url=http://localhost:8000/v1/completions,num_concurrent=32,max_retries=5,tokenized_requests=False,tokenizer_backend=huggingface" \
    --tasks "$LL_TASKS" --batch_size 1 "${LIM[@]}" \
    --output_path "$OUT/$v/ll" > "$OUT/$v/ll.log" 2>&1 || log "$v: loglikelihood FAILED"
  grep -E '^\|' "$OUT/$v/ll.log" | tail -30 >> "$OUT/progress.log"
  fi

  if [ -n "$GEN_TASKS" ] && [[ " $GEN_VERSIONS " == *" $v "* ]]; then
  log "$v: generation tasks ($GEN_TASKS)"
  lm_eval --model local-chat-completions \
    --model_args "model=$MODEL,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=32,max_retries=5" \
    --tasks "$GEN_TASKS" --apply_chat_template --gen_kwargs "temperature=0,max_gen_toks=1280" "${LIM[@]}" \
    --output_path "$OUT/$v/gen" > "$OUT/$v/gen.log" 2>&1 || log "$v: generation FAILED"
  grep -E '^\|' "$OUT/$v/gen.log" | tail -10 >> "$OUT/progress.log"
  fi

  if [ -n "$CODE_TASKS" ]; then
  log "$v: code tasks ($CODE_TASKS)"
  for t in ${CODE_TASKS//,/ }; do  # one task per run: the stop-list fix keeps one list at a time
    python3 bench/lm_eval_vllm.py --model local-completions       --model_args "model=$MODEL,base_url=http://localhost:8000/v1/completions,num_concurrent=32,max_retries=5,tokenized_requests=False,tokenizer_backend=huggingface"       --tasks "$t" --gen_kwargs "temperature=0" --confirm_run_unsafe_code "${LIM[@]}"       --output_path "$OUT/$v/code_$t" > "$OUT/$v/code_$t.log" 2>&1 || log "$v: $t FAILED"
    grep -E '^\|' "$OUT/$v/code_$t.log" | tail -6 >> "$OUT/progress.log"
  done
  fi

  # the precision must not have moved during the evaluation
  curl -s localhost:8000/v1/precision > "$OUT/$v/precision_after.json"
  kill $SERVER; wait $SERVER 2>/dev/null; sleep 5
  log "== $v done"
done
log "EVALS_DONE"

if [ "${STOP_WHEN_DONE:-0}" = 1 ]; then
  log "stopping pod $RUNPOD_POD_ID"
  sync
  python3 deploy/pod/self_stop.py >> "$OUT/progress.log" 2>&1
fi
