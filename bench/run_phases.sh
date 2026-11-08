#!/usr/bin/env bash
# Runs ON the pod: phased load (quiet -> busy -> quiet) per server config.
#   CONFIGS="stock-bf16 stock-bf16+eager dual-auto swap-auto" bash bench/run_phases.sh
set -uo pipefail
cd /workspace/vpa
export HF_HOME=/workspace/hf
MODEL="${MODEL:-Qwen/Qwen3-4B}"
CONFIGS="${CONFIGS:-stock-bf16 dual-auto swap-auto}"
PHASES="${PHASES:-4:45,96:60,4:60}"
OUT=results/phases
mkdir -p "$OUT"
EXT=vllm_precision_autoscaler.worker_ext.PrecisionAutoscalerWorkerExtension
COMMON=(--max-model-len 4096 --gpu-memory-utilization ${GPU_UTIL:-0.90} --port 8000 --seed 0)
NOGRAPH='{"cudagraph_mode": "NONE"}'

for cfg in $CONFIGS; do
  base=${cfg%%+*}
  unset VPA_MODE VPA_FORMAT VPA_THRESHOLD
  export VLLM_SERVER_DEV_MODE=1
  unset VLLM_PLUGINS CTL_PID
  case "$base" in
    swap-ctl)
      # externally controlled: plugin exposes /v1/precision, controller decides
      unset VLLM_SERVER_DEV_MODE
      export VPA_FORMAT=swap VPA_MODE=high VLLM_PLUGINS=precision_autoscaler,precision_autoscaler_api
      args=(--quantization precision_autoscaler) ;;
    stock-bf16) args=() ;;
    stock-fp8)  args=(--quantization fp8_per_channel) ;;
    dual-*|swap-*)
      export VPA_FORMAT="${base%%-*}" VPA_MODE="${base#*-}" VPA_THRESHOLD="${THRESHOLD:-32}"
      args=(--quantization precision_autoscaler --worker-extension-cls "$EXT") ;;
    *) echo "unknown $cfg"; continue ;;
  esac
  if [[ "$cfg" == *"+eager"* ]]; then args+=(--compilation-config "$NOGRAPH")
  elif [[ "$base" == swap-* || "$cfg" == *"+fdo"* ]]; then args+=(--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'); fi
  [[ "$base" != stock-* ]] && export VLLM_DISABLE_COMPILE_CACHE=1 || unset VLLM_DISABLE_COMPILE_CACHE
  [[ "$cfg" == *"+kv8"* ]] && args+=(--kv-cache-dtype fp8)
  echo "== $cfg  (${args[*]:-no extra args})"
  vllm serve "$MODEL" "${COMMON[@]}" "${args[@]}" > "$OUT/$cfg.server.log" 2>&1 &
  pid=$!
  ok=0
  for _ in $(seq 1 180); do
    curl -sf localhost:8000/health >/dev/null && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 5
  done
  if [ $ok != 1 ]; then echo "server $cfg failed"; tail -30 "$OUT/$cfg.server.log"; kill $pid 2>/dev/null; continue; fi
  grep -E "GPU KV cache size|Model loading took|precision_autoscaler swap" "$OUT/$cfg.server.log" | sed -E 's/.*\] //' | head -4
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,pcie.link.gen.current,pcie.link.width.current \
    --format=csv,noheader,nounits -l 1 > "$OUT/$cfg.gpu.csv" 2>/dev/null &
  smi=$!
  if [ "$base" = swap-ctl ]; then
    python3 controller/precision_controller.py --url http://localhost:8000       --up "${CTL_UP:-16}" --down "${CTL_DOWN:-6}" --dwell "${CTL_DWELL:-20}"       --log "$OUT/$cfg.controller.jsonl" > "$OUT/$cfg.controller.out" 2>&1 &
    CTL_PID=$!
    sleep 2
  fi
  python3 bench/phase_load.py --name "$cfg" --model "$MODEL" --phases "$PHASES" --out "$OUT" 2>&1 | tail -8
  kill $smi 2>/dev/null
  [ -n "${CTL_PID:-}" ] && { kill $CTL_PID 2>/dev/null; echo "controller decisions:"; grep -E '"event": "(switch|migrated)"' "$OUT/$cfg.controller.jsonl" | cut -c1-220; }
  grep -E "precision_autoscaler swap: (now|migrating)" "$OUT/$cfg.server.log" | sed -E 's/.*\] //' | head -10
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "vllm serve" >/dev/null || break; sleep 2; done
done
echo PHASES_DONE
