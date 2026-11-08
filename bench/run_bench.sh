#!/usr/bin/env bash
# Runs ON the pod. For each server config: start vllm serve, sweep
# concurrency with `vllm bench serve`, save JSON + server log + telemetry.
#
#   CONFIGS="stock-bf16 dual-auto stock-bf16+kv8 dual-auto+kv8 dual-auto+acct" bash bench/run_bench.sh
#
# Config = <format>-<mode>[+kv8][+acct]
#   stock-bf16 | stock-fp8           plain vLLM (no plugin) / vLLM online FP8
#   {dual,nested}-{high,fp8,auto,dial} the plugin with that format and mode
#   swap-{high,fp8,auto}            the swap format (CUDA graphs are turned off automatically)
#   swap-* uses two CUDA graph sets (cudagraph_mode FULL_DECODE_ONLY) unless +eager
#   +eager turn CUDA graphs off
#   +fdo   cudagraph_mode FULL_DECODE_ONLY (fair stock baseline for swap)
#   +kv8   add --kv-cache-dtype fp8 (FP8 KV cache)
#   +acct  enable the plugin's GPU counters and record the FP8 share for each
#          concurrency level (counters add a little overhead: separate run)
#
# Telemetry per config: nvidia-smi samples every second (gpu.csv),
# /metrics snapshot after each concurrency level, server log.
set -uo pipefail
cd /workspace/vpa
export HF_HOME=/workspace/hf

MODEL="${MODEL:-Qwen/Qwen3-4B}"
CONFIGS="${CONFIGS:-stock-bf16 stock-fp8 dual-high dual-fp8 dual-auto}"
CONCURRENCY="${CONCURRENCY:-1 8 32 64 128}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
THRESHOLD="${THRESHOLD:-32}"
OUT="${OUT:-results/bench/$(echo "$MODEL" | tr / _)}"
COMMON=(--max-model-len 4096 --gpu-memory-utilization ${GPU_UTIL:-0.90} --port 8000 --seed 0)
EXT=vllm_precision_autoscaler.worker_ext.PrecisionAutoscalerWorkerExtension
mkdir -p "$OUT"

rpc() { curl -s localhost:8000/collective_rpc -H 'Content-Type: application/json' -d "{\"method\":\"$1\"}"; }

start_server() {
  local cfg=$1 base=${1%%+*}
  unset VPA_MODE VPA_THRESHOLD VPA_COUNT VPA_FORMAT VLLM_SERVER_DEV_MODE VLLM_DISABLE_COMPILE_CACHE
  ACCT=0
  case "$base" in
    stock-bf16) args=() ;;
    stock-fp8)  args=(--quantization fp8_per_channel) ;;
    nested-*|dual-*|swap-*)
      export VPA_FORMAT="${base%%-*}" VPA_MODE="${base#*-}" VPA_THRESHOLD=$THRESHOLD
      args=(--quantization precision_autoscaler) ;;
    *) echo "unknown config $cfg"; return 1 ;;
  esac
  [[ "$cfg" == *"+kv8"* ]] && args+=(--kv-cache-dtype fp8)
  if [[ "$cfg" == *"+eager"* ]]; then
    args+=(--compilation-config '{"cudagraph_mode": "NONE"}')
  elif [[ "$base" == swap-* || "$cfg" == *"+fdo"* ]]; then
    args+=(--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}')
  fi
  # vLLM's compile cache key ignores VPA_* settings (see FINDINGS 9.3)
  [[ "$base" != stock-* ]] && export VLLM_DISABLE_COMPILE_CACHE=1 || unset VLLM_DISABLE_COMPILE_CACHE
  if [[ "$cfg" == *"+acct"* ]]; then
    ACCT=1
    export VPA_COUNT=1 VLLM_SERVER_DEV_MODE=1
    args+=(--worker-extension-cls "$EXT")
  fi
  echo "  args: ${args[*]:-(none)}  VPA_MODE=${VPA_MODE:-} VPA_FORMAT=${VPA_FORMAT:-}"
  vllm serve "$MODEL" "${COMMON[@]}" "${args[@]}" > "$OUT/$cfg.server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 180); do
    curl -sf localhost:8000/health >/dev/null && return 0
    kill -0 $SERVER_PID 2>/dev/null || { echo "server $cfg died"; tail -40 "$OUT/$cfg.server.log"; return 1; }
    sleep 5
  done
  echo "server $cfg did not become healthy"; return 1
}

stop_server() {
  kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "vllm serve" >/dev/null || break; sleep 2; done
}

for cfg in $CONFIGS; do
  echo "== $cfg"
  start_server "$cfg" || continue
  grep -E "GPU KV cache size|Model loading took" "$OUT/$cfg.server.log" | sed -E 's/.*\] //' | head -3
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm \
    --format=csv,noheader,nounits -l 1 > "$OUT/$cfg.gpu.csv" 2>/dev/null &
  SMI_PID=$!
  vllm bench serve --model "$MODEL" --dataset-name random --random-input-len "$INPUT_LEN" \
    --random-output-len 32 --num-prompts 16 --max-concurrency 8 --ignore-eos --seed 1 >/dev/null 2>&1
  for c in $CONCURRENCY; do
    n=$(( c * 8 )); [ $n -lt ${MIN_PROMPTS:-16} ] && n=${MIN_PROMPTS:-16}
    [ "$ACCT" = 1 ] && { n=$(( c * 4 )); [ $n -lt 16 ] && n=16; rpc vpa_reset >/dev/null; }
    echo "$(date -u +%FT%TZ) start c=$c" >> "$OUT/$cfg.timeline.txt"
    vllm bench serve --model "$MODEL" --dataset-name random \
      --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
      --max-concurrency "$c" --num-prompts "$n" --ignore-eos --seed 42 \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
      --goodput ttft:2000 tpot:50 \
      --save-result --result-dir "$OUT" --result-filename "$cfg.c$c.json" \
      > "$OUT/$cfg.c$c.txt" 2>&1
    echo "$(date -u +%FT%TZ) end c=$c" >> "$OUT/$cfg.timeline.txt"
    curl -s localhost:8000/metrics > "$OUT/$cfg.c$c.metrics.txt"
    line=$(grep -E "Output token throughput|Mean TPOT|P95 TTFT|Request goodput" "$OUT/$cfg.c$c.txt" \
      | tr -s ' ' | paste -sd'|')
    if [ "$ACCT" = 1 ]; then
      rpc vpa_stats > "$OUT/$cfg.c$c.vpa_stats.json"
      share=$(python3 -c "import json;print(round(json.load(open('$OUT/$cfg.c$c.vpa_stats.json'))['results'][0]['fp8_token_share'],3))" 2>/dev/null)
      line="$line| fp8 token share: $share"
    fi
    echo "  c=$c | $line"
  done
  kill $SMI_PID 2>/dev/null
  stop_server
done
echo "done -> $OUT"
