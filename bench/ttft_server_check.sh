#!/usr/bin/env bash
# Runs ON the pod. Server-mode TTFT check: stock vs swap (16-bit), 1 and 4
# users, and swap again while a poller calls GET /v1/precision every 0.5 s
# (like the telemetry sampler in phase tests). Same graph mode everywhere.
set -uo pipefail
cd /workspace/vpa
export HF_HOME=/workspace/hf VLLM_DISABLE_COMPILE_CACHE=1
MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUT=results/ttft_check; mkdir -p "$OUT"
FDO='{"cudagraph_mode": "FULL_DECODE_ONLY"}'

serve() {  # $1 = name, rest = extra args
  local name=$1; shift
  vllm serve "$MODEL" --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8000 --seed 0 \
    --compilation-config "$FDO" "$@" > "$OUT/$name.server.log" 2>&1 &
  SPID=$!
  for _ in $(seq 1 120); do curl -sf localhost:8000/health >/dev/null && return 0; sleep 5; done
  echo "server $name failed"; tail -20 "$OUT/$name.server.log"; return 1
}
stop() { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 5; }
bench() {  # $1 = name, $2 = concurrency
  vllm bench serve --model "$MODEL" --dataset-name random --random-input-len 512 --random-output-len 64 \
    --max-concurrency "$2" --num-prompts $(( $2 * 12 )) --ignore-eos --seed $((70 + $2)) \
    --percentile-metrics ttft,tpot --metric-percentiles 50,95 \
    --save-result --result-dir "$OUT" --result-filename "$1.c$2.json" > "$OUT/$1.c$2.txt" 2>&1
  echo "$1 c=$2 | $(grep -E 'Median TTFT|P95 TTFT|Median TPOT' "$OUT/$1.c$2.txt" | tr -s ' ' | paste -sd'|')"
}
warm() { vllm bench serve --model "$MODEL" --dataset-name random --random-input-len 512 \
  --random-output-len 16 --num-prompts 8 --max-concurrency 4 --ignore-eos --seed 1 >/dev/null 2>&1; }

unset VLLM_PLUGINS VPA_FORMAT VPA_MODE
serve stock && { warm; bench stock 1; bench stock 4; stop; }

export VLLM_PLUGINS=precision_autoscaler,precision_autoscaler_api VPA_FORMAT=swap VPA_MODE=high
if serve swap --quantization precision_autoscaler; then
  warm; bench swap 1; bench swap 4
  ( while true; do curl -s localhost:8000/v1/precision >/dev/null; sleep 0.5; done ) &
  POLL=$!
  bench swap-polled 1; bench swap-polled 4
  kill $POLL
  stop
fi
echo TTFT_CHECK_DONE
