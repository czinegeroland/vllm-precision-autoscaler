#!/usr/bin/env bash
# Quick A/B without CUDA graphs: is the slowdown the host or the plugin?
#   CONFIGS="stock-bf16 dual-high swap-high" bash bench/ab_eager.sh
set -uo pipefail
cd /workspace/vpa
export HF_HOME=/workspace/hf
MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUT=results/ab_eager; mkdir -p "$OUT"
NOGRAPH='{"cudagraph_mode": "NONE"}'
for cfg in ${CONFIGS:-stock-bf16 dual-high swap-high}; do
  unset VPA_MODE VPA_FORMAT
  args=(--compilation-config "$NOGRAPH")
  case "$cfg" in
    stock-bf16) ;;
    *) export VPA_FORMAT="${cfg%%-*}" VPA_MODE="${cfg#*-}"; args+=(--quantization precision_autoscaler) ;;
  esac
  vllm serve "$MODEL" --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000 "${args[@]}" \
    > "$OUT/$cfg.server.log" 2>&1 &
  pid=$!
  for _ in $(seq 1 120); do curl -sf localhost:8000/health >/dev/null && break; sleep 5; done
  for c in 1 8; do
    vllm bench serve --model "$MODEL" --dataset-name random --random-input-len 512 --random-output-len 128 \
      --max-concurrency $c --num-prompts $((c * 6)) --ignore-eos --seed $((100 + c)) > "$OUT/$cfg.c$c.txt" 2>&1
    echo "$cfg c=$c $(grep -E 'Output token throughput|Mean TPOT' "$OUT/$cfg.c$c.txt" | tr -s ' ' | paste -sd'|')"
  done
  kill $pid; wait $pid 2>/dev/null; sleep 5
done
echo AB_DONE
