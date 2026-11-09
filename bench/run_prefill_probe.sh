#!/usr/bin/env bash
# Runs ON the pod: prefill probe for each config, then profile stock vs swap.
set -uo pipefail
cd /workspace/vpa
export HF_HOME=/workspace/hf
MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUT=results/prefill; mkdir -p "$OUT"
for cfg in stock dual-high swap-high swap-eager; do
  python3 bench/prefill_probe.py --config $cfg --model "$MODEL" --out "$OUT" > "$OUT/$cfg.log" 2>&1
  grep -E '^RESULT' "$OUT/$cfg.log" || { echo "FAILED $cfg"; tail -20 "$OUT/$cfg.log"; }
done
for cfg in stock swap-high; do
  python3 bench/prefill_probe.py --config $cfg --model "$MODEL" --profile --repeats 3 --out "$OUT" \
    > "$OUT/$cfg.profile.log" 2>&1 || { echo "PROFILE FAILED $cfg"; tail -20 "$OUT/$cfg.profile.log"; }
done
python3 bench/analyze_trace.py "$OUT/trace_stock" "$OUT/trace_swap-high" | tee "$OUT/trace_summary.txt"
echo PREFILL_PROBE_DONE
