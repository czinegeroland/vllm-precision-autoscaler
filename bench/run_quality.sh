#!/usr/bin/env bash
# Runs ON the pod: GSM8K quality for each precision configuration.
set -uo pipefail
cd /workspace/vpa
export HF_HOME=/workspace/hf
OUT=results/quality
mkdir -p "$OUT"
MODEL="${MODEL:-Qwen/Qwen3-4B}"

run() {
  local name=$1; shift
  echo "== $name"
  python3 bench/quality_gsm8k.py --name "$name" --model "$MODEL" --out "$OUT" "$@" \
    > "$OUT/$name.log" 2>&1 || { echo "FAILED $name"; tail -30 "$OUT/$name.log"; return 1; }
  tail -1 "$OUT/$name.log" | cut -c1-400
}

run stock-bf16
run stock-fp8 --quant fp8_per_channel
run dual-high --quant precision_autoscaler --mode high
run dual-fp8  --quant precision_autoscaler --mode fp8
run dual-auto-batch --quant precision_autoscaler --mode auto       # heavy load: all at once
run dual-auto-seq   --quant precision_autoscaler --mode auto --limit 100 --sequential  # low load
python3 bench/quality_compare.py "$OUT" | tee "$OUT/compare.txt"
