#!/usr/bin/env bash
# Runs ON the pod: GSM8K (full test set, 1319 questions) for every precision version.
#   stock-bf16  : reference, no plugin
#   swap-high/fp8/int4 : the plugin's three versions (VPA_FORMAT=swap)
#   stock-awq   : a calibrated INT4 checkpoint (AWQ) for comparison with our online RTN INT4
set -uo pipefail
cd /workspace/vpa
export HF_HOME=/workspace/hf
OUT="${OUT:-results/quality_versions}"; mkdir -p "$OUT"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
AWQ_MODEL="${AWQ_MODEL:-Qwen/Qwen3-4B-AWQ}"
SWAP="--quant precision_autoscaler --format swap --versions high,fp8,int4 --cudagraph-mode FULL_DECODE_ONLY"

run() {
  local name=$1; shift
  echo "== $name"
  python3 bench/quality_gsm8k.py --name "$name" --out "$OUT" "$@"     > "$OUT/$name.log" 2>&1 || { echo "FAILED $name"; tail -30 "$OUT/$name.log"; return 1; }
  grep -E '^all [0-9]+ layers' "$OUT/$name.log"
  tail -1 "$OUT/$name.log" | cut -c1-300
}

run stock-bf16 --model "$MODEL"
run swap-high  --model "$MODEL" $SWAP --mode high
run swap-fp8   --model "$MODEL" $SWAP --mode fp8
run swap-int4  --model "$MODEL" $SWAP --mode int4
run stock-awq  --model "$AWQ_MODEL"
python3 bench/quality_compare.py "$OUT" | tee "$OUT/compare.txt"
echo QUALITY_DONE
