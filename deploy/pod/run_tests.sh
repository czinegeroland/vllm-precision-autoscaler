#!/usr/bin/env bash
# Runs ON the pod: unit tests, op-level GPU tests, then end-to-end switching
# checks in a real vLLM engine (3 engine launches).
set -uo pipefail
cd /workspace/vpa/plugin
MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUT=/workspace/vpa/results/e2e
mkdir -p "$OUT"
export HF_HOME=/workspace/hf

echo "== unit tests";     python3 -m unittest discover -s tests -p "test_policy.py" || exit 1
echo "== op GPU tests"
python3 -m pytest -q -s tests/gpu/test_op_gpu.py tests/gpu/test_swap_gpu.py > "$OUT/pytest.log" 2>&1
grep -E 'PINNED|passed|failed|error' "$OUT/pytest.log"
grep -qE '[0-9]+ passed' "$OUT/pytest.log" && ! grep -qE 'failed|error' "$OUT/pytest.log" || exit 1

echo "== e2e swap"
for cg in FULL_DECODE_ONLY NONE; do
  python3 tests/gpu/e2e_swap.py --model "$MODEL" --cudagraph-mode $cg --out "$OUT/swap_$cg.json" \
    > "$OUT/swap_$cg.log" 2>&1 || { echo "e2e swap ($cg) failed"; tail -60 "$OUT/swap_$cg.log"; exit 1; }
  grep -E '^\{' "$OUT/swap_$cg.log" | tail -1
  grep -E 'precision_autoscaler' "$OUT/swap_$cg.log" | sed -E 's/.*\] //' | grep -v "step with" | head -6
done
[ "${SWAP_ONLY:-0}" = 1 ] && exit 0

export VPA_FORMAT="${VPA_FORMAT:-dual}"
echo "== e2e format=$VPA_FORMAT"

for mode in high fp8 auto; do
  echo "== e2e mode=$mode"
  python3 tests/gpu/e2e_switching.py --model "$MODEL" --mode "$mode" --threshold 64 \
      --out "$OUT/$mode.json" > "$OUT/$mode.log" 2>&1 \
    || { echo "e2e $mode failed"; tail -60 "$OUT/$mode.log"; exit 1; }
  grep -E "^(single|batch|\{)" "$OUT/$mode.log"
  grep -E "KV cache|precision_autoscaler" "$OUT/$mode.log" | head -12
done

echo "== compare"
python3 tests/gpu/check_e2e.py "$OUT/high.json" "$OUT/fp8.json" "$OUT/auto.json"
