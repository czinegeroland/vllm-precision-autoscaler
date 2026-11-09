#!/usr/bin/env bash
# Runs ON the pod: INT4 version (v0.5.0): GPU tests, 3-version round trip, and a
# load sweep 16-bit vs FP8 vs INT4 (same 3 versions built in every config).
set -uo pipefail
cd /workspace/vpa
export HF_HOME=/workspace/hf VLLM_DISABLE_COMPILE_CACHE=1
MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUT=results/int4; mkdir -p "$OUT"

echo "== 1. GPU tests"
(cd plugin && python3 -m pytest -q tests/gpu/test_op_gpu.py tests/gpu/test_swap_gpu.py tests/gpu/test_int4_gpu.py \
  > "../$OUT/pytest.log" 2>&1)
grep -E 'passed|failed|error' "$OUT/pytest.log" | tail -3

echo "== 2. 3-version round trip (high -> fp8 -> int4 -> high)"
python3 plugin/tests/gpu/e2e_swap.py --model "$MODEL" --versions high,fp8,int4 \
  --out "$OUT/e2e.json" > "$OUT/e2e.log" 2>&1 || { echo "e2e failed"; tail -30 "$OUT/e2e.log"; }
grep -E '^\{' "$OUT/e2e.log" | tail -1

echo "== 3. load sweep"
VPA_VERSIONS=high,fp8,int4 OUT="$OUT/sweep" CONFIGS="swap-high swap-fp8 swap-int4" \
  CONCURRENCY="1 8 32 64 128" bash bench/run_bench.sh 2>&1 | grep -E '^==|c=|KV cache|died|failed'
echo INT4_CHECK_DONE
