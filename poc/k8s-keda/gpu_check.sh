#!/usr/bin/env bash
# Runs ON a GPU machine before the KEDA experiment, when the GPU is new to the plugin
# (e.g. RTX 5090, Blackwell sm_120): does every precision work, and how fast is each one?
#   1. GPU tests (swap, INT4 W4A16, INT4 W4A8)
#   2. load sweep: stock BF16 and the plugin's BF16 / FP8 / INT4 W4A16 / INT4 W4A8
# The sweep gives the KEDA target (users per replica) and the controller's thresholds.
# The agent must be idle (no vLLM running): this uses the whole GPU.
set -uo pipefail
cd /workspace/vpa
source deploy/pod/env.sh
export HF_HOME=/workspace/hf
OUT="${OUT:-results/gpu_check}"; mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv | tee "$OUT/gpu.txt"
# the benchmark servers use port 8000, where the agent listens: stop it, restart it at the end
pkill -f "[d]eploy/k8s/agent.py" 2>/dev/null || true
trap 'nohup setsid python3 poc/k8s-keda/agent.py --port 8000 >> /workspace/agent.log 2>&1 < /dev/null &' EXIT
sleep 2

echo "== 1. GPU tests"
(cd plugin && python3 -m pytest -q tests/gpu/test_swap_gpu.py tests/gpu/test_int4_gpu.py \
   tests/gpu/test_int4a8_gpu.py > "../$OUT/pytest.log" 2>&1)
grep -E 'passed|failed|error' "$OUT/pytest.log" | tail -3
VERS=high,fp8,int4,int4a8
if grep -qE '[0-9]+ failed|error' "$OUT/pytest.log" && grep -q int4a8 "$OUT/pytest.log"; then
  if grep -E 'FAILED.*int4a8|ERROR.*int4a8' "$OUT/pytest.log" >/dev/null; then
    echo "INT4 W4A8 does not work on this GPU: sweep without it"
    VERS=high,fp8,int4
  fi
fi
echo "VERSIONS=$VERS" | tee "$OUT/versions.txt"

echo "== 2. load sweep"
CFGS="stock-bf16"
for v in ${VERS//,/ }; do CFGS="$CFGS swap-$v"; done
VPA_VERSIONS=$VERS OUT="$OUT/sweep" CONFIGS="$CFGS" CONCURRENCY="${CONCURRENCY:-1 16 32 48 64 96 128}" \
  bash bench/run_bench.sh 2>&1 | grep -E '^==|c=|KV cache|died|failed|did not'
echo GPU_CHECK_DONE
