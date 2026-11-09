#!/usr/bin/env bash
# Runs ON the pod: verify the N-versions refactor (v0.4.0) behaves like v0.3.2.
set -uo pipefail
cd /workspace/vpa
export HF_HOME=/workspace/hf VLLM_DISABLE_COMPILE_CACHE=1
MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUT=results/phase2; mkdir -p "$OUT"

echo "== 1. GPU tests"
(cd plugin && python3 -m pytest -q tests/gpu/test_op_gpu.py tests/gpu/test_swap_gpu.py > "../$OUT/pytest.log" 2>&1)
grep -E 'passed|failed|error' "$OUT/pytest.log" | tail -2

echo "== 2. e2e round trip with graph sets"
python3 plugin/tests/gpu/e2e_swap.py --model "$MODEL" --cudagraph-mode FULL_DECODE_ONLY \
  --out "$OUT/e2e.json" > "$OUT/e2e.log" 2>&1 || { echo "e2e failed"; tail -30 "$OUT/e2e.log"; }
grep -E '^\{' "$OUT/e2e.log" | tail -1 | cut -c1-400

echo "== 3. API check"
export VLLM_PLUGINS=precision_autoscaler,precision_autoscaler_api VPA_FORMAT=swap VPA_MODE=high
vllm serve "$MODEL" --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8000 --seed 0 \
  --quantization precision_autoscaler --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  > "$OUT/server.log" 2>&1 &
SPID=$!
for _ in $(seq 1 120); do curl -sf localhost:8000/health >/dev/null && break; sleep 5; done
echo "GET:            $(curl -s localhost:8000/v1/precision | python3 -c 'import sys,json; d=json.load(sys.stdin); print({k: d[k] for k in ("version","versions","migrating")})')"
echo "POST int3:      HTTP $(curl -s -o "$OUT/bad.json" -w '%{http_code}' localhost:8000/v1/precision -H 'Content-Type: application/json' -d '{"version":"int3"}') $(cut -c1-160 "$OUT/bad.json")"
echo "POST fp8:       $(curl -s localhost:8000/v1/precision -H 'Content-Type: application/json' -d '{"version":"fp8"}' | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["version"], d["fp8_layers"], d["migrating"], d["migrations"][-1:] )')"
echo "POST bf16 (alias): $(curl -s localhost:8000/v1/precision -H 'Content-Type: application/json' -d '{"version":"bf16"}' | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["version"], d["fp8_layers"], d["migrating"], d["migrations"][-1:])')"
echo "chat after round trip: $(curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"'"$MODEL"'","prompt":"The capital of France is","max_tokens":8,"temperature":0}' | python3 -c 'import sys,json; print(repr(json.load(sys.stdin)["choices"][0]["text"]))')"
kill $SPID; wait $SPID 2>/dev/null; sleep 5

echo "== 4. migration speed (must match v0.3.2)"
VARIANTS=pack bash bench/migration_bench.sh > "$OUT/migration.log" 2>&1
grep -E '^\{' "$OUT/migration.log" | cut -c1-200
echo PHASE2_CHECK_DONE
