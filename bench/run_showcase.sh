#!/usr/bin/env bash
# Runs ON the pod: precision showcase with full telemetry (bench/showcase_load.py).
#   1. baseline : same load profile, stays 16-bit (no switches)
#   2. showcase : 16-bit -> INT4 (medium load) -> FP8 (high load) -> 16-bit (quiet)
set -uo pipefail
cd /workspace/vpa
export HF_HOME=/workspace/hf VLLM_DISABLE_COMPILE_CACHE=1
MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUT="${OUT:-results/showcase}"; mkdir -p "$OUT"
PHASES="${PHASES:-4:60,32:90,128:90,4:60}"
SWITCHES="${SWITCHES:-75:int4,165:fp8,255:high}"
RUNS="${RUNS:-baseline showcase}"
export VPA_FORMAT=swap VPA_VERSIONS=high,fp8,int4 VPA_MODE=high
export VLLM_PLUGINS=precision_autoscaler,precision_autoscaler_api

for run in $RUNS; do
  sw=""; [ "$run" != baseline ] && sw="$SWITCHES"
  echo "== $run (switches: ${sw:-none})"
  vllm serve "$MODEL" --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000 --seed 0 \
    --quantization precision_autoscaler --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    > "$OUT/$run.server.log" 2>&1 &
  pid=$!
  ok=0
  for _ in $(seq 1 180); do
    curl -sf localhost:8000/health >/dev/null && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 5
  done
  if [ $ok != 1 ]; then echo "server failed"; tail -30 "$OUT/$run.server.log"; kill $pid 2>/dev/null; continue; fi
  grep -E "GPU KV cache size|precision_autoscaler swap:" "$OUT/$run.server.log" | sed -E 's/.*\] //' | head -2 | cut -c1-200
  python3 bench/showcase_load.py --name "$run" --model "$MODEL" --phases "$PHASES" \
    --switches "$sw" --out "$OUT" 2>&1 | tee "$OUT/$run.load.log" | grep -vE '^\s*$' | cut -c1-220
  grep -E "precision_autoscaler swap: (now|migrat)" "$OUT/$run.server.log" | sed -E 's/.*\] //' | head -12 | cut -c1-200
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "[v]llm serve" >/dev/null || break; sleep 2; done
  pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null; sleep 3
done
echo SHOWCASE_DONE
