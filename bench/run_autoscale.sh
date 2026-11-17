#!/usr/bin/env bash
# Runs ON the pod: "hold per-user speed while users grow" showcase, with Prometheus + Grafana.
#   one vLLM server (swap format, 16-bit/FP8/INT4), the same user ramp twice:
#   1. baseline   : no controller, stays 16-bit
#   2. autoscaled : controller/precision_controller.py steps the precision ladder
# Telemetry: Prometheus (scraped every 2 s) + bench/showcase_load.py JSON for static charts.
set -uo pipefail
cd /workspace/vpa
export HF_HOME=/workspace/hf VLLM_DISABLE_COMPILE_CACHE=1
MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUT="${OUT:-results/autoscale}"; mkdir -p "$OUT"
RAMP="${RAMP:-4:40,8:40,16:40,32:60,48:60,64:60,32:40,16:40,4:60}"
CONFIG="${CONFIG:-controller/config.example.yaml}"
GAP="${GAP:-30}"
export VPA_FORMAT=swap VPA_VERSIONS=high,fp8,int4 VPA_MODE=high
export VLLM_PLUGINS=precision_autoscaler,precision_autoscaler_api

vllm serve "$MODEL" --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000 --seed 0 \
  --quantization precision_autoscaler --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  > "$OUT/server.log" 2>&1 &
SPID=$!
bash deploy/pod/observability.sh > "$OUT/observability.log" 2>&1 &
OPID=$!
ok=0
for _ in $(seq 1 180); do
  curl -sf localhost:8000/health >/dev/null && { ok=1; break; }
  kill -0 $SPID 2>/dev/null || break
  sleep 5
done
[ $ok = 1 ] || { echo "server failed"; tail -30 "$OUT/server.log"; exit 1; }
wait $OPID; grep -vE '^downloading' "$OUT/observability.log"
grep -E "GPU KV cache size|precision_autoscaler swap:" "$OUT/server.log" | sed -E 's/.*\] //' | head -2 | cut -c1-200
echo "plugin metrics: $(curl -s localhost:8000/v1/precision/metrics | grep -c '^vpa_')"

echo "== baseline (16-bit only) $(date -u +%T)"
python3 bench/showcase_load.py --name baseline --model "$MODEL" --phases "$RAMP" --switches "" \
  --out "$OUT" 2>&1 | grep -E 'load ->|saved' | cut -c1-200
sleep "$GAP"

echo "== autoscaled $(date -u +%T)"
python3 controller/precision_controller.py --config "$CONFIG" --log "$OUT/controller.jsonl" \
  > "$OUT/controller.out" 2>&1 &
CPID=$!
sleep 3
python3 bench/showcase_load.py --name autoscaled --model "$MODEL" --phases "$RAMP" --switches "" \
  --out "$OUT" 2>&1 | grep -E 'load ->|saved' | cut -c1-200
sleep 25   # let the controller step back up to 16-bit while idle
kill $CPID 2>/dev/null
echo "controller decisions:"
grep -E '"event": "(switch|migrated)"' "$OUT/controller.jsonl" | cut -c1-260

bash deploy/pod/observability.sh snapshot
cp deploy/observability/grafana/dashboards/precision.json "$OUT/"
echo AUTOSCALE_DONE
