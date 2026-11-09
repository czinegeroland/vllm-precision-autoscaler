#!/usr/bin/env bash
# Runs ON the pod: how fast is a precision switch? Old (one layer per staging
# buffer) vs new (packed) migrations, under load and while idle.
#   VARIANTS="nopack pack pack-1g" bash bench/migration_bench.sh
set -uo pipefail
cd /workspace/vpa
export HF_HOME=/workspace/hf VLLM_DISABLE_COMPILE_CACHE=1
MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUT=results/migration; mkdir -p "$OUT"
FDO='{"cudagraph_mode": "FULL_DECODE_ONLY"}'

for v in ${VARIANTS:-nopack pack pack-1g}; do
  unset VPA_SWAP_PACK VPA_SWAP_STAGING_MIB VPA_SWAP_SLOTS
  case "$v" in
    nopack)  export VPA_SWAP_PACK=0 VPA_SWAP_SLOTS=${SLOTS:-4} ;;   # v0.3.0 behaviour
    pack)    export VPA_SWAP_PACK=1 VPA_SWAP_SLOTS=${SLOTS:-4} ;;   # same memory, packed
    pack-1g) export VPA_SWAP_PACK=1 VPA_SWAP_STAGING_MIB=1024 ;;    # 1 GiB staging budget
  esac
  export VLLM_PLUGINS=precision_autoscaler,precision_autoscaler_api VPA_FORMAT=swap VPA_MODE=high
  vllm serve "$MODEL" --max-model-len 4096 --gpu-memory-utilization ${GPU_UTIL:-0.85} --port 8000 \
    --seed 0 --quantization precision_autoscaler --compilation-config "$FDO" > "$OUT/$v.server.log" 2>&1 &
  SPID=$!
  ok=0; for _ in $(seq 1 150); do curl -sf localhost:8000/health >/dev/null && { ok=1; break; }; sleep 5; done
  [ $ok = 1 ] || { echo "server $v failed"; tail -20 "$OUT/$v.server.log"; kill $SPID; continue; }
  echo "== $v  staging: $(curl -s localhost:8000/v1/precision | python3 -c 'import sys,json; print(json.load(sys.stdin).get("workers"))') worker(s)"
  grep -E 'precision_autoscaler swap: [0-9]+ layers' "$OUT/$v.server.log" | sed -E 's/.*\] //'
  # under load: 8 users generating continuously
  vllm bench serve --model "$MODEL" --dataset-name random --random-input-len 256 --random-output-len 1024 \
    --max-concurrency 8 --num-prompts 64 --ignore-eos --seed 5 > "$OUT/$v.load.txt" 2>&1 &
  LPID=$!
  sleep 20
  python3 bench/migration_probe.py --mode fp8  --label "$v/loaded" --out "$OUT/migrations.jsonl"
  sleep 10
  python3 bench/migration_probe.py --mode high --label "$v/loaded" --out "$OUT/migrations.jsonl"
  kill $LPID 2>/dev/null; wait $LPID 2>/dev/null; sleep 5
  # idle: only the controller's GET polling advances the migration
  python3 bench/migration_probe.py --mode fp8  --label "$v/idle" --out "$OUT/migrations.jsonl"
  python3 bench/migration_probe.py --mode high --label "$v/idle" --out "$OUT/migrations.jsonl"
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 5
done
echo MIGRATION_BENCH_DONE
