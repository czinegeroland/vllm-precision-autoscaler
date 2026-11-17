#!/usr/bin/env bash
# Whole real run, unattended: verified machines -> prewarm -> both scenarios -> save -> terminate.
# The pods are terminated at the end whatever happens (trap). Log: poc/k8s-keda/.state/run_all.log
set -uo pipefail
cd "$(dirname "$0")/../.."
export N="${N:-2}" OUT="${OUT:-results/k8s_burst}"
export PHASES="${PHASES:-16:60,56:300,16:240,56:300,16:120}"
export PATH="$PWD/poc/k8s-keda/.bin:$PATH"
log() { echo "[$(date +%H:%M:%S)] $*"; }
finish() {
  log "SAVE"; bash poc/k8s-keda/run.sh save || log "save failed"
  log "TERMINATE"; bash poc/k8s-keda/run.sh pods-terminate || python deploy/runpod/rp.py list
  bash poc/k8s-keda/run.sh report || log "report failed"
  log "RUN_ALL_END"
}
mkdir -p "$OUT"
bash poc/k8s-keda/acquire.sh || { log "ACQUIRE_FAILED"; bash poc/k8s-keda/run.sh pods-terminate; exit 1; }
trap finish EXIT
log "PREWARM"; bash poc/k8s-keda/run.sh prewarm | tee "$OUT/prewarm.txt"
log "VLLM_UP"; MODE=stock bash poc/k8s-keda/run.sh vllm-up || exit 1
log "SCENARIO_A"; bash poc/k8s-keda/run.sh scenario stock keda-only || log "scenario A failed"
log "SCENARIO_B"; bash poc/k8s-keda/run.sh scenario plugin keda-plugin || log "scenario B failed"
