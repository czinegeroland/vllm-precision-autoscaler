#!/usr/bin/env bash
# Get N verified GPU machines on RunPod, set them up, and replace any that are not right.
#
#   N=2 bash poc/k8s-keda/acquire.sh
#   N=2 ADOPT="<pod id> ..." bash poc/k8s-keda/acquire.sh   # also take pods that already exist
#       (they must run the pinned image with the bootstrap command, e.g. after a pod edit)
#
# Community hosts are not always what they claim: we got an RTX 5090 listed as a 4090, and a
# 4090 whose CUDA would not start ("CUDA unknown error"). So every new pod is checked with
# nvidia-smi (EXPECT_GPU, EXPECT_MIB) right after SSH comes up, then set up
# (poc/k8s-keda/pod_setup.sh, which also checks that CUDA starts). A pod that fails any check
# is terminated at once and another one is requested.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
PY="${PY:-python}"
N="${N:-2}"
GPU="${GPU:-NVIDIA GeForce RTX 4090}"
EXPECT_GPU="${EXPECT_GPU:-$GPU}"
EXPECT_MIB="${EXPECT_MIB:-24564}"          # 24 GB card (not the 48 GB variant)
CLOUDS="${CLOUDS:-SECURE COMMUNITY}"
DEADLINE=$(( $(date +%s) + ${MAX_MIN:-120} * 60 ))
S="$HERE/.state"; mkdir -p "$S/setup"
PODS="$S/pods.txt"; touch "$PODS"
TOKEN=$(tr -d '\r\n' < "$S/agent_token")
KEY="$ROOT/deploy/runpod/.keys/vpa_ed25519"
SSHO=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15)
log() { echo "[$(date +%H:%M:%S)] $*"; }

reject() {  # reject <id> <reason>
  log "REJECT $1: $2"
  # KEEP_PODS=1: never terminate (e.g. pods the user deployed and wants to keep)
  [ "${KEEP_PODS:-0}" = 1 ] || "$PY" deploy/runpod/rp.py terminate "$1" >/dev/null
  sed -i "/^$1 /d" "$PODS"
}

# One create call (one cloud), then verify the new pod.
candidate() {  # candidate <slot> <cloud>; returns 0 if a pod was accepted
  local slot=$1 cloud=$2 res id
  if ! res=$("$PY" deploy/runpod/rp.py create --gpu "$GPU" --cloud "$cloud" --name "vpa-k8s-$slot" --disk 80 \
       --ports "22/tcp,8000/tcp" --env "AGENT_TOKEN=$TOKEN" --env "MAX_HOURS=${MAX_HOURS:-4}" --json 2>&1); then
    log "slot $slot: none on $cloud ($(echo "$res" | tr -d '\r' | tail -1 | cut -c1-300))"
    return 1
  fi
  id=$(echo "$res" | "$PY" -c "import sys, json; print(json.load(sys.stdin)['id'])")
  log "created $id on $cloud for slot $slot"
  verify "$id"
}

# Wait for SSH and check the GPU. An accepted pod is added to pods.txt and its setup starts
# in the background, so the search for the next one goes on.
verify() {  # verify <pod id>; returns 0 if the pod was accepted
  local id=$1 info ip port gpu
  info=""
  for _ in $(seq 1 30); do   # up to 10 min for the container to start (public IP + ports)
    info=$("$PY" deploy/runpod/rp.py status "$id" 2>/dev/null | "$PY" -c "
import sys, json; p = json.load(sys.stdin); m = p.get('portMappings') or {}
print(p['id'], p.get('publicIp') or '', m.get('22', ''), m.get('8000', ''))" 2>/dev/null)
    [ "$(echo "$info" | wc -w)" = 4 ] && break
    sleep 20
  done
  [ "$(echo "$info" | wc -w)" = 4 ] || { reject "$id" "container not started after 10 min"; return 1; }
  ip=$(echo "$info" | cut -d' ' -f2); port=$(echo "$info" | cut -d' ' -f3)
  # hosts that already failed (CUDA does not start there): BAD_HOSTS="ip ip ..."
  case " ${BAD_HOSTS:-} " in *" $ip "*) reject "$id" "known bad host $ip"; return 1 ;; esac
  gpu=""
  for _ in $(seq 1 40); do
    gpu=$(ssh "${SSHO[@]}" -p "$port" "root@$ip" \
      "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits" 2>/dev/null) && [ -n "$gpu" ] && break
    sleep 15
  done
  [ -n "$gpu" ] || { reject "$id" "SSH never answered"; return 1; }
  case "$gpu" in
    "$EXPECT_GPU, $EXPECT_MIB") ;;
    *) reject "$id" "wrong GPU: $gpu"; return 1 ;;
  esac
  echo "$info" >> "$PODS"
  log "accepted $id: $gpu at $ip; setup started"
  ( if ( export POD_SSH="root@$ip -p $port"
         bash deploy/runpod/pod.sh push && bash deploy/runpod/pod.sh run "KEEP_POD=${KEEP_PODS:-0} bash poc/k8s-keda/pod_setup.sh"
       ) > "$S/setup/$id.log" 2>&1 && grep -q POD_READY "$S/setup/$id.log"; then
      log "READY $id"
    else
      reject "$id" "setup failed: $(grep -E 'CUDA_BROKEN|Error|error' "$S/setup/$id.log" | tail -1)"
    fi ) &
  echo $! >> "$S/setup/pids"
  return 0
}

in_stock() {  # one GraphQL call (not rate limited like REST); "-" = none in stock
  local st
  st=$("$PY" deploy/runpod/rp.py prices 2>/dev/null | grep -F "$GPU " | awk '{print $NF}')
  log "stock check: ${st:-?}"
  [ -n "$st" ] && [ "$st" != "-" ]
}

# RunPod's REST API has an hourly rate limit: one create call per round, clouds alternating,
# RETRY_S between rounds. Accepted pods set up in the background while the search goes on.
: > "$S/setup/pids"
for id in ${ADOPT:-}; do
  grep -q "^$id " "$PODS" || { log "adopting $id"; verify "$id" || true; }
done
round=0
while true; do
  while [ "$(grep -c . "$PODS")" -lt "$N" ]; do
    [ "$(date +%s)" -gt "$DEADLINE" ] && { log "gave up after ${MAX_MIN:-120} min with $(grep -c . "$PODS") pods"; exit 3; }
    if in_stock; then
      set -- $CLOUDS; shift $(( round % $# )); cloud=$1; round=$((round + 1))
      candidate "$(grep -c . "$PODS")" "$cloud" && continue
    fi
    sleep "${RETRY_S:-240}"
  done
  log "have $N machines; waiting for their setup"
  for p in $(cat "$S/setup/pids"); do wait "$p" 2>/dev/null; done
  : > "$S/setup/pids"
  [ "$(grep -c . "$PODS")" -ge "$N" ] && break   # a failed setup removed its pod: search again
done
awk '{print "http://" $2 ":" $4}' "$PODS" > "$S/backends.txt"
rm -f "$S/fake"
log "ALL_READY: $N machines"; cat "$PODS"
