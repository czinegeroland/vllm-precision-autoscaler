#!/usr/bin/env bash
# Kubernetes + KEDA burst experiment, run from the laptop (Git Bash). See poc/k8s-keda/README.md.
#
#   bash poc/k8s-keda/run.sh tools            # k3d into poc/k8s-keda/.bin (kubectl + Docker must exist)
#   bash poc/k8s-keda/run.sh cluster-up       # k3d cluster, KEDA, Prometheus, Grafana, image
#   bash poc/k8s-keda/run.sh fake-up          # local test: 4 fake GPU machines in the cluster
#   bash poc/k8s-keda/run.sh pods-create      # real: N RunPod RTX 4090 pods (warm GPU machines)
#   bash poc/k8s-keda/run.sh pods-setup       # real: push code, install, download model, start agents
#   bash poc/k8s-keda/run.sh prewarm          # real: start/stop each vLLM mode twice, print cold starts
#   bash poc/k8s-keda/run.sh vllm-up          # replicas + KEDA ScaledObject
#   bash poc/k8s-keda/run.sh scenario stock  keda-only     # scenario 1
#   bash poc/k8s-keda/run.sh scenario plugin keda-plugin   # scenario 2
#   bash poc/k8s-keda/run.sh report           # chart + numbers
#   bash poc/k8s-keda/run.sh pull-logs        # real: agent/controller logs of every pod
#   bash poc/k8s-keda/run.sh pods-terminate   # real: delete the RunPod pods (do not forget)
#   bash poc/k8s-keda/run.sh cluster-down
#   bash poc/k8s-keda/run.sh status
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
STATE="$HERE/.state"; mkdir -p "$STATE"
BIN="$HERE/.bin"; export PATH="$BIN:$PATH"
PY="${PY:-python}"
CLUSTER=vpa
K3D_VERSION="${K3D_VERSION:-v5.8.3}"
KEDA_VERSION="${KEDA_VERSION:-2.17.2}"
N="${N:-4}"                                       # GPU machines = max replicas
GPU="${GPU:-NVIDIA GeForce RTX 4090}"
CLOUD="${CLOUD:-SECURE}"
OUT="${OUT:-results/k8s_burst}"
PHASES="${PHASES:-16:60,56:300,112:300,16:240}"  # users:seconds
PODS="$STATE/pods.txt"                            # id ip ssh_port agent_port (one line per GPU pod)
BACKENDS_FILE="$STATE/backends.txt"
KEY="$ROOT/deploy/runpod/.keys/vpa_ed25519"
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR
          -o ServerAliveInterval=30)

log() { echo "[$(date +%H:%M:%S)] $*"; }
token() {
  [ -s "$STATE/agent_token" ] || "$PY" -c "import secrets; print(secrets.token_hex(24))" > "$STATE/agent_token"
  tr -d '\r\n' < "$STATE/agent_token"
}
is_fake() { [ -f "$STATE/fake" ]; }

# Call agent i: agent_call <i> <GET|POST> <path> [json]. Fake agents are only reachable inside
# the cluster, real ones on the pod's public IP.
agent_call() {
  local i=$1 method=$2 path=$3 body=${4:-}
  if is_fake; then
    kubectl -n fake exec "agent-$i" -- python -c "
import sys, urllib.request
req = urllib.request.Request('http://127.0.0.1:8000$path', method='$method',
      data=(sys.argv[1].encode() if sys.argv[1] else None),
      headers={'X-Agent-Token': '$(token)', 'Content-Type': 'application/json'})
try:
    print(urllib.request.urlopen(req, timeout=90).read().decode())
except urllib.error.HTTPError as e:
    print(e.read().decode())" "$body"
  else
    local url; url=$(sed -n "$((i + 1))p" "$BACKENDS_FILE")
    curl -s -m 90 -X "$method" -H "X-Agent-Token: $(token)" -H "Content-Type: application/json" \
      ${body:+-d "$body"} "$url$path"
  fi
}
agent_state() { agent_call "$1" GET /agent/status | "$PY" -c "import sys, json; print(json.load(sys.stdin)['state'])"; }

set_backends() {
  local mode=$1
  kubectl -n vpa create configmap vllm-config --from-literal=BACKENDS="$(paste -sd, "$BACKENDS_FILE")" \
    --from-literal=MODE="$mode" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
}

secrets() {
  for ns in vpa fake; do
    kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    kubectl -n "$ns" create secret generic agent-token --from-literal=token="$(token)" \
      --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  done
}

ssh_pod() {  # ssh_pod <i> <command>
  local line; line=$(sed -n "$(( $1 + 1 ))p" "$PODS")
  set -- "$2"
  # shellcheck disable=SC2086
  ssh "${SSH_OPTS[@]}" -p "$(echo "$line" | cut -d' ' -f3)" "root@$(echo "$line" | cut -d' ' -f2)" "$1"
}

cmd="${1:-}"; shift || true
case "$cmd" in
tools)
  mkdir -p "$BIN"
  if ! command -v k3d >/dev/null; then
    log "downloading k3d $K3D_VERSION"
    curl -sSfL -o "$BIN/k3d.exe" "https://github.com/k3d-io/k3d/releases/download/$K3D_VERSION/k3d-windows-amd64.exe"
  fi
  k3d version | head -1; kubectl version --client | head -1; docker version --format 'docker {{.Server.Version}}'
  ;;

cluster-up)
  k3d cluster list 2>/dev/null | grep -q "^$CLUSTER " || k3d cluster create "$CLUSTER" --servers 1 --agents 0 \
    --api-port 127.0.0.1:6550 \
    -p "30080:30080@server:0" -p "30090:30090@server:0" -p "30030:30030@server:0" \
    --k3s-arg "--disable=traefik@server:0" \
    --k3s-arg "--kube-controller-manager-arg=horizontal-pod-autoscaler-sync-period=5s@server:0" --wait
  kubectl config use-context "k3d-$CLUSTER" >/dev/null
  "$0" image
  log "KEDA $KEDA_VERSION"
  kubectl apply --server-side -f "https://github.com/kedacore/keda/releases/download/v$KEDA_VERSION/keda-$KEDA_VERSION.yaml" >/dev/null
  "$PY" poc/k8s-keda/make_dashboard.py >/dev/null
  kubectl apply -f poc/k8s-keda/manifests/monitoring.yaml >/dev/null
  kubectl -n monitoring create configmap grafana-dashboards --from-file=poc/k8s-keda/grafana/k8s-burst.json \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  secrets
  kubectl -n keda rollout status deploy/keda-operator --timeout=300s
  kubectl -n monitoring rollout status deploy/prometheus --timeout=300s
  kubectl -n monitoring rollout status deploy/grafana --timeout=300s
  log "cluster ready. Prometheus http://localhost:30090  Grafana http://localhost:30030"
  ;;

image)  # (re)build the replica/fake image and load it into the cluster
  docker build -q -t vpa-k8s:dev -f poc/k8s-keda/Dockerfile . >/dev/null
  k3d image import vpa-k8s:dev -c "$CLUSTER" >/dev/null
  log "image vpa-k8s:dev imported"
  ;;

fake-up)  # local test: fake GPU machines; COLD_START seconds for the plugin's vLLM start
  touch "$STATE/fake"; secrets
  sed "s/COLD_START/${COLD_START:-40}/" poc/k8s-keda/manifests/fake-backends.yaml \
    | sed "s/replicas: 4/replicas: $N/" | kubectl apply -f - >/dev/null
  kubectl -n fake rollout status statefulset/agent --timeout=300s
  : > "$BACKENDS_FILE"
  for i in $(seq 0 $((N - 1))); do echo "http://agent-$i.agent.fake.svc.cluster.local:8000" >> "$BACKENDS_FILE"; done
  log "fake backends:"; cat "$BACKENDS_FILE"
  ;;

pods-create)  # real: N warm GPU machines on RunPod
  rm -f "$STATE/fake"
  [ -s "$PODS" ] && { echo "pods already listed in $PODS; terminate them first"; exit 1; }
  # Retries until N pods exist: every RETRY_S seconds, each cloud in CLOUDS. Stops (keeping the
  # pods it has) if the rest don't appear within GIVEUP_MIN minutes of the first one.
  CLOUDS="${CLOUDS:-$CLOUD}"
  log "creating $N x $GPU (clouds: $CLOUDS), check the price below"
  "$PY" deploy/runpod/rp.py prices | grep -E "GPU|4090" || true
  touch "$STATE/created.jsonl"
  first=""
  while [ "$(wc -l < "$STATE/created.jsonl")" -lt "$N" ]; do
    for cloud in $CLOUDS; do
      i=$(wc -l < "$STATE/created.jsonl"); [ "$i" -ge "$N" ] && break
      if res=$("$PY" deploy/runpod/rp.py create --gpu "$GPU" --cloud "$cloud" --name "vpa-k8s-$i" --disk 80 \
           --ports "22/tcp,8000/tcp" --env "AGENT_TOKEN=$(token)" --env "MAX_HOURS=${MAX_HOURS:-4}" \
           --json 2>&1); then
        echo "$res" >> "$STATE/created.jsonl"; log "created pod $i on $cloud: $res"
        first=${first:-$(date +%s)}
        continue
      fi
      echo "$res" | grep -q "no instances currently available" || log "create failed: $res"
    done
    [ "$(wc -l < "$STATE/created.jsonl")" -ge "$N" ] && break
    if [ -n "$first" ] && [ $(( $(date +%s) - first )) -gt $(( ${GIVEUP_MIN:-30} * 60 )) ]; then
      log "only $(wc -l < "$STATE/created.jsonl") of $N pods after ${GIVEUP_MIN:-30} min; they keep running"
      exit 3
    fi
    sleep "${RETRY_S:-120}"
  done
  : > "$PODS"
  while read -r line; do
    id=$(echo "$line" | "$PY" -c "import sys, json; print(json.load(sys.stdin)['id'])")
    "$PY" deploy/runpod/rp.py wait "$id" >/dev/null
    for _ in $(seq 1 30); do
      info=$("$PY" deploy/runpod/rp.py status "$id" | "$PY" -c "
import sys, json; p = json.load(sys.stdin); m = p.get('portMappings') or {}
print(p['id'], p.get('publicIp') or '', m.get('22', ''), m.get('8000', ''))")
      [ "$(echo "$info" | wc -w)" = 4 ] && break
      sleep 10
    done
    echo "$info" | tee -a "$PODS"
  done < "$STATE/created.jsonl"
  awk '{print "http://" $2 ":" $4}' "$PODS" > "$BACKENDS_FILE"
  rm -f "$STATE/created.jsonl"
  log "pods: $(wc -l < "$PODS"); backends in $BACKENDS_FILE"
  ;;

pods-setup)  # real: push this checkout to every pod and start the agents (in parallel)
  mkdir -p "$STATE/setup"
  for i in $(seq 0 $(( $(wc -l < "$PODS") - 1 ))); do
    (
      line=$(sed -n "$((i + 1))p" "$PODS")
      export POD_SSH="root@$(echo "$line" | cut -d' ' -f2) -p $(echo "$line" | cut -d' ' -f3)"
      for _ in $(seq 1 30); do ssh_pod "$i" true 2>/dev/null && break; sleep 10; done
      bash deploy/runpod/pod.sh push
      bash deploy/runpod/pod.sh run "bash poc/k8s-keda/pod_setup.sh"
    ) > "$STATE/setup/pod$i.log" 2>&1 &
  done
  wait
  for i in $(seq 0 $(( $(wc -l < "$PODS") - 1 ))); do
    echo "pod $i: $(grep -E 'GPU_MEMORY_MIB|RAM_GIB|POD_READY' "$STATE/setup/pod$i.log" | tr '\n' ' ')"
    mem=$(grep -oE "GPU_MEMORY_MIB=[0-9]+" "$STATE/setup/pod$i.log" | cut -d= -f2 || true)
    [ -n "$mem" ] && [ "$mem" -gt 30000 ] && echo "  WARNING: pod $i has a ${mem} MiB GPU (48 GB 4090?): KV cache differs"
  done
  ;;

prewarm)  # start/stop each mode twice on every machine (fills compile caches), print cold starts
  for i in $(seq 0 $((N - 1))); do
    (
      for mode in stock plugin stock plugin; do
        agent_call "$i" POST /agent/start "{\"mode\": \"$mode\"}" >/dev/null
        for _ in $(seq 1 600); do
          s=$(agent_state "$i"); [ "$s" = ready ] || [ "$s" = failed ] && break; sleep 2
        done
        echo "machine $i $mode: $(agent_call "$i" GET /agent/status | "$PY" -c \
          "import sys, json; s = json.load(sys.stdin); print(s['state'], s['cold_start_s'], 's', s['error'])")"
        agent_call "$i" POST /agent/stop >/dev/null
      done
    ) &
  done
  wait
  ;;

vllm-up)
  set_backends "${MODE:-stock}"
  sed "s/maxReplicaCount: 4 /maxReplicaCount: $N /" poc/k8s-keda/manifests/vllm.yaml | kubectl apply -f - >/dev/null
  log "waiting for vllm-0 (cold start)"
  kubectl -n vpa wait --for=condition=Ready pod/vllm-0 --timeout=900s
  ;;

scenario)  # scenario <stock|plugin> <name>
  mode=${1:?mode stock|plugin}; name=${2:?name}
  log "== $name: reset to 1 replica, mode $mode"
  kubectl -n vpa annotate scaledobject vllm autoscaling.keda.sh/paused-replicas=1 --overwrite >/dev/null
  for _ in $(seq 1 120); do
    n=$(kubectl -n vpa get pods -l app=vllm --no-headers 2>/dev/null | wc -l)
    [ "$n" -le 1 ] && break; sleep 2
  done
  for i in $(seq 1 $((N - 1))); do agent_call "$i" POST /agent/stop >/dev/null; done
  set_backends "$mode"
  kubectl -n vpa delete pod vllm-0 --wait=true >/dev/null
  sleep 3
  log "waiting for vllm-0 to serve in $mode mode"
  kubectl -n vpa wait --for=condition=Ready pod/vllm-0 --timeout=900s >/dev/null
  sleep "${SETTLE:-20}"
  kubectl -n vpa annotate scaledobject vllm autoscaling.keda.sh/paused-replicas- >/dev/null
  sleep 5
  # record everything Kubernetes does during the run (pods disappear on scale-down, and
  # their logs with them): events, HPA/ScaledObject state, every replica's log, KEDA's log
  K8S="$OUT/$name.k8s"; mkdir -p "$K8S"
  kubectl get events -A -w -o wide > "$K8S/events.txt" 2>&1 &
  watchers=$!
  (
    while true; do
      for p in $(kubectl -n vpa get pods -l app=vllm -o name 2>/dev/null); do
        p=${p#pod/}
        uid=$(kubectl -n vpa get pod "$p" -o jsonpath='{.metadata.uid}' 2>/dev/null) || continue
        [ -z "$uid" ] || [ -f "$K8S/following.$p.$uid" ] && continue
        touch "$K8S/following.$p.$uid"
        kubectl -n vpa logs -f "$p" --timestamps >> "$K8S/replica-$p-$uid.log" 2>&1 &
        echo $! >> "$K8S/pids"
      done
      echo "$(date +%s) $(kubectl -n vpa get hpa keda-hpa-vllm -o jsonpath='{.status.currentMetrics[0].external.current.averageValue} {.status.currentReplicas} {.status.desiredReplicas}' 2>/dev/null)" >> "$K8S/hpa.txt"
      sleep 2
    done
  ) &
  follower=$!
  log "== $name: load $PHASES"
  "$PY" poc/k8s-keda/burst_load.py --name "$name" --phases "$PHASES" --out "$OUT" 2>&1 | tee "$OUT/$name.load.log"
  sleep 5
  kill "$follower" "$watchers" 2>/dev/null || true
  [ -f "$K8S/pids" ] && kill $(cat "$K8S/pids") 2>/dev/null; rm -f "$K8S/pids"
  kubectl -n keda logs deploy/keda-operator --since=1h --timestamps > "$K8S/keda-operator.log" 2>&1 || true
  kubectl -n vpa get scaledobject,hpa,statefulset,pods -o yaml > "$K8S/objects.yaml" 2>&1 || true
  rm -f "$K8S"/following.*
  "$PY" poc/k8s-keda/k8s_export.py --run "$OUT/$name.json"
  ;;

save)  # everything else: full Prometheus TSDB snapshot + raw export, GPU machine logs, config
  mkdir -p "$OUT/prometheus-snapshot" "$OUT/config"
  "$PY" bench/prom_export.py --prom http://localhost:30090 --out "$OUT/prometheus" --minutes "${MINUTES:-240}" || true
  snap=$(curl -s -XPOST localhost:30090/api/v1/admin/tsdb/snapshot | "$PY" -c "import sys, json; print(json.load(sys.stdin)['data']['name'])")
  prom=$(kubectl -n monitoring get pod -l app=prometheus -o jsonpath='{.items[0].metadata.name}')
  # MSYS_NO_PATHCONV: Git Bash would rewrite /prometheus/... into C:/Program Files/Git/...
  MSYS_NO_PATHCONV=1 kubectl -n monitoring exec "$prom" -- tar -C "/prometheus/snapshots/$snap" -czf - . \
    | tar -C "$OUT/prometheus-snapshot" -xzf -
  log "prometheus snapshot -> $OUT/prometheus-snapshot ($(du -sh "$OUT/prometheus-snapshot" | cut -f1))"
  cp poc/k8s-keda/grafana/k8s-burst.json poc/k8s-keda/manifests/*.yaml poc/k8s-keda/controller.yaml "$OUT/config/"
  is_fake || "$0" pull-logs
  [ -s "$PODS" ] && cut -d' ' -f1 "$PODS" > "$OUT/config/runpod_pod_ids.txt"
  cp "$STATE"/*.log "$STATE/setup"/*.log "$OUT/config/" 2>/dev/null || true
  ;;

report)
  "$PY" poc/k8s-keda/plot_k8s_burst.py --dir "$OUT" --a "${A:-keda-only}" --b "${B:-keda-plugin}"
  ;;

pull-logs)
  for i in $(seq 0 $(( $(wc -l < "$PODS") - 1 ))); do
    mkdir -p "$OUT/machine$i"
    ssh_pod "$i" "cd /workspace && tar -czf - agent-logs agent.log" | tar -C "$OUT/machine$i" -xzf - || true
  done
  log "logs -> $OUT/machine*/"
  ;;

pods-terminate)
  while read -r id _; do "$PY" deploy/runpod/rp.py terminate "$id"; done < "$PODS"
  mv "$PODS" "$STATE/pods.terminated.$(date +%s).txt"
  "$PY" deploy/runpod/rp.py list
  ;;

cluster-down)
  k3d cluster delete "$CLUSTER"
  rm -f "$STATE/fake"
  ;;

status)
  kubectl -n vpa get pods -o wide 2>/dev/null || true
  kubectl -n vpa get scaledobject,hpa 2>/dev/null || true
  for i in $(seq 0 $((N - 1))); do echo "machine $i: $(agent_call "$i" GET /agent/status 2>/dev/null | cut -c1-160)"; done
  ;;

*)
  sed -n '2,20p' "$0"; exit 2 ;;
esac
