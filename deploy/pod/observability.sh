#!/usr/bin/env bash
# Runs ON the pod: Prometheus + Grafana + node_exporter + nvidia_gpu_exporter as plain
# binaries (the pod is already a container: no Docker). Idempotent: re-running restarts them.
#
#   bash deploy/pod/observability.sh            # install + start
#   bash deploy/pod/observability.sh snapshot   # Prometheus TSDB snapshot -> results/prometheus-snapshot
#
# Grafana: port 3000 (RunPod proxy: https://<pod-id>-3000.proxy.runpod.net), anonymous
# read-only viewers, dashboards provisioned from deploy/observability/grafana.
set -euo pipefail
cd /workspace/vpa
OBS=/workspace/obs
REPO_OBS=/workspace/vpa/deploy/observability
mkdir -p "$OBS/bin" "$OBS/data" "$OBS/logs"

if [ "${1:-}" = snapshot ]; then
  name=$(curl -s -XPOST localhost:9090/api/v1/admin/tsdb/snapshot | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"]["name"])')
  mkdir -p results/prometheus-snapshot
  cp -r "$OBS/data/prometheus/snapshots/$name/." results/prometheus-snapshot/
  echo "snapshot -> results/prometheus-snapshot ($(du -sh results/prometheus-snapshot | cut -f1))"
  exit 0
fi

PROM=3.5.0 NODE=1.9.1 GRAFANA=12.1.0 GPUEXP=1.3.2
fetch() {  # url dir
  [ -d "$OBS/$2" ] && return 0
  echo "downloading $2"
  curl -sSfL "$1" | tar -xz -C "$OBS"
  local d; d=$(ls -d "$OBS"/"$3"* | head -1)
  [ "$d" != "$OBS/$2" ] && mv "$d" "$OBS/$2"
  return 0
}
fetch "https://github.com/prometheus/prometheus/releases/download/v$PROM/prometheus-$PROM.linux-amd64.tar.gz" prometheus prometheus-
fetch "https://github.com/prometheus/node_exporter/releases/download/v$NODE/node_exporter-$NODE.linux-amd64.tar.gz" node_exporter node_exporter-
fetch "https://dl.grafana.com/oss/release/grafana-$GRAFANA.linux-amd64.tar.gz" grafana grafana-
if [ ! -x "$OBS/bin/nvidia_gpu_exporter" ]; then
  echo "downloading nvidia_gpu_exporter"
  curl -sSfL "https://github.com/utkuozdemir/nvidia_gpu_exporter/releases/download/v$GPUEXP/nvidia_gpu_exporter_${GPUEXP}_linux_x86_64.tar.gz" \
    | tar -xz -C "$OBS/bin" nvidia_gpu_exporter
fi

# dashboards: ours + vLLM's official one (its ${DS_PROMETHEUS} placeholder -> our datasource)
DASH="$OBS/dashboards"; mkdir -p "$DASH"
cp "$REPO_OBS/grafana/dashboards/"*.json "$DASH/"
curl -sSfL https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/examples/observability/prometheus_grafana/grafana.json \
  | sed 's/\${DS_PROMETHEUS}/prometheus/g' > "$DASH/vllm-official.json" || echo "vLLM dashboard download failed (ours still works)"
PROV="$OBS/provisioning"; rm -rf "$PROV"; cp -r "$REPO_OBS/grafana/provisioning" "$PROV"
sed -i "s|__DASHBOARDS_DIR__|$DASH|" "$PROV/dashboards/dashboards.yaml"

cat > "$OBS/grafana.ini" <<EOF
[server]
http_port = 3000
[paths]
data = $OBS/data/grafana
logs = $OBS/logs
provisioning = $PROV
[security]
admin_password = $(head -c 18 /dev/urandom | base64 | tr -d '/+=')
[auth.anonymous]
enabled = true
org_role = Viewer
[dashboards]
default_home_dashboard_path = $DASH/precision.json
[analytics]
reporting_enabled = false
check_for_updates = false
check_for_plugin_updates = false
[news]
news_feed_enabled = false
EOF

for p in prometheus node_exporter nvidia_gpu_exporter grafana; do pkill -f "[o]bs/.*$p" 2>/dev/null || true; done
pkill -f "[o]bs/bin/nvidia_gpu_exporter" 2>/dev/null || true
sleep 1
nohup "$OBS/prometheus/prometheus" --config.file="$REPO_OBS/prometheus.yml" \
  --storage.tsdb.path="$OBS/data/prometheus" --web.enable-admin-api \
  > "$OBS/logs/prometheus.log" 2>&1 &
nohup "$OBS/node_exporter/node_exporter" > "$OBS/logs/node_exporter.log" 2>&1 &
nohup "$OBS/bin/nvidia_gpu_exporter" > "$OBS/logs/gpu_exporter.log" 2>&1 &
nohup "$OBS/grafana/bin/grafana" server --homepath "$OBS/grafana" --config "$OBS/grafana.ini" \
  > "$OBS/logs/grafana.log" 2>&1 &

for _ in $(seq 1 30); do
  curl -sf localhost:9090/-/ready >/dev/null && curl -sf localhost:3000/api/health >/dev/null && break
  sleep 2
done
echo "prometheus: $(curl -s localhost:9090/-/ready || echo DOWN)"
echo "grafana:    $(curl -s localhost:3000/api/health | tr -d '\n ' || echo DOWN)"
echo "gpu exporter metrics: $(curl -s localhost:9835/metrics | grep -c '^nvidia_smi_')"
echo "node exporter metrics: $(curl -s localhost:9100/metrics | grep -c '^node_')"
[ -n "${RUNPOD_POD_ID:-}" ] && echo "Grafana: https://${RUNPOD_POD_ID}-3000.proxy.runpod.net"
echo OBS_READY
