#!/usr/bin/env bash
# Runs ON each RunPod GPU pod (after `pod.sh push`): install the plugin from the pushed
# checkout, download the model, start the agent on port 8000. vLLM itself is NOT started;
# the Kubernetes replica that owns this machine starts it through the agent.
#   MAX_HOURS (default 4): safety net, the pod stops itself after this long.
set -euo pipefail
cd /workspace/vpa
VPA_INSTALL=local bash deploy/pod/setup.sh
source deploy/pod/env.sh
# "CUDA available" is not enough: one host passed that and then failed every allocation
# ("CUDA error: out of memory" with 2 MiB in use). Allocate 1 GiB and multiply.
if ! python3 -c "
import torch
free, total = torch.cuda.mem_get_info()
assert free > 0.9 * total, (free, total)
a = torch.randn(16384, 16384, device='cuda', dtype=torch.bfloat16)
b = (a @ a).float().abs().mean().item()
assert b == b and b > 0
print('CUDA_OK', torch.cuda.get_device_name(0), round(free / 2**30, 1), 'GiB free')
" 2>&1 | tail -1 | tee /tmp/cuda_check.txt | grep -q CUDA_OK; then
  echo "CUDA_BROKEN: $(cat /tmp/cuda_check.txt); replace the pod"
  exit 1
fi
cat /tmp/cuda_check.txt
export HF_HOME=/workspace/hf
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B')" >/dev/null
echo "GPU_MEMORY_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
echo "RAM_GIB=$(free -g | awk '/^Mem:/ {print $2}')"

pkill -f "[d]eploy/k8s/agent.py" 2>/dev/null || true
sleep 1
nohup setsid python3 poc/k8s-keda/agent.py --port 8000 > /workspace/agent.log 2>&1 < /dev/null &

if [ "${KEEP_POD:-0}" != 1 ] && [ ! -f /workspace/self_stop.armed ]; then   # KEEP_POD=1: no timer
  touch /workspace/self_stop.armed
  nohup setsid bash -c "sleep $(( ${MAX_HOURS:-4} * 3600 )); python3 /workspace/vpa/deploy/pod/self_stop.py" \
    > /workspace/self_stop.log 2>&1 < /dev/null &
fi

for _ in $(seq 1 30); do
  curl -sf -H "X-Agent-Token: ${AGENT_TOKEN:-}" localhost:8000/agent/status >/dev/null && break
  sleep 1
done
curl -sf -H "X-Agent-Token: ${AGENT_TOKEN:-}" localhost:8000/agent/status && echo && echo POD_READY
