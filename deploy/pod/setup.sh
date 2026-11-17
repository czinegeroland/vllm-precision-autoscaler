#!/usr/bin/env bash
# Runs ON the pod after the repo is copied to /workspace/vpa.
# Installs the plugin into the image's Python (next to vLLM) and test deps.
set -euo pipefail
cd /workspace/vpa

source deploy/pod/env.sh  # SSH sessions lack the container's Docker ENV

# Image bug (vllm/vllm-openai:v0.32.0-cu129): torchcodec 0.17.0 is a CUDA 13
# build (needs libnvrtc.so.13) inside a CUDA 12.9 image, so importing it raises
# OSError, which vllm.multimodal does not catch. No cu129 build of 0.17 exists;
# vLLM handles a *missing* torchcodec (audio input only), so remove it.
if ! python3 -c "import torchcodec" >/dev/null 2>&1; then
  echo "removing broken torchcodec (CUDA 13 build in a CUDA 12.9 image)"
  python3 -m pip uninstall -y -q torchcodec 2>/dev/null || python3 -m pip uninstall -y -q --break-system-packages torchcodec
fi

# Hosts with an older driver (e.g. 570 = CUDA 12.8) and a GeForce GPU: the
# image's CUDA forward-compat libcuda fails with "Error 804: forward
# compatibility was attempted on non supported HW". Use the host driver.
if ! python3 -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
  echo "CUDA unavailable: disabling the image's forward-compat libcuda"
  grep -l compat /etc/ld.so.conf.d/*.conf 2>/dev/null | xargs -r sed -i '/compat/d'
  ldconfig
fi

python3 -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu ok', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv

pip_install() {
  python3 -m pip install -q "$@" 2>/dev/null || python3 -m pip install -q --break-system-packages "$@"
}
if [ "${VPA_INSTALL:-git}" = local ]; then
  pip_install --no-deps -e plugin                      # development: the checkout on the pod
else
  REF="${VPA_REF:-main}"
  pip_install --no-deps "git+https://github.com/czinegeroland/vllm-precision-autoscaler@${REF}#subdirectory=plugin"
fi
pip_install pytest

python3 - <<'EOF'
from importlib.metadata import entry_points
eps = [e for e in entry_points(group="vllm.general_plugins") if e.name == "precision_autoscaler"]
assert eps, "entry point not registered"
import vllm_precision_autoscaler as v; v.register()
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
assert "precision_autoscaler" in QUANTIZATION_METHODS
import torch; print("plugin OK, op:", torch.ops.precision_autoscaler.linear)
EOF
