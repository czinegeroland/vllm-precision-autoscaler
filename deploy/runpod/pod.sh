#!/usr/bin/env bash
# Local (Git Bash) helpers to work with a running pod over SSH.
#
#   export POD_SSH="root@1.2.3.4 -p 12345"     # from: rp.py wait <id>
#   bash deploy/runpod/pod.sh push              # copy repo -> /workspace/vpa
#   bash deploy/runpod/pod.sh run "bash deploy/pod/setup.sh"
#   bash deploy/runpod/pod.sh pull              # copy /workspace/vpa/results -> ./results
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
KEY="$HERE/.keys/vpa_ed25519"
: "${POD_SSH:?set POD_SSH, e.g. 'root@1.2.3.4 -p 12345'}"
# shellcheck disable=SC2086
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -o LogLevel=ERROR -o ServerAliveInterval=30 $POD_SSH)

case "${1:-}" in
  push)
    tar -C "$ROOT" --exclude=./.git --exclude=./results --exclude=./deploy/runpod/.keys \
        --exclude=./deploy/runpod/.runpod_api_key --exclude='*/__pycache__' \
        --exclude=./controller/*/bin --exclude=./controller/*/obj -czf - . \
      | "${SSH[@]}" "mkdir -p /workspace/vpa && tar --no-same-owner -C /workspace/vpa -xzf -"
    echo "pushed $ROOT -> /workspace/vpa" ;;
  pull)
    mkdir -p "$ROOT/results"
    "${SSH[@]}" "cd /workspace/vpa && tar -czf - results" | tar -C "$ROOT" -xzf -
    echo "pulled results -> $ROOT/results" ;;
  run)
    shift; "${SSH[@]}" "cd /workspace/vpa && source deploy/pod/env.sh && $*" ;;
  shell)
    "${SSH[@]}" ;;
  *) echo "usage: pod.sh push|pull|run <cmd>|shell"; exit 2 ;;
esac
