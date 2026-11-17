# Source me on the pod. SSH sessions don't inherit the container's Docker ENV
# (LD_LIBRARY_PATH, CUDA_HOME, ...); copy it from PID 1.
eval "$(python3 - <<'EOF'
import shlex
skip = {"HOME", "PWD", "SHLVL", "HOSTNAME", "TERM", "_", "PUBLIC_KEY"}
for kv in open("/proc/1/environ", "rb").read().split(b"\0"):
    if b"=" not in kv:
        continue
    k, v = kv.split(b"=", 1)
    k = k.decode()
    if k in skip or not k.replace("_", "").isalnum():
        continue
    print(f"export {k}={shlex.quote(v.decode(errors='replace'))}")
EOF
)"
