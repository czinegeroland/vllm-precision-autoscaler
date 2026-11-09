#!/usr/bin/env python3
"""Minimal RunPod helper (stdlib only): price check, create, wait, stop, terminate.

API key: env var RUNPOD_API_KEY, or the file deploy/runpod/.runpod_api_key
(gitignored). Every pod action is appended to deploy/runpod/ledger.csv.

    python deploy/runpod/rp.py prices
    python deploy/runpod/rp.py create --gpu "NVIDIA GeForce RTX 4090" --cloud COMMUNITY
    python deploy/runpod/rp.py wait <pod_id>        # prints the ssh command
    python deploy/runpod/rp.py terminate <pod_id>
    python deploy/runpod/rp.py list
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
LEDGER = HERE / "ledger.csv"
KEY_FILE = HERE / ".runpod_api_key"
SSH_KEY = HERE / ".keys" / "vpa_ed25519"

IMAGE = "vllm/vllm-openai:v0.32.0-cu129"  # pinned; CUDA 12.9 build runs on more hosts than cu13
WATCH_GPUS = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA H100 PCIe",
]

# Container start: install sshd, authorize our key, keep the container alive.
# vLLM itself is started later over SSH so we control every flag.
BOOTSTRAP = r"""
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq openssh-server git >/dev/null
mkdir -p /root/.ssh /run/sshd
echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
env | grep -E '^(RUNPOD_|HF_|PUBLIC_KEY|VPA_REF)' | sed 's/^/export /' > /etc/profile.d/runpod_env.sh || true
/usr/sbin/sshd
# Get the benchmark scripts from the public repo (no SSH copying needed).
if [ -n "${VPA_REF:-}" ] && [ ! -d /workspace/vpa ]; then
  mkdir -p /workspace && git clone -q --depth 1 --branch "$VPA_REF"     https://github.com/czinegeroland/vllm-precision-autoscaler /workspace/vpa || true
fi
echo BOOTSTRAP_DONE
sleep infinity
"""


def api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8-sig").strip()
    if not key:
        sys.exit(f"No API key: set RUNPOD_API_KEY or create {KEY_FILE}")
    return key


def request(method: str, url: str, body: dict | None = None) -> dict | list | None:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {api_key()}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "vllm-precision-autoscaler/0.1")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} {method} {url}: {e.read().decode(errors='replace')[:800]}")
    return json.loads(raw) if raw.strip() else None


def ledger(event: str, pod: dict) -> None:
    new = not LEDGER.exists()
    with LEDGER.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "event", "pod_id", "name", "gpu", "cost_per_hr"])
        gpu = (pod.get("machine") or {}).get("gpuTypeId") or (pod.get("gpu") or {}).get("id") or ""
        w.writerow([dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), event,
                    pod.get("id", ""), pod.get("name", ""), gpu, pod.get("costPerHr", "")])


def cmd_prices(_args) -> None:
    query = """query { gpuTypes { id displayName memoryInGb secureCloud communityCloud
                securePrice communityPrice
                lowestPrice(input: {gpuCount: 1}) { uninterruptablePrice stockStatus } } }"""
    req = urllib.request.Request(GRAPHQL, data=json.dumps({"query": query}).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {api_key()}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "vllm-precision-autoscaler/0.1")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    if "errors" in data:
        sys.exit(json.dumps(data["errors"], indent=2))
    rows = [g for g in data["data"]["gpuTypes"] if g["id"] in WATCH_GPUS]
    print(f"{'GPU':34} {'VRAM':>5} {'secure $/h':>10} {'community $/h':>13}  stock")
    for g in sorted(rows, key=lambda g: g.get("communityPrice") or g.get("securePrice") or 99):
        lp = g.get("lowestPrice") or {}
        print(f"{g['id']:34} {g['memoryInGb']:>5} {g.get('securePrice') or '-':>10} "
              f"{g.get('communityPrice') or '-':>13}  {lp.get('stockStatus') or '-'}")


def cmd_create(args) -> None:
    pub = SSH_KEY.with_suffix(".pub")
    if not pub.exists():
        sys.exit(f"Missing {pub}; run: ssh-keygen -t ed25519 -N '' -f {SSH_KEY}")
    body = {
        "name": args.name,
        "imageName": IMAGE,
        "cloudType": args.cloud,
        "gpuTypeIds": [args.gpu],
        "gpuCount": 1,
        "minRAMPerGPU": args.min_ram,
        "containerDiskInGb": args.disk,
        "volumeInGb": 0,
        "ports": ["22/tcp", "8000/http", "3000/http"],
        "env": {"PUBLIC_KEY": pub.read_text().strip(), "VPA_REF": args.ref},
        "dockerEntrypoint": ["bash", "-c"],
        "dockerStartCmd": [BOOTSTRAP],
        "allowedCudaVersions": ["12.8", "12.9", "13.0"],
    }
    if args.cloud == "COMMUNITY":
        body["supportPublicIp"] = True
    if args.any_cuda:
        body.pop("allowedCudaVersions")
    pod = request("POST", f"{REST}/pods", body)
    ledger("create", pod)
    print(json.dumps({k: pod.get(k) for k in ("id", "name", "costPerHr", "desiredStatus")}, indent=2))


def get_pod(pod_id: str) -> dict:
    return request("GET", f"{REST}/pods/{pod_id}")


def cmd_wait(args) -> None:
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        pod = get_pod(args.pod_id)
        ip, ports = pod.get("publicIp"), pod.get("portMappings") or {}
        if ip and "22" in ports:
            print(f"ssh -i {SSH_KEY} -p {ports['22']} -o StrictHostKeyChecking=no root@{ip}")
            print(json.dumps({"id": pod["id"], "ip": ip, "ssh_port": ports["22"],
                              "costPerHr": pod.get("costPerHr")}))
            return
        time.sleep(10)
    sys.exit("timed out waiting for public IP / SSH port")


def cmd_status(args) -> None:
    pod = get_pod(args.pod_id)
    print(json.dumps({k: pod.get(k) for k in ("id", "desiredStatus", "publicIp", "portMappings",
                                              "costPerHr", "lastStatusChange")}, indent=2))


def cmd_terminate(args) -> None:
    pod = get_pod(args.pod_id)
    request("DELETE", f"{REST}/pods/{args.pod_id}")
    ledger("terminate", pod)
    print(f"terminated {args.pod_id}")


def cmd_stop(args) -> None:
    pod = get_pod(args.pod_id)
    request("POST", f"{REST}/pods/{args.pod_id}/stop")
    ledger("stop", pod)
    print(f"stopped {args.pod_id}")


def cmd_list(_args) -> None:
    pods = request("GET", f"{REST}/pods") or []
    for p in pods:
        print(p.get("id"), p.get("name"), p.get("desiredStatus"), p.get("costPerHr"))
    if not pods:
        print("no pods")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prices").set_defaults(fn=cmd_prices)
    c = sub.add_parser("create")
    c.add_argument("--gpu", required=True)
    c.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    c.add_argument("--name", default="vpa-dev")
    c.add_argument("--disk", type=int, default=60)
    c.add_argument("--min-ram", type=int, default=32, help="GB of system RAM per GPU")
    c.add_argument("--ref", default="main", help="git branch/tag the pod clones")
    c.add_argument("--any-cuda", action="store_true", help="drop the host CUDA version filter")
    c.set_defaults(fn=cmd_create)
    for name, fn in (("wait", cmd_wait), ("status", cmd_status), ("stop", cmd_stop),
                     ("terminate", cmd_terminate)):
        p = sub.add_parser(name)
        p.add_argument("pod_id")
        if name == "wait":
            p.add_argument("--timeout", type=int, default=900)
        p.set_defaults(fn=fn)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
