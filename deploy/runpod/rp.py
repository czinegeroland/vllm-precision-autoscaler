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
    for attempt in range(21):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            break
        except urllib.error.HTTPError as e:
            # 429 = hourly rate limit: wait it out (a terminate must not be lost to it)
            if e.code == 429 and attempt < 20:
                print(f"HTTP 429 {method} {url}: rate limited, retry in 60 s", file=sys.stderr)
                time.sleep(60)
                continue
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
        "volumeInGb": args.volume,
        "ports": args.ports.split(","),
        "env": {"PUBLIC_KEY": pub.read_text().strip(), "VPA_REF": args.ref,
                **dict(kv.split("=", 1) for kv in args.env)},
        "dockerEntrypoint": ["bash", "-c"],
        "dockerStartCmd": [BOOTSTRAP],
        "allowedCudaVersions": ["12.8", "12.9", "13.0"],
    }
    if args.volume:
        body["volumeMountPath"] = "/workspace"  # survives a stop (GPU billing ends)
    if args.cloud == "COMMUNITY":
        body["supportPublicIp"] = True
    if args.any_cuda:
        body.pop("allowedCudaVersions")
    try:
        pod = request("POST", f"{REST}/pods", body)
    except urllib.error.URLError as e:  # REST host unreachable (its DNS name has gone missing before)
        print(f"REST API unreachable ({e.reason}); creating through GraphQL", file=sys.stderr)
        pod = graphql_create(body)
    ledger("create", pod)
    print(json.dumps({k: pod.get(k) for k in ("id", "name", "costPerHr", "desiredStatus")},
                     indent=None if args.json else 2))


def graphql_create(body: dict) -> dict:
    """The same pod as the REST create call, through the GraphQL API."""
    inp = {
        "cloudType": body["cloudType"],
        "gpuCount": 1,
        "gpuTypeId": body["gpuTypeIds"][0],
        "name": body["name"],
        "imageName": body["imageName"],
        # GraphQL has no entrypoint field: RunPod reads a JSON start command as entrypoint + cmd
        "dockerArgs": json.dumps({"entrypoint": body["dockerEntrypoint"], "cmd": body["dockerStartCmd"]}),
        "ports": ",".join(body["ports"]),
        "env": [{"key": k, "value": v} for k, v in body["env"].items()],
        "containerDiskInGb": body["containerDiskInGb"],
        "volumeInGb": body["volumeInGb"],
        "minMemoryInGb": body["minRAMPerGPU"],
    }
    for key in ("volumeMountPath", "supportPublicIp", "allowedCudaVersions"):
        if key in body:
            inp[key] = body[key]
    query = """mutation($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) { id name costPerHr desiredStatus machine { gpuTypeId } } }"""
    try:
        pod = graphql(query, {"input": inp})["podFindAndDeployOnDemand"]
    except (RuntimeError, urllib.error.URLError) as e:
        sys.exit(f"GraphQL create failed: {e}")
    if not pod:
        sys.exit("GraphQL create failed: no pod returned (none in stock?)")
    return pod


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
    try:  # GraphQL (no hourly rate limit), shaped like the REST answer
        p = graphql('{ pod(input: {podId: "%s"}) { id desiredStatus costPerHr lastStatusChange '
                    'runtime { ports { ip publicPort privatePort isIpPublic } } } }' % args.pod_id)["pod"]
        ports = [x for x in ((p.get("runtime") or {}).get("ports") or []) if x.get("isIpPublic")]
        pod = {**p, "publicIp": ports[0]["ip"] if ports else None,
               "portMappings": {str(x["privatePort"]): x["publicPort"] for x in ports}}
    except Exception:  # noqa: BLE001
        pod = get_pod(args.pod_id)
    print(json.dumps({k: pod.get(k) for k in ("id", "desiredStatus", "publicIp", "portMappings",
                                              "costPerHr", "lastStatusChange")}, indent=2))


def graphql(query: str, variables: dict | None = None) -> dict:
    payload = {"query": query, **({"variables": variables} if variables else {})}
    req = urllib.request.Request(GRAPHQL, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {api_key()}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "vllm-precision-autoscaler/0.1")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:  # GraphQL validation errors come back as HTTP 400
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from None
    if "errors" in data:
        raise RuntimeError(json.dumps(data["errors"])[:500])
    return data["data"]


def cmd_terminate(args) -> None:
    """GraphQL first: the REST API has an hourly rate limit, and a terminate must never wait."""
    try:
        graphql('mutation { podTerminate(input: {podId: "%s"}) }' % args.pod_id)
    except Exception as e:  # noqa: BLE001
        print(f"GraphQL terminate failed ({e}); trying REST", file=sys.stderr)
        request("DELETE", f"{REST}/pods/{args.pod_id}")
    ledger("terminate", {"id": args.pod_id})
    print(f"terminated {args.pod_id}")


def cmd_stop(args) -> None:
    pod = get_pod(args.pod_id)
    request("POST", f"{REST}/pods/{args.pod_id}/stop")
    ledger("stop", pod)
    print(f"stopped {args.pod_id}")


def cmd_start(args) -> None:
    """Resume a stopped pod (e.g. one that stopped itself; its volume holds results)."""
    pod = get_pod(args.pod_id)
    request("POST", f"{REST}/pods/{args.pod_id}/start")
    ledger("start", pod)
    print(f"started {args.pod_id}")


def cmd_list(_args) -> None:
    try:  # GraphQL: not affected by the REST API's hourly rate limit
        pods = graphql("{ myself { pods { id name desiredStatus costPerHr } } }")["myself"]["pods"]
    except Exception:  # noqa: BLE001
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
    c.add_argument("--volume", type=int, default=0, help="GB of pod volume at /workspace (kept when stopped)")
    c.add_argument("--min-ram", type=int, default=32, help="GB of system RAM per GPU")
    c.add_argument("--ref", default="main", help="git branch/tag the pod clones")
    c.add_argument("--any-cuda", action="store_true", help="drop the host CUDA version filter")
    c.add_argument("--ports", default="22/tcp,8000/http,3000/http",
                   help="exposed ports; /tcp ones get a public IP:port mapping")
    c.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="extra container env var (repeatable)")
    c.add_argument("--json", action="store_true", help="print the result on one line")
    c.set_defaults(fn=cmd_create)
    for name, fn in (("wait", cmd_wait), ("status", cmd_status), ("stop", cmd_stop), ("start", cmd_start),
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
