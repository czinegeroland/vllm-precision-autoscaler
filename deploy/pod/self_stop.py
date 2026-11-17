"""Runs ON a RunPod pod: stop this pod (GPU billing ends, the /workspace volume is kept).

Uses the pod's own RUNPOD_POD_ID and RUNPOD_API_KEY environment variables (RunPod
provides them inside pods), via runpodctl when installed, else the GraphQL API.
    python3 deploy/pod/self_stop.py [--check]   # --check: only report whether it can
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request


def main() -> int:
    pod = os.environ.get("RUNPOD_POD_ID")
    key = os.environ.get("RUNPOD_API_KEY")
    ctl = shutil.which("runpodctl")
    if "--check" in sys.argv:
        print(json.dumps({"pod_id": bool(pod), "api_key": bool(key), "runpodctl": bool(ctl)}))
        return 0 if pod and (key or ctl) else 1
    if not pod:
        print("no RUNPOD_POD_ID")
        return 1
    # GraphQL with the pod's own key first: verified to authenticate inside pods,
    # while runpodctl there has no config ("Unauthorized").
    if key:
        q = {"query": f'mutation {{ podStop(input: {{podId: "{pod}"}}) {{ id desiredStatus }} }}'}
        for attempt in range(5):
            try:
                req = urllib.request.Request(f"https://api.runpod.io/graphql?api_key={key}",
                                             data=json.dumps(q).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = resp.read().decode()
                print("graphql:", body[:300])
                if '"errors"' not in body:
                    return 0
            except Exception as e:  # noqa: BLE001
                print("graphql error:", e)
            time.sleep(10)
    if ctl:
        r = subprocess.run([ctl, "stop", "pod", pod], capture_output=True, text=True)
        print("runpodctl:", r.returncode, r.stdout.strip(), r.stderr.strip())
        return r.returncode
    return 1


if __name__ == "__main__":
    sys.exit(main())
