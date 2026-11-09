"""Switch precision through the API and time the migration (stdlib only).

    python3 bench/migration_probe.py --mode fp8 --label loaded --out results/migration/x.jsonl

POSTs /v1/precision, then polls GET /v1/precision every --poll seconds (a GET
also advances a pending migration while the engine is idle, like a real
controller would) until the migration is finished, and appends one JSON line
with the plugin's own migration record plus the wall time seen by the client.
"""

import argparse
import json
import time
import urllib.request


def call(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 method="POST" if body else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--mode", required=True, choices=["fp8", "high"])
    ap.add_argument("--label", default="")
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    before = len(call(f"{args.url}/v1/precision").get("migrations", []))
    t0 = time.monotonic()
    call(f"{args.url}/v1/precision", {"mode": args.mode})
    while True:
        s = call(f"{args.url}/v1/precision")
        if not s["migrating"] and len(s.get("migrations", [])) > before:
            break
        if time.monotonic() - t0 > 300:
            raise SystemExit("migration did not finish in 300 s")
        time.sleep(args.poll)
    rec = {"label": args.label, "mode": args.mode, "client_s": round(time.monotonic() - t0, 3),
           **s["migrations"][-1]}
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
