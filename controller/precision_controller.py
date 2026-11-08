#!/usr/bin/env python3
"""Reference external controller for vllm-precision-autoscaler (stdlib only).

vLLM only *executes* precision switches; this separate process *decides*.
It reads vLLM's standard Prometheus metrics and calls the plugin's API:

    signal  = vllm:num_requests_running + vllm:num_requests_waiting
    to FP8  when signal >= --up   for at least --hold-up   seconds
    to BF16 when signal <= --down for at least --hold-down seconds
    never switch again within --dwell seconds of the last switch

Every decision is printed as one JSON line (and appended to --log).

    python3 controller/precision_controller.py --url http://localhost:8000 --up 16 --down 6

A production controller (e.g. the planned C#/.NET service) can replace this; the
contract is just GET /metrics, GET /v1/precision and POST /v1/precision.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request

METRIC = re.compile(r"^(vllm:num_requests_(?:running|waiting))(?:\{[^}]*\})? ([0-9.eE+-]+)$", re.M)


def http(method: str, url: str, body: dict | None = None, timeout: float = 5.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode()
    return raw if not raw.startswith("{") else json.loads(raw)


def load_signal(base: str) -> tuple[float, float]:
    text = http("GET", f"{base}/metrics")
    vals = {"vllm:num_requests_running": 0.0, "vllm:num_requests_waiting": 0.0}
    for name, value in METRIC.findall(text):
        vals[name] += float(value)
    return vals["vllm:num_requests_running"], vals["vllm:num_requests_waiting"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--up", type=float, default=16, help="running+waiting to go FP8")
    ap.add_argument("--down", type=float, default=6, help="running+waiting to go back to BF16")
    ap.add_argument("--hold-up", type=float, default=1.0, help="seconds above --up before switching")
    ap.add_argument("--hold-down", type=float, default=10.0, help="seconds below --down before switching")
    ap.add_argument("--dwell", type=float, default=20.0, help="minimum seconds between switches")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    if args.down >= args.up:
        sys.exit("--down must be < --up")

    log = open(args.log, "a") if args.log else None

    def emit(event: dict) -> None:
        event = {"t": round(time.time(), 3), **event}
        line = json.dumps(event)
        print(line, flush=True)
        if log:
            log.write(line + "\n")
            log.flush()

    state = http("GET", f"{args.url}/v1/precision")
    mode = state["mode"]
    emit({"event": "start", "mode": mode, "up": args.up, "down": args.down})
    last_switch = -1e9
    above_since = below_since = None
    while True:
        try:
            running, waiting = load_signal(args.url)
            state = http("GET", f"{args.url}/v1/precision")  # also advances idle migrations
        except Exception as e:  # noqa: BLE001
            emit({"event": "error", "error": str(e)})
            time.sleep(args.interval)
            continue
        now = time.monotonic()
        signal = running + waiting
        above_since = (above_since or now) if signal >= args.up else None
        below_since = (below_since or now) if signal <= args.down else None
        want = mode
        if mode == "high" and above_since is not None and now - above_since >= args.hold_up:
            want = "fp8"
        elif mode == "fp8" and below_since is not None and now - below_since >= args.hold_down:
            want = "high"
        if want != mode and now - last_switch >= args.dwell:
            reply = http("POST", f"{args.url}/v1/precision", {"mode": want})
            emit({"event": "switch", "from": mode, "to": want, "running": running,
                  "waiting": waiting, "fp8_layers": reply.get("fp8_layers"),
                  "migrating": reply.get("migrating")})
            mode, last_switch = want, now
        elif state.get("migrating") is False and state.get("migrations"):
            last = state["migrations"][-1]
            key = (last["target"], round(last["seconds"], 3))
            if getattr(main, "_last_done", None) != key:
                main._last_done = key
                emit({"event": "migrated", **last})
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
