"""Burst load against the Kubernetes Service, with a record of every request and of the replicas.

Closed-loop users (as in phase_load.py): during each phase `users` clients send streaming
completions back to back (random 512-token prompts, 256 output tokens). Every request opens
a new connection, so the Service spreads requests over the replicas that are Ready at that
moment (keep-alive connections would stay pinned to the first replica). Every 2 s it also
records the StatefulSet's desired and ready replicas with kubectl.

    python poc/k8s-keda/burst_load.py --name keda-only --phases 16:60,56:300,112:300,16:240 \
        --out results/k8s_burst
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import random
import statistics
import subprocess
import time

import aiohttp


def parse_phases(text):
    return [(int(u), float(s)) for u, s in (p.split(":") for p in text.split(","))]


async def one_request(session, args, rng, phase, records, t0):
    prompt = [rng.randint(1000, 100000) for _ in range(args.prompt_len)]
    body = {"model": args.model, "prompt": prompt, "max_tokens": args.output_len,
            "temperature": 0.0, "ignore_eos": True, "stream": True}
    start = time.monotonic()
    first = last = None
    n = 0
    replica = status = None
    err = None
    try:
        async with session.post(f"{args.url}/v1/completions", json=body) as resp:
            replica, status = resp.headers.get("X-Replica"), resp.status
            if resp.status == 200:
                buf = b""
                async for chunk in resp.content.iter_any():
                    now = time.monotonic()
                    buf += chunk
                    *events, buf = buf.split(b"\n\n")
                    for ev in events:
                        if ev.startswith(b"data:") and not ev.endswith(b"[DONE]"):
                            n += 1
                            first = first or now
                            last = now
            else:
                err = (await resp.text())[:200]
    except Exception as e:  # noqa: BLE001
        err = repr(e)[:200]
    ok = status == 200 and n >= args.output_len
    records.append({"phase": phase, "start": start - t0, "ok": ok, "status": status, "replica": replica,
                    "tokens": n, "first": None if first is None else first - t0,
                    "last": None if last is None else last - t0, "error": err})
    if not ok:
        await asyncio.sleep(0.5)  # don't spin on a failing Service


async def user(i, session, args, load, end, rng, records, t0):
    """User i is active while the current phase has more than i users. One pool for all
    phases: going from 16 to 64 users adds 48, it doesn't replace 16 with 64 (which would
    briefly put 80 requests in flight and make KEDA over-scale)."""
    while time.monotonic() < end:
        if i < load["users"]:
            await one_request(session, args, rng, load["phase"], records, t0)
        else:
            await asyncio.sleep(0.2)


async def replica_sampler(args, t0, rows, stop):
    while not stop.is_set():
        t = time.monotonic() - t0
        try:
            proc = await asyncio.create_subprocess_exec(
                "kubectl", "-n", args.namespace, "get", "statefulset", args.statefulset, "-o", "json",
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            out, _ = await proc.communicate()
            sts = json.loads(out)
            rows.append({"t": t, "desired": sts["spec"].get("replicas", 0),
                         "ready": sts["status"].get("readyReplicas", 0)})
        except Exception:  # noqa: BLE001
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass


async def main_async(args):
    phases = parse_phases(args.phases)
    records, replicas = [], []
    rng = random.Random(args.seed)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    conn = aiohttp.TCPConnector(limit=0, force_close=True)  # new connection per request
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        t0 = time.monotonic()
        t0_wall = time.time()
        stop = asyncio.Event()
        sampler = asyncio.create_task(replica_sampler(args, t0, replicas, stop))
        load = {"users": 0, "phase": 0}
        end = t0 + sum(s for _, s in phases)
        tasks = [asyncio.create_task(user(i, session, args, load, end, random.Random(rng.random()),
                                          records, t0)) for i in range(max(u for u, _ in phases))]
        for i, (users, secs) in enumerate(phases):
            deadline = time.monotonic() + secs
            load.update(users=users, phase=i)
            print(f"[{time.monotonic() - t0:6.0f} s] phase {i}: {users} users for {secs:.0f} s", flush=True)
            while time.monotonic() < deadline:
                await asyncio.sleep(min(10.0, max(0.0, deadline - time.monotonic())))
                done = [r for r in records if r["ok"]]
                rep = replicas[-1] if replicas else {}
                print(f"[{time.monotonic() - t0:6.0f} s] requests ok {len(done)}, failed "
                      f"{sum(not r['ok'] for r in records)}, replicas ready {rep.get('ready')}/"
                      f"{rep.get('desired')}", flush=True)
        await asyncio.gather(*tasks)
        end_wall = time.time()
        stop.set()
        await sampler
    return phases, records, replicas, t0_wall, end_wall


def decode_speed(r):
    """Tokens/s one user saw while their answer streamed (first token excluded)."""
    if not r["ok"] or r["last"] is None or r["last"] <= r["first"]:
        return None
    return (r["tokens"] - 1) / (r["last"] - r["first"])


def summarize(phases, records):
    out, t = [], 0.0
    for i, (users, secs) in enumerate(phases):
        rs = [r for r in records if r["phase"] == i]
        sp = [s for s in (decode_speed(r) for r in rs) if s]
        ttft = sorted(r["first"] - r["start"] for r in rs if r["ok"])
        out.append({"phase": i, "users": users, "seconds": secs, "requests_ok": len(sp),
                    "requests_failed": sum(not r["ok"] for r in rs),
                    "per_user_tok_s_median": statistics.median(sp) if sp else None,
                    "ttft_p95_s": ttft[int(0.95 * (len(ttft) - 1))] if ttft else None})
        t += secs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--url", default="http://localhost:30080")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--phases", default="16:60,56:300,112:300,16:240")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--namespace", default="vpa")
    ap.add_argument("--statefulset", default="vllm")
    ap.add_argument("--out", default="results/k8s_burst")
    args = ap.parse_args()
    phases, records, replicas, t0_wall, end_wall = asyncio.run(main_async(args))
    summary = summarize(phases, records)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(
        {"name": args.name, "phases": phases, "t0_wall": t0_wall, "end_wall": end_wall,
         "summary": summary, "replicas": replicas, "records": records}))
    for row in summary:
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}))
    print(f"saved {out / (args.name + '.json')}")


if __name__ == "__main__":
    main()
