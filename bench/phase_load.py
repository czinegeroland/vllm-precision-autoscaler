"""Phased load (quiet -> busy -> quiet) with a telemetry timeline.

Closed-loop clients: during each phase, `concurrency` clients send streaming
completions back-to-back (random 512-token prompts, so no prefix-cache hits,
256 output tokens). Every 0.5 s it samples vLLM /metrics and, if available,
the plugin's state over /collective_rpc (server needs VLLM_SERVER_DEV_MODE=1
and the worker extension).

    python3 bench/phase_load.py --name swap-auto --phases 4:45,96:60,4:60 \
        --out results/phases
"""

import argparse
import asyncio
import json
import pathlib
import random
import re
import statistics
import time

import aiohttp

METRICS = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc")


def parse_phases(text):
    return [(int(c), float(s)) for c, s in (p.split(":") for p in text.split(","))]


async def one_request(session, url, model, prompt_len, out_len, rng, phase, records, t0):
    prompt = [rng.randint(1000, 100000) for _ in range(prompt_len)]
    body = {"model": model, "prompt": prompt, "max_tokens": out_len, "temperature": 0.0,
            "ignore_eos": True, "stream": True}
    start = time.monotonic()
    token_times = []
    try:
        async with session.post(f"{url}/v1/completions", json=body) as resp:
            async for raw in resp.content:
                line = raw.decode(errors="ignore").strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                token_times.append(time.monotonic())
        ok = True
    except Exception:  # noqa: BLE001
        ok = False
    records.append({
        "phase": phase, "start": start - t0, "ok": ok, "tokens": len(token_times),
        "ttft": (token_times[0] - start) if token_times else None,
        "e2e": (token_times[-1] - start) if token_times else None,
        "token_times": [t - t0 for t in token_times],
    })


async def client(session, args, phase, deadline, rng, records, t0):
    while time.monotonic() < deadline:
        await one_request(session, args.url, args.model, args.prompt_len, args.output_len,
                          rng, phase, records, t0)


async def sampler(session, args, t0, samples, stop):
    while not stop.is_set():
        row = {"t": time.monotonic() - t0}
        try:
            async with session.get(f"{args.url}/metrics") as r:
                text = await r.text()
            for name in METRICS:
                m = re.search(rf"^{re.escape(name)}(?:{{[^}}]*}})? ([0-9.eE+-]+)$", text, re.M)
                if m:
                    row[name.split(":")[1]] = float(m.group(1))
        except Exception:  # noqa: BLE001
            pass
        try:
            async with session.post(f"{args.url}/collective_rpc", json={"method": "vpa_stats"}) as r:
                if r.status == 200:
                    s = (await r.json())["results"][0]
                    sw = s.get("swap") or {}
                    row.update({"target": sw.get("target"), "fp8_layers": sw.get("fp8_layers"),
                                "layers": sw.get("layers"), "migrating": sw.get("migrating"),
                                "ema": sw.get("ema_tokens_per_step"),
                                "migrations": len(sw.get("migrations") or [])})
                    samples_last = sw.get("migrations")
                    if samples_last:
                        row["last_migration"] = samples_last[-1]
        except Exception:  # noqa: BLE001
            pass
        samples.append(row)
        await asyncio.sleep(0.5)


def summarize(records, samples, phases):
    out = []
    t = 0.0
    for i, (c, secs) in enumerate(phases):
        rs = [r for r in records if r["phase"] == i and r["ok"] and r["ttft"] is not None]
        itl = []
        for r in rs:
            tt = r["token_times"]
            itl += [b - a for a, b in zip(tt, tt[1:])]
        tok = sum(len([x for x in r["token_times"] if t <= x < t + secs]) for r in records)
        q = lambda xs, p: (sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else None)  # noqa: E731
        out.append({
            "phase": i, "concurrency": c, "seconds": secs, "requests": len(rs),
            "output_tok_s": tok / secs,
            "ttft_p50_ms": 1000 * statistics.median([r["ttft"] for r in rs]) if rs else None,
            "ttft_p95_ms": 1000 * q([r["ttft"] for r in rs], 0.95) if rs else None,
            "itl_p50_ms": 1000 * q(itl, 0.50) if itl else None,
            "itl_p99_ms": 1000 * q(itl, 0.99) if itl else None,
            "itl_max_ms": 1000 * max(itl) if itl else None,
        })
        t += secs
    return out


async def main_async(args):
    phases = parse_phases(args.phases)
    records, samples = [], []
    rng = random.Random(args.seed)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as s:
        stop = asyncio.Event()
        t0 = time.monotonic()
        samp = asyncio.create_task(sampler(s, args, t0, samples, stop))
        tasks = []
        for i, (c, secs) in enumerate(phases):
            deadline = time.monotonic() + secs
            tasks += [asyncio.create_task(client(s, args, i, deadline, random.Random(rng.random()),
                                                 records, t0)) for _ in range(c)]
            await asyncio.sleep(max(0.0, deadline - time.monotonic()))
        await asyncio.gather(*tasks)
        await asyncio.sleep(2)
        stop.set()
        await samp
    return phases, records, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--phases", default="4:45,96:60,4:60")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/phases")
    args = ap.parse_args()
    phases, records, samples = asyncio.run(main_async(args))
    summary = summarize(records, samples, phases)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(
        {"name": args.name, "phases": phases, "summary": summary, "samples": samples,
         "records": records}))
    for row in summary:
        print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()}))
    migs = [s.get("last_migration") for s in samples if s.get("last_migration")]
    seen = []
    for m in migs:
        if m not in seen:
            seen.append(m)
    for m in seen:
        print("migration", json.dumps(m))


if __name__ == "__main__":
    main()
