"""Precision showcase: changing load + scheduled precision switches + full telemetry.

Closed-loop clients (same as phase_load.py) follow a load profile; at fixed
times the script POSTs /v1/precision to switch version. Every 0.5 s it samples:

  vLLM /metrics      running, waiting, KV cache usage, generation token counter
  plugin API         target version, active layers per version, migrating, migrations
  GPU (NVML)         SM utilization, memory used, power, temperature, SM clock,
                     PCIe host->GPU and GPU->host throughput
  host (psutil)      total CPU %, CPU % and RSS of the vLLM processes, RAM used

    python3 bench/showcase_load.py --name showcase \
        --phases 4:60,32:90,128:90,4:60 --switches 75:int4,165:fp8,255:high \
        --out results/showcase

Server: VPA_FORMAT=swap VPA_VERSIONS=high,fp8,int4 VPA_MODE=high
        VLLM_PLUGINS=precision_autoscaler,precision_autoscaler_api (see run_showcase.sh).
"""

import argparse
import asyncio
import json
import pathlib
import random
import re
import time

import aiohttp
import psutil

from phase_load import client, parse_phases

METRICS = ("vllm:num_requests_running", "vllm:num_requests_waiting",
           "vllm:kv_cache_usage_perc", "vllm:generation_tokens_total")


class Host:
    """GPU (NVML) + host (psutil) probes. Missing NVML just leaves GPU fields out."""

    def __init__(self):
        self.nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001
            pass
        self.procs: dict[int, psutil.Process] = {}
        psutil.cpu_percent(None)

    def _vllm_procs(self):
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            if p.info["pid"] in self.procs:
                continue
            cmd = " ".join(p.info["cmdline"] or [])
            if "vllm" in cmd.lower() or "VLLM" in (p.info["name"] or ""):
                if "showcase_load" in cmd:
                    continue
                try:
                    p.cpu_percent(None)
                    self.procs[p.info["pid"]] = p
                except psutil.Error:
                    pass
        return list(self.procs.values())

    def sample(self) -> dict:
        row = {}
        if self.nvml:
            n, h = self.nvml, self.gpu
            try:
                row["gpu_util"] = n.nvmlDeviceGetUtilizationRates(h).gpu
                mem = n.nvmlDeviceGetMemoryInfo(h)
                row["vram_used_gib"] = mem.used / 2**30
                row["vram_total_gib"] = mem.total / 2**30
                row["power_w"] = n.nvmlDeviceGetPowerUsage(h) / 1000
                row["temp_c"] = n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)
                row["sm_clock_mhz"] = n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)
                # KB/s over a ~20 ms window: noisy, but shows migration bursts
                row["pcie_h2d_gbs"] = n.nvmlDeviceGetPcieThroughput(h, n.NVML_PCIE_UTIL_RX_BYTES) / 1e6
                row["pcie_d2h_gbs"] = n.nvmlDeviceGetPcieThroughput(h, n.NVML_PCIE_UTIL_TX_BYTES) / 1e6
            except Exception:  # noqa: BLE001
                pass
        row["cpu_total_pct"] = psutil.cpu_percent(None)
        vm = psutil.virtual_memory()
        row["ram_used_gib"] = (vm.total - vm.available) / 2**30
        row["ram_total_gib"] = vm.total / 2**30
        cpu = rss = 0.0
        for p in self._vllm_procs():
            try:
                cpu += p.cpu_percent(None)
                rss += p.memory_info().rss
            except psutil.Error:
                self.procs.pop(p.pid, None)
        row["vllm_cpu_pct"] = cpu          # 100 = one core
        row["vllm_rss_gib"] = rss / 2**30
        row["cpu_cores"] = psutil.cpu_count()
        return row


async def sampler(session, args, t0, samples, stop, host):
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
            async with session.get(f"{args.url}/v1/precision") as r:
                api = await r.json()
            row["target"] = api.get("version")
            row["migrating"] = api.get("migrating")
            row["layers"] = api.get("layers")
            row["versions"] = [{k: v[k] for k in ("name", "active_layers", "ram_gib")}
                               for v in api.get("versions", [])]
            row["n_migrations"] = len(api.get("migrations") or [])
            if api.get("migrations"):
                row["last_migration"] = api["migrations"][-1]
        except Exception:  # noqa: BLE001
            pass
        row.update(host.sample())
        samples.append(row)
        await asyncio.sleep(max(0.0, 0.5 - (time.monotonic() - t0 - row["t"])))


async def switcher(session, args, t0, events):
    for at, version in sorted(args.switch_list):
        await asyncio.sleep(max(0.0, at - (time.monotonic() - t0)))
        t = time.monotonic() - t0
        async with session.post(f"{args.url}/v1/precision", json={"version": version}) as r:
            body = await r.json()
        events.append({"t": t, "version": version, "status": r.status,
                       "post_s": time.monotonic() - t0 - t})
        print(f"t={t:6.1f}s switch -> {version} (HTTP {r.status})", flush=True)


async def main_async(args):
    phases = parse_phases(args.phases)
    records, samples, events = [], [], []
    rng = random.Random(args.seed)
    host = Host()
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as s:
        stop = asyncio.Event()
        t0 = time.monotonic()
        samp = asyncio.create_task(sampler(s, args, t0, samples, stop, host))
        sw = asyncio.create_task(switcher(s, args, t0, events))
        tasks = []
        for i, (c, secs) in enumerate(phases):
            print(f"t={time.monotonic() - t0:6.1f}s load -> {c} users", flush=True)
            deadline = time.monotonic() + secs
            tasks += [asyncio.create_task(client(s, args, i, deadline, random.Random(rng.random()),
                                                 records, t0)) for _ in range(c)]
            await asyncio.sleep(max(0.0, deadline - time.monotonic()))
        await asyncio.gather(*tasks)
        await sw
        await asyncio.sleep(3)
        stop.set()
        await samp
    return phases, records, samples, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--phases", default="4:60,32:90,128:90,4:60")
    ap.add_argument("--switches", default="75:int4,165:fp8,255:high")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/showcase")
    args = ap.parse_args()
    args.switch_list = [(float(a), v) for a, v in (x.split(":") for x in args.switches.split(",") if x)]
    phases, records, samples, events = asyncio.run(main_async(args))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(
        {"name": args.name, "model": args.model, "phases": phases, "switches": events,
         "samples": samples, "records": records}))
    migs = []
    for s in samples:
        m = s.get("last_migration")
        if m and m not in migs:
            migs.append(m)
    for m in migs:
        print("migration", json.dumps(m))
    print(f"saved {len(samples)} samples, {len(records)} requests -> {out / (args.name + '.json')}")


if __name__ == "__main__":
    main()
