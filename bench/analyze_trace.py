"""Summarize torch-profiler traces (Chrome trace JSON, .json or .json.gz).

    python3 bench/analyze_trace.py results/prefill/trace_stock results/prefill/trace_swap-high

For each trace directory (worker trace only): wall span of the profiled region,
total GPU kernel time and count, GPU busy ratio, the top GPU kernels, the top
CPU ops, and how many memcpy / sync events occurred. Comparing stock vs plugin
shows where extra prefill time comes from (more kernels? idle GPU = CPU-bound?
copies? syncs?).
"""

import collections
import gzip
import json
import pathlib
import sys


def load(path: pathlib.Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def pick_worker_trace(d: pathlib.Path) -> pathlib.Path:
    files = sorted([*d.rglob("*.json.gz"), *d.rglob("*.json")], key=lambda p: p.stat().st_size)
    if not files:
        sys.exit(f"no trace files in {d}")
    return files[-1]  # the worker (GPU) trace is the largest


def summarize(d: pathlib.Path) -> dict:
    path = pick_worker_trace(d)
    ev = [e for e in load(path) if e.get("ph") == "X" and "dur" in e]
    kernels = [e for e in ev if e.get("cat") in ("kernel", "gpu_memset")]
    copies = [e for e in ev if e.get("cat") == "gpu_memcpy"]
    cpu_ops = [e for e in ev if e.get("cat") == "cpu_op"]
    runtime = [e for e in ev if e.get("cat") == "cuda_runtime"]
    if not kernels:
        return {"trace": str(path), "error": "no GPU kernel events"}
    t0 = min(e["ts"] for e in kernels)
    t1 = max(e["ts"] + e["dur"] for e in kernels)
    busy = sum(e["dur"] for e in kernels)
    by_kernel = collections.Counter()
    for e in kernels:
        by_kernel[e["name"][:90]] += e["dur"]
    by_cpu = collections.Counter()
    for e in cpu_ops:
        by_cpu[e["name"][:70]] += e["dur"]
    syncs = [e for e in runtime if "Synchronize" in e["name"] or "EventQuery" in e["name"]]
    launches = [e for e in runtime if "LaunchKernel" in e["name"] or "Launch" in e["name"]]
    return {
        "trace": path.name,
        "gpu_span_ms": (t1 - t0) / 1000,
        "gpu_kernel_ms": busy / 1000,
        "gpu_busy_ratio": busy / max(t1 - t0, 1),
        "kernel_count": len(kernels),
        "launch_calls": len(launches),
        "memcpy_count": len(copies),
        "memcpy_ms": sum(e["dur"] for e in copies) / 1000,
        "sync_calls": len(syncs),
        "sync_ms": sum(e["dur"] for e in syncs) / 1000,
        "top_kernels_ms": [(k, round(v / 1000, 2)) for k, v in by_kernel.most_common(8)],
        "top_cpu_ops_ms": [(k, round(v / 1000, 2)) for k, v in by_cpu.most_common(12)],
    }


def main() -> None:
    for d in sys.argv[1:]:
        s = summarize(pathlib.Path(d))
        print(f"\n=== {d}")
        for k, v in s.items():
            if isinstance(v, list):
                print(f"  {k}:")
                for name, ms in v:
                    print(f"      {ms:9.2f} ms  {name}")
            else:
                print(f"  {k}: {round(v, 3) if isinstance(v, float) else v}")


if __name__ == "__main__":
    main()
