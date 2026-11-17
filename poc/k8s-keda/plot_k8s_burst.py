"""Chart + numbers for the Kubernetes burst experiment: KEDA only vs KEDA + precision plugin.

    python poc/k8s-keda/plot_k8s_burst.py --dir results/k8s_burst --a keda-only --b keda-plugin

Panels (one time axis):
  1. per-user speed users saw (median decode speed of the answers streaming at that moment),
     both runs, with the 60 tok/s target and the user count of each phase
  2. ready replicas in both runs (from kubectl)
  3. precision of each replica in the plugin run (from Prometheus: vpa_active_bits)
Prints per-phase medians, time below target and completed requests.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BLUE, ORANGE, PURPLE, RED, GREY, INK = "#2a78d6", "#eb6834", "#7b5cd6", "#d03b3b", "#c8c8c8", "#1a1a1a"
BITS_COLOR = {16: BLUE, 8: ORANGE, 4: PURPLE}
plt.rcParams.update({"font.family": "Arial", "font.size": 10.5, "axes.edgecolor": "#888",
                     "axes.labelcolor": "#444", "xtick.color": "#444", "ytick.color": "#444"})


def load(d: pathlib.Path, name: str):
    run = json.loads((d / f"{name}.json").read_text())
    prom_path = d / f"{name}.prom.json"
    prom = json.loads(prom_path.read_text())["series"] if prom_path.exists() else {}
    return run, prom


def complete(r) -> bool:
    """A finished answer: HTTP 200 and the stream ended without an error. Every request
    uses ignore_eos, so a finished answer has exactly OUTPUT_LEN tokens.

    burst_load.py counts SSE events, not tokens. vLLM merges several tokens into one event
    when the reader falls behind (fast INT4 decode, proxies over the internet), so a
    complete answer can arrive in far fewer than OUTPUT_LEN events. Those were marked
    failed in the raw records although nothing failed."""
    return r["status"] == 200 and not r["error"] and r["tokens"] > 0


OUTPUT_LEN = 256


def speed_timeline(run, bin_s=10.0):
    """Median decode speed of the requests whose answer was streaming during each bin."""
    reqs = [(r["first"], r["last"], (OUTPUT_LEN - 1) / (r["last"] - r["first"]))
            for r in run["records"] if complete(r) and r["last"] and r["last"] > r["first"]]
    end = sum(s for _, s in run["phases"])
    xs, ys = [], []
    t = 0.0
    while t < end:
        sp = [s for a, b, s in reqs if a < t + bin_s and b > t]
        if sp:
            xs.append(t + bin_s / 2)
            ys.append(statistics.median(sp))
        t += bin_s
    return xs, ys


def phase_bounds(run):
    out, t = [], 0.0
    for users, secs in run["phases"]:
        out.append((t, t + secs, users))
        t += secs
    return out


def numbers(run, xs, ys, target):
    below = sum(10.0 for y in ys if y < target)
    ok = sum(complete(r) for r in run["records"])
    failed = sum(not complete(r) for r in run["records"])
    return {"requests_ok": ok, "requests_failed": failed, "seconds_below_target": below,
            "phases": run["summary"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/k8s_burst")
    ap.add_argument("--a", default="keda-only", help="run without the plugin")
    ap.add_argument("--b", default="keda-plugin", help="run with the plugin")
    ap.add_argument("--target", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    run_a, _ = load(d, args.a)
    run_b, prom_b = load(d, args.b)
    xa, ya = speed_timeline(run_a)
    xb, yb = speed_timeline(run_b)

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 7.4), dpi=200, sharex=True,
                             gridspec_kw={"height_ratios": [3.2, 1.2, 1.3], "hspace": 0.18})
    fig.subplots_adjust(left=0.11, right=0.98, top=0.9, bottom=0.08)
    ax = axes[0]
    for t0, t1, users in phase_bounds(run_b):
        ax.axvline(t0, color="#e0e0e0", lw=0.8, zorder=0)
        ax.text((t0 + t1) / 2, 1.01, f"{users} users", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, color="#666")
    ax.axhline(args.target, color=RED, ls=(0, (4, 3)), lw=1.2, zorder=1)
    ax.text(0.995, args.target, f"target {args.target:.0f} tok/s", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8.5, color=RED)
    ax.plot(xa, ya, color=BLUE, ls="--", lw=1.8, label="KEDA only (stock vLLM)")
    ax.plot(xb, yb, color=ORANGE, lw=2.2, label="KEDA + precision autoscaler")
    ax.set_ylabel("tokens/s per user")
    ax.set_ylim(0, max(ya + yb + [args.target]) * 1.12)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.09, 0.995),
               ncol=2, frameon=False, fontsize=10, handlelength=2.2)

    ax = axes[1]
    for run, col, ls in ((run_a, BLUE, "--"), (run_b, ORANGE, "-")):
        rs = run["replicas"]
        ax.step([r["t"] for r in rs], [r["ready"] for r in rs], where="post", color=col, ls=ls, lw=1.8)
    rs = run_b["replicas"]
    ax.step([r["t"] for r in rs], [r["desired"] for r in rs], where="post", color="#999", lw=1.0,
            ls=":", label="asked for by KEDA")
    ax.set_ylabel("ready\nreplicas")
    ax.set_ylim(0, max([r["desired"] for r in rs] + [1]) + 0.6)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    pods = sorted({s["labels"].get("pod", "?") for s in prom_b.get("active_bits_by_pod", [])})
    for row, pod in enumerate(pods):
        for s in prom_b["active_bits_by_pod"]:
            if s["labels"].get("pod") != pod:
                continue
            vals = [tv for tv in s["values"] if tv[0] >= 0]
            for (t, v), nxt in zip(vals, vals[1:] + [[vals[-1][0] + 2, None]] if vals else []):
                if nxt[0] - t <= 6:  # gaps = replica not running
                    ax.barh(row, nxt[0] - t, left=t, height=0.7, color=BITS_COLOR.get(int(v), GREY),
                            lw=0)
    ax.set_yticks(range(len(pods)), [p.replace("vllm-", "replica ") for p in pods], fontsize=8.5)
    ax.set_ylim(len(pods) - 0.4, -0.6)
    ax.set_xlabel("seconds")
    ax.legend(handles=[Patch(color=BLUE, label="BF16"), Patch(color=ORANGE, label="FP8"),
                       Patch(color=PURPLE, label="INT4 W4A8")], loc="lower right", ncol=3,
              frameon=False, fontsize=8.5, bbox_to_anchor=(1.0, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    out = pathlib.Path(args.out) if args.out else d / "k8s_burst.png"
    fig.savefig(out, facecolor="white")
    print("wrote", out)

    for name, run, xs, ys in ((args.a, run_a, xa, ya), (args.b, run_b, xb, yb)):
        print(name, json.dumps(numbers(run, xs, ys, args.target), indent=1))


if __name__ == "__main__":
    main()
