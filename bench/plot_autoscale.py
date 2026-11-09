"""Article chart for the autoscaling run (bench/run_autoscale.sh output).

    python bench/plot_autoscale.py results/autoscale --target 60 --out docs/img

Writes:
  autoscale-hero.png     users ramp / per-user tokens/s vs target / total throughput,
                         16-bit only vs autoscaled, background = autoscaled precision
  autoscale-steps.csv    per load step: per-user tok/s (median), target met?, version
"""

import argparse
import csv
import json
import pathlib
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_showcase import (BASE, INK, INK2, SURFACE, VERSION_LABEL, active_version,  # noqa: E402
                           gaps_by_time, label_bands, legend, phase_bounds, shade,
                           throughput, version_spans, windowed)


def request_speeds(records, min_tokens=16):
    """[(t_finish, tokens/s)] per request: decode speed the user saw while streaming."""
    out = []
    for r in records:
        tt = r["token_times"]
        if len(tt) >= min_tokens and tt[-1] > tt[0]:
            out.append((tt[-1], (len(tt) - 1) / (tt[-1] - tt[0])))
    out.sort()
    return out


def per_user(records, end, window=10.0):
    return windowed(request_speeds(records), end, window, 1.0, statistics.median)


def switch_marks(ax, data):
    prev = None
    for s in data["samples"]:
        v = active_version(s)
        if prev and v and v != prev:
            ax.axvline(s["t"], color=INK2, lw=0.9, ls=(0, (3, 3)), zorder=1)
        prev = v or prev


def steps(data, target):
    rows = []
    speeds = request_speeds(data["records"])
    for a, b, users in phase_bounds(data):
        lo = a + 10  # skip the ramp into each step
        vals = [v for x, v in speeds if lo <= x < b]
        ss = [s for s in data["samples"] if lo <= s["t"] < b]
        vers = [active_version(s) for s in ss if active_version(s)]
        order = ["high", "fp8", "int4"]
        used = [v for v in order if vers and vers.count(v) / len(vers) >= 0.25]
        tok = statistics.median(vals) if vals else None
        rows.append({"start_s": a, "users": users,
                     "version": "/".join(VERSION_LABEL[v] for v in used) or "-",
                     "per_user_tok_s": round(tok, 1) if tok else None,
                     "meets_target": bool(tok and tok >= target)})
    return rows


def users_chart(base, auto, target, path):
    b_rows, a_rows = steps(base, target), steps(auto, target)
    up = [i for i in range(len(b_rows)) if i == 0 or b_rows[i]["users"] > b_rows[i - 1]["users"]]
    xs = [b_rows[i]["users"] for i in up]
    yb = [b_rows[i]["per_user_tok_s"] for i in up]
    ya = [a_rows[i]["per_user_tok_s"] for i in up]
    va = [a_rows[i]["version"] for i in up]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(xs, yb, color=BASE, ls="--", lw=1.8, marker="o", ms=8, label="16-bit only")
    ax.plot(xs, ya, color=INK, lw=2.2, marker="o", ms=8, label="autoscaled precision")
    for x, y, v in zip(xs, ya, va):
        ax.annotate(v, (x, y), xytext=(0, 10), textcoords="offset points", ha="center",
                    fontsize=9.5, color=INK)
    for x, y in zip(xs, yb):
        ax.annotate(f"{y:.0f}", (x, y), xytext=(0, -16), textcoords="offset points", ha="center",
                    fontsize=9, color=INK2)
    held_b = max([x for x, y in zip(xs, yb) if y >= target], default=0)
    held_a = max([x for x, y in zip(xs, ya) if y >= target], default=0)
    fig.subplots_adjust(top=0.83)
    fig.suptitle("Per-user speed as users grow (tokens/s each user receives, median)",
                 x=0.125, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.97)
    fig.text(0.125, 0.895, f"Target {target:.0f} tok/s per user: 16-bit only holds it up to {held_b} users, "
             f"autoscaled precision up to {held_a}", fontsize=10.5, color=INK2)
    ax.axhline(target, color=INK2, lw=1.2, ls=":", zorder=1)
    ax.text(xs[0], target, f"target {target:.0f}", va="bottom", ha="left", fontsize=9.5,
            color=INK2)
    ax.set_xlabel("concurrent users")
    ax.set_xticks(xs)
    ax.set_ylim(0, max(ya + yb) * 1.18)
    legend(ax)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5, labelcolor=INK2)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--target", type=float, default=60)
    ap.add_argument("--out", default="docs/img")
    args = ap.parse_args()
    run = pathlib.Path(args.run_dir)
    base = json.loads((run / "baseline.json").read_text())
    auto = json.loads((run / "autoscaled.json").read_text())
    end = min(max(s["t"] for s in base["samples"]), max(s["t"] for s in auto["samples"]))
    spans = version_spans(auto["samples"], end)
    vram = next((x.get("vram_total_gib") for x in auto["samples"] if x.get("vram_total_gib")), None)
    gpu_note = f" ({vram:.0f} GB)" if vram else ""

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, gridspec_kw={"hspace": 0.45,
                                                                               "height_ratios": [1, 1.6, 1.1]})
    fig.subplots_adjust(top=0.87)
    fig.suptitle("More users, (almost) the same per-user speed",
                 x=0.06, ha="left", fontsize=15, fontweight="bold", color=INK, y=0.985)
    fig.text(0.06, 0.935, f"{auto['model'].split('/')[-1]} on one RTX 4090{gpu_note}. Background = precision "
             "chosen by the controller; dashed lines = switches", fontsize=10, color=INK2)

    ax = axes[0]
    xs, ys = [], []
    for a, b, c in phase_bounds(auto):
        xs += [a, b]
        ys += [c, c]
    ax.plot(xs, ys, color=INK)
    ax.set_title("Load: concurrent users", pad=22)
    ax.set_ylim(bottom=0)
    label_bands(ax, spans)

    ax = axes[1]
    bx, by = per_user(base["records"], end)
    ax.plot(bx, by, color=BASE, ls="--", lw=1.6, label="16-bit only")
    x, y = per_user(auto["records"], end)
    ax.plot(x, y, color=INK, label="autoscaled precision")
    ax.axhline(args.target, color=INK2, lw=1.2, ls=":", zorder=2)
    ax.text(end, args.target, f" target {args.target:.0f}", va="center", ha="left", fontsize=9.5,
            color=INK2)
    ax.set_title("Per-user speed: tokens/s each user receives while streaming "
                 "(median of requests, 10 s; higher is better)")
    ax.set_ylim(bottom=0)
    legend(ax)

    ax = axes[2]
    bx, by = throughput(base["records"], end, smooth=10)
    ax.plot(bx, by, color=BASE, ls="--", lw=1.6, label="16-bit only")
    x, y = throughput(auto["records"], end, smooth=10)
    ax.plot(x, y, color=INK, label="autoscaled precision")
    ax.set_title("Total output throughput (tokens/s, 10 s average)")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("time (s)")
    legend(ax)

    for ax in axes:
        shade(ax, spans)
        switch_marks(ax, auto)
        ax.set_xlim(0, end)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "autoscale-hero.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    users_chart(base, auto, args.target, out / "autoscale-users.png")

    rows = []
    for name, d in (("16-bit only", base), ("autoscaled", auto)):
        for r in steps(d, args.target):
            rows.append({"run": name, **r})
    with open(out / "autoscale-steps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'users':>5} | {'16-bit only':>16} | {'autoscaled':>22}")
    b_rows, a_rows = steps(base, args.target), steps(auto, args.target)
    for b, a in zip(b_rows, a_rows):
        mark = lambda r: "ok " if r["meets_target"] else "MISS"  # noqa: E731
        print(f"{b['users']:>5} | {b['per_user_tok_s']!s:>7} tok/s {mark(b)} | "
              f"{a['per_user_tok_s']!s:>7} tok/s {mark(a)} {a['version']:>6}")
    for name, rs in (("16-bit only", b_rows), ("autoscaled", a_rows)):
        ok = [r["users"] for r in rs if r["meets_target"]]
        print(f"{name}: most users within target = {max(ok) if ok else 0}")
    print("wrote", out / "autoscale-hero.png")


if __name__ == "__main__":
    main()
