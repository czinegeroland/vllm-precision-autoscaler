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


def per_user(records, end, window=5.0):
    g = gaps_by_time(records)
    return windowed(g, end, window, 1.0, lambda v: 1.0 / statistics.median(v))


def switch_marks(ax, data):
    prev = None
    for s in data["samples"]:
        v = active_version(s)
        if prev and v and v != prev:
            ax.axvline(s["t"], color=INK2, lw=0.9, ls=(0, (3, 3)), zorder=1)
        prev = v or prev


def steps(data, target):
    rows = []
    g = gaps_by_time(data["records"])
    for a, b, users in phase_bounds(data):
        lo = a + 10  # skip the ramp into each step
        vals = [v for x, v in g if lo <= x < b]
        ss = [s for s in data["samples"] if lo <= s["t"] < b]
        vers = [active_version(s) for s in ss if active_version(s)]
        main = max(set(vers), key=vers.count) if vers else None
        tok = 1.0 / statistics.median(vals) if vals else None
        rows.append({"start_s": a, "users": users, "version": VERSION_LABEL.get(main, "-"),
                     "per_user_tok_s": round(tok, 1) if tok else None,
                     "meets_target": bool(tok and tok >= target)})
    return rows


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

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, gridspec_kw={"hspace": 0.45,
                                                                               "height_ratios": [1, 1.6, 1.1]})
    fig.subplots_adjust(top=0.87)
    fig.suptitle("More users, same per-user speed: precision autoscaling in vLLM",
                 x=0.06, ha="left", fontsize=15, fontweight="bold", color=INK, y=0.985)
    fig.text(0.06, 0.935, f"{auto['model'].split('/')[-1]} on one RTX 4090. Background = precision "
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
    ax.set_title("Per-user speed: tokens/s each user receives (median over 5 s; higher is better)")
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
