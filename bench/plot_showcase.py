"""Charts for the precision showcase (bench/showcase_load.py output).

    python bench/plot_showcase.py results/showcase/showcase.json \
        --baseline results/showcase/baseline.json --out docs/img

Writes:
  showcase-hero.png       4 panels: users, throughput, per-user latency, weights in VRAM
  showcase-telemetry.png  every signal: load, serving, GPU, PCIe, host CPU/RAM
  showcase-summary.csv    per-phase numbers (for the article's tables)

Style: small multiples, one y-axis per panel (no dual axes), shared time axis,
the active precision shown as a shaded band behind every panel and labeled in text.
"""

import argparse
import csv
import json
import pathlib
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical slots 1-3 (all-pairs pass, light surface); see dataviz palette.
VERSION_COLOR = {"high": "#2a78d6", "fp8": "#eb6834", "int4": "#1baf7a"}
VERSION_LABEL = {"high": "16-bit", "fp8": "FP8", "int4": "INT4"}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BASE = "#9b9a94"   # baseline series: neutral, recessive

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "font.size": 10.5,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
})


def load(path):
    return json.loads(pathlib.Path(path).read_text()) if path else None


def bins(end, step):
    n = int(end // step) + 1
    return [i * step for i in range(n)]


def throughput(records, end, step=1.0, smooth=5):
    edges = bins(end, step)
    counts = [0] * len(edges)
    for r in records:
        for t in r["token_times"]:
            i = int(t // step)
            if 0 <= i < len(counts):
                counts[i] += 1
    rate = [c / step for c in counts]
    k = max(1, int(smooth / step))
    sm = [statistics.fmean(rate[max(0, i - k + 1): i + 1]) for i in range(len(rate))]
    return edges, sm


def gaps_by_time(records):
    out = []
    for r in records:
        tt = r["token_times"]
        out += [(b, b - a) for a, b in zip(tt, tt[1:])]
    out.sort()
    return out


def windowed(points, end, window, step, fn):
    """points: [(t, value)] sorted by t -> fn(values in (t-window, t]) per step."""
    xs, ys = [], []
    j0 = 0
    for x in bins(end, step):
        while j0 < len(points) and points[j0][0] <= x - window:
            j0 += 1
        vals = []
        j = j0
        while j < len(points) and points[j][0] <= x:
            vals.append(points[j][1])
            j += 1
        if vals:
            xs.append(x)
            ys.append(fn(vals))
    return xs, ys


def p(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def tpot_series(records, end):
    g = gaps_by_time(records)
    return windowed(g, end, 5.0, 1.0, lambda v: 1000 * statistics.median(v))


def ttft_series(records, end):
    pts = sorted((r["start"] + r["ttft"], r["ttft"]) for r in records if r.get("ttft") is not None)
    return windowed(pts, end, 10.0, 1.0, lambda v: 1000 * p(v, 0.95))


def users_series(samples):
    return [s["t"] for s in samples], [s.get("num_requests_running") for s in samples]


def weights_gib(s):
    vs, layers = s.get("versions"), s.get("layers")
    if not vs or not layers:
        return None
    return sum(v["ram_gib"] * v["active_layers"] / layers for v in vs)


def active_version(s):
    vs = s.get("versions") or []
    return max(vs, key=lambda v: v["active_layers"])["name"] if vs else None


def version_spans(samples, end):
    spans, cur, start = [], None, 0.0
    for s in samples:
        v = active_version(s)
        if v and v != cur:
            if cur:
                spans.append((start, s["t"], cur))
            cur, start = v, s["t"]
    if cur:
        spans.append((start, end, cur))
    return spans


def migrations(data):
    """[(t_switch, target, seconds, gib)] matching POSTs with the migration records."""
    recs = []
    for s in data["samples"]:
        m = s.get("last_migration")
        if m and m not in recs:
            recs.append(m)
    out = []
    for ev in data.get("switches", []):
        m = next((r for r in recs if r["target"] == ev["version"]), None)
        if m:
            recs.remove(m)
        out.append((ev["t"], ev["version"], m["seconds"] if m else None, m["gib"] if m else None))
    return out


def shade(ax, spans):
    for a, b, v in spans:
        ax.axvspan(a, b, color=VERSION_COLOR[v], alpha=0.10, lw=0, zorder=0)


def label_bands(ax, spans):
    y = 1.02
    for a, b, v in spans:
        if b - a < 8:
            continue
        ax.text((a + b) / 2, y, VERSION_LABEL[v], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=INK)
        ax.plot([a + 1, b - 1], [y - 0.005, y - 0.005], transform=ax.get_xaxis_transform(),
                color=VERSION_COLOR[v], lw=4, solid_capstyle="butt", clip_on=False)


def mark_switches(ax, migs, text=False):
    for t, v, secs, gib in migs:
        ax.axvline(t, color=INK2, lw=0.9, ls=(0, (3, 3)), zorder=1)
        if text:
            s = f"→ {VERSION_LABEL[v]}" + (f"\n{secs:.2f} s" if secs is not None else "")
            ax.text(t + 2, 0.95, s, transform=ax.get_xaxis_transform(), ha="left", va="top",
                    fontsize=9, color=INK2, zorder=5,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.9, "pad": 1.5})


def legend(ax):
    ax.legend(loc="best", frameon=True, facecolor=SURFACE, edgecolor="none",
              framealpha=0.9, fontsize=9, labelcolor=INK2, handlelength=2.4)


def end_label(ax, xs, ys, text, color):
    if xs:
        ax.annotate(text, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=INK2 if color == BASE else INK)


def phase_bounds(data):
    t, out = 0.0, []
    for c, secs in data["phases"]:
        out.append((t, t + secs, c))
        t += secs
    return out


def hero(data, base, out, compact=False):
    end = max(s["t"] for s in data["samples"])
    spans, migs = version_spans(data["samples"], end), migrations(data)
    n = 3 if compact else 4
    fig, axes = plt.subplots(n, 1, figsize=(10, 8.2) if compact else (11, 10.5), sharex=True,
                             gridspec_kw={"hspace": 0.45})
    if compact:
        fig.subplots_adjust(top=0.86)
    fig.suptitle(f"Precision follows load: {data['model'].split('/')[-1]} on one GPU",
                 x=0.06, ha="left", fontsize=15, fontweight="bold", color=INK, y=0.985)
    fig.text(0.06, 0.945, "Background = precision of the weights in VRAM; dashed lines = "
             "switch requests (migration time below)", fontsize=10, color=INK2)

    ax = axes[0]
    xs, ys = [], []
    for a, b, c in phase_bounds(data):
        xs += [a, b]
        ys += [c, c]
    ax.plot(xs, ys, color=INK)
    ax.set_title("Load: concurrent users sending requests", pad=22)
    ax.set_ylim(bottom=0)
    label_bands(ax, spans)

    ax = axes[1]
    if base:
        bx, by = throughput(base["records"], end, smooth=10)
        ax.plot(bx, by, color=BASE, ls="--", lw=1.6, label="16-bit only (baseline run)")
    x, y = throughput(data["records"], end, smooth=10)
    ax.plot(x, y, color=INK, label="with precision switching")
    ax.set_title("Output throughput (tokens/s, 10 s average)")
    ax.set_ylim(bottom=0)
    legend(ax)

    ax = axes[2]
    if base:
        bx, by = tpot_series(base["records"], end)
        ax.plot(bx, by, color=BASE, ls="--", lw=1.6, label="16-bit only (baseline run)")
    x, y = tpot_series(data["records"], end)
    ax.plot(x, y, color=INK, label="with precision switching")
    legend(ax)
    ax.set_title("Per-user latency: time between tokens (ms, median over 5 s; lower is better)")
    ax.set_ylim(bottom=0)

    if compact:
        ax.set_xlabel("time (s)")
    else:
        ax = axes[3]
        weights_panel(ax, data)
    for i, ax in enumerate(axes):
        shade(ax, spans)
        mark_switches(ax, migs, text=(i == n - 1))
        ax.set_xlim(0, end)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def weights_panel(ax, data):
    if True:
        xs = [s["t"] for s in data["samples"]]
        ax.plot(xs, [weights_gib(s) for s in data["samples"]], color=INK)
        ax.set_title("Model weights in VRAM (GiB): the active precision")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("time (s)")


def telemetry(data, base, out):
    S = data["samples"]
    end = max(s["t"] for s in S)
    spans, migs = version_spans(S, end), migrations(data)
    t = [s["t"] for s in S]

    def col(key, scale=1.0):
        return [(s[key] * scale) if s.get(key) is not None else None for s in S]

    vram_total = next((s["vram_total_gib"] for s in S if s.get("vram_total_gib")), None)
    ram_total = next((s["ram_total_gib"] for s in S if s.get("ram_total_gib")), None)
    cores = next((s["cpu_cores"] for s in S if s.get("cpu_cores")), None)
    panels = [
        ("Users running", [(t, col("num_requests_running"), "running", INK),
                           (t, col("num_requests_waiting"), "waiting (queued)", BASE)], None),
        ("Output throughput (tokens/s, 10 s average)", [(*throughput(data["records"], end, smooth=10), "", INK)], 0),
        ("Time between tokens, median (ms)", [(*tpot_series(data["records"], end), "", INK)], 0),
        ("Time to first token, p95 over 10 s (ms)", [(*ttft_series(data["records"], end), "", INK)], 0),
        ("KV cache in use (%)", [(t, col("kv_cache_usage_perc", 100), "", INK)], (0, 100)),
        ("GPU utilization (%)", [(t, col("gpu_util"), "", INK)], (0, 105)),
        ("GPU power (W)", [(t, col("power_w"), "", INK)], 0),
        ("GPU memory (GiB)" + (f", of {vram_total:.0f}" if vram_total else ""),
         [(t, col("vram_used_gib"), "allocated (vLLM reserves it up front)", BASE),
          (t, [weights_gib(s) for s in S], "weights of the active precision", INK)], 0),
        ("PCIe host → GPU (GB/s): weight migrations", [(t, col("pcie_h2d_gbs"), "", INK)], 0),
        ("CPU: vLLM processes (% of one core)" + (f", host has {cores} cores" if cores else ""),
         [(t, col("vllm_cpu_pct"), "", INK)], 0),
        ("Host RAM (GiB)" + (f", of {ram_total:.0f}" if ram_total else ""),
         [(t, col("vllm_rss_gib"), "vLLM processes (incl. pinned copies of all versions)", INK),
          (t, col("ram_used_gib"), "whole host", BASE)], 0),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.05 * len(panels)), sharex=True,
                             gridspec_kw={"hspace": 0.62})
    fig.subplots_adjust(top=0.955)
    fig.suptitle(f"Telemetry: switching precision under changing load "
                 f"({data['model'].split('/')[-1]})", x=0.06, ha="left", fontsize=15,
                 fontweight="bold", color=INK, y=0.99)
    for i, (ax, (title, series, ylim)) in enumerate(zip(axes, panels)):
        for xs, ys, lab, color in series:
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not pts:
                continue
            ax.plot([a for a, _ in pts], [b for _, b in pts], color=color,
                    lw=1.6 if color == BASE else 2.0, ls="--" if color == BASE else "-")
            ax.lines[-1].set_label(lab or "_")
        if any(lab for *_, lab, _c in series):
            legend(ax)
        ax.set_title(title, pad=22 if i == 0 else 6)
        if isinstance(ylim, tuple):
            ax.set_ylim(*ylim)
        elif ylim == 0:
            ax.set_ylim(bottom=0)
        shade(ax, spans)
        mark_switches(ax, migs, text=(i == 0))
        ax.set_xlim(0, end)
    label_bands(axes[0], spans)
    axes[-1].set_xlabel("time (s)")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summary(data, base, out):
    rows = []
    for name, d in (("switching", data), ("16-bit only", base)):
        if not d:
            continue
        for a, b, users in phase_bounds(d):
            # skip the first 10 s of each phase (ramp) for steady-state numbers
            lo = a + 10
            toks = sum(1 for r in d["records"] for x in r["token_times"] if lo <= x < b)
            g = [v for x, v in gaps_by_time(d["records"]) if lo <= x < b]
            tt = [r["ttft"] for r in d["records"] if r.get("ttft") is not None and lo <= r["start"] < b]
            ss = [s for s in d["samples"] if lo <= s["t"] < b]
            vers = sorted({active_version(s) for s in ss if active_version(s)})
            rows.append({
                "run": name, "phase_s": f"{a:.0f}-{b:.0f}", "users": users,
                "version(s)": "/".join(VERSION_LABEL[v] for v in vers),
                "tok_s": round(toks / (b - lo), 1),
                "tpot_p50_ms": round(1000 * statistics.median(g), 2) if g else "",
                "tpot_p99_ms": round(1000 * p(g, 0.99), 2) if g else "",
                "ttft_p95_ms": round(1000 * p(tt, 0.95), 1) if tt else "",
                "gpu_util": round(statistics.fmean(s.get("gpu_util", 0) for s in ss), 1) if ss else "",
                "power_w": round(statistics.fmean(s.get("power_w", 0) for s in ss), 1) if ss else "",
                "vllm_cpu_pct": round(statistics.fmean(s.get("vllm_cpu_pct", 0) for s in ss), 1) if ss else "",
            })
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    for t, v, secs, gib in migrations(data):
        print(f"switch at {t:.1f}s -> {v}: migration {secs} s, {gib} GiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("showcase")
    ap.add_argument("--baseline")
    ap.add_argument("--out", default="docs/img")
    args = ap.parse_args()
    data, base = load(args.showcase), load(args.baseline)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hero(data, base, out / "showcase-hero.png")
    hero(data, base, out / "showcase-compact.png", compact=True)
    telemetry(data, base, out / "showcase-telemetry.png")
    summary(data, base, out / "showcase-summary.csv")
    print("wrote", out)


if __name__ == "__main__":
    main()
