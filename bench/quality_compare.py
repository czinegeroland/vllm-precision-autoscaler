"""Compare GSM8K runs: accuracy with 95% CI and paired disagreement.

    python bench/quality_compare.py results/quality
"""

import json
import math
import pathlib
import sys


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    runs = {}
    for p in sorted(root.glob("*.json")):
        data = json.loads(p.read_text())
        if "summary" in data:
            runs[data["summary"]["name"]] = data
    print(f"{'run':16} {'n':>5} {'acc %':>7} {'95% CI':>15} {'fp8 token share':>16} {'gen s':>7}")
    for name, d in runs.items():
        s = d["summary"]
        lo, hi = wilson(s["correct"], s["n"])
        share = (s.get("vpa_stats") or {}).get("fp8_token_share")
        share_s = f"{share:.3f}" if share is not None else "-"
        print(f"{name:16} {s['n']:>5} {100 * s['accuracy']:>7.2f} "
              f"{100 * lo:>6.1f}-{100 * hi:<6.1f}  {share_s:>15} {s['gen_s']:>7.0f}")

    print("\npaired comparison on shared questions (a_only = right only in A):")
    names = list(runs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = runs[a]["rows"], runs[b]["rows"]
            n = min(len(ra), len(rb))
            a_only = sum(ra[k]["ok"] and not rb[k]["ok"] for k in range(n))
            b_only = sum(rb[k]["ok"] and not ra[k]["ok"] for k in range(n))
            same_text = sum(ra[k]["text"] == rb[k]["text"] for k in range(n))
            print(f"  {a:>16} vs {b:<16} n={n:<5} a_only={a_only:<4} b_only={b_only:<4} "
                  f"identical replies={same_text}/{n}")


if __name__ == "__main__":
    main()
