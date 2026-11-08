"""Summarize `vllm bench serve` JSON files into a table (and a CSV).

    python bench/summarize.py results/bench/Qwen_Qwen3-4B
"""

import csv
import json
import pathlib
import re
import sys

FIELDS = [
    ("output_throughput", "out tok/s"),
    ("median_tpot_ms", "TPOT p50 ms"),
    ("p95_tpot_ms", "TPOT p95 ms"),
    ("median_ttft_ms", "TTFT p50 ms"),
    ("p95_ttft_ms", "TTFT p95 ms"),
    ("request_goodput", "goodput req/s"),
]


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    rows = []
    for path in sorted(root.glob("*.c*.json")):
        m = re.match(r"(.+)\.c(\d+)\.json$", path.name)
        if not m:
            continue
        data = json.loads(path.read_text())
        row = {"config": m.group(1), "concurrency": int(m.group(2))}
        for key, _ in FIELDS:
            row[key] = data.get(key)
        rows.append(row)
    rows.sort(key=lambda r: (r["concurrency"], r["config"]))

    header = ["conc", "config"] + [label for _, label in FIELDS]
    print(" | ".join(f"{h:>13}" for h in header))
    for r in rows:
        vals = [r["concurrency"], r["config"]] + [
            f"{r[k]:.1f}" if isinstance(r[k], (int, float)) else "-" for k, _ in FIELDS
        ]
        print(" | ".join(f"{str(v):>13}" for v in vals))

    out = root / "summary.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "concurrency"] + [k for k, _ in FIELDS])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
