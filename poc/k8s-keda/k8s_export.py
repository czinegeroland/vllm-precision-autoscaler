"""Saves the Prometheus series of one burst run (the window recorded by burst_load.py).

    python poc/k8s-keda/k8s_export.py --run results/k8s_burst/keda-only.json

Writes results/k8s_burst/<name>.prom.json: {series name: [{"labels": {...}, "values": [[t, v], ...]}]}
with t in seconds since the run started (same clock as the load generator's records).
"""

import argparse
import json
import pathlib
import urllib.parse
import urllib.request

PER_USER = ("sum(rate(vllm:generation_tokens_total[15s])) / "
            "avg_over_time(sum(vllm:num_requests_running)[15s:2s])")
QUERIES = {
    "per_user_tok_s": PER_USER,
    "throughput_tok_s": "sum(rate(vllm:generation_tokens_total[15s]))",
    "users": "(sum(vllm:num_requests_running) or vector(0)) + (sum(vllm:num_requests_waiting) or vector(0))",
    "users_by_pod": "sum by (pod) (vllm:num_requests_running) + sum by (pod) (vllm:num_requests_waiting)",
    "replicas_scheduled": "count(vpa_proxy_ready)",
    "replicas_ready": "sum(vpa_proxy_ready)",
    "active_bits_by_pod": "vpa_active_bits",
    "target_bits_by_pod": "vpa_ctl_target_bits",
    "ttft_p95_s": "histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[30s])))",
    "kv_cache_by_pod": "vllm:kv_cache_usage_perc",
    "cold_start_s_by_pod": "vpa_agent_last_cold_start_seconds",
}


def query_range(prom, expr, start, end, step):
    qs = urllib.parse.urlencode({"query": expr, "start": start, "end": end, "step": step})
    with urllib.request.urlopen(f"{prom}/api/v1/query_range?{qs}", timeout=60) as r:
        return json.load(r)["data"]["result"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the JSON written by burst_load.py")
    ap.add_argument("--prom", default="http://localhost:30090")
    ap.add_argument("--step", default="2s")
    args = ap.parse_args()
    run_path = pathlib.Path(args.run)
    run = json.loads(run_path.read_text())
    t0, end = run["t0_wall"], run["end_wall"] + 5
    out = {}
    for name, expr in QUERIES.items():
        res = query_range(args.prom, expr, t0 - 10, end, args.step)
        out[name] = [{"labels": s["metric"],
                      "values": [[float(t) - t0, float(v)] for t, v in s["values"] if v not in ("NaN", "+Inf")]}
                     for s in res]
    dest = run_path.with_suffix(".prom.json")
    dest.write_text(json.dumps({"t0_wall": t0, "queries": QUERIES, "series": out}))
    print(f"exported {sum(len(v) for v in out.values())} series -> {dest}")


if __name__ == "__main__":
    main()
