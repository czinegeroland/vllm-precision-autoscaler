"""Export Prometheus data to JSON files (for charts after the pod is gone).

    python3 bench/prom_export.py --out results/autoscale/prometheus --minutes 120

Two kinds of output:
  raw/<metric>.json       every series of the metrics below, raw samples at --step
  derived/<name>.json     the dashboard's computed series (per-user tok/s, throughput, ...)
Re-running overwrites the files, so it can be called periodically during a run.
"""

import argparse
import json
import pathlib
import time
import urllib.parse
import urllib.request

RAW = [
    # vLLM
    "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
    "vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:request_success_total",
    "vllm:num_preemptions_total", "vllm:inter_token_latency_seconds_bucket",
    "vllm:time_to_first_token_seconds_bucket", "vllm:e2e_request_latency_seconds_bucket",
    "vllm:request_queue_time_seconds_bucket",
    # plugin
    "vpa_active_bits", "vpa_target_bits", "vpa_version_active_layers", "vpa_weights_vram_bytes",
    "vpa_migrating", "vpa_migrations_total", "vpa_last_migration_seconds", "vpa_version_ram_bytes",
    # controller
    "vpa_ctl_level", "vpa_ctl_target_bits", "vpa_ctl_users", "vpa_ctl_per_user_tok_s",
    "vpa_ctl_slo_tok_s", "vpa_ctl_switches_total",
    # GPU + host
    "nvidia_smi_utilization_gpu_ratio", "nvidia_smi_memory_used_bytes", "nvidia_smi_power_draw_watts",
    "nvidia_smi_temperature_gpu", "nvidia_smi_clocks_current_sm_clock_hz",
    "node_memory_MemAvailable_bytes", "node_memory_MemTotal_bytes", "process_cpu_seconds_total",
    "process_resident_memory_bytes",
]
ITL = "sum by (le) (rate(vllm:inter_token_latency_seconds_bucket[15s]))"
DERIVED = {
    "per_user_tok_s": ("sum(rate(vllm:generation_tokens_total[15s])) / "
                       "avg_over_time(sum(vllm:num_requests_running)[15s:2s])"),
    # coarse (buckets 10 ms -> 25 ms), kept only to show why it is not used:
    "per_user_tok_s_from_itl_histogram": f"1 / histogram_quantile(0.5, {ITL})",
    "throughput_tok_s": "sum(rate(vllm:generation_tokens_total[15s]))",
    "ttft_p95_s": "histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[30s])))",
    "host_cpu_cores_busy": 'sum(rate(node_cpu_seconds_total{mode!="idle"}[15s]))',
    "users": "sum(vllm:num_requests_running) + sum(vllm:num_requests_waiting)",
}


def query_range(base, expr, start, end, step):
    qs = urllib.parse.urlencode({"query": expr, "start": start, "end": end, "step": step})
    with urllib.request.urlopen(f"{base}/api/v1/query_range?{qs}", timeout=60) as r:
        return json.load(r)["data"]["result"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prom", default="http://localhost:9090")
    ap.add_argument("--out", default="results/autoscale/prometheus")
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--step", default="2s")
    args = ap.parse_args()
    end = time.time()
    start = end - args.minutes * 60
    out = pathlib.Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "derived").mkdir(parents=True, exist_ok=True)
    n = 0
    for name in RAW:
        res = query_range(args.prom, name, start, end, args.step)
        if res:
            (out / "raw" / f"{name.replace(':', '_')}.json").write_text(json.dumps(res))
            n += 1
    for name, expr in DERIVED.items():
        res = query_range(args.prom, expr, start, end, args.step)
        (out / "derived" / f"{name}.json").write_text(json.dumps({"expr": expr, "result": res}))
    (out / "export.json").write_text(json.dumps({"exported_at": end, "start": start, "step": args.step,
                                                 "raw_metrics": n}))
    print(f"exported {n} raw metrics + {len(DERIVED)} derived series -> {out}")


if __name__ == "__main__":
    main()
