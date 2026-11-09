"""Generates grafana/dashboards/precision.json (the precision autoscaling dashboard).

    python deploy/observability/make_dashboard.py

Top row answers the article's question: is per-user speed held while users grow?
Then serving, memory, GPU and host panels. Precision colors match the article
charts (16-bit blue, FP8 orange, INT4 aqua).
"""

import json
import pathlib

DS = {"type": "prometheus", "uid": "prometheus"}
BLUE, ORANGE, AQUA, INK, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d8d9da", "#e34948"
SLO = 60
_id = 0


def q(expr, legend, ref="A"):
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref, "range": True}


def panel(title, x, y, w, h, targets, unit="short", kind="timeseries", desc="", **extra):
    global _id
    _id += 1
    p = {
        "id": _id, "type": kind, "title": title, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [q(e, leg, chr(65 + i)) for i, (e, leg) in enumerate(targets)],
        "fieldConfig": {"defaults": {"unit": unit, "custom": {"lineWidth": 2, "fillOpacity": 0,
                                                              "spanNulls": True}},
                        "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "multi"}},
    }
    for k, v in extra.items():
        p[k] = v
    return p


def color(p, name, c, dashed=False):
    props = [{"id": "color", "value": {"mode": "fixed", "fixedColor": c}}]
    if dashed:
        props.append({"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [8, 6]}})
    p["fieldConfig"]["overrides"].append({"matcher": {"id": "byName", "options": name},
                                          "properties": props})
    return p


ITL = "sum by (le) (rate(vllm:inter_token_latency_seconds_bucket[15s]))"
panels = []

# --- row 1: the question ---------------------------------------------------
PER_USER = ("sum(rate(vllm:generation_tokens_total[15s])) / "
            "avg_over_time(sum(vllm:num_requests_running)[15s:2s])")
p = panel("Per-user speed (tokens/s each user receives) vs target", 0, 0, 14, 10,
          [(PER_USER, "per-user tok/s"),
           ("vpa_ctl_per_user_tok_s", "as measured by the controller"),
           (f"vector({SLO})", f"target ({SLO} tok/s)")],
          desc="Generated tokens/s divided by running requests. (vLLM's inter-token latency "
          "histogram is too coarse for this: its buckets jump from 10 ms to 25 ms.)")
color(p, "per-user tok/s", INK)
color(p, "as measured by the controller", "#8a8984", dashed=True)
color(p, f"target ({SLO} tok/s)", RED, dashed=True)
p["fieldConfig"]["defaults"]["min"] = 0
panels.append(p)

p = panel("Precision of the weights in VRAM", 14, 0, 10, 5,
          [("vpa_active_bits", "active"), ("vpa_ctl_target_bits", "controller target")],
          kind="state-timeline", desc="Bits of the version most layers hold (plugin) and the "
          "version the controller asked for.")
p["fieldConfig"]["defaults"]["mappings"] = [{"type": "value", "options": {
    "16": {"text": "16-bit", "color": BLUE, "index": 0},
    "8": {"text": "FP8", "color": ORANGE, "index": 1},
    "4": {"text": "INT4", "color": AQUA, "index": 2}}}]
p["fieldConfig"]["defaults"]["color"] = {"mode": "thresholds"}
p["fieldConfig"]["defaults"]["thresholds"] = {"mode": "absolute", "steps": [{"color": "#555", "value": None}]}
p["options"] = {"showValue": "auto", "mergeValues": True, "rowHeight": 0.8,
                "legend": {"showLegend": False}}
panels.append(p)

p = panel("Users (requests running + waiting)", 14, 5, 10, 5,
          [("sum(vllm:num_requests_running)", "running"),
           ("sum(vllm:num_requests_waiting)", "waiting (queued)")])
color(p, "running", INK)
color(p, "waiting (queued)", ORANGE)
panels.append(p)

# --- row 2: serving ----------------------------------------------------------
p = panel("Output throughput (tokens/s, all users)", 0, 10, 8, 8,
          [("sum(rate(vllm:generation_tokens_total[15s]))", "tokens/s")])
color(p, "tokens/s", INK)
panels.append(p)
p = panel("Time to first token (s)", 8, 10, 8, 8,
          [("histogram_quantile(0.5, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[30s])))", "p50"),
           ("histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[30s])))", "p95")],
          unit="s")
color(p, "p50", INK)
color(p, "p95", ORANGE)
panels.append(p)
p = panel("KV cache in use", 16, 10, 8, 8,
          [("max(vllm:kv_cache_usage_perc)", "KV cache"),
           ("sum(rate(vllm:num_preemptions_total[30s]))", "preemptions/s")], unit="percentunit")
p["fieldConfig"]["overrides"].append({"matcher": {"id": "byName", "options": "preemptions/s"},
                                      "properties": [{"id": "unit", "value": "short"},
                                                     {"id": "custom.axisPlacement", "value": "hidden"}]})
color(p, "KV cache", INK)
panels.append(p)

# --- row 3: memory + migrations ------------------------------------------------
p = panel("GPU memory", 0, 18, 8, 8,
          [("max(nvidia_smi_memory_used_bytes)", "allocated (vLLM reserves it up front)"),
           ("vpa_weights_vram_bytes", "weights of the active precision")], unit="bytes")
color(p, "allocated (vLLM reserves it up front)", "#8a8984", dashed=True)
color(p, "weights of the active precision", INK)
p["fieldConfig"]["defaults"]["min"] = 0
panels.append(p)
p = panel("Precision migrations", 8, 18, 8, 8,
          [("vpa_last_migration_seconds", "last migration ({{version}})"),
           ("vpa_migrating", "migrating now")], unit="s",
          desc="Duration of the most recent weight migration (host RAM -> VRAM).")
panels.append(p)
p = panel("Host RAM", 16, 18, 8, 8,
          [("node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes", "used (whole host)"),
           ("sum(vpa_version_ram_bytes)", "pinned copies of all versions")], unit="bytes")
color(p, "used (whole host)", "#8a8984", dashed=True)
color(p, "pinned copies of all versions", INK)
p["fieldConfig"]["defaults"]["min"] = 0
panels.append(p)

# --- row 4: GPU + CPU -------------------------------------------------------------
p = panel("GPU utilization (nvidia-smi)", 0, 26, 6, 7,
          [("max(nvidia_smi_utilization_gpu_ratio)", "utilization")], unit="percentunit",
          desc="Share of time any kernel ran; near 100% whenever serving. Not a load signal.")
color(p, "utilization", INK)
panels.append(p)
p = panel("GPU power", 6, 26, 6, 7, [("max(nvidia_smi_power_draw_watts)", "power")], unit="watt")
color(p, "power", INK)
panels.append(p)
p = panel("GPU temperature / SM clock", 12, 26, 6, 7,
          [("max(nvidia_smi_temperature_gpu)", "temperature °C"),
           ("max(nvidia_smi_clocks_current_sm_clock_hz) / 1e9", "SM clock GHz")])
panels.append(p)
p = panel("CPU cores busy", 18, 26, 6, 7,
          [('sum(rate(node_cpu_seconds_total{mode!="idle"}[15s]))', "whole host"),
           ('rate(process_cpu_seconds_total{job="vllm"}[15s])', "vLLM API server process")])
color(p, "whole host", "#8a8984", dashed=True)
color(p, "vLLM API server process", INK)
panels.append(p)

dash = {
    "uid": "vpa-precision", "title": "vLLM precision autoscaling", "editable": True,
    "schemaVersion": 39, "tags": ["vllm", "precision"], "refresh": "5s",
    "time": {"from": "now-20m", "to": "now"}, "timezone": "browser",
    "annotations": {"list": [{
        "name": "precision switch", "datasource": DS, "enable": True, "iconColor": ORANGE,
        "expr": "changes(vpa_ctl_target_bits[4s]) > 0", "step": "2s",
        "titleFormat": "precision switch", "textFormat": "controller changed precision"}]},
    "panels": panels,
}
out = pathlib.Path(__file__).parent / "grafana" / "dashboards" / "precision.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(dash, indent=1), encoding="utf-8")
print("wrote", out, len(panels), "panels")
