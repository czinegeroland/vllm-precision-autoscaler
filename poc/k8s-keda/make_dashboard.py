"""Generates poc/k8s-keda/grafana/k8s-burst.json: the Grafana dashboard of the KEDA experiment.

    python poc/k8s-keda/make_dashboard.py

Top: per-user speed over the whole cluster vs the 60 tok/s target, the precision of each
replica, and the replicas (scraped vs ready). Then users per replica, throughput, TTFT.
Colours: BF16 blue, FP8 orange, INT4 W4A8 purple.
"""

import json
import pathlib

DS = {"type": "prometheus", "uid": "prometheus"}
BLUE, ORANGE, PURPLE, GREY, RED = "#2a78d6", "#eb6834", "#7b5cd6", "#8a8984", "#e34948"
SLO = 60
PER_USER = ("sum(rate(vllm:generation_tokens_total[15s])) / "
            "avg_over_time(sum(vllm:num_requests_running)[15s:2s])")
_id = 0


def panel(title, x, y, w, h, targets, unit="short", kind="timeseries", desc=""):
    global _id
    _id += 1
    return {
        "id": _id, "type": kind, "title": title, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"datasource": DS, "expr": e, "legendFormat": leg, "refId": chr(65 + i),
                     "range": True} for i, (e, leg) in enumerate(targets)],
        "fieldConfig": {"defaults": {"unit": unit, "custom": {"lineWidth": 2, "fillOpacity": 0,
                                                              "spanNulls": False}},
                        "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "multi"}},
    }


def color(p, name, c, dashed=False):
    props = [{"id": "color", "value": {"mode": "fixed", "fixedColor": c}}]
    if dashed:
        props.append({"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [8, 6]}})
    p["fieldConfig"]["overrides"].append({"matcher": {"id": "byName", "options": name},
                                          "properties": props})
    return p


panels = []
p = panel("Per-user speed, whole cluster (tokens/s each user receives)", 0, 0, 14, 10,
          [(PER_USER, "per-user tok/s"), (f"vector({SLO})", f"target ({SLO} tok/s)")],
          desc="Generated tokens/s of all replicas divided by the requests they are running.")
color(p, "per-user tok/s", BLUE)
color(p, f"target ({SLO} tok/s)", RED, dashed=True)
p["fieldConfig"]["defaults"]["min"] = 0
panels.append(p)

p = panel("Precision per replica", 14, 0, 10, 5, [("vpa_active_bits", "{{pod}}")],
          kind="state-timeline", desc="Bits of the weights each replica holds in VRAM "
          "(only replicas running the plugin report it).")
p["fieldConfig"]["defaults"]["mappings"] = [{"type": "value", "options": {
    "16": {"text": "BF16", "color": BLUE, "index": 0},
    "8": {"text": "FP8", "color": ORANGE, "index": 1},
    "4": {"text": "INT4", "color": PURPLE, "index": 2}}}]
p["fieldConfig"]["defaults"]["color"] = {"mode": "thresholds"}
p["fieldConfig"]["defaults"]["thresholds"] = {"mode": "absolute",
                                              "steps": [{"color": GREY, "value": None}]}
p["options"] = {"showValue": "auto", "mergeValues": True, "rowHeight": 0.8,
                "legend": {"showLegend": False}}
panels.append(p)

p = panel("Replicas (KEDA)", 14, 5, 10, 5,
          [("count(vpa_proxy_ready)", "scheduled"), ("sum(vpa_proxy_ready)", "ready (vLLM serving)")],
          desc="Scheduled = pods KEDA asked for; ready = their vLLM answers. The gap is the cold start.")
color(p, "scheduled", GREY, dashed=True)
color(p, "ready (vLLM serving)", BLUE)
p["fieldConfig"]["defaults"]["min"] = 0
p["fieldConfig"]["defaults"]["decimals"] = 0
panels.append(p)

p = panel("Users per replica (requests running + waiting)", 0, 10, 8, 8,
          [("sum by (pod) (vllm:num_requests_running) + sum by (pod) (vllm:num_requests_waiting)",
            "{{pod}}"),
           ("(sum(vllm:num_requests_running) or vector(0)) + (sum(vllm:num_requests_waiting) or vector(0))",
            "all users (KEDA's metric)")])
color(p, "all users (KEDA's metric)", GREY, dashed=True)
panels.append(p)

p = panel("Output throughput (tokens/s, all users)", 8, 10, 8, 8,
          [("sum(rate(vllm:generation_tokens_total[15s]))", "tokens/s")])
color(p, "tokens/s", BLUE)
panels.append(p)

p = panel("Time to first token, p95 (s)", 16, 10, 8, 8,
          [("histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[30s])))",
            "p95")], unit="s")
color(p, "p95", ORANGE)
panels.append(p)

dash = {
    "uid": "vpa-k8s-burst", "title": "vLLM on Kubernetes: KEDA + precision autoscaling",
    "editable": True, "schemaVersion": 39, "tags": ["vllm", "keda", "precision"],
    "refresh": "5s", "time": {"from": "now-20m", "to": "now"}, "timezone": "browser",
    "annotations": {"list": [{
        "name": "precision switch", "datasource": DS, "enable": True, "iconColor": ORANGE,
        "expr": "changes(vpa_ctl_target_bits[4s]) > 0", "step": "2s",
        "titleFormat": "precision switch", "textFormat": "{{pod}}"}]},
    "panels": panels,
}
out = pathlib.Path(__file__).parent / "grafana" / "k8s-burst.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(dash, indent=1), encoding="utf-8")
print("wrote", out, len(panels), "panels")
