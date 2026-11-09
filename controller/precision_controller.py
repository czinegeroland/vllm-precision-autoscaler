#!/usr/bin/env python3
"""Precision controller for vllm-precision-autoscaler.

vLLM + the plugin only *execute* precision switches; this process *decides*.
Goal: keep per-user tokens/s at or above a target while the number of users
grows, by stepping down a precision ladder (e.g. 16-bit -> FP8 -> INT4), and
step back up when load falls so quiet periods get full quality.

Signals (from vLLM's own Prometheus endpoint, GET /metrics):
    users        = vllm:num_requests_running + vllm:num_requests_waiting
    per-user tok/s = 1 / median(vllm:inter_token_latency_seconds) over `window_s`

Policy (config file, see controller/config.example.yaml):
    ladder: [high, fp8, int4]               index = level (0 = best quality)
    step_down_users: [24, 40]               go below level i when users >= [i]
                                            (jumps several levels if load is far above)
    step_up_users:   [16, 32]               return to level i when users <= [i]
    hold_down_s / hold_up_s                 how long a condition must hold
    slo_tok_s, slo_hold_s                   guard: if per-user tok/s stays below the
                                            target, step down one more level
    dwell_s                                 minimum time between switches

Actions: POST /v1/precision {"version": ...}. Every decision is printed as one
JSON line (and appended to --log). With prometheus_client installed, the
controller's own state is served on --metrics-port for Prometheus.

    python3 controller/precision_controller.py --config controller/config.example.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from dataclasses import dataclass, field

GAUGE_RE = re.compile(r"^(vllm:num_requests_(?:running|waiting))(?:\{[^}]*\})? ([0-9.eE+-]+)$", re.M)
BUCKET_RE = re.compile(r'^vllm:inter_token_latency_seconds_bucket\{([^}]*)\} ([0-9.eE+-]+)$', re.M)
LE_RE = re.compile(r'le="([^"]+)"')


@dataclass
class Config:
    url: str = "http://localhost:8000"
    ladder: list = field(default_factory=lambda: ["high", "fp8", "int4"])
    step_down_users: list = field(default_factory=lambda: [24, 40])
    step_up_users: list = field(default_factory=lambda: [16, 32])
    hold_down_s: float = 2.0
    hold_up_s: float = 20.0
    dwell_s: float = 10.0
    slo_tok_s: float = 60.0
    slo_hold_s: float = 8.0
    window_s: float = 10.0
    interval_s: float = 0.5

    def validate(self) -> None:
        n = len(self.ladder) - 1
        if len(self.step_down_users) != n or len(self.step_up_users) != n:
            raise ValueError(f"need {n} step_down_users and step_up_users for {len(self.ladder)} versions")
        for i in range(n):
            if self.step_up_users[i] >= self.step_down_users[i]:
                raise ValueError(f"step_up_users[{i}] must be < step_down_users[{i}] (hysteresis)")
        if sorted(self.step_down_users) != list(self.step_down_users):
            raise ValueError("step_down_users must increase along the ladder")


def load_config(path: str | None, overrides: dict) -> Config:
    data = {}
    if path:
        text = open(path, encoding="utf-8").read()
        try:
            import yaml  # vLLM depends on PyYAML
            data = yaml.safe_load(text) or {}
        except ImportError:
            data = json.loads(text)
    data.update({k: v for k, v in overrides.items() if v is not None})
    cfg = Config(**data)
    cfg.validate()
    return cfg


# -- pure policy (unit-tested in plugin/tests/test_controller.py) -------------
class Policy:
    def __init__(self, cfg: Config, level: int = 0):
        self.cfg, self.level = cfg, level
        self.down_since = self.up_since = self.slow_since = None
        self.last_switch = -1e9

    def desired_down(self, users: float) -> int:
        """Deepest level whose step-down threshold the load reaches."""
        return sum(1 for t in self.cfg.step_down_users if users >= t)

    def decide(self, now: float, users: float, per_user_tok_s: float | None):
        """Returns (new_level, reason) or None."""
        c, lv = self.cfg, self.level
        last = len(c.ladder) - 1
        want_down = self.desired_down(users)
        def since(prev, cond):
            return (now if prev is None else prev) if cond else None

        self.down_since = since(self.down_since, want_down > lv)
        self.up_since = since(self.up_since, lv > 0 and users <= c.step_up_users[lv - 1])
        slow = per_user_tok_s is not None and per_user_tok_s < c.slo_tok_s and lv < last
        self.slow_since = since(self.slow_since, slow)
        if now - self.last_switch < c.dwell_s:
            return None
        if self.down_since is not None and now - self.down_since >= c.hold_down_s:
            return want_down, f"users {users:.0f} >= {c.step_down_users[want_down - 1]}"
        if self.slow_since is not None and now - self.slow_since >= c.slo_hold_s:
            return lv + 1, f"per-user {per_user_tok_s:.0f} tok/s < target {c.slo_tok_s:.0f}"
        if self.up_since is not None and now - self.up_since >= c.hold_up_s:
            return lv - 1, f"users {users:.0f} <= {c.step_up_users[lv - 1]} for {c.hold_up_s:.0f}s"
        return None

    def switched(self, now: float, level: int) -> None:
        self.level, self.last_switch = level, now
        self.down_since = self.up_since = self.slow_since = None


def itl_buckets(text: str) -> dict[float, float]:
    out: dict[float, float] = {}
    for labels, value in BUCKET_RE.findall(text):
        m = LE_RE.search(labels)
        if m:
            le = float("inf") if m.group(1) == "+Inf" else float(m.group(1))
            out[le] = out.get(le, 0.0) + float(value)
    return out


def quantile(prev: dict, cur: dict, q: float) -> float | None:
    """q-quantile of the observations between two cumulative histogram snapshots."""
    les = sorted(cur)
    delta = [(le, cur[le] - prev.get(le, 0.0)) for le in les]
    total = delta[-1][1] if delta else 0
    if total <= 0:
        return None
    rank, lo_le, lo_n = q * total, 0.0, 0.0
    for le, n in delta:
        if n >= rank:
            if le == float("inf"):
                return lo_le
            return lo_le + (le - lo_le) * (rank - lo_n) / max(n - lo_n, 1e-12)
        lo_le, lo_n = le, n
    return None


def http(method: str, url: str, body: dict | None = None, timeout: float = 5.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode()
    return json.loads(raw) if raw.startswith("{") else raw


class Exporter:
    """Controller state for Prometheus (no-op without prometheus_client)."""

    def __init__(self, port: int, cfg: Config):
        self.ok = False
        if not port:
            return
        try:
            from prometheus_client import Counter, Gauge, start_http_server
        except ImportError:
            return
        self.level = Gauge("vpa_ctl_level", "Ladder level (0 = best quality)")
        self.bits = Gauge("vpa_ctl_target_bits", "Bits of the version the controller asked for")
        self.users = Gauge("vpa_ctl_users", "running + waiting requests seen by the controller")
        self.tok = Gauge("vpa_ctl_per_user_tok_s", "Per-user tokens/s (1 / median inter-token latency)")
        self.slo = Gauge("vpa_ctl_slo_tok_s", "Per-user tokens/s target")
        self.switches = Counter("vpa_ctl_switches", "Switch decisions", ["to", "reason"])
        self.slo.set(cfg.slo_tok_s)
        start_http_server(port)
        self.ok = True


BITS = {"high": 16, "fp8": 8, "int4": 4}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--url")
    ap.add_argument("--slo-tok-s", type=float)
    ap.add_argument("--metrics-port", type=int, default=9101)
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    cfg = load_config(args.config, {"url": args.url, "slo_tok_s": args.slo_tok_s})
    log = open(args.log, "a") if args.log else None

    def emit(event: dict) -> None:
        line = json.dumps({"t": round(time.time(), 3), **event})
        print(line, flush=True)
        if log:
            log.write(line + "\n")
            log.flush()

    state = http("GET", f"{cfg.url}/v1/precision")
    current = state["version"]
    policy = Policy(cfg, cfg.ladder.index(current) if current in cfg.ladder else 0)
    exp = Exporter(args.metrics_port, cfg)
    emit({"event": "start", "version": current, "config": cfg.__dict__})
    history: list[tuple[float, dict]] = []
    last_mig = None
    while True:
        now = time.monotonic()
        try:
            text = http("GET", f"{cfg.url}/metrics")
            state = http("GET", f"{cfg.url}/v1/precision")  # also finishes idle migrations
        except Exception as e:  # noqa: BLE001
            emit({"event": "error", "error": str(e)})
            time.sleep(cfg.interval_s)
            continue
        vals = {"vllm:num_requests_running": 0.0, "vllm:num_requests_waiting": 0.0}
        for name, value in GAUGE_RE.findall(text):
            vals[name] += float(value)
        users = vals["vllm:num_requests_running"] + vals["vllm:num_requests_waiting"]
        history.append((now, itl_buckets(text)))
        while len(history) > 1 and history[0][0] < now - cfg.window_s:
            history.pop(0)
        p50 = quantile(history[0][1], history[-1][1], 0.5) if len(history) > 1 else None
        tok = (1.0 / p50) if p50 else None
        if users == 0:
            tok = None  # idle: no tokens, nothing to judge
        decision = policy.decide(now, users, tok)
        if decision:
            level, reason = decision
            version = cfg.ladder[level]
            reply = http("POST", f"{cfg.url}/v1/precision", {"version": version})
            emit({"event": "switch", "from": cfg.ladder[policy.level], "to": version,
                  "reason": reason, "users": users,
                  "per_user_tok_s": round(tok, 1) if tok else None,
                  "migrating": reply.get("migrating")})
            if exp.ok:
                exp.switches.labels(version, reason.split()[0]).inc()
            policy.switched(now, level)
        migs = state.get("migrations") or []
        if migs and not state.get("migrating") and migs[-1] != last_mig:
            last_mig = migs[-1]
            emit({"event": "migrated", **last_mig})
        if exp.ok:
            exp.level.set(policy.level)
            exp.bits.set(BITS.get(cfg.ladder[policy.level], 0))
            exp.users.set(users)
            if tok:
                exp.tok.set(tok)
        time.sleep(cfg.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
