"""Runtime settings, read from environment variables.

vLLM builds a custom quantization config with no arguments when the
checkpoint has no quantization config of its own, so environment variables
are the simplest way to configure the plugin. They are inherited by the
engine-core and worker processes that vLLM spawns.

    VPA_MODE          high | fp8 | auto | dial      (default: auto)
                      high  = always 16-bit (the model's dtype, BF16 or FP16)
                      fp8   = always FP8
                      auto  = FP8 when a step has more than VPA_THRESHOLD tokens
                      dial  = per-layer thresholds between VPA_DIAL_LOW and
                              VPA_DIAL_HIGH; MLP layers switch before attention
    VPA_THRESHOLD     tokens per step above which `auto` uses FP8 (default 256)
    VPA_DIAL_LOW      lowest per-layer threshold for `dial` (default 64)
    VPA_DIAL_HIGH     highest per-layer threshold for `dial` (default 512)
    VPA_COUNT         1 = count calls/tokens per precision on the GPU (default 0)
                      adds one tiny kernel per linear call; use for accounting
                      runs, not for speed benchmarks
    VPA_LOG_DECISIONS 1 = log the precision chosen for each new step size
    VPA_FORMAT        dual | swap | nested          (default: dual)
                      dual   = BF16 weight + separate FP8 copy (v0, +50% weight
                               memory; uses only vLLM's existing kernels)
                      swap   = one precision in VRAM, both in pinned CPU RAM;
                               layers migrate in the background when the
                               target changes (needs CUDA graphs off).
                               VPA_MODE: high | fp8 = fixed start, switch via
                               RPC; auto = hysteresis policy below
    VPA_SWAP_POLICY   swap: external (default) | hysteresis
                      external   = the plugin only switches when told to, via
                                   POST /v1/precision (api.py) or vpa_set_mode;
                                   an outside controller decides when
                      hysteresis = optional built-in policy (VPA_SWAP_UP/DOWN/
                                   DWELL_S/EMA_S below), started with VPA_MODE=auto
    VPA_SWAP_UP       swap+auto: go FP8 when smoothed tokens/step > this (64)
    VPA_SWAP_DOWN     swap+auto: back to 16-bit when below this (32)
    VPA_SWAP_DWELL_S  swap+auto: minimum seconds between target changes (5)
    VPA_SWAP_EMA_S    swap+auto: smoothing time constant in seconds (2)
    VPA_SWAP_SLOTS    swap: staging buffers for background copies (4)
    VPA_SWAP_GRAPHS   swap: 1 = record two CUDA graph sets, one per precision
                      (needs cudagraph_mode FULL_DECODE_ONLY or FULL); 0 = never
                      use graphs (cudagraph_mode NONE required). Default 1.
                      nested = experimental: one 16-bit-per-weight copy holding the
                               FP8 weight inside it; needs the Triton kernel in
                               kernels/ (v2, BF16 only, not GPU-validated yet)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Mapping

MODES = ("high", "fp8", "auto", "dial")
FORMATS = ("nested", "dual", "swap")
_ALIASES = {"bf16": "high", "fp16": "high", "16": "high", "8": "fp8"}


def normalize_mode(raw: str) -> str:
    mode = raw.strip().lower()
    mode = _ALIASES.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f"VPA_MODE must be one of {MODES} (or bf16/fp16), got {raw!r}")
    return mode


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{key} must be >= 0, got {value}")
    return value


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    value = float(raw)
    if value < 0:
        raise ValueError(f"{key} must be >= 0, got {value}")
    return value


def _flag(env: Mapping[str, str], key: str) -> bool:
    return env.get(key, "0").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    mode: str = "auto"
    threshold: int = 256
    dial_low: int = 64
    dial_high: int = 512
    count: bool = False
    log_decisions: bool = False
    format: str = "dual"
    swap_up: float = 64.0
    swap_down: float = 32.0
    swap_dwell_s: float = 5.0
    swap_ema_s: float = 2.0
    swap_slots: int = 4
    swap_graphs: bool = True
    swap_policy: str = "external"

    def __post_init__(self) -> None:
        if self.swap_policy not in ("external", "hysteresis"):
            raise ValueError(f"VPA_SWAP_POLICY must be external or hysteresis, got {self.swap_policy!r}")
        if self.format not in FORMATS:
            raise ValueError(f"VPA_FORMAT must be one of {FORMATS}, got {self.format!r}")
        if self.dial_low > self.dial_high:
            raise ValueError(
                f"VPA_DIAL_LOW ({self.dial_low}) must be <= VPA_DIAL_HIGH ({self.dial_high})"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        return cls(
            mode=normalize_mode(env.get("VPA_MODE", "auto")),
            threshold=_int(env, "VPA_THRESHOLD", 256),
            dial_low=_int(env, "VPA_DIAL_LOW", 64),
            dial_high=_int(env, "VPA_DIAL_HIGH", 512),
            count=_flag(env, "VPA_COUNT"),
            log_decisions=_flag(env, "VPA_LOG_DECISIONS"),
            format=env.get("VPA_FORMAT", "dual").strip().lower(),
            swap_up=_float(env, "VPA_SWAP_UP", 64.0),
            swap_down=_float(env, "VPA_SWAP_DOWN", 32.0),
            swap_dwell_s=_float(env, "VPA_SWAP_DWELL_S", 5.0),
            swap_ema_s=_float(env, "VPA_SWAP_EMA_S", 2.0),
            swap_slots=max(1, _int(env, "VPA_SWAP_SLOTS", 4)),
            swap_graphs=env.get("VPA_SWAP_GRAPHS", "1").strip().lower() not in ("0", "false", "no", "off"),
            swap_policy=env.get("VPA_SWAP_POLICY", "external").strip().lower(),
        )

    def with_mode(self, mode: str, threshold: int | None = None) -> "Settings":
        changes: dict = {"mode": normalize_mode(mode)}
        if threshold is not None:
            changes["threshold"] = threshold
        return replace(self, **changes)
