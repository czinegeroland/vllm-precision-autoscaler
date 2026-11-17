"""Precision policy: pure Python, no torch or vLLM imports, so it is unit
testable anywhere.

The decision is a function of the number of tokens in the current engine
step. That is what makes it compatible with CUDA graphs: vLLM records one
graph per (padded) batch size, and each size always maps to the same
precision, so a recorded graph never contains the "wrong" kernel.
"""

from __future__ import annotations

import re

from .settings import Settings

_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def layer_index(prefix: str) -> int | None:
    """Decoder layer index from a module prefix like 'model.layers.12.mlp.down_proj'."""
    match = _LAYER_INDEX.search(prefix)
    return int(match.group(1)) if match else None


def layer_kind(prefix: str) -> str:
    """'mlp', 'attn' or 'other'. MoE expert/shared-expert projections count as mlp."""
    parts = prefix.split(".")
    if any(p in ("mlp", "feed_forward", "ffn", "shared_expert", "shared_experts") for p in parts):
        return "mlp"
    if any("attn" in p or p == "attention" for p in parts):
        return "attn"
    return "other"


def dial_fraction(prefix: str, num_layers: int) -> float:
    """Position of a layer on the dial, in [0, 1].

    MLP layers take the lower half (they switch to FP8 first), attention
    projections the upper half, both in depth order. Layers outside the
    decoder stack sit at the very top (switch last).
    """
    idx = layer_index(prefix)
    kind = layer_kind(prefix)
    if idx is None or num_layers <= 0 or kind == "other":
        return 1.0
    idx = min(idx, num_layers - 1)
    base = 0.0 if kind == "mlp" else 0.5
    return base + 0.5 * idx / num_layers


def layer_threshold(settings: Settings, prefix: str, num_layers: int) -> int:
    """Token threshold above which this layer runs in FP8."""
    if settings.mode == "dial":
        frac = dial_fraction(prefix, num_layers)
        return round(settings.dial_low + (settings.dial_high - settings.dial_low) * frac)
    return settings.threshold


def use_fp8(mode: str, num_tokens: int, threshold: int) -> bool:
    if mode == "high":
        return False
    if mode == "fp8":
        return True
    # auto and dial: FP8 only when the step is busier than the threshold.
    return num_tokens > threshold
