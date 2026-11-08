"""Per-process runtime state shared by the linear method, the custom op and
the worker extension.

Each vLLM worker process has its own copy. Layer ids are assigned in the
order vLLM builds the layers, which is identical in every worker.
"""

from __future__ import annotations

import logging
from typing import Any

try:  # vLLM's logger has handlers configured; a bare stdlib logger prints nothing
    from vllm.logger import init_logger as _get_logger
except Exception:  # noqa: BLE001
    _get_logger = logging.getLogger

from . import policy
from .settings import Settings

# Must live under "vllm." so vLLM's handlers print our INFO lines.
logger = _get_logger("vllm.precision_autoscaler")

# Large enough for any dense model (layers x ~7 projections); counters are
# a fixed-size tensor so they can be allocated before CUDA graph capture.
MAX_LAYERS = 8192
# Counter columns.
HIGH_CALLS, FP8_CALLS, HIGH_TOKENS, FP8_TOKENS = range(4)

settings: Settings = Settings.from_env()
num_layers_hint: int = 0
prefixes: list[str] = []
thresholds: list[int] = []
eligible: list[bool] = []
counters: Any = None  # torch.Tensor [MAX_LAYERS, 4] int64 on the GPU, or None
_logged_sizes: set[tuple[int, bool]] = set()
swap_manager: Any = None  # swap.SwapManager when VPA_FORMAT=swap
# Two-graph-set mode (graphs.py): set while recording one precision's graphs.
capture_precision: str | None = None
graphs_enabled: bool = False
graph_set_counts: dict | None = None


def get_swap_manager() -> Any:
    global swap_manager
    if swap_manager is None:
        from .swap import SwapManager

        swap_manager = SwapManager(settings, logger)
    return swap_manager


def register_layer(prefix: str) -> int:
    layer_id = len(prefixes)
    if layer_id >= MAX_LAYERS:
        raise RuntimeError(f"more than {MAX_LAYERS} linear layers; raise MAX_LAYERS")
    prefixes.append(prefix)
    thresholds.append(policy.layer_threshold(settings, prefix, num_layers_hint))
    eligible.append(True)
    return layer_id


def mark_ineligible(layer_id: int, reason: str) -> None:
    eligible[layer_id] = False
    logger.warning("precision_autoscaler: %s stays 16-bit (%s)", prefixes[layer_id], reason)


def ensure_counters(device: Any) -> None:
    global counters
    if settings.count and counters is None:
        import torch

        counters = torch.zeros((MAX_LAYERS, 4), dtype=torch.int64, device=device)


def decide(layer_id: int, num_tokens: int) -> bool:
    fp8 = policy.use_fp8(settings.mode, num_tokens, thresholds[layer_id])
    if settings.log_decisions and layer_id == 0 and (num_tokens, fp8) not in _logged_sizes:
        _logged_sizes.add((num_tokens, fp8))
        logger.info(
            "precision_autoscaler: step with %d tokens -> %s (mode=%s)",
            num_tokens, "fp8" if fp8 else "high", settings.mode,
        )
    return fp8


def record(layer_id: int, fp8: bool, num_tokens: int) -> None:
    # In-place adds on a GPU tensor: they are recorded into CUDA graphs and
    # replayed with them, so the counts stay exact even when the Python
    # code of the op no longer runs.
    row = counters[layer_id]
    row[FP8_CALLS if fp8 else HIGH_CALLS].add_(1)
    row[FP8_TOKENS if fp8 else HIGH_TOKENS].add_(num_tokens)


def set_mode(mode: str, threshold: int | None = None) -> None:
    """Change the policy at runtime.

    Takes effect for steps that run the op's Python code: eager steps, and
    every step when CUDA graphs are disabled. Sizes already recorded in a
    CUDA graph keep the precision they were recorded with.
    """
    global settings
    settings = settings.with_mode(mode, threshold)
    if swap_manager is not None:
        if settings.mode == "auto":
            if settings.swap_policy != "hysteresis":
                raise ValueError("swap format is externally controlled: set mode high or fp8 "
                                 "(or start vLLM with VPA_SWAP_POLICY=hysteresis)")
            if swap_manager.policy is None:
                from .swap_policy import HysteresisPolicy

                swap_manager.policy = HysteresisPolicy(
                    settings.swap_up, settings.swap_down, settings.swap_dwell_s,
                    settings.swap_ema_s, start=swap_manager.target)
        else:
            swap_manager.policy = None  # manual control
            swap_manager.set_target(settings.mode)
    for i, prefix in enumerate(prefixes):
        thresholds[i] = policy.layer_threshold(settings, prefix, num_layers_hint)
    _logged_sizes.clear()


def reset_counters() -> None:
    if counters is not None:
        counters.zero_()


def stats(per_layer: bool = False) -> dict:
    out: dict = {
        "mode": settings.mode,
        "threshold": settings.threshold,
        "layers": len(prefixes),
        "eligible_layers": sum(eligible),
        "counting": counters is not None,
        "format": settings.format,
        "nested": nested_stats(),
        "swap": swap_manager.stats() if swap_manager is not None else None,
        "cuda_graph_sets": graph_set_counts if graphs_enabled else None,
    }
    if counters is None:
        return out
    rows = counters[: len(prefixes)].cpu()
    totals = rows.sum(dim=0).tolist()
    out["calls"] = {"high": totals[HIGH_CALLS], "fp8": totals[FP8_CALLS]}
    out["tokens"] = {"high": totals[HIGH_TOKENS], "fp8": totals[FP8_TOKENS]}
    tok = totals[HIGH_TOKENS] + totals[FP8_TOKENS]
    out["fp8_token_share"] = (totals[FP8_TOKENS] / tok) if tok else None
    if per_layer:
        out["per_layer"] = [
            {"prefix": p, "threshold": t, "eligible": e, "counts": r}
            for p, t, e, r in zip(prefixes, thresholds, eligible, rows.tolist())
        ]
    return out


# --- nested format bookkeeping ---------------------------------------------

_logged_messages: set[str] = set()
nested_totals: Any = None  # nested_format.NestedStats
_nested_layers_done = 0


def log_once(message: str) -> None:
    if message not in _logged_messages:
        _logged_messages.add(message)
        logger.warning("precision_autoscaler: %s", message)


def add_nested_stats(stats: Any) -> None:
    global nested_totals, _nested_layers_done
    if nested_totals is None:
        from .nested_format import NestedStats

        nested_totals = NestedStats()
    nested_totals.add(stats)
    _nested_layers_done += 1
    if _nested_layers_done == sum(eligible):
        t = nested_totals
        logger.info(
            "precision_autoscaler: nested format on %d layers, %d weights; "
            "%d tiny (%.4f%%, %d non-zero) rebuilt approximately in 16-bit mode, "
            "max abs err %.3g (%.3g of channel max)",
            _nested_layers_done, t.weights, t.tiny, 100.0 * t.tiny / max(t.weights, 1),
            t.tiny_nonzero, t.max_abs_err, t.max_rel_err_vs_channel_max,
        )


def nested_stats() -> dict | None:
    if nested_totals is None:
        return None
    t = nested_totals
    return {
        "weights": t.weights,
        "tiny": t.tiny,
        "tiny_nonzero": t.tiny_nonzero,
        "tiny_share": t.tiny / max(t.weights, 1),
        "max_abs_err": t.max_abs_err,
        "max_rel_err_vs_channel_max": t.max_rel_err_vs_channel_max,
    }
