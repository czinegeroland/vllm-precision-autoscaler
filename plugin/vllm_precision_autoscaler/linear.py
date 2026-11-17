"""Linear method: one layer, several precisions.

Two storage formats (VPA_FORMAT):
  dual (default): BF16 weight + separate FP8 copy in VRAM, about +50% weight
      memory; the precision is chosen per engine step from the tokens in the step.
  swap: one precision version in VRAM (in the loaded weight's own memory),
      every version (16-bit, FP8, INT4, ...) in pinned CPU RAM; layers migrate
      in the background when the target version changes (swap.py).
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.linear import (
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)

from . import ops, state
from .fp8_weights import quantize_weight_per_channel


@register_weight_loader_v2_supported_method
class PrecisionAutoscalerLinearMethod(UnquantizedLinearMethod):
    # We transform weights after loading; don't claim support for loading
    # already-processed weights.
    supports_pre_processed_weights = False

    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.layer_id = state.register_layer(prefix)
        self.format = state.settings.format

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        weight = layer.weight.data  # [N, K], model dtype
        n, k = weight.shape
        if weight.dtype not in (torch.bfloat16, torch.float16):
            state.mark_ineligible(self.layer_id, f"dtype {weight.dtype}")
            return
        if n % 16 or k % 16:
            # The CUTLASS FP8 GEMM wants both dims aligned to 16.
            state.mark_ineligible(self.layer_id, f"shape {tuple(weight.shape)} not 16-aligned")
            return
        if self.format == "swap":
            self._process_swap(layer, weight)
        else:
            q, scale = quantize_weight_per_channel(weight)
            # CUTLASS expects B as [K, N] column-major: the transpose of a
            # contiguous [N, K] tensor is exactly that, no copy needed.
            layer.register_buffer("vpa_weight_fp8_t", q.t(), persistent=False)
            layer.register_buffer("vpa_weight_scale", scale, persistent=False)
        state.ensure_counters(weight.device)

    def _process_swap(self, layer: torch.nn.Module, weight: torch.Tensor) -> None:
        from .swap import build_swap_layer

        n, k = weight.shape
        # The GPU region *is* the loaded 16-bit weight's memory: no extra VRAM
        # (every version is at most as large as the 16-bit one).
        region = weight.view(torch.uint8).reshape(-1)
        self.swap_n, self.swap_k = n, k
        layer.register_buffer("vpa_region", region, persistent=False)
        mgr = state.get_swap_manager()
        mgr.register(build_swap_layer(self.layer_id, weight, region, mgr.version_names))
        if len(mgr.layers) == sum(state.eligible):
            mgr.finalize(weight.device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not state.eligible[self.layer_id]:
            return super().apply(layer, x, bias)
        if self.format == "swap":
            return ops.swap_linear(
                x, layer.vpa_region, bias, self.layer_id, self.swap_n, self.swap_k,
            )
        return ops.linear(
            x, layer.weight, layer.vpa_weight_fp8_t, layer.vpa_weight_scale, bias, self.layer_id
        )
