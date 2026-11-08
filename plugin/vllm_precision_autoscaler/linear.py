"""Linear method: 16-bit and FP8 from one layer, chosen per engine step.

Two storage formats (VPA_FORMAT):
  nested (v2, default): one 2-byte-per-weight copy (A = FP8 byte, B = rest);
      FP8 mode reads A only, 16-bit mode rebuilds BF16 in a Triton kernel.
      Same weight memory as BF16. BF16 models only.
  dual (v0): BF16 weight + separate FP8 copy, about +50% weight memory
      ("separate storage" baseline from the NestedFP paper).
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
        if self.format == "nested" and weight.dtype != torch.bfloat16:
            state.log_once(f"nested format needs BF16 weights; {weight.dtype} uses dual format")
            self.format = "dual"

        if self.format == "nested":
            self._process_nested(layer, weight)
        elif self.format == "swap":
            self._process_swap(layer, weight)
        else:
            q, scale = quantize_weight_per_channel(weight)
            # CUTLASS expects B as [K, N] column-major: the transpose of a
            # contiguous [N, K] tensor is exactly that, no copy needed.
            layer.register_buffer("vpa_weight_fp8_t", q.t(), persistent=False)
            layer.register_buffer("vpa_weight_scale", scale, persistent=False)
        state.ensure_counters(weight.device)

    def _process_swap(self, layer: torch.nn.Module, weight: torch.Tensor) -> None:
        from .swap import SwapLayer

        n, k = weight.shape
        q, scale = quantize_weight_per_channel(weight)
        cpu_high = torch.empty(2 * n * k, dtype=torch.uint8, pin_memory=True)
        cpu_high.copy_(weight.contiguous().view(torch.uint8).reshape(-1))
        cpu_fp8 = torch.empty(n * k, dtype=torch.uint8, pin_memory=True)
        cpu_fp8.copy_(q.view(torch.uint8).reshape(-1))
        del q
        # The GPU region *is* the loaded weight's memory: no extra VRAM.
        region = weight.view(torch.uint8).reshape(-1)
        self.swap_n, self.swap_k = n, k
        layer.register_buffer("vpa_region", region, persistent=False)
        layer.register_buffer("vpa_scale", scale, persistent=False)
        mgr = state.get_swap_manager()
        mgr.register(SwapLayer(self.layer_id, n, k, region, scale, cpu_high, cpu_fp8))
        if len(mgr.layers) == sum(state.eligible):
            mgr.finalize(weight.device)

    def _process_nested(self, layer: torch.nn.Module, weight: torch.Tensor) -> None:
        from . import nested_format
        from .kernels import nested_gemm

        a, b, off, scale, stats = nested_format.encode(weight)
        layer.register_buffer("vpa_a", a, persistent=False)
        layer.register_buffer("vpa_b", b, persistent=False)
        layer.register_buffer("vpa_off", off, persistent=False)
        layer.register_buffer("vpa_scale", scale, persistent=False)
        state.add_nested_stats(stats)
        # Free the BF16 copy: A and B are the only weights now. Keep a
        # 0-element parameter so code that looks up `layer.weight` still works.
        has_bias = getattr(layer, "bias", None) is not None
        layer.weight = torch.nn.Parameter(
            torch.empty(0, dtype=weight.dtype, device=weight.device), requires_grad=False
        )
        del weight
        nested_gemm.warmup_once(a.shape[0], a.shape[1], has_bias, a.device)

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
                x, layer.vpa_region, layer.vpa_scale, bias, self.layer_id,
                self.swap_n, self.swap_k,
            )
        if self.format == "nested":
            return ops.nested_linear(
                x, layer.vpa_a, layer.vpa_b, layer.vpa_off, layer.vpa_scale, bias, self.layer_id
            )
        return ops.linear(
            x, layer.weight, layer.vpa_weight_fp8_t, layer.vpa_weight_scale, bias, self.layer_id
        )
