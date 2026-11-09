"""Precision versions for the swap format.

A *version* is one precision of a linear layer, stored in the exact format its
kernel needs. Each version kind knows three things:

  build(weight)       -> pinned CPU bytes (what gets copied into the GPU region)
                         + small GPU-resident extras (e.g. FP8 scales)
  nbytes(n, k, esize) -> bytes it occupies in the GPU region
  apply(...)          -> run the matrix multiply on the region's bytes

Every version of a layer shares one GPU region, sized for the largest version;
switching = copying another version's bytes into the region (swap.py). All
kernels are vLLM's or PyTorch's own: no custom kernel.

Adding a precision (e.g. INT4 Marlin) = adding one kind here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .fp8_weights import quantize_weight_per_channel
from .versions_names import KNOWN, canonical, parse_versions  # noqa: F401


@dataclass
class LayerVersion:
    cpu: torch.Tensor                      # pinned uint8, copied into the region
    nbytes: int
    aux: dict[str, torch.Tensor] = field(default_factory=dict)  # GPU-resident extras


class HighKind:
    """The model's own 16-bit weights (BF16/FP16), PyTorch/cuBLAS GEMM."""

    name = "high"
    bits = 16
    method = "unquantized"

    def nbytes(self, n: int, k: int, esize: int) -> int:
        return n * k * esize

    def build(self, weight: torch.Tensor) -> LayerVersion:
        cpu = torch.empty(weight.numel() * weight.element_size(), dtype=torch.uint8, pin_memory=True)
        cpu.copy_(weight.contiguous().view(torch.uint8).reshape(-1))
        return LayerVersion(cpu=cpu, nbytes=cpu.numel())

    def apply(self, x: torch.Tensor, region: torch.Tensor, aux: dict, bias: Optional[torch.Tensor],
              n: int, k: int) -> torch.Tensor:
        w = region[: n * k * x.element_size()].view(x.dtype).view(n, k)
        return torch.nn.functional.linear(x, w, bias)


class Fp8Kind:
    """FP8 E4M3 weights with per-channel scales, dynamic per-token activation
    scales (W8A8), vLLM's CUTLASS scaled-mm kernel. Needs compute capability >= 8.9."""

    name = "fp8"
    bits = 8
    method = "fp8_per_channel_w8a8"

    def nbytes(self, n: int, k: int, esize: int) -> int:
        return n * k

    def build(self, weight: torch.Tensor) -> LayerVersion:
        q, scale = quantize_weight_per_channel(weight)
        cpu = torch.empty(q.numel(), dtype=torch.uint8, pin_memory=True)
        cpu.copy_(q.view(torch.uint8).reshape(-1))
        return LayerVersion(cpu=cpu, nbytes=cpu.numel(), aux={"scale": scale})

    def apply(self, x: torch.Tensor, region: torch.Tensor, aux: dict, bias: Optional[torch.Tensor],
              n: int, k: int) -> torch.Tensor:
        from .ops import _fp8_gemm

        w = region[: n * k].view(torch.float8_e4m3fn).view(n, k)
        return _fp8_gemm(x, w.t(), aux["scale"], bias, n)


KINDS = {kind.name: kind for kind in (HighKind(), Fp8Kind())}
assert set(KINDS) == set(KNOWN), "versions_names.KNOWN must list every kind"
