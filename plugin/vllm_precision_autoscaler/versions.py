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


class Int4Kind:
    """INT4 weight-only (W4A16): symmetric round-to-nearest, group size 128,
    uint4b8 (values -8..7 stored +8), run by vLLM's Marlin kernel. Activations
    stay 16-bit. Quarter of the 16-bit bytes; best at small batches, where
    reading weights dominates. Quantized on the GPU at startup (vLLM's own
    test helpers pack on the CPU with NumPy, too slow/large for 32B models).
    Quality: plain RTN, no calibration (AWQ/GPTQ checkpoints are a later option)."""

    name = "int4"
    bits = 4
    method = "int4_w4a16_rtn_g128_marlin"
    group = 128

    def nbytes(self, n: int, k: int, esize: int) -> int:
        return n * k // 2

    def check(self, n: int, k: int) -> None:
        if k % self.group or not ((n % 64 == 0 and k % 128 == 0) or (n % 128 == 0 and k % 64 == 0)):
            raise ValueError(f"int4 (Marlin) needs n%64==0 and k%128==0, got n={n}, k={k}")

    @torch.no_grad()
    def quantize(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[N, K] 16-bit -> (Marlin-repacked int32 [K/16, 2N], Marlin scales [K/128, N])."""
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales

        n, k = weight.shape
        self.check(n, k)
        g = self.group
        packed = torch.empty((k // 8, n), dtype=torch.int32, device=weight.device)
        scales = torch.empty((k // g, n), dtype=weight.dtype, device=weight.device)
        chunk = 4096  # output channels per chunk: bounds the fp32 temporaries
        for c0 in range(0, n, chunk):
            w = weight[c0 : c0 + chunk].float().t().contiguous()       # [K, nc]
            wg = w.view(k // g, g, -1)
            s = (wg.abs().amax(dim=1) / 7.0).clamp(min=1e-8)            # [K/g, nc]
            q = torch.round(wg / s[:, None, :]).clamp(-8, 7).to(torch.int32) + 8
            q = q.view(k, -1)                                           # [K, nc], 0..15
            p = torch.zeros((k // 8, q.shape[1]), dtype=torch.int32, device=weight.device)
            for i in range(8):  # GPTQ layout: 8 consecutive k values per int32
                p |= q[i::8] << (4 * i)
            packed[:, c0 : c0 + chunk] = p
            scales[:, c0 : c0 + chunk] = s.to(weight.dtype)
        marlin_w = ops.gptq_marlin_repack(packed, k, n, 4)
        marlin_s = marlin_permute_scales(scales, k, n, g)
        return marlin_w, marlin_s

    def build(self, weight: torch.Tensor) -> LayerVersion:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_empty

        marlin_w, marlin_s = self.quantize(weight)
        cpu = torch.empty(marlin_w.numel() * 4, dtype=torch.uint8, pin_memory=True)
        cpu.copy_(marlin_w.view(torch.uint8).reshape(-1))
        return LayerVersion(cpu=cpu, nbytes=cpu.numel(),
                            aux={"scale": marlin_s, "zp": marlin_make_empty(weight.device).data})

    def apply(self, x: torch.Tensor, region: torch.Tensor, aux: dict, bias: Optional[torch.Tensor],
              n: int, k: int) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            apply_gptq_marlin_linear,
            get_marlin_workspace,
        )
        from vllm.scalar_type import scalar_types

        w = region[: n * k // 2].view(torch.int32).view(k // 16, 2 * n)
        if bias is not None:
            # Marlin adds the bias in its own column order: permute once per
            # layer (the bias is a fixed parameter) and reuse it.
            cached = aux.get("_bias")
            if cached is None or cached[0] != bias.data_ptr():
                from vllm.model_executor.layers.quantization.utils.marlin_utils import (
                    marlin_permute_bias,
                )

                cached = (bias.data_ptr(), marlin_permute_bias(bias))
                aux["_bias"] = cached
            bias = cached[1]
        return apply_gptq_marlin_linear(
            input=x, weight=w, weight_scale=aux["scale"], weight_zp=aux["zp"],
            workspace=get_marlin_workspace(x.device), wtype=scalar_types.uint4b8,
            output_size_per_partition=n, input_size_per_partition=k, bias=bias,
        )


KINDS = {kind.name: kind for kind in (HighKind(), Fp8Kind(), Int4Kind())}
assert set(KINDS) == set(KNOWN), "versions_names.KNOWN must list every kind"
