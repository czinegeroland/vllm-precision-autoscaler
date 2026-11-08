"""Nested BF16 format: one 16-bit-per-weight storage that serves both FP8
and (near-exact) BF16 inference.

Idea after NestedFP (Lee et al., arXiv:2506.02024), which nests FP8 in FP16.
NestedFP's trick does not carry over to BF16 (BF16's upper byte is sign + 7
exponent bits), so this is a different construction:

Each BF16 weight w of output channel n is split into two bytes:

  A  = E4M3 (round-to-nearest-even) encoding of w / 2^k[n]
       -> exactly what a standard FP8 cast with a power-of-two per-channel
          scale would produce, so FP8 mode uses A as-is with vLLM's FP8 GEMM.
  B  = what is needed to rebuild the BF16 bits exactly:
         bit 7     "tiny" flag (w below the FP8 normal range, see below)
         bits 0-4  d + 8, where d in [-8, 8] is the rounding remainder

Derivation (normal range). Write the BF16 magnitude bits as u = E<<7 | m7
(E = 8-bit exponent, m7 = 7-bit mantissa) and let off[n] = 120 + k[n].
Then t = u - (off << 7) = (E4 << 7 | m7) with E4 = E - 127 - k + 7, the
E4M3 exponent. Dropping 4 mantissa bits with round-to-nearest-even is
integer RNE of t / 16, and a mantissa carry into the exponent falls out of
the integer arithmetic:   v = rne(t / 16),  d = t - 16 v  in [-8, 8],
A = sign << 7 | v,   and exactly   u = 16 v + d + (off << 7).

Values with E4 < 1 ("tiny": below 2^-6 * 2^k, i.e. ~6e-5 of the channel max
or less) cannot be rebuilt from 8 extra bits. For those, A holds the proper
FP8 (subnormal/zero) value and 16-bit mode uses that value: a tiny absolute
error, reported by `encode` stats. Zeros are tiny and rebuilt exactly.

k[n] is the smallest exponent such that the channel max fits E4M3 (v <= 126,
i.e. <= 448 * 2^k).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FP8 = torch.float8_e4m3fn
TINY_FLAG = 0x80


def _rne_div16(t: torch.Tensor) -> torch.Tensor:
    q = t >> 4  # arithmetic shift (floor) for int32
    r = t & 15
    up = (r > 8) | ((r == 8) & ((q & 1) == 1))
    return q + up.to(t.dtype)


@dataclass
class NestedStats:
    weights: int = 0
    tiny: int = 0
    tiny_nonzero: int = 0
    max_abs_err: float = 0.0
    max_rel_err_vs_channel_max: float = 0.0

    def add(self, other: "NestedStats") -> None:
        self.weights += other.weights
        self.tiny += other.tiny
        self.tiny_nonzero += other.tiny_nonzero
        self.max_abs_err = max(self.max_abs_err, other.max_abs_err)
        self.max_rel_err_vs_channel_max = max(
            self.max_rel_err_vs_channel_max, other.max_rel_err_vs_channel_max
        )


@torch.no_grad()
def encode(weight: torch.Tensor, chunk_rows: int = 2048):
    """BF16 [N, K] -> (A float8_e4m3fn [N, K], B uint8 [N, K],
    off int32 [N], scale float32 [N, 1] (= 2^k), stats)."""
    assert weight.dtype == torch.bfloat16 and weight.ndim == 2, (weight.dtype, weight.shape)
    n, k_dim = weight.shape
    dev = weight.device
    a_out = torch.empty((n, k_dim), dtype=torch.uint8, device=dev)
    b_out = torch.empty((n, k_dim), dtype=torch.uint8, device=dev)
    off_out = torch.empty((n,), dtype=torch.int32, device=dev)
    stats = NestedStats(weights=n * k_dim)

    for start in range(0, n, chunk_rows):
        w = weight[start : start + chunk_rows]
        bits = w.view(torch.int16).to(torch.int32) & 0xFFFF
        sign = bits >> 15
        u = bits & 0x7FFF
        if bool(((u >> 7) == 0xFF).any()):
            raise ValueError("weight contains inf/nan")

        umax = u.amax(dim=1)
        off = (umax >> 7) - 15
        vmax = _rne_div16(umax - (off << 7))
        off = off + (vmax > 126).to(torch.int32)
        # All-zero (or absurdly small) rows: keep 2^k a normal float32.
        off = off.clamp(min=0)

        t = u - (off[:, None] << 7)
        normal = t >= 128
        v = _rne_div16(t)
        d = t - 16 * v
        a_normal = (sign << 7) | v
        b_normal = d + 8

        scale = torch.exp2((off - 120).to(torch.float32))[:, None]
        tiny_fp8 = (w.float() / scale).to(FP8)
        a_tiny = tiny_fp8.view(torch.uint8).to(torch.int32)

        a = torch.where(normal, a_normal, a_tiny)
        b = torch.where(normal, b_normal, torch.full_like(b_normal, TINY_FLAG))
        a_out[start : start + chunk_rows] = a.to(torch.uint8)
        b_out[start : start + chunk_rows] = b.to(torch.uint8)
        off_out[start : start + chunk_rows] = off

        tiny = ~normal
        stats.tiny += int(tiny.sum())
        nz = tiny & (u != 0)
        stats.tiny_nonzero += int(nz.sum())
        if bool(nz.any()):
            err = (tiny_fp8.float() * scale - w.float()).abs() * nz
            stats.max_abs_err = max(stats.max_abs_err, float(err.max()))
            cmax = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
            stats.max_rel_err_vs_channel_max = max(
                stats.max_rel_err_vs_channel_max, float((err / cmax).max())
            )

    scale_out = torch.exp2((off_out - 120).to(torch.float32))[:, None].contiguous()
    return a_out.view(FP8), b_out, off_out, scale_out, stats


@torch.no_grad()
def decode(a: torch.Tensor, b: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
    """Reference (torch) reconstruction of BF16 weights; the Triton kernel
    does the same in registers."""
    a_i = a.view(torch.uint8).to(torch.int32)
    b_i = b.to(torch.int32)
    sign = (a_i >> 7) & 1
    v = a_i & 0x7F
    d = (b_i & 0x1F) - 8
    u = 16 * v + d + (off[:, None].to(torch.int32) << 7)
    bits = (sign << 15) | u
    normal = bits.to(torch.int16).view(torch.bfloat16)  # int32 -> int16 wraps to the bit pattern
    scale = torch.exp2((off - 120).to(torch.float32))[:, None]
    tiny = (a.view(FP8).float() * scale).to(torch.bfloat16)
    return torch.where((b_i & TINY_FLAG) != 0, tiny, normal)
