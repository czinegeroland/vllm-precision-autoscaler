"""Weight conversion helpers (torch only)."""

from __future__ import annotations

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max  # 448.0


@torch.no_grad()
def quantize_weight_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel E4M3 quantization of a [N, K] weight.

    Returns (q [N, K] float8_e4m3fn, scale [N, 1] float32) such that
    weight ~= q.float() * scale. Rows are converted in chunks to keep the
    temporary FP32 copy small.
    """
    assert weight.ndim == 2, weight.shape
    n = weight.shape[0]
    q = torch.empty_like(weight, dtype=FP8_DTYPE)
    scale = torch.empty((n, 1), dtype=torch.float32, device=weight.device)
    chunk = 4096
    for start in range(0, n, chunk):
        w = weight[start : start + chunk].float()
        s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / FP8_MAX
        q[start : start + chunk] = (w / s).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        scale[start : start + chunk] = s
    return q, scale
