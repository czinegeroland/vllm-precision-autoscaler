"""Triton GEMM for the 16-bit mode of the nested format:

    y[M, N] = x[M, K] @ W^T,   W rebuilt in registers from A (fp8 bits) and B.

Weights are read as two uint8 tiles (2 bytes/weight, same traffic as BF16),
BF16 bits are rebuilt with integer ops, then fed to tl.dot. In `auto` mode
this kernel only runs for small steps (memory-bound decode), where a simple
tiled kernel with split-K can get close to cuBLAS.
"""

from __future__ import annotations

import functools

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _rebuild_bf16(wa, wb, off, kscale):
    """wa, wb: uint8 [BN, BK]; off int32 [BN, 1]; kscale fp32 [BN, 1]."""
    a = wa.to(tl.int32)
    b = wb.to(tl.int32)
    sign = (a >> 7) & 1
    v = a & 0x7F
    d = (b & 0x1F) - 8
    u = 16 * v + d + (off << 7)
    bits = (sign << 15) | u
    normal = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    tiny = (wa.to(tl.float8e4nv, bitcast=True).to(tl.float32) * kscale).to(tl.bfloat16)
    return tl.where((b & 0x80) != 0, tiny, normal)


@triton.jit(do_not_specialize=["M"])
def _nested_gemm_kernel(
    x_ptr, a_ptr, b_ptr, off_ptr, kscale_ptr, bias_ptr, y_ptr,
    M, N, K, K_PER_SPLIT,
    stride_xm, stride_xk, stride_wn, stride_wk, stride_ym, stride_yn,
    HAS_BIAS: tl.constexpr, SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    mask_m = rm < M
    mask_n = rn < N

    off = tl.load(off_ptr + rn, mask=mask_n, other=0)[:, None]
    kscale = tl.load(kscale_ptr + rn, mask=mask_n, other=0.0)[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_start = pid_k * K_PER_SPLIT
    k_end = tl.minimum(k_start + K_PER_SPLIT, K)
    for k0 in range(k_start, k_end, BLOCK_K):
        kk = k0 + rk
        mask_k = kk < k_end
        x = tl.load(
            x_ptr + rm[:, None] * stride_xm + kk[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        w_off = rn[:, None] * stride_wn + kk[None, :] * stride_wk
        w_mask = mask_n[:, None] & mask_k[None, :]
        wa = tl.load(a_ptr + w_off, mask=w_mask, other=0)
        wb = tl.load(b_ptr + w_off, mask=w_mask, other=0x80)  # masked -> tiny 0
        w = _rebuild_bf16(wa, wb, off, kscale)  # [BN, BK]
        acc += tl.dot(x, tl.trans(w))

    if HAS_BIAS:
        if pid_k == 0:
            bias = tl.load(bias_ptr + rn, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias[None, :]

    y_ptrs = y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    y_mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)
    else:
        tl.atomic_add(y_ptrs, acc, mask=y_mask)


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def pick_config(m: int, n: int, k: int, sms: int) -> dict:
    """Fixed heuristics (no autotuning: autotune would benchmark during CUDA
    graph capture). Small M: narrow N tiles + split-K to fill the GPU."""
    if m <= 16:
        cfg = dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=4)
    elif m <= 32:
        cfg = dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=4)
    elif m <= 64:
        cfg = dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=4)
    else:
        cfg = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=3)
    tiles = triton.cdiv(m, cfg["BLOCK_M"]) * triton.cdiv(n, cfg["BLOCK_N"])
    split = 1
    if m <= 64:
        while (
            split < 8
            and tiles * split < 2 * sms
            and k // (split * 2) >= 4 * cfg["BLOCK_K"]
        ):
            split *= 2
    cfg["SPLIT_K"] = split
    return cfg


def nested_gemm(
    x: torch.Tensor,
    a_u8: torch.Tensor,
    b_u8: torch.Tensor,
    off: torch.Tensor,
    kscale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """x [M, K] bf16; a_u8, b_u8 [N, K] uint8 (row-major); off int32 [N];
    kscale fp32 [N]. Returns [M, N] bf16."""
    assert x.dtype == torch.bfloat16 and x.ndim == 2
    m, k = x.shape
    n = a_u8.shape[0]
    cfg = pick_config(m, n, k, _num_sms(x.device.index or 0))
    split = cfg.pop("SPLIT_K")
    k_per_split = triton.cdiv(triton.cdiv(k, split), cfg["BLOCK_K"]) * cfg["BLOCK_K"]
    if split == 1:
        y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    else:
        y = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, cfg["BLOCK_M"]), triton.cdiv(n, cfg["BLOCK_N"]), split)
    _nested_gemm_kernel[grid](
        x, a_u8, b_u8, off, kscale, bias if bias is not None else x, y,
        m, n, k, k_per_split,
        x.stride(0), x.stride(1), a_u8.stride(0), a_u8.stride(1), y.stride(0), y.stride(1),
        HAS_BIAS=bias is not None, SPLIT_K=split, **cfg,
    )
    return y if split == 1 else y.to(x.dtype)


def warmup(n: int, k: int, has_bias: bool, device: torch.device) -> None:
    """Compile every kernel variant this weight shape can hit, before vLLM
    records CUDA graphs (JIT compilation must not happen during capture)."""
    a = torch.zeros((n, k), dtype=torch.uint8, device=device)
    b = torch.full((n, k), 0x80, dtype=torch.uint8, device=device)
    off = torch.zeros((n,), dtype=torch.int32, device=device)
    ks = torch.ones((n,), dtype=torch.float32, device=device)
    bias = torch.zeros((n,), dtype=torch.bfloat16, device=device) if has_bias else None
    seen = set()
    for m in (1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 512):
        cfg = pick_config(m, n, k, _num_sms(device.index or 0))
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        x = torch.zeros((m, k), dtype=torch.bfloat16, device=device)
        nested_gemm(x, a, b, off, ks, bias)
    torch.cuda.synchronize(device)


_warmed: set = set()


def warmup_once(n: int, k: int, has_bias: bool, device: torch.device) -> None:
    key = (n, k, has_bias, str(device))
    if key not in _warmed:
        _warmed.add(key)
        warmup(n, k, has_bias, device)
