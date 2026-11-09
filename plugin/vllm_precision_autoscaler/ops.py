"""The dual-precision linear, registered as a custom torch op.

Why a custom op: vLLM compiles the model with torch.compile once for all
batch sizes and does not re-check guards. A Python `if` on the batch size
inside the model code could get frozen into the compiled graph. As an
opaque custom op, the branch below runs for real:
  * when vLLM records a CUDA graph for a batch size, the branch picks the
    precision for that size and the graph keeps it;
  * in eager steps (sizes without a graph), it runs every step.

Note: no `from __future__ import annotations` here - torch infers the op
schema from real type annotations.
"""

from typing import Optional

import torch
from torch.library import Library
from vllm import _custom_ops as ops
from vllm.utils.torch_utils import direct_register_custom_op

from . import state

_LIB = Library("precision_autoscaler", "FRAGMENT")  # keep a reference: op lifetime


def _linear_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_fp8_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    layer_id: int,
) -> torch.Tensor:
    num_tokens = x.numel() // x.shape[-1]
    fp8 = state.decide(layer_id, num_tokens)
    if state.counters is not None:
        state.record(layer_id, fp8, num_tokens)

    if not fp8:
        return torch.nn.functional.linear(x, weight, bias)

    x2d = x.reshape(-1, x.shape[-1])
    # Dynamic per-token activation scales, per-channel weight scales.
    x_q, x_scale = ops.scaled_fp8_quant(x2d, use_per_token_if_dynamic=True)
    out = ops.cutlass_scaled_mm(
        x_q, weight_fp8_t, scale_a=x_scale, scale_b=weight_scale, out_dtype=x.dtype, bias=bias
    )
    return out.view(*x.shape[:-1], weight.shape[0])


def _linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_fp8_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    layer_id: int,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="linear",
    op_func=_linear_impl,
    mutates_args=[],
    fake_impl=_linear_fake,
    target_lib=_LIB,
)


def linear(x, weight, weight_fp8_t, weight_scale, bias, layer_id: int) -> torch.Tensor:
    return torch.ops.precision_autoscaler.linear(
        x, weight, weight_fp8_t, weight_scale, bias, layer_id
    )


# --- v2: nested format (one 16-bit-per-weight copy) -------------------------

def _fp8_gemm(x: torch.Tensor, weight_fp8_t: torch.Tensor, scale: torch.Tensor,
              bias: Optional[torch.Tensor], n: int) -> torch.Tensor:
    x2d = x.reshape(-1, x.shape[-1])
    x_q, x_scale = ops.scaled_fp8_quant(x2d, use_per_token_if_dynamic=True)
    out = ops.cutlass_scaled_mm(
        x_q, weight_fp8_t, scale_a=x_scale, scale_b=scale, out_dtype=x.dtype, bias=bias
    )
    return out.view(*x.shape[:-1], n)


def _nested_linear_impl(
    x: torch.Tensor,
    a_fp8: torch.Tensor,
    b: torch.Tensor,
    off: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    layer_id: int,
) -> torch.Tensor:
    num_tokens = x.numel() // x.shape[-1]
    fp8 = state.decide(layer_id, num_tokens)
    if state.counters is not None:
        state.record(layer_id, fp8, num_tokens)
    n = a_fp8.shape[0]
    if fp8:
        # A *is* a standard per-channel FP8 weight (power-of-two scales).
        return _fp8_gemm(x, a_fp8.t(), scale, bias, n)
    from .kernels.nested_gemm import nested_gemm

    x2d = x.reshape(-1, x.shape[-1])
    y = nested_gemm(x2d, a_fp8.view(torch.uint8), b, off, scale.view(-1), bias)
    return y.view(*x.shape[:-1], n)


def _nested_linear_fake(
    x: torch.Tensor,
    a_fp8: torch.Tensor,
    b: torch.Tensor,
    off: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    layer_id: int,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], a_fp8.shape[0]))


direct_register_custom_op(
    op_name="nested_linear",
    op_func=_nested_linear_impl,
    mutates_args=[],
    fake_impl=_nested_linear_fake,
    target_lib=_LIB,
)


def nested_linear(x, a_fp8, b, off, scale, bias, layer_id: int) -> torch.Tensor:
    return torch.ops.precision_autoscaler.nested_linear(x, a_fp8, b, off, scale, bias, layer_id)


# --- swap format: one version resident, all versions in pinned RAM -------------

def _swap_linear_impl(
    x: torch.Tensor,
    region: torch.Tensor,
    bias: Optional[torch.Tensor],
    layer_id: int,
    n: int,
    k: int,
) -> torch.Tensor:
    from .versions import KINDS

    num_tokens = x.numel() // x.shape[-1]
    mgr = state.swap_manager
    if not state.graphs_enabled and layer_id == mgr.first_id:
        mgr.on_step(num_tokens)  # once per engine step, before any layer runs
    layer = mgr.layers[layer_id]
    # While vLLM records the graph set for one version, run that version's
    # kernels (the region may still hold another one: recorded outputs are
    # discarded).
    version = state.capture_precision or layer.active
    if state.counters is not None:
        state.record(layer_id, version != "high", num_tokens)
    return KINDS[version].apply(x, region, layer.versions[version].aux, bias, n, k)


def _swap_linear_fake(
    x: torch.Tensor,
    region: torch.Tensor,
    bias: Optional[torch.Tensor],
    layer_id: int,
    n: int,
    k: int,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op(
    op_name="swap_linear",
    op_func=_swap_linear_impl,
    mutates_args=[],
    fake_impl=_swap_linear_fake,
    target_lib=_LIB,
)


def swap_linear(x, region, bias, layer_id: int, n: int, k: int) -> torch.Tensor:
    return torch.ops.precision_autoscaler.swap_linear(x, region, bias, layer_id, n, k)
