"""Nested format + Triton kernel tests (needs vLLM + FP8-capable GPU).

    pytest -q tests/gpu/test_nested_gpu.py
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from vllm_precision_autoscaler import nested_format, ops, state  # noqa: E402
from vllm_precision_autoscaler.kernels.nested_gemm import nested_gemm  # noqa: E402

FP8 = torch.float8_e4m3fn


def realistic_weight(n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", generator=g) * 0.02
    # a few outliers per row, like real LLM weights
    idx = torch.randint(0, k, (n, 4), device="cuda", generator=g)
    w.scatter_(1, idx, torch.randn(n, 4, device="cuda", generator=g) * 0.5)
    return w.to(torch.bfloat16)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("shape", [(256, 512), (1024, 2560)])
def test_roundtrip_exact_outside_tiny(shape):
    w = realistic_weight(*shape)
    a, b, off, scale, stats = nested_format.encode(w)
    rebuilt = nested_format.decode(a, b, off)
    tiny = (b & nested_format.TINY_FLAG) != 0
    exact = rebuilt.view(torch.int16) == w.view(torch.int16)
    assert bool(exact[~tiny].all()), "normal-range weights must be bit-exact"
    assert stats.tiny / stats.weights < 1e-3
    # tiny values: error far below the channel's max
    assert stats.max_rel_err_vs_channel_max < 1e-4


def test_a_is_standard_fp8_cast():
    """A must equal torch's RNE cast of w / 2^k, so FP8 mode == plain FP8."""
    w = realistic_weight(512, 1024, seed=1)
    a, b, off, scale, _ = nested_format.encode(w)
    ref = (w.float() / scale).to(FP8)
    assert torch.equal(a.view(torch.uint8), ref.view(torch.uint8))
    assert int(a.view(torch.uint8).max()) != 0x7F and not torch.isnan(a.float()).any()


def test_edge_values():
    rows = [
        torch.zeros(64),                                    # all-zero row
        torch.full((64,), -0.0),                            # negative zeros
        torch.linspace(-448.0, 448.0, 64),                  # hits the E4M3 max
        torch.tensor([1.0 + i / 128 for i in range(64)]),   # every mantissa value
        torch.tensor([2.0 ** -e for e in range(64)]),       # wide dynamic range
    ]
    w = torch.stack(rows).cuda().to(torch.bfloat16)
    a, b, off, scale, _ = nested_format.encode(w)
    rebuilt = nested_format.decode(a, b, off)
    tiny = (b & nested_format.TINY_FLAG) != 0
    assert torch.equal(rebuilt.view(torch.int16)[~tiny], w.view(torch.int16)[~tiny])
    assert torch.equal(rebuilt[:2].float(), w[:2].float())  # zeros rebuilt exactly
    ref = (w.float() / scale).to(FP8)
    assert torch.equal(a.view(torch.uint8), ref.view(torch.uint8))


@pytest.mark.parametrize("m", [1, 7, 16, 24, 33, 64, 200])
@pytest.mark.parametrize("nk", [(2560, 9728), (6144, 2560)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_kernel_matches_reference(m, nk, with_bias):
    n, k = nk
    w = realistic_weight(n, k, seed=m)
    a, b, off, scale, _ = nested_format.encode(w)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) if with_bias else None
    y = nested_gemm(x, a.view(torch.uint8), b, off, scale.view(-1), bias)
    ref = torch.nn.functional.linear(x, nested_format.decode(a, b, off), bias)
    assert y.shape == ref.shape and y.dtype == torch.bfloat16
    assert _rel(y, ref) < 5e-3
    # and vs the original BF16 weights
    assert _rel(y, torch.nn.functional.linear(x, w, bias)) < 5e-3


def test_op_modes():
    w = realistic_weight(1024, 2560, seed=3)
    a, b, off, scale, _ = nested_format.encode(w)
    lid = state.register_layer("model.layers.0.mlp.nested_test")
    x = torch.randn(40, 2560, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, w)
    state.set_mode("high")
    y_high = ops.nested_linear(x, a, b, off, scale, None, lid)
    state.set_mode("fp8")
    y_fp8 = ops.nested_linear(x, a, b, off, scale, None, lid)
    assert _rel(y_high, ref) < 5e-3
    assert 5e-3 < _rel(y_fp8, ref) < 0.08  # FP8 path is really taken and accurate


def test_kernel_in_cuda_graph():
    n, k = 2560, 9728
    w = realistic_weight(n, k, seed=4)
    a, b, off, scale, _ = nested_format.encode(w)
    au = a.view(torch.uint8)
    for m in (4, 128):  # split-K (atomics + zeros) and plain paths
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        nested_gemm(x, au, b, off, scale.view(-1))  # compile outside capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = nested_gemm(x, au, b, off, scale.view(-1))
        x.copy_(torch.randn_like(x))
        g.replay()
        g.replay()  # replay twice: split-K buffer must be re-zeroed each time
        torch.cuda.synchronize()
        assert _rel(y, torch.nn.functional.linear(x, w)) < 5e-3
