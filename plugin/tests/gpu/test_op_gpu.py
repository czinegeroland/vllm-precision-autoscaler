"""Op-level GPU tests (needs vLLM + an FP8-capable NVIDIA GPU).

    pytest -q tests/gpu/test_op_gpu.py
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from vllm_precision_autoscaler import ops, state  # noqa: E402
from vllm_precision_autoscaler.fp8_weights import quantize_weight_per_channel  # noqa: E402


@pytest.fixture()
def layer():
    torch.manual_seed(0)
    n, k = 1024, 2048
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    q, s = quantize_weight_per_channel(w)
    lid = state.register_layer(f"model.layers.0.mlp.test_{len(state.prefixes)}")
    return w, q.t(), s, lid


def _rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_weight_quantization_roundtrip(layer):
    w, q_t, s, _ = layer
    deq = q_t.t().float() * s
    assert _rel_err(deq, w) < 0.04


@pytest.mark.parametrize("tokens", [1, 7, 64, 300])
def test_high_mode_matches_f_linear_exactly(layer, tokens):
    w, q_t, s, lid = layer
    state.set_mode("high")
    x = torch.randn(tokens, w.shape[1], device="cuda", dtype=torch.bfloat16)
    out = ops.linear(x, w, q_t, s, None, lid)
    torch.testing.assert_close(out, torch.nn.functional.linear(x, w), rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 16, 300])
def test_fp8_mode_close_to_reference(layer, tokens):
    w, q_t, s, lid = layer
    state.set_mode("fp8")
    x = torch.randn(tokens, w.shape[1], device="cuda", dtype=torch.bfloat16)
    out = ops.linear(x, w, q_t, s, None, lid)
    ref = torch.nn.functional.linear(x, w)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert _rel_err(out, ref) < 0.08
    assert not torch.equal(out, ref)  # really took the FP8 path


def test_auto_switches_on_token_count(layer):
    w, q_t, s, lid = layer
    state.set_mode("auto", threshold=32)
    small = torch.randn(32, w.shape[1], device="cuda", dtype=torch.bfloat16)
    big = torch.randn(33, w.shape[1], device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        ops.linear(small, w, q_t, s, None, lid), torch.nn.functional.linear(small, w), rtol=0, atol=0
    )
    assert not torch.equal(ops.linear(big, w, q_t, s, None, lid), torch.nn.functional.linear(big, w))


def test_bias_supported(layer):
    w, q_t, s, lid = layer
    bias = torch.randn(w.shape[0], device="cuda", dtype=torch.bfloat16)
    x = torch.randn(40, w.shape[1], device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, w, bias)
    for mode in ("high", "fp8"):
        state.set_mode(mode)
        assert _rel_err(ops.linear(x, w, q_t, s, bias, lid), ref) < 0.08


def test_cuda_graph_freezes_per_size_decision(layer):
    """The core claim: a recorded CUDA graph keeps the precision chosen
    for its size, and replays it correctly."""
    w, q_t, s, lid = layer
    state.set_mode("auto", threshold=32)
    outs = {}
    for tokens in (8, 64):
        x = torch.randn(tokens, w.shape[1], device="cuda", dtype=torch.bfloat16)
        ops.linear(x, w, q_t, s, None, lid)  # warm-up outside capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = ops.linear(x, w, q_t, s, None, lid)
        x.copy_(torch.randn_like(x))
        g.replay()
        torch.cuda.synchronize()
        outs[tokens] = (y.clone(), torch.nn.functional.linear(x, w))
    y8, ref8 = outs[8]
    y64, ref64 = outs[64]
    torch.testing.assert_close(y8, ref8, rtol=0, atol=0)  # graph for 8 -> 16-bit
    assert not torch.equal(y64, ref64) and _rel_err(y64, ref64) < 0.08  # graph for 64 -> FP8


def test_torch_compile_does_not_freeze_branch(layer):
    w, q_t, s, lid = layer

    @torch.compile(dynamic=True, fullgraph=True)
    def f(x):
        return ops.linear(x, w, q_t, s, None, lid) * 1.0

    state.set_mode("auto", threshold=32)
    small = torch.randn(16, w.shape[1], device="cuda", dtype=torch.bfloat16)
    big = torch.randn(128, w.shape[1], device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(f(small), torch.nn.functional.linear(small, w), rtol=0, atol=0)
    assert not torch.equal(f(big), torch.nn.functional.linear(big, w))
    state.set_mode("high")  # runtime change is visible through the compiled fn
    torch.testing.assert_close(f(big), torch.nn.functional.linear(big, w), rtol=0, atol=0)
