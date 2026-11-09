"""Swap format: background migration between BF16 and FP8 (needs vLLM + FP8 GPU).

    pytest -q tests/gpu/test_swap_gpu.py
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from vllm_precision_autoscaler import ops, state  # noqa: E402
from vllm_precision_autoscaler.fp8_weights import quantize_weight_per_channel  # noqa: E402
from vllm_precision_autoscaler.settings import Settings  # noqa: E402
from vllm_precision_autoscaler.swap import SwapLayer, SwapManager  # noqa: E402


class _Log:
    def info(self, *a):
        pass


SHAPES = [(6144, 2560), (19456, 2560), (2560, 9728)]


@pytest.fixture()
def setup():
    torch.manual_seed(0)
    mgr = SwapManager(Settings(mode="high", format="swap", swap_slots=2), _Log())
    weights, layers = [], []
    for n, k in SHAPES:
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        q, scale = quantize_weight_per_channel(w)
        cpu_high = torch.empty(2 * n * k, dtype=torch.uint8, pin_memory=True)
        cpu_high.copy_(w.view(torch.uint8).reshape(-1))
        cpu_fp8 = torch.empty(n * k, dtype=torch.uint8, pin_memory=True)
        cpu_fp8.copy_(q.view(torch.uint8).reshape(-1))
        region = w.clone().view(torch.uint8).reshape(-1)
        lid = state.register_layer(f"model.layers.0.mlp.swap_test_{len(state.prefixes)}")
        mgr.register(SwapLayer(lid, n, k, region, scale, cpu_high, cpu_fp8))
        weights.append((w, q, scale))
        layers.append(lid)
    mgr.finalize(torch.device("cuda"))
    state.swap_manager = mgr
    yield mgr, weights, layers
    state.swap_manager = None


def forward(mgr, layers, xs):
    """One 'engine step': every layer once, first layer triggers on_step."""
    outs = []
    for lid, x in zip(layers, xs):
        layer = mgr.layers[lid]
        outs.append(ops.swap_linear(x, layer.region, layer.scale, None, lid, layer.n, layer.k))
    return outs


def run_until_settled(mgr, layers, xs, max_steps=500):
    for step in range(max_steps):
        forward(mgr, layers, xs)
        torch.cuda.synchronize()
        if not mgr.inflight and not mgr._pending() and mgr.current is None:
            return step + 1
    raise AssertionError("migration did not finish")


def fp8_ref(x, q, scale):
    from vllm import _custom_ops as vops

    xq, xs = vops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
    return vops.cutlass_scaled_mm(xq, q.t(), scale_a=xs, scale_b=scale, out_dtype=x.dtype)


def test_roundtrip_bf16_fp8_bf16(setup):
    mgr, weights, layers = setup
    xs = [torch.randn(8, k, device="cuda", dtype=torch.bfloat16) for _, k in SHAPES]
    ptrs = [mgr.layers[l].region.data_ptr() for l in layers]

    out = forward(mgr, layers, xs)
    for o, x, (w, _, _) in zip(out, xs, weights):
        torch.testing.assert_close(o, torch.nn.functional.linear(x, w), rtol=0, atol=0)

    mgr.set_target("fp8")
    steps = run_until_settled(mgr, layers, xs)
    assert all(mgr.layers[l].mode == "fp8" for l in layers)
    out = forward(mgr, layers, xs)
    for o, x, (w, q, s) in zip(out, xs, weights):
        torch.testing.assert_close(o, fp8_ref(x, q, s), rtol=0, atol=0)  # exactly the FP8 path
    assert steps >= 2  # really migrated in the background, not in one go

    mgr.set_target("high")
    run_until_settled(mgr, layers, xs)
    out = forward(mgr, layers, xs)
    for o, x, (w, _, _) in zip(out, xs, weights):
        # restored from CPU RAM: bit-exact BF16 again
        torch.testing.assert_close(o, torch.nn.functional.linear(x, w), rtol=0, atol=0)
    assert [mgr.layers[l].region.data_ptr() for l in layers] == ptrs  # no reallocation
    assert [m.target for m in mgr.migrations] == ["fp8", "high"]


def test_every_step_is_consistent_during_migration(setup):
    """While migrating, each layer's output is exactly BF16 or exactly FP8.
    (The BF16 direction is used: those layers don't all fit in one staging
    buffer, so there are steps with mixed precisions.)"""
    mgr, weights, layers = setup
    xs = [torch.randn(4, k, device="cuda", dtype=torch.bfloat16) for _, k in SHAPES]
    mgr.set_target("fp8")
    run_until_settled(mgr, layers, xs)
    mgr.set_target("high")
    seen_mixed = False
    for _ in range(200):
        modes = [mgr.layers[l].mode for l in layers]
        out = forward(mgr, layers, xs)
        torch.cuda.synchronize()
        for o, x, (w, q, s), lid in zip(out, xs, weights, layers):
            if torch.equal(o, torch.nn.functional.linear(x, w)):
                continue
            assert torch.equal(o, fp8_ref(x, q, s)), "output is neither BF16 nor FP8"
        if len(set(modes)) == 2:
            seen_mixed = True
        if mgr.current is None and not mgr.inflight:
            break
    assert all(mgr.layers[l].mode == "high" for l in layers)
    assert seen_mixed, "expected at least one step with mixed precisions"


def test_packing_reduces_transfers(setup):
    """FP8 versions of all three test layers fit one staging buffer together."""
    mgr, weights, layers = setup
    xs = [torch.randn(2, k, device="cuda", dtype=torch.bfloat16) for _, k in SHAPES]
    mgr.set_target("fp8")
    run_until_settled(mgr, layers, xs)
    m = mgr.migrations[-1]
    assert m.layers == len(layers)
    assert m.transfers < len(layers), m


def test_pcie_bandwidth_report(setup):
    """Not an assertion: prints pinned host->device bandwidth for the findings."""
    src = torch.empty(512 * 2**20, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty_like(src, device="cuda")
    dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        dst.copy_(src, non_blocking=True)
    end.record()
    torch.cuda.synchronize()
    gbs = 5 * src.numel() / (start.elapsed_time(end) / 1000) / 1e9
    print(f"\nPINNED_H2D_GBPS={gbs:.1f}")
    assert gbs > 1
