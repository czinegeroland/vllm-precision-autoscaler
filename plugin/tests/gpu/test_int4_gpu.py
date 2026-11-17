"""INT4 (W4A16, Marlin) version kind + 3-version swap (needs vLLM + CUDA GPU).

    pytest -q tests/gpu/test_int4_gpu.py
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from vllm_precision_autoscaler import ops, state  # noqa: E402
from vllm_precision_autoscaler.settings import Settings  # noqa: E402
from vllm_precision_autoscaler.swap import SwapManager, build_swap_layer  # noqa: E402
from vllm_precision_autoscaler.versions import KINDS  # noqa: E402

INT4 = KINDS["int4"]
SHAPES = [(6144, 2560), (2560, 9728), (19456, 2560)]


class _Log:
    def info(self, *a):
        pass


def weight(n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)


def dequant_ref(w: torch.Tensor) -> torch.Tensor:
    """Same RTN as Int4Kind.quantize, dequantized back to bf16 (reference)."""
    n, k = w.shape
    wg = w.float().t().reshape(k // 128, 128, n)
    s = (wg.abs().amax(dim=1) / 7.0).clamp(min=1e-8)
    q = torch.round(wg / s[:, None, :]).clamp(-8, 7)
    return (q * s[:, None, :]).reshape(k, n).t().to(torch.bfloat16)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def region_with(version) -> torch.Tensor:
    r = torch.empty(version.cpu.numel(), dtype=torch.uint8, device="cuda")
    r.copy_(version.cpu)
    return r


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [1, 7, 64, 300])
def test_int4_matches_dequantized_reference(n, k, m):
    w = weight(n, k, seed=n + k)
    v = INT4.build(w)
    assert v.nbytes == n * k // 2
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    y = INT4.apply(x, region_with(v), v.aux, None, n, k)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16
    assert rel(y, torch.nn.functional.linear(x, dequant_ref(w))) < 1e-2   # kernel == math
    assert rel(y, torch.nn.functional.linear(x, w)) < 0.15                # RTN int4 error


def test_int4_bias():
    n, k = 2560, 2560
    w = weight(n, k, seed=1)
    v = INT4.build(w)
    x = torch.randn(9, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    y = INT4.apply(x, region_with(v), v.aux, b, n, k)
    assert rel(y, torch.nn.functional.linear(x, dequant_ref(w), b)) < 1e-2


def test_three_version_round_trip():
    """high -> int4 -> fp8 -> high while 'serving': every step is consistent,
    and the final 16-bit outputs equal the first ones exactly."""
    mgr = SwapManager(Settings(mode="high", format="swap", swap_slots=1,
                               swap_versions=("high", "fp8", "int4")), _Log())
    ws, lids = [], []
    for n, k in SHAPES:
        w = weight(n, k, seed=7 * n)
        region = w.clone().view(torch.uint8).reshape(-1)
        lid = state.register_layer(f"model.layers.0.mlp.int4_test_{len(state.prefixes)}")
        mgr.register(build_swap_layer(lid, w, region, mgr.version_names))
        ws.append(w)
        lids.append(lid)
    mgr.finalize(torch.device("cuda"))
    state.swap_manager = mgr
    try:
        xs = [torch.randn(4, k, device="cuda", dtype=torch.bfloat16) for _, k in SHAPES]

        def forward():
            return [ops.swap_linear(x, mgr.layers[l].region, None, l, mgr.layers[l].n, mgr.layers[l].k)
                    for x, l in zip(xs, lids)]

        first = forward()
        for target in ("int4", "fp8", "high"):
            mgr.set_target(target)
            for _ in range(200):
                outs = forward()
                torch.cuda.synchronize()
                for o, x, l in zip(outs, xs, lids):  # each layer: one version's math
                    layer = mgr.layers[l]
                    errs = {v: rel(o, KINDS[v].apply(x, region_with(layer.versions[v]),
                                                     layer.versions[v].aux, None, layer.n, layer.k))
                            for v in mgr.version_names}
                    # Marlin may reduce with atomics (last-bit differences), so
                    # "matches" means within 1e-3; other versions are ~1e-2..1e-1 off.
                    assert min(errs.values()) < 1e-3, errs
                if mgr.current is None and not mgr.inflight:
                    break
            assert all(mgr.layers[l].active == target for l in lids), target
        last = forward()
        for a, b in zip(first, last):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert [m.target for m in mgr.migrations] == ["int4", "fp8", "high"]
    finally:
        state.swap_manager = None
