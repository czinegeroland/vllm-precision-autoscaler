"""INT4 W4A8-FP8 (Marlin) version kind (needs vLLM + an SM89/SM12x CUDA GPU).

    pytest -q tests/gpu/test_int4a8_gpu.py
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((8, 9),)
    and torch.cuda.get_device_capability()[0] != 12,
    reason="needs CUDA SM89 or SM12x (Marlin W4A8-FP8)")

from vllm_precision_autoscaler import ops, state  # noqa: E402
from vllm_precision_autoscaler.settings import Settings  # noqa: E402
from vllm_precision_autoscaler.swap import SwapManager, build_swap_layer  # noqa: E402
from vllm_precision_autoscaler.versions import KINDS  # noqa: E402

@pytest.fixture(autouse=True, scope="module")
def _vllm_config():
    """Marlin's FP8 input quantizer (a vLLM CustomOp) needs a current vLLM config."""
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        yield


A8 = KINDS["int4a8"]
INT4 = KINDS["int4"]
SHAPES = [(6144, 2560), (2560, 9728), (19456, 2560)]


class _Log:
    def info(self, *a):
        pass


def weight(n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)


def dequant_ref(w: torch.Tensor) -> torch.Tensor:
    n, k = w.shape
    wg = w.float().t().reshape(k // 128, 128, n)
    s = (wg.abs().amax(dim=1) / 7.0).clamp(min=1e-8)
    q = torch.round(wg / s[:, None, :]).clamp(-8, 7)
    return (q * s[:, None, :]).reshape(k, n).t().to(torch.bfloat16)


def fp8_per_token(x: torch.Tensor) -> torch.Tensor:
    """Per-token FP8 E4M3 round trip of the activations (reference)."""
    s = (x.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    return ((x.float() / s).to(torch.float8_e4m3fn).float() * s).to(x.dtype)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def region_with(version) -> torch.Tensor:
    r = torch.empty(version.cpu.numel(), dtype=torch.uint8, device="cuda")
    r.copy_(version.cpu)
    return r


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [1, 7, 64, 300])
def test_int4a8_matches_reference(n, k, m):
    w = weight(n, k, seed=n + k)
    v = A8.build(w)
    assert v.nbytes == n * k // 2
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    y = A8.apply(x, region_with(v), v.aux, None, n, k)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16
    ref = torch.nn.functional.linear(fp8_per_token(x), dequant_ref(w))
    assert rel(y, ref) < 3e-2, rel(y, ref)                                 # kernel == math
    assert rel(y, torch.nn.functional.linear(x, w)) < 0.2                  # int4 + fp8 error
    y16 = INT4.apply(x, region_with(INT4.build(w)), INT4.build(w).aux, None, n, k)
    assert rel(y, y16) < 0.1                                               # close to W4A16


def test_int4a8_bias():
    n, k = 2560, 2560
    w = weight(n, k, seed=1)
    v = A8.build(w)
    x = torch.randn(9, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    y = A8.apply(x, region_with(v), v.aux, b, n, k)
    assert rel(y, torch.nn.functional.linear(fp8_per_token(x), dequant_ref(w), b)) < 3e-2


def test_four_version_round_trip():
    """high -> int4a8 -> int4 -> fp8 -> high: each step consistent, 16-bit exact at the end."""
    names = ("high", "fp8", "int4", "int4a8")
    mgr = SwapManager(Settings(mode="high", format="swap", swap_slots=1, swap_versions=names), _Log())
    lids = []
    for n, k in SHAPES:
        w = weight(n, k, seed=7 * n)
        region = w.clone().view(torch.uint8).reshape(-1)
        lid = state.register_layer(f"model.layers.0.mlp.int4a8_test_{len(state.prefixes)}")
        mgr.register(build_swap_layer(lid, w, region, mgr.version_names))
        lids.append(lid)
    mgr.finalize(torch.device("cuda"))
    state.swap_manager = mgr
    try:
        xs = [torch.randn(4, k, device="cuda", dtype=torch.bfloat16) for _, k in SHAPES]

        def forward():
            return [ops.swap_linear(x, mgr.layers[l].region, None, l, mgr.layers[l].n, mgr.layers[l].k)
                    for x, l in zip(xs, lids)]

        first = forward()
        for target in ("int4a8", "int4", "fp8", "high"):
            mgr.set_target(target)
            for _ in range(200):
                outs = forward()
                torch.cuda.synchronize()
                for o, x, l in zip(outs, xs, lids):
                    layer = mgr.layers[l]
                    errs = {v: rel(o, KINDS[v].apply(x, region_with(layer.versions[v]),
                                                     layer.versions[v].aux, None, layer.n, layer.k))
                            for v in mgr.version_names}
                    assert min(errs.values()) < 1e-3, errs
                if mgr.current is None and not mgr.inflight:
                    break
            assert all(mgr.layers[l].active == target for l in lids), target
        last = forward()
        for a, b in zip(first, last):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    finally:
        state.swap_manager = None
