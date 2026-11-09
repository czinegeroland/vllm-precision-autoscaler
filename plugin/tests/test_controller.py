"""Unit tests for the precision controller policy (stdlib only, no vLLM)."""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "controller"))

import precision_controller as pc  # noqa: E402


def cfg(**kw):
    c = pc.Config(**kw)
    c.validate()
    return c


def run(policy, timeline):
    """timeline: [(t, users, tok)] -> list of (t, level, reason) switches."""
    out = []
    for t, users, tok in timeline:
        d = policy.decide(t, users, tok)
        if d:
            policy.switched(t, d[0])
            out.append((t, d[0], d[1]))
    return out


def test_config_validation():
    with pytest.raises(ValueError):
        cfg(step_down_users=[24], step_up_users=[16])            # wrong length
    with pytest.raises(ValueError):
        cfg(step_down_users=[24, 40], step_up_users=[30, 32])    # no hysteresis
    with pytest.raises(ValueError):
        cfg(step_down_users=[40, 24], step_up_users=[16, 20])    # not increasing


def test_steps_down_after_hold():
    p = pc.Policy(cfg(hold_down_s=2, dwell_s=0))
    sw = run(p, [(t, 30, 100) for t in range(0, 5)])
    assert sw[0][:2] == (2, 1)                                   # high -> fp8 after 2 s


def test_jumps_several_levels_on_a_burst():
    p = pc.Policy(cfg(hold_down_s=1, dwell_s=0))
    sw = run(p, [(t, 64, 100) for t in range(0, 3)])
    assert sw[0][1] == 2                                         # straight to int4


def test_steps_up_one_level_at_a_time_after_hold():
    p = pc.Policy(cfg(hold_up_s=20, dwell_s=0), level=2)
    sw = run(p, [(t, 4, 100) for t in range(0, 50)])
    assert [s[1] for s in sw] == [1, 0]
    assert sw[0][0] == 20 and sw[1][0] == 41      # hold restarts after a switch


def test_hysteresis_band_holds_level():
    p = pc.Policy(cfg(dwell_s=0), level=1)                       # fp8, band 16 < users < 40
    assert run(p, [(t, 25, 100) for t in range(0, 60)]) == []


def test_slo_guard_steps_down_one_more_level():
    p = pc.Policy(cfg(slo_tok_s=60, slo_hold_s=8, dwell_s=0))
    sw = run(p, [(t, 10, 50) for t in range(0, 12)])             # few users but slow
    assert sw[0][:2] == (8, 1) and "tok/s" in sw[0][2]


def test_slo_guard_cannot_go_below_last_level():
    p = pc.Policy(cfg(dwell_s=0), level=2)
    assert run(p, [(t, 50, 10) for t in range(0, 30)]) == []


def test_dwell_blocks_rapid_switches():
    p = pc.Policy(cfg(hold_down_s=0, hold_up_s=0, dwell_s=10))
    sw = run(p, [(0, 30, 100), (1, 50, 100), (5, 50, 100), (11, 50, 100)])
    assert [(s[0], s[1]) for s in sw] == [(0, 1), (11, 2)]


def test_quantile_from_histogram_delta():
    prev = {0.01: 0, 0.02: 0, float("inf"): 0}
    cur = {0.01: 50, 0.02: 100, float("inf"): 100}
    assert pc.quantile(prev, cur, 0.5) == pytest.approx(0.01)
    assert pc.quantile(prev, cur, 0.75) == pytest.approx(0.015)
    assert pc.quantile(cur, cur, 0.5) is None                    # no new observations


def test_itl_bucket_parsing():
    text = ('vllm:inter_token_latency_seconds_bucket{engine="0",le="0.01",model_name="m"} 3.0\n'
            'vllm:inter_token_latency_seconds_bucket{engine="0",le="+Inf",model_name="m"} 5.0\n'
            'vllm:num_requests_running{engine="0"} 2.0\n')
    assert pc.itl_buckets(text) == {0.01: 3.0, float("inf"): 5.0}


def test_example_config_loads():
    pytest.importorskip("yaml")
    c = pc.load_config(str(ROOT / "controller" / "config.example.yaml"), {})
    assert c.ladder == ["high", "fp8", "int4"]


def test_per_user_tok_s_from_counter_and_running():
    hist = [(0.0, 1000.0, 4), (5.0, 3000.0, 4), (10.0, 5000.0, 4)]
    assert pc.per_user_tok_s(hist) == pytest.approx(100.0)       # 400 tok/s / 4 users
    assert pc.per_user_tok_s([(0.0, 0.0, 0), (10.0, 0.0, 0)]) is None   # idle
    assert pc.per_user_tok_s(hist[:1]) is None
