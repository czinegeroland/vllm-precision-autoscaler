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


def test_metric_parsing():
    text = ('vllm:num_requests_running{engine="0",model_name="m"} 3.0\n'
            'vllm:num_requests_waiting{engine="0",model_name="m"} 2.0\n'
            'vllm:generation_tokens_total{engine="0",model_name="m"} 1234.0\n'
            'vllm:generation_tokens_created{engine="0",model_name="m"} 1.7e+09\n')
    assert dict(pc.GAUGE_RE.findall(text)) == {"vllm:num_requests_running": "3.0",
                                               "vllm:num_requests_waiting": "2.0"}
    assert pc.GEN_RE.findall(text) == ["1234.0"]


def test_configs_load():
    pytest.importorskip("yaml")
    c = pc.load_config(str(ROOT / "controller" / "config.example.yaml"), {})
    assert c.ladder == ["high", "fp8", "int4"]
    k = pc.load_config(str(ROOT / "poc" / "k8s-keda" / "controller.yaml"), {})
    assert k.ladder == ["high", "fp8", "int4a8"] and k.slo_tok_s == 0


def test_slo_guard_off_when_target_is_zero():
    p = pc.Policy(cfg(slo_tok_s=0, dwell_s=0))
    assert run(p, [(t, 10, 5) for t in range(0, 30)]) == []      # slow, but the guard is off


def test_per_user_tok_s_from_counter_and_running():
    hist = [(0.0, 1000.0, 4), (5.0, 3000.0, 4), (10.0, 5000.0, 4)]
    assert pc.per_user_tok_s(hist) == pytest.approx(100.0)       # 400 tok/s / 4 users
    assert pc.per_user_tok_s([(0.0, 0.0, 0), (10.0, 0.0, 0)]) is None   # idle
    assert pc.per_user_tok_s(hist[:1]) is None
    assert pc.per_user_tok_s([(0.0, 5000.0, 4), (10.0, 100.0, 4)]) is None   # counter reset
