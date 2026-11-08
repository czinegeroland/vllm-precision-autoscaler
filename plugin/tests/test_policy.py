"""Pure-Python tests: no torch, no vLLM, no GPU.

Run anywhere:  python -m unittest discover -s tests -p "test_policy.py"
They import the two modules by file path so the package __init__ (which
needs vLLM at register time) is not required.
"""

import importlib.util
import pathlib
import sys
import types
import unittest

PKG_DIR = pathlib.Path(__file__).resolve().parents[1] / "vllm_precision_autoscaler"


def _load(name: str):
    pkg = sys.modules.setdefault("vpa_testpkg", types.ModuleType("vpa_testpkg"))
    pkg.__path__ = [str(PKG_DIR)]
    spec = importlib.util.spec_from_file_location(f"vpa_testpkg.{name}", PKG_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


settings_mod = _load("settings")
policy = _load("policy")
Settings = settings_mod.Settings


class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        s = Settings.from_env({})
        self.assertEqual((s.mode, s.threshold, s.count), ("auto", 256, False))

    def test_aliases_and_case(self):
        self.assertEqual(Settings.from_env({"VPA_MODE": "BF16"}).mode, "high")
        self.assertEqual(Settings.from_env({"VPA_MODE": "fp16"}).mode, "high")
        self.assertEqual(Settings.from_env({"VPA_MODE": " FP8 "}).mode, "fp8")

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            Settings.from_env({"VPA_MODE": "int4"})

    def test_flags_and_ints(self):
        s = Settings.from_env({"VPA_COUNT": "1", "VPA_THRESHOLD": "64", "VPA_LOG_DECISIONS": "true"})
        self.assertEqual((s.count, s.threshold, s.log_decisions), (True, 64, True))

    def test_negative_threshold_rejected(self):
        with self.assertRaises(ValueError):
            Settings.from_env({"VPA_THRESHOLD": "-1"})

    def test_dial_bounds(self):
        with self.assertRaises(ValueError):
            Settings.from_env({"VPA_DIAL_LOW": "600", "VPA_DIAL_HIGH": "100"})

    def test_with_mode(self):
        s = Settings.from_env({}).with_mode("fp8", 32)
        self.assertEqual((s.mode, s.threshold), ("fp8", 32))


class PolicyTests(unittest.TestCase):
    def test_use_fp8(self):
        self.assertFalse(policy.use_fp8("high", 10_000, 0))
        self.assertTrue(policy.use_fp8("fp8", 1, 10_000))
        self.assertFalse(policy.use_fp8("auto", 64, 64))  # strictly greater
        self.assertTrue(policy.use_fp8("auto", 65, 64))
        self.assertTrue(policy.use_fp8("dial", 65, 64))

    def test_layer_index_and_kind(self):
        self.assertEqual(policy.layer_index("model.layers.12.mlp.down_proj"), 12)
        self.assertIsNone(policy.layer_index("lm_head"))
        self.assertEqual(policy.layer_kind("model.layers.3.mlp.gate_up_proj"), "mlp")
        self.assertEqual(policy.layer_kind("model.layers.3.self_attn.qkv_proj"), "attn")
        self.assertEqual(policy.layer_kind("model.layers.3.self_attn.o_proj"), "attn")
        self.assertEqual(policy.layer_kind("visual.merger"), "other")

    def test_auto_threshold_is_global(self):
        s = Settings(mode="auto", threshold=128)
        self.assertEqual(policy.layer_threshold(s, "model.layers.0.mlp.down_proj", 36), 128)
        self.assertEqual(policy.layer_threshold(s, "model.layers.35.self_attn.o_proj", 36), 128)

    def test_dial_orders_mlp_before_attention(self):
        s = Settings(mode="dial", dial_low=64, dial_high=512)
        n = 36
        mlp = [policy.layer_threshold(s, f"model.layers.{i}.mlp.down_proj", n) for i in range(n)]
        attn = [policy.layer_threshold(s, f"model.layers.{i}.self_attn.o_proj", n) for i in range(n)]
        self.assertEqual(mlp[0], 64)
        self.assertEqual(mlp, sorted(mlp))
        self.assertEqual(attn, sorted(attn))
        self.assertLess(max(mlp), min(attn))  # every MLP switches before any attention
        self.assertLessEqual(max(attn), 512)
        self.assertEqual(policy.layer_threshold(s, "lm_head", n), 512)




swap_policy = _load("swap_policy")


class SwapPolicyTests(unittest.TestCase):
    def make(self, **kw):
        args = dict(up=64, down=32, dwell_s=5, ema_s=2, start="high")
        args.update(kw)
        return swap_policy.HysteresisPolicy(**args)

    def feed(self, p, t0, seconds, tokens, step_s=0.02):
        t = t0
        while t < t0 + seconds:
            target = p.update(t, tokens)
            t += step_s
        return t, target

    def test_rejects_bad_band(self):
        with self.assertRaises(ValueError):
            self.make(up=32, down=32)

    def test_switches_up_then_down_with_hysteresis(self):
        p = self.make()
        t, target = self.feed(p, 0.0, 10, 8)
        self.assertEqual(target, "high")
        t, target = self.feed(p, t, 10, 128)      # sustained load
        self.assertEqual(target, "fp8")
        t, target = self.feed(p, t, 10, 48)       # between down and up: stay
        self.assertEqual(target, "fp8")
        t, target = self.feed(p, t, 10, 4)        # quiet again
        self.assertEqual(target, "high")

    def test_single_long_prompt_does_not_switch(self):
        p = self.make()
        t, _ = self.feed(p, 0.0, 10, 8)
        self.assertEqual(p.update(t, 4096), "high")  # one 4k-token prefill step
        t, target = self.feed(p, t + 0.02, 1, 8)
        self.assertEqual(target, "high")

    def test_first_big_step_does_not_switch(self):
        p = self.make()
        self.assertEqual(p.update(0.0, 2048), "high")  # e.g. 4 prompts prefilled at once
        t, target = self.feed(p, 0.02, 3, 8)
        self.assertEqual(target, "high")

    def test_dwell_time_limits_flapping(self):
        p = self.make(dwell_s=5, ema_s=0.01)
        t, target = self.feed(p, 0.0, 1, 128)
        self.assertEqual(target, "fp8")
        switched_at = t
        t, target = self.feed(p, t, 2, 4)          # quiet, but within dwell
        self.assertEqual(target, "fp8")
        t, target = self.feed(p, switched_at, 6, 4)
        self.assertEqual(target, "high")


class SwapSettingsTests(unittest.TestCase):
    def test_swap_settings(self):
        s = Settings.from_env({"VPA_FORMAT": "swap", "VPA_SWAP_UP": "100", "VPA_SWAP_SLOTS": "2"})
        self.assertEqual((s.format, s.swap_up, s.swap_slots), ("swap", 100.0, 2))

    def test_swap_policy_defaults_to_external(self):
        self.assertEqual(Settings.from_env({"VPA_FORMAT": "swap"}).swap_policy, "external")
        self.assertEqual(Settings.from_env({"VPA_SWAP_POLICY": "hysteresis"}).swap_policy, "hysteresis")
        with self.assertRaises(ValueError):
            Settings.from_env({"VPA_SWAP_POLICY": "magic"})

    def test_unknown_format(self):
        with self.assertRaises(ValueError):
            Settings.from_env({"VPA_FORMAT": "int4"})


if __name__ == "__main__":
    unittest.main()
