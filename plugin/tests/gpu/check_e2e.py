"""Compare e2e_switching.py outputs for modes high, fp8 and auto.

    python tests/gpu/check_e2e.py /tmp/high.json /tmp/fp8.json /tmp/auto.json
"""

import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def share(run, name):
    return run[name]["stats"]["fp8_token_share"]


def main() -> int:
    high, fp8, auto = (load(p) for p in sys.argv[1:4])
    checks = []

    def check(desc, ok, detail=""):
        checks.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {desc} {detail}")

    for name in ("single", "batch"):
        check(f"high/{name}: no FP8 tokens", share(high, name) == 0.0, f"share={share(high, name)}")
        check(f"fp8/{name}: only FP8 tokens", share(fp8, name) == 1.0, f"share={share(fp8, name)}")

    check("auto/single: all 16-bit (small steps)", share(auto, "single") == 0.0,
          f"share={share(auto, 'single')}")
    check("auto/batch: mostly FP8 (large steps)", share(auto, "batch") >= 0.9,
          f"share={share(auto, 'batch'):.3f}")

    # Same kernels for the same step sizes -> identical greedy tokens.
    check("auto/single tokens == high/single tokens",
          auto["single"]["token_ids"] == high["single"]["token_ids"])
    # A request whose prefill landed in a small step (e.g. the very first
    # step, before the rest of the batch arrived) is prefilled in 16-bit, so
    # its output may legitimately differ. Require most samples to match.
    pairs = list(zip(auto["batch"]["token_ids"], fp8["batch"]["token_ids"]))
    same = sum(a == b for a, b in pairs)
    check("auto/batch tokens == fp8/batch tokens for most samples", same >= len(pairs) - 1,
          f"{same}/{len(pairs)} identical")
    check("fp8 differs from high somewhere (FP8 path is real)",
          fp8["batch"]["token_ids"] != high["batch"]["token_ids"]
          or fp8["single"]["token_ids"] != high["single"]["token_ids"])

    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
