"""End-to-end swap test in a real vLLM engine (CUDA graphs off).

1. generate in 16-bit -> tokens A
2. switch to FP8 over RPC, keep serving until the migration finishes
3. generate -> tokens B (FP8)
4. switch back to 16-bit, migrate
5. generate -> tokens C; must equal A exactly (weights restored from CPU RAM)

    python tests/gpu/e2e_swap.py --out /workspace/vpa/results/e2e_swap.json
"""

import argparse
import json
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cudagraph-mode", default="FULL_DECODE_ONLY", help="FULL_DECODE_ONLY or NONE")
    args = ap.parse_args()
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

    os.environ["VPA_FORMAT"] = "swap"
    os.environ["VPA_MODE"] = "high"
    os.environ["VPA_COUNT"] = "1"

    from vllm import LLM, SamplingParams

    from vllm_precision_autoscaler.worker_ext import EXTENSION

    t0 = time.time()
    llm = LLM(model=args.model, quantization="precision_autoscaler", worker_extension_cls=EXTENSION,
              max_model_len=1024, gpu_memory_utilization=0.85, seed=0,
              compilation_config={"cudagraph_mode": args.cudagraph_mode})
    load_s = time.time() - t0
    prompts = [f"Write one sentence about topic number {i}:" for i in range(16)]
    sp = SamplingParams(temperature=0.0, max_tokens=48, ignore_eos=True)
    filler = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)

    def gen():
        return [list(o.outputs[0].token_ids) for o in llm.generate(prompts, sp, use_tqdm=False)]

    def decode_speed():
        """Single-request decode tok/s (graph replay dominates if graphs work)."""
        long = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)
        llm.generate(prompts[:1], long, use_tqdm=False)  # warm
        t = time.time()
        out = llm.generate(prompts[:1], long, use_tqdm=False)
        return len(out[0].outputs[0].token_ids) / (time.time() - t)

    def stats():
        return llm.collective_rpc("vpa_stats")[0]

    def migrate(mode):
        llm.collective_rpc("vpa_set_mode", args=(mode,))
        t = time.time()
        steps = 0
        while True:  # migration advances on engine steps: keep the engine busy
            llm.generate(prompts[:2], filler, use_tqdm=False)
            steps += 1
            s = stats()["swap"]
            if not s["migrating"] and (s["fp8_layers"] == (s["layers"] if mode == "fp8" else 0)):
                return {"wall_s": time.time() - t, "filler_batches": steps, "swap": s}
            if time.time() - t > 120:
                raise RuntimeError(f"migration to {mode} did not finish: {s}")

    result = {"model": args.model, "cudagraph_mode": args.cudagraph_mode, "load_s": load_s,
              "initial": stats()}
    a = gen()
    result["decode_tok_s_high"] = decode_speed()
    result["to_fp8"] = migrate("fp8")
    b = gen()
    result["decode_tok_s_fp8"] = decode_speed()
    result["to_high"] = migrate("high")
    c = gen()
    result["checks"] = {
        "fp8_differs_from_high": a != b,
        "high_restored_exactly": a == c,
        "samples_identical_after_roundtrip": sum(x == y for x, y in zip(a, c)),
    }
    result["final"] = stats()
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"load_s": load_s, "cudagraph_mode": args.cudagraph_mode,
                      "graph_sets": result["final"].get("cuda_graph_sets"),
                      "decode_tok_s": [round(result["decode_tok_s_high"], 1),
                                       round(result["decode_tok_s_fp8"], 1)],
                      "checks": result["checks"],
                      "to_fp8": result["to_fp8"]["swap"]["migrations"][-1],
                      "to_high": result["to_high"]["swap"]["migrations"][-1]}))


if __name__ == "__main__":
    main()
