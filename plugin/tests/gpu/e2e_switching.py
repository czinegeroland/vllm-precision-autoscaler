"""End-to-end check inside a real vLLM engine (torch.compile + CUDA graphs on).

Runs one engine with the given VPA_MODE and two workloads:
  single: 1 short prompt   -> prefill and decode steps are small
  batch:  128 short prompts -> prefill and decode steps are large
and records generated token ids plus the plugin's GPU-side counters.

    python tests/gpu/e2e_switching.py --mode auto --threshold 64 --out /tmp/auto.json

check_e2e.py then compares the runs.
"""

import argparse
import json
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--threshold", type=int, default=64)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Must be set before vLLM spawns its engine process.
    os.environ["VPA_MODE"] = args.mode
    os.environ["VPA_THRESHOLD"] = str(args.threshold)
    os.environ["VPA_COUNT"] = "1"
    os.environ["VPA_LOG_DECISIONS"] = "1"

    from vllm import LLM, SamplingParams

    from vllm_precision_autoscaler.worker_ext import EXTENSION

    t0 = time.time()
    llm = LLM(
        model=args.model,
        quantization="precision_autoscaler",
        worker_extension_cls=EXTENSION,
        max_model_len=1024,
        gpu_memory_utilization=0.85,
        seed=0,
    )
    load_s = time.time() - t0

    topics = ["the ocean", "volcanoes", "bread", "trains", "chess", "the moon", "bees", "rain"]
    prompts = [f"Write one sentence about {topics[i % len(topics)]} (#{i}):" for i in range(args.batch)]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    result = {"mode": args.mode, "threshold": args.threshold, "model": args.model, "load_s": load_s}
    for name, batch in (("single", prompts[:1]), ("batch", prompts)):
        llm.collective_rpc("vpa_reset")
        t = time.time()
        outs = llm.generate(batch, sp, use_tqdm=False)
        elapsed = time.time() - t
        stats = llm.collective_rpc("vpa_stats")[0]
        result[name] = {
            "seconds": elapsed,
            "stats": stats,
            "token_ids": [list(o.outputs[0].token_ids) for o in outs[:4]],
            "texts": [o.outputs[0].text for o in outs[:2]],
        }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k in ("mode", "load_s")}))
    for name in ("single", "batch"):
        print(name, json.dumps(result[name]["stats"]), repr(result[name]["texts"][0][:80]))


if __name__ == "__main__":
    main()
