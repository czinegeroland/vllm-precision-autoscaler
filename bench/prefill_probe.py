"""Prefill latency probe: why is prefill slower with the plugin?

Runs ONE configuration per process (the plugin reads VPA_* at import time):

    python3 bench/prefill_probe.py --config stock      --out results/prefill
    python3 bench/prefill_probe.py --config dual-high  --out results/prefill
    python3 bench/prefill_probe.py --config swap-high  --out results/prefill
    python3 bench/prefill_probe.py --config swap-eager --out results/prefill
    python3 bench/prefill_probe.py --config swap-high  --profile --out results/prefill

Measures single-request prefill time (generate with max_tokens=1) for several
prompt lengths, median of N repeats, all with cudagraph_mode FULL_DECODE_ONLY
(so prefill runs without graphs in every config) except swap-eager (NONE).
With --profile, wraps one 512-token prefill in vLLM's torch profiler.
"""

import argparse
import json
import os
import pathlib
import statistics
import time

CONFIGS = {
    # name: (env, quantization, cudagraph_mode)
    "stock": ({}, None, "FULL_DECODE_ONLY"),
    "dual-high": ({"VPA_FORMAT": "dual", "VPA_MODE": "high"}, "precision_autoscaler", "FULL_DECODE_ONLY"),
    "swap-high": ({"VPA_FORMAT": "swap", "VPA_MODE": "high"}, "precision_autoscaler", "FULL_DECODE_ONLY"),
    "swap-eager": ({"VPA_FORMAT": "swap", "VPA_MODE": "high", "VPA_SWAP_GRAPHS": "0"},
                   "precision_autoscaler", "NONE"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--lengths", default="128,512,2048")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", default="results/prefill")
    args = ap.parse_args()

    env, quant, cg_mode = CONFIGS[args.config]
    os.environ.update(env)
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams

    kwargs = dict(model=args.model, max_model_len=4096, gpu_memory_utilization=0.85, seed=0,
                  compilation_config={"cudagraph_mode": cg_mode})
    if quant:
        kwargs["quantization"] = quant
    trace_dir = out / f"trace_{args.config}"
    if args.profile:
        kwargs["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": str(trace_dir)}
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=1)

    def prompt(n: int, salt: int) -> dict:
        # distinct token ids per call so prefix caching never hits
        return {"prompt_token_ids": [1000 + ((salt * 7919 + i * 31) % 50000) for i in range(n)]}

    lengths = [int(x) for x in args.lengths.split(",")]
    for n in lengths:  # warm up every length (compilation, autotuning)
        llm.generate([prompt(n, 999_000 + n)], sp, use_tqdm=False)

    result = {"config": args.config, "model": args.model, "cudagraph_mode": cg_mode, "prefill_ms": {}}
    salt = 0
    for n in lengths:
        times = []
        for _ in range(args.repeats):
            salt += 1
            t = time.perf_counter()
            llm.generate([prompt(n, salt)], sp, use_tqdm=False)
            times.append((time.perf_counter() - t) * 1000)
        result["prefill_ms"][str(n)] = {"median": statistics.median(times), "min": min(times),
                                         "max": max(times)}

    if args.profile:
        llm.start_profile()
        for k in range(3):
            llm.generate([prompt(512, 555_000 + k)], sp, use_tqdm=False)
        llm.stop_profile()
        result["trace_dir"] = str(trace_dir)

    (out / f"{args.config}.json").write_text(json.dumps(result, indent=2))
    print("RESULT " + json.dumps({"config": args.config,
                                  **{k: round(v["median"], 1) for k, v in result["prefill_ms"].items()}}))


if __name__ == "__main__":
    main()
