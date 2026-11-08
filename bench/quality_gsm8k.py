"""GSM8K accuracy for one serving configuration (vLLM offline engine).

    python3 bench/quality_gsm8k.py --name stock-bf16 --out results/quality
    python3 bench/quality_gsm8k.py --name dual-fp8 --quant precision_autoscaler --mode fp8
    python3 bench/quality_gsm8k.py --name dual-auto-seq --quant precision_autoscaler \
        --mode auto --limit 100 --sequential        # low load: one request at a time

Greedy decoding, Qwen3 thinking disabled, answer = number after '####'
(fallback: last number in the reply). Saves per-question results so runs can
be compared pairwise (quality_compare.py).
"""

import argparse
import json
import os
import pathlib
import re
import time
import urllib.request

DATA_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
            "grade_school_math/data/test.jsonl")
PROMPT = ("Solve the following math problem step by step. "
          "End your reply with a final line of the form '#### <number>'.\n\n{q}")
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def load_gsm8k(cache: pathlib.Path) -> list[dict]:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, cache)
    return [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]


def norm(s: str) -> str:
    s = s.replace(",", "").strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def extract(text: str) -> str | None:
    if "####" in text:
        m = NUM.search(text.split("####")[-1])
        if m:
            return norm(m.group(0))
    nums = NUM.findall(text)
    return norm(nums[-1]) if nums else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--quant", default="none")
    ap.add_argument("--mode", default="")
    ap.add_argument("--threshold", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sequential", action="store_true", help="one request at a time (low load)")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--out", default="results/quality")
    args = ap.parse_args()

    use_plugin = args.quant == "precision_autoscaler"
    if use_plugin:
        os.environ["VPA_MODE"] = args.mode or "auto"
        os.environ["VPA_THRESHOLD"] = str(args.threshold)
        os.environ["VPA_COUNT"] = "1"
        os.environ.setdefault("VPA_FORMAT", "dual")

    from vllm import LLM, SamplingParams

    data = load_gsm8k(pathlib.Path(args.out) / "gsm8k_test.jsonl")
    if args.limit:
        data = data[: args.limit]

    kwargs = dict(model=args.model, max_model_len=2048, gpu_memory_utilization=0.90, seed=0)
    if args.quant != "none":
        kwargs["quantization"] = args.quant
    if use_plugin:
        from vllm_precision_autoscaler.worker_ext import EXTENSION
        kwargs["worker_extension_cls"] = EXTENSION
    t0 = time.time()
    llm = LLM(**kwargs)
    load_s = time.time() - t0

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    msgs = [[{"role": "user", "content": PROMPT.format(q=d["question"])}] for d in data]
    chat_kw = dict(use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
    if use_plugin:
        llm.collective_rpc("vpa_reset")
    t1 = time.time()
    if args.sequential:
        outs = [llm.chat(m, sp, **chat_kw)[0] for m in msgs]
    else:
        outs = llm.chat(msgs, sp, **chat_kw)
    gen_s = time.time() - t1
    stats = llm.collective_rpc("vpa_stats")[0] if use_plugin else None

    rows, correct = [], 0
    for d, o in zip(data, outs):
        text = o.outputs[0].text
        gold = norm(d["answer"].split("####")[-1])
        pred = extract(text)
        ok = pred == gold
        correct += ok
        rows.append({"gold": gold, "pred": pred, "ok": ok, "tokens": len(o.outputs[0].token_ids),
                     "text": text})
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "name": args.name, "model": args.model, "quant": args.quant, "mode": args.mode,
        "sequential": args.sequential, "n": len(rows), "correct": correct,
        "accuracy": correct / len(rows), "load_s": load_s, "gen_s": gen_s,
        "gen_tokens": sum(r["tokens"] for r in rows), "vpa_stats": stats,
    }
    (out_dir / f"{args.name}.json").write_text(json.dumps({"summary": summary, "rows": rows}))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
