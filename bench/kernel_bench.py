"""GEMM microbenchmark on the pod: BF16 (cuBLAS) vs nested 16-bit (Triton)
vs FP8 (vLLM CUTLASS, incl. activation quantization), for Qwen3-4B shapes.

    python3 bench/kernel_bench.py --out results/kernel_bench.csv
"""

import argparse
import csv

import torch
from vllm import _custom_ops as vops

from vllm_precision_autoscaler import nested_format
from vllm_precision_autoscaler.kernels.nested_gemm import nested_gemm

# (name, N, K) for Qwen3-4B: hidden 2560, intermediate 9728, 32 q heads, 8 kv heads, dim 128
SHAPES = [("qkv_proj", 6144, 2560), ("o_proj", 2560, 4096),
          ("gate_up_proj", 19456, 2560), ("down_proj", 2560, 9728)]
MS = [1, 4, 8, 16, 32, 64, 128, 256, 512, 2048]


def bench(fn, iters=200):
    """Time with a CUDA graph so launch overhead matches vLLM's decode path."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    g.replay()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters // 10):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters  # microseconds per call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/kernel_bench.csv")
    args = ap.parse_args()
    rows = []
    print(f"{'layer':>13} {'M':>5} {'bf16 us':>9} {'nested us':>10} {'fp8 us':>8} {'nested/bf16':>12} {'fp8/bf16':>9}")
    for name, n, k in SHAPES:
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        a, b, off, scale, _ = nested_format.encode(w)
        au, ks, a_t = a.view(torch.uint8), scale.view(-1), a.t()
        for m in MS:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            t_bf16 = bench(lambda: torch.nn.functional.linear(x, w))
            t_nested = bench(lambda: nested_gemm(x, au, b, off, ks))

            def fp8():
                xq, xs = vops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                return vops.cutlass_scaled_mm(xq, a_t, scale_a=xs, scale_b=scale, out_dtype=x.dtype)

            t_fp8 = bench(fp8)
            row = dict(layer=name, N=n, K=k, M=m, bf16_us=t_bf16, nested_us=t_nested, fp8_us=t_fp8)
            rows.append(row)
            print(f"{name:>13} {m:>5} {t_bf16:>9.1f} {t_nested:>10.1f} {t_fp8:>8.1f} "
                  f"{t_nested / t_bf16:>12.2f} {t_fp8 / t_bf16:>9.2f}")
        del w, a, b
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
