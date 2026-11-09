# Findings log: "Autoscale LLMs by lowering precision with vLLM"

Running notes for the Medium article. Every number here comes from a file
under `results/`; the file is named next to each table. Newest run at the
bottom. Times are UTC.

---

## 1. The idea in one paragraph

An LLM server is busy at some moments and quiet at others. When it is quiet,
the GPU has spare capacity, so it can afford the slower, more accurate 16-bit
(BF16) weights. When it is busy, it should switch to FP8: half the bytes per
weight and twice the tensor-core throughput, at a small accuracy cost. The
plugin `vllm-precision-autoscaler` makes that decision **on every engine step,
inside one vLLM process, with no restart and no second replica**.

### Prior work (for the "related work" paragraph)

- **NestedFP** (Lee et al., NeurIPS 2025, arXiv:2506.02024): FP16 when quiet,
  FP8 under load, using one memory format and a custom CUTLASS kernel. FP16
  models only; a fork of vLLM 0.8.5; switches when a step has more than 1024
  tokens. Our project borrows the idea, not the code.
- **MorphServe** (arXiv:2506.02006): swaps individual layers to quantized
  versions under load, and resizes the KV cache.
- **BrownoutServe** (arXiv:2507.17133): "brownout" degradation for MoE models
  under bursts (not quantization-based).
- In web searches on 2026-11-07/08, no vLLM plugin, RFC or practitioner write-up did
  per-step precision switching with stock vLLM.

### Why it can work (the roofline argument)

Each decode step reads every weight once, whether it serves 1 user or 100.
With few users, the step is **memory-bound**: speed ≈ memory bandwidth ÷
model size. Example (RTX 4090, ~1,008 GB/s):

| Format | Weights | Upper limit, 1 user | Measured (run 1) |
|---|---|---|---|
| BF16 | 7.56 GiB | ~132 tok/s | 101 tok/s (77%) |
| FP8 | 5.14 GiB | ~196 tok/s | 132 tok/s |

With many users or long prompts the step becomes **compute-bound**, and FP8
tensor cores do 2× the math of BF16.

---

## 2. Design (v0, the version used in the article)

- **Plugin, not a fork.** A Python package registered through vLLM's
  `vllm.general_plugins` entry point. `--quantization precision_autoscaler`
  routes every linear layer to our `LinearMethod`.
- **Two weight copies.** After loading, each BF16 weight also gets a
  per-channel FP8 (E4M3) copy. Only vLLM's existing kernels are used: cuBLAS
  for BF16, vLLM's CUTLASS FP8 GEMM (dynamic per-token activation scales) for
  FP8. **No custom kernel.**
- **The switch lives inside a custom torch op**
  (`torch.ops.precision_autoscaler.linear`). Reason: vLLM compiles the model
  with `torch.compile` once and skips guard checks, so a plain Python `if` in
  the layer could be frozen into the compiled graph. Inside an opaque custom
  op the branch really runs.
- **The decision depends only on tokens in the step** (`auto`: FP8 when the
  step has more than `VPA_THRESHOLD` tokens). That makes it compatible with
  **CUDA graphs**: vLLM records one graph per batch size, and each size
  always maps to one precision.
- **Exact telemetry inside CUDA graphs.** With `VPA_COUNT=1`, each call adds
  to a counter tensor on the GPU. The adds are recorded into the graphs, so
  the counts stay exact even when Python no longer runs. Read through a
  vLLM worker extension (`collective_rpc` → `vpa_stats`).
- Modes: `high` (always 16-bit), `fp8` (always FP8), `auto` (threshold),
  `dial` (per-layer thresholds, MLP layers switch before attention).

Stack decision: Python for the plugin (required by vLLM), C#/.NET planned for
the external controller and load generator.

---

## 3. Environment

| Item | Value |
|---|---|
| Cloud | RunPod Secure Cloud (Community Cloud had no RTX 4090 capacity both days) |
| GPU | 1× NVIDIA GeForce RTX 4090, 24 GB, compute capability 8.9, driver 595.91 |
| Price | $0.89/h (Community would be $0.34/h) |
| Image | `vllm/vllm-openai:v0.32.0-cu129` (pinned) |
| vLLM / torch / CUDA | 0.32.0 / 2.13.0+cu129 / 12.9 |
| Model | `Qwen/Qwen3-4B` (BF16). Qwen3-8B with two weight copies would not leave room for KV cache on 24 GB |
| Serve flags | `--max-model-len 4096 --gpu-memory-utilization 0.90 --seed 0` |
| Benchmark | `vllm bench serve`, random dataset, 512 input / 256 output tokens, `--ignore-eos`, prompts = 8 × concurrency (min 16), SLO for goodput: TTFT ≤ 2 s and TPOT ≤ 50 ms |

---

## 4. Problems hit along the way (useful "lessons learned" section)

1. **RunPod capacity.** Community 4090s were unavailable; with a host CUDA
   filter (12.8+) Secure Cloud was also "no instances". Dropping the filter
   worked; the host driver supported CUDA 12.9.
2. **SSH sessions don't get the container's Docker `ENV`** (`LD_LIBRARY_PATH`,
   `CUDA_HOME`, …). Fix: `deploy/pod/env.sh` copies the environment of PID 1.
3. **Broken `torchcodec` in the official image.** `v0.32.0-cu129` ships
   torchcodec 0.17.0 built for CUDA 13 (`libnvrtc.so.13`) inside a CUDA 12.9
   image. Importing it raises `OSError`, which `vllm.multimodal` doesn't catch,
   so even `import vllm.multimodal` fails. No cu129 build of 0.17 exists.
   Fix: uninstall torchcodec (vLLM treats a missing torchcodec as "no audio
   support").
4. **Windows line endings.** A Python edit on Windows wrote CRLF into a shell
   script; bash on the pod failed (`set: pipefail: invalid option name`).
   Fix: convert to LF, add `.gitattributes` (`eol=lf`).
5. **Test expectation too strict.** In `auto`, one request's prefill landed in
   a small first step (before the rest of the batch arrived), so it ran in
   16-bit and its output differed from pure FP8. That is the policy working
   correctly under realistic arrival; the check now allows it.
6. **CUDA "Error 804" on an older-driver host (run 2).** The Community host
   had driver 570 (CUDA 12.8). The image's CUDA 12.9 forward-compatibility
   `libcuda` (in `/usr/local/cuda/compat`, registered with `ldconfig`) only
   works on datacenter GPUs, so on a GeForce card PyTorch failed with
   `Error 804: forward compatibility was attempted on non supported HW` and
   every GPU test was silently *skipped*. Fix: remove the compat path from
   `/etc/ld.so.conf.d/` and run `ldconfig`; the host driver then runs the 12.9
   runtime fine (CUDA minor-version compatibility). Now automatic in
   `setup.sh`. Lesson: check that GPU tests *passed*, not just "didn't fail".
7. **Community pods can be slow to start.** Run 2's pod took ~15 min to get
   a public IP and SSH (slow 10 GB image pull), then worked normally.

---

## 5. Run 1 (2026-11-07/08): correctness and speed

Pod `36p144a9nftuws`, 55 min, ~$0.82. Files: `results/e2e_v0/`,
`results/bench_v0/Qwen_Qwen3-4B/`.

### 5.1 Correctness

- Unit tests (policy, settings): 11/11 pass.
- GPU op tests: 12/12 pass, including:
  - a recorded **CUDA graph keeps the precision chosen for its size** and
    replays correctly;
  - **`torch.compile` does not freeze the branch** (a runtime mode change is
    visible through a compiled function).
- End-to-end in a real vLLM engine (torch.compile + CUDA graphs on), 9/9:

| Run | Workload | 16-bit calls | FP8 calls | FP8 token share |
|---|---|---|---|---|
| `high` | 1 prompt | 6,912 | 0 | 0.0 |
| `high` | 128 prompts | 7,200 | 0 | 0.0 |
| `fp8` | 1 prompt | 0 | 6,912 | 1.0 |
| `fp8` | 128 prompts | 0 | 7,200 | 1.0 |
| `auto` (T=64) | 1 prompt | 6,912 | 0 | **0.0** |
| `auto` (T=64) | 128 prompts | 288 | 6,912 | **0.99** |

  Greedy outputs: `auto` with 1 prompt produced **exactly the same tokens** as
  `high`; `auto` with 128 prompts matched `fp8` for 3 of 4 sampled outputs
  (the 4th was the request whose prefill ran in a small 16-bit step).
  Qwen3-4B has 144 linear layers (36 decoder layers × 4: qkv, o, gate_up, down).

### 5.2 Memory: the cost of v0

From the server logs (`*.server.log`):

| Config | Weight memory | KV cache (tokens) |
|---|---|---|
| stock BF16 | 7.56 GiB | 88,592 |
| stock FP8 (`--quantization fp8_per_channel`) | 5.14 GiB | 104,192 |
| plugin, 16-bit mode | 11.57 GiB | 59,376 |
| plugin, FP8 mode | 11.57 GiB | 62,048 |
| plugin, auto | 11.57 GiB | 62,112 |

Two weight copies cost ~4 GiB, about 30% of the KV cache.

### 5.3 Throughput and latency (output tokens/s)

`results/bench_v0/Qwen_Qwen3-4B/summary.csv`

| Users | stock BF16 | stock FP8 | plugin 16-bit | plugin FP8 | plugin auto (T=32) |
|---|---|---|---|---|---|
| 1 | 101.1 | 132.4 | 101.3 | 129.7 | **101.4** |
| 8 | 684.7 | 896.1 | 685.3 | 879.3 | **691.4** |
| 32 | 1,964.6 | 2,386.0 | 1,966.0 | 2,333.5 | **2,039.8** |
| 64 | 2,647.8 | 3,132.4 | 2,642.1 | 3,071.1 | **3,077.4** |
| 128 | 2,931.8 | 3,557.1 | 2,546.6 | 2,962.1 | 2,958.6 |

p95 time to first token (ms):

| Users | stock BF16 | stock FP8 | plugin 16-bit | plugin FP8 | plugin auto |
|---|---|---|---|---|---|
| 1 | 54 | 42 | 47 | 42 | 42 |
| 8 | 230 | 188 | 228 | 186 | 182 |
| 32 | 733 | 445 | 626 | 500 | 540 |
| 64 | 1,180 | 922 | 1,180 | 920 | 992 |
| 128 | 2,423 | 1,783 | 8,058 | 7,279 | 7,279 |

Mean time per output token (ms): stock BF16 9.7 / 11.1 / 14.9 / 22.1 / 38.8;
stock FP8 7.4 / 8.5 / 12.6 / 18.9 / 34.0; plugin auto 9.7 / 11.2 / 14.6 /
19.1 / 32.1 (for 1 / 8 / 32 / 64 / 128 users).

### 5.4 What run 1 shows

1. **The switch works.** `auto` runs at BF16 speed up to 32 users and at FP8
   speed from 64 users: **+16% throughput vs BF16 at 64 users**, same GPU, no
   restart.
2. **The plugin itself costs ~nothing.** Forced 16-bit matches stock BF16
   (101.3 vs 101.1 tok/s at 1 user); forced FP8 is ~2% behind vLLM's own FP8.
3. **The v0 memory cost shows up at 128 users.** A request needs ~768 tokens
   of KV cache (512 + 256). 128 users need ~98k tokens; v0 has ~62k, so only
   ~80 run at once and the rest queue → p95 TTFT 7–8 s (stock: 1.8–2.4 s).
   At 64 users (~49k tokens) everything fits and there is no penalty.
4. **Counting tokens is not counting users.** A 512-token prompt exceeds the
   32-token threshold, so prompt processing (prefill) runs in FP8 even with
   one user; only generation stays 16-bit. Visible in TTFT: `auto` at 8 users
   has FP8-like TTFT (182 ms) but BF16-like TPOT (11.2 ms).
5. **The threshold is strict:** 32 users → decode steps of exactly 32 tokens →
   still 16-bit (`> 32` is required). That's why `auto` at 32 users is close to
   BF16.

---

## 6. Side track: v2 "nested" format (shelved)

To remove the memory cost, a BF16-native nested format was designed (FP8
byte + 1-byte remainder, exact BF16 rebuild for normal-range weights) with a
Triton kernel for the 16-bit path. Code is in the repo
(`nested_format.py`, `kernels/nested_gemm.py`, `VPA_FORMAT=nested`) but was
**not run on a GPU**. Decision (2026-11-08): keep the article kernel-free;
v0 (`VPA_FORMAT=dual`) is the default.

---

## 7. Run 2 (2026-11-08): quality and the KV cache question

Plan:
- GSM8K accuracy (1,319 questions, greedy, thinking off) for stock BF16,
  stock FP8, plugin 16-bit, plugin FP8, plugin auto under batch load, and
  plugin auto at low load (100 questions, one at a time).
- Benchmark on one host: stock BF16 and plugin auto, each with and without
  `--kv-cache-dtype fp8`, plus an accounting run that records the FP8 token
  share for each user count.
- Telemetry: `nvidia-smi` every second, `/metrics` after each user count.

Environment for run 2: RunPod **Community** Cloud, RTX 4090 at **$0.34/h**,
driver 570.211.01 (pod `yytqrag5mrr6c6`, rented 16:36 UTC). Same image,
vLLM, model and serve flags as run 1. Op GPU tests: 12/12 pass (after the
error-804 fix).

Note: run 1 and run 2 used different physical machines, so the run-2
benchmark repeats stock BF16 and plugin auto to compare on one host.

Results: sections 7.1-7.4 below.

### 7.1 Quality: GSM8K (run 2)

Files: `results/quality/*.json` (per-question replies), `compare.txt`.
Qwen3-4B, GSM8K test set (1,319 questions), greedy decoding, thinking
disabled, max 768 new tokens, answer = number after `####`. All 1,319
questions sent at once ("batch") unless noted.

| Run | What it is | n | Accuracy | 95% CI | FP8 token share |
|---|---|---|---|---|---|
| stock-bf16 | plain vLLM, BF16 | 1,319 | **92.12%** | 90.5–93.5 | – |
| stock-fp8 | vLLM online FP8 (`fp8_per_channel`) | 1,319 | 91.74% | 90.1–93.1 | – |
| dual-high | plugin forced 16-bit | 1,319 | 92.12% | 90.5–93.5 | 0.000 |
| dual-fp8 | plugin forced FP8 | 1,319 | 92.19% | 90.6–93.5 | 1.000 |
| dual-auto-batch | plugin auto, heavy load | 1,319 | 91.89% | 90.3–93.2 | 0.995 |
| dual-auto-seq | plugin auto, **one request at a time** | 100 | 96.0% | 90.2–98.4 | 0.246 |

Same first 100 questions for every run: stock-bf16 95, stock-fp8 94,
dual-high 94, dual-fp8 95, dual-auto-batch 95, dual-auto-seq 96 correct.
Mean reply length 260–265 tokens everywhere; no run had unparseable answers.

**Paired comparison** (questions right in A but wrong in B, and vice versa):

| A vs B | right only in A | right only in B | identical replies |
|---|---|---|---|
| dual-high vs stock-bf16 (same precision, two runs) | 11 | 11 | 688 / 1,319 |
| stock-bf16 vs stock-fp8 | 27 | 22 | 96 / 1,319 |
| dual-fp8 vs stock-bf16 | 24 | 23 | 109 / 1,319 |
| dual-fp8 vs stock-fp8 | 30 | 24 | 132 / 1,319 |

**Where auto at low load spent its steps** (`dual-auto-seq`, GPU counters):
16-bit 3,745,008 layer calls, FP8 14,400 calls = 14,400 / 144 layers =
**exactly 100 FP8 steps, one prefill per question**. Every generation step
(26,007 = 3,745,008 / 144) ran in 16-bit. Token counters: 8,496 prompt tokens
(~85 per question) went through FP8 and 26,007 generated tokens through
16-bit, so the FP8 *token* share is 0.246, while only 0.4% of *steps*
(100 of 26,107) were FP8.

What this shows:

1. **On GSM8K, FP8 costs Qwen3-4B no measurable accuracy.** All runs are
   within 91.7–92.2%, well inside each other's 95% intervals.
   stock FP8 vs BF16: −0.4 points, not significant.
2. **The noise floor is large.** Two 16-bit runs of the *same* model (plugin
   16-bit vs stock) gave identical replies on only 52% of questions and
   disagreed on correctness 22 times, because batched inference is not
   bit-reproducible when memory layout and batch composition differ (v0 has a
   smaller KV cache, so batches are scheduled differently). FP8 vs BF16
   disagreements (~47) are about 2× that noise floor: FP8 changes many
   replies, but not their correctness on this task.
3. **The plugin's precisions match vLLM's.** Plugin 16-bit = stock BF16
   accuracy exactly (92.12%); plugin FP8 ≈ stock FP8.
4. **Under heavy load, auto = FP8** (99.5% of tokens in FP8), as designed.
5. **At low load, auto generates every token in 16-bit**; only prompt
   prefill used FP8 (the token-threshold effect from run 1, now measured
   exactly).
6. **Honest caveat for the article:** GSM8K on a 4B model is too coarse to
   show the quality benefit of 16-bit at low load. A harder or more sensitive
   test (longer reasoning, larger model, perplexity/KL vs BF16 logits) would
   be needed to measure it. The argument for `auto` here rests on (a) FP8's
   known small accuracy cost in the literature (NestedFP Table 1: −0.1 to
   −1.6 points across models) and (b) the proof that low-load generation is
   bit-for-bit the 16-bit model (run 1 e2e test).

### 7.2 Benchmark on one host: does FP8 KV cache cancel the memory cost?

Files: `results/bench/Qwen_Qwen3-4B/` (`summary.csv`, per-run JSON/TXT,
`*.server.log`, `*.gpu.csv`, `*.c<N>.metrics.txt`, `*.vpa_stats.json`).
Same workload as run 1 (512 in / 256 out, prompts = 8 × users).
`+kv8` = `--kv-cache-dtype fp8`. Plugin threshold `VPA_THRESHOLD=32`.

**Memory** (server logs):

| Config | Weights | KV cache (tokens) |
|---|---|---|
| stock BF16 | 7.56 GiB | 88,400 |
| stock BF16 + FP8 KV | 7.56 GiB | 170,240 |
| plugin auto | 11.57 GiB | 57,520 |
| plugin auto + FP8 KV | 11.57 GiB | **109,984** |
| plugin auto (accounting run) | 11.57 GiB | 60,912 |

FP8 KV cache halves the bytes per cached token. With it, the plugin's KV
cache (110k) is **24% larger than stock BF16 without it (88k)**, so the cost
of the second weight copy is more than recovered. Note: two identical plugin
configs got 57.5k vs 60.9k tokens: vLLM's startup memory profiling varies by a
few percent between launches.

**Output throughput (tokens/s):**

| Users | stock BF16 | stock BF16 + FP8 KV | plugin auto | **plugin auto + FP8 KV** |
|---|---|---|---|---|
| 1 | 102.0 | 103.5 | 101.7 | 103.8 |
| 8 | 689.2 | 713.0 | 694.6 | 723.9 |
| 32 | 1,977.1 | 2,182.2 | 2,050.4 | 2,277.3 |
| 64 | 2,646.4 | 3,079.1 | 3,080.1 | **3,687.1** |
| 128 | 2,933.9 | 3,979.3 | 2,919.9 | **4,609.5** |

**p95 time to first token (ms):**

| Users | stock BF16 | stock BF16 + FP8 KV | plugin auto | plugin auto + FP8 KV |
|---|---|---|---|---|
| 1 | 44 | 49 | 39 | 42 |
| 8 | 227 | 230 | 179 | 181 |
| 32 | 465 | 530 | 319 | 397 |
| 64 | 1,202 | 1,241 | 946 | 955 |
| 128 | 2,485 | 2,347 | 7,160 | **1,867** |

**Mean time per output token (ms):** stock BF16 9.7 / 10.9 / 15.5 / 22.7 / 39.3;
plugin auto + FP8 KV 9.5 / 10.6 / 13.6 / 15.8 / 26.3 (1 / 8 / 32 / 64 / 128 users).

**Goodput** (requests/s meeting TTFT ≤ 2 s and TPOT ≤ 50 ms) at 128 users:
stock BF16 10.7, stock BF16 + FP8 KV 14.6, plugin auto 6.8,
**plugin auto + FP8 KV 17.2** (+61% vs stock BF16).

Run 1 vs run 2 (different machines): stock BF16 and plugin auto agree within
~1–2% at every user count, so results are reproducible across hosts.

### 7.3 Telemetry: where FP8 was actually used

Accounting run (`dual-auto+acct`, GPU counters on, prompts = 4 × users).
FP8 token share = tokens processed by FP8 layers / all tokens, from
`*.vpa_stats.json`:

| Users | FP8 token share | Explanation |
|---|---|---|
| 1 | 0.668 | = 512 / (512 + 256): every prompt token in FP8 (prefill steps > 32 tokens), every generated token in 16-bit |
| 8 | 0.507 | lower than 0.667 because ~24% of prompt tokens were prefix-cache hits (see below) and were never recomputed |
| 32 | 0.611 | decode steps have exactly 32 tokens → still 16-bit (threshold is `> 32`); prefill FP8 |
| 64 | **0.999** | decode steps of 64 tokens → FP8: the switch has happened |
| 128 | **0.998** | FP8 |

Counter overhead: ~2.5% throughput at 1 user (99.1 vs 101.7 tok/s), which is
why accounting ran separately.

**Prefix caching (methodology note).** vLLM caches prompt prefixes by default,
and `vllm bench serve --seed 42` generates the same first prompts at every
user count, so later runs reuse prompts from earlier ones. From
`vllm:prefix_cache_hits_total` / `queries_total` (stock BF16): 0% at 1 user,
**~24% of prompt tokens at 8 and 32 users**, 0% at 64 and ~0.1% at 128 (the
full KV cache had evicted the cached prefixes). Same for every config, so the
comparisons are fair, but absolute TTFT at 8–32 users is a little
optimistic. Next time: different seed per user count, or
`--no-enable-prefix-caching`.

**GPU telemetry** (`nvidia-smi` every second, samples with utilization > 5%):

| Config | GPU util (mean) | Max memory used | Mean power | Max power | Mean SM clock |
|---|---|---|---|---|---|
| stock BF16 | 100.0% | 22.2 GiB | 384 W | 465 W | 2,664 MHz |
| stock BF16 + FP8 KV | 99.4% | 22.4 GiB | 381 W | 455 W | 2,661 MHz |
| plugin auto | 99.7% | 21.8 GiB | 349 W | 439 W | 2,715 MHz |
| plugin auto + FP8 KV | 99.7% | 22.6 GiB | 347 W | 434 W | 2,701 MHz |

`nvidia-smi` utilization only says "a kernel was running", so it reads ~100%
for every config and is not a useful efficiency signal. **Power is**: the
plugin configs drew ~9% less average power than stock BF16 while producing
more tokens, i.e. better tokens per joule (rough: averages over whole runs,
not per user count). Memory used is ~22 GiB everywhere because vLLM always
claims 90% of the GPU; how it is split between weights and KV cache is
what differs.

### 7.4 Conclusions for the article

1. **It works, and it is cheap to run.** Per-step BF16↔FP8 switching inside
   one vLLM process, as a plugin, with no custom kernel. At low load it is the
   BF16 model (same speed, identical generated tokens); under load it runs at
   FP8 speed.
2. **Alone, the plugin's gain is capped by memory.** Two weight copies cost
   ~30% of the KV cache. Up to 64 users it wins (+16% vs BF16); at 128 users
   requests queue (p95 TTFT 7 s).
3. **With FP8 KV cache it is the best configuration tested.** At 128 users:
   4,609 tok/s (+57% vs stock BF16, +16% vs stock BF16 + FP8 KV), p95 TTFT
   1.87 s (within a 2 s SLO), goodput 17.2 req/s (+61%).
4. **Fair comparison point:** much of the gain at high load comes from FP8 KV
   cache alone (stock BF16 + FP8 KV: 3,979 tok/s at 128 users). The plugin
   adds +20% at 64 users and +16% at 128 on top of it.
5. **Quality:** on GSM8K, FP8 cost Qwen3-4B no measurable accuracy (all runs
   91.7–92.2%), so this benchmark can't show the benefit of 16-bit at low
   load; the low-load path is provably the BF16 model, though. A more
   sensitive quality test is future work. FP8 KV cache's own quality effect
   was **not** measured in this run.
6. **Tokens vs users:** the token threshold sends every prompt (prefill)
   through FP8 even at 1 user. A policy based on the number of decoding
   requests would keep low-load prefill in 16-bit too.

### 7.5 Cost so far

| Run | Pod | GPU | $/h | Duration | Cost |
|---|---|---|---|---|---|
| 1 | `36p144a9nftuws` | RTX 4090 Secure | 0.89 | 55 min | ~$0.82 |
| (v2 attempt) | `isswavgnzoosbu` | RTX 4090 Secure | 0.89 | 6 min | ~$0.09 |
| 2 | `yytqrag5mrr6c6` | RTX 4090 Community | 0.34 | 83 min | ~$0.47 |
| **Total** | | | | | **~$1.38** |

### 7.6 Open items / next experiments

- Quality of FP8 KV cache (GSM8K with `--kv-cache-dtype fp8`).
- A more sensitive quality metric (perplexity or KL divergence vs BF16 logits;
  a longer-reasoning benchmark; a larger model).
- "Decoding requests" policy instead of "tokens per step".
- Bursty traffic (C# load generator replaying a real trace) with a Grafana
  timeline: request rate, FP8 share, queue length, TTFT.
- Rerun with prefix caching off or a different seed per user count.

---

## 8. The swap format: your original idea (one copy in VRAM, both in CPU RAM)

Clarification first: runs 1–2 kept **both** weight copies in VRAM (`dual`).
FP8 KV cache (`--kv-cache-dtype fp8`, an existing vLLM setting) shrank the
memory per cached token instead; it does not move weights anywhere, and with it
on, low-load mode is BF16 weights + FP8 KV cache, not pure BF16.

The swap format (`VPA_FORMAT=swap`) implements the original idea: **only one
precision in VRAM, both kept in CPU RAM, copied in when the precision
changes, with the copy prefetched in the background so the switch has no
pause.**

### 8.1 Design

- **VRAM:** each linear layer owns one GPU "region" the size of its BF16
  weight: the loaded weight's own memory, so **no extra VRAM** (VRAM weights
  = stock BF16, 7.56 GiB, so the KV cache should be ~stock BF16's 88k tokens).
  The region holds either the BF16 weight or, in its first half, the FP8
  weight. FP8 per-channel scales (tiny) stay resident.
- **CPU RAM:** pinned copies of both versions (Qwen3-4B: ~7.5 GB BF16 +
  ~3.8 GB FP8 for the linear layers). Pinned memory is needed for fast
  asynchronous PCIe copies.
- **Background migration (the prefetch idea):** when the target precision
  changes, layers migrate one by one:
  1. on a separate CUDA stream, copy the layer's new weights CPU → a small
     staging buffer (4 layer-sized buffers ≈ 400 MB, not a second model);
     this overlaps with the model's own kernels;
  2. at the start of a later engine step, once that copy finished: staging →
     region (GPU-to-GPU, fast) on the main stream, then flip that layer's
     precision.
  Every step therefore sees a consistent set of weights; during a migration
  some layers are FP8 and some 16-bit, for a short time.
- **Trigger:** `VPA_MODE=auto` uses a hysteresis policy on smoothed tokens per
  step (EMA, 2 s time constant): FP8 above 64, back to 16-bit below 32, at most
  one change per 5 s. `VPA_MODE=high|fp8` + `vpa_set_mode` over
  `/collective_rpc` lets an external controller (later: C#) decide instead.
- **Constraint:** **CUDA graphs must be off**
  (`--compilation-config '{"cudagraph_mode": "NONE"}'`): a recorded graph
  replays fixed kernels on fixed addresses, so it would keep running the old
  kernel on the new bytes. The plugin refuses to start otherwise. The cost of
  running without graphs is measured below (stock BF16 with and without
  graphs).
- Difference from `dual`: switching is **per load period** (seconds), not per
  step, because a migration takes time. That is closer to an autoscaler.

### 8.2 Tests written for it

- Unit tests (17 total now): policy hysteresis, a single long prompt does not
  trigger a switch, dwell time prevents flapping.
- GPU tests: BF16 → FP8 → BF16 round trip with **bit-exact restore** of BF16
  outputs, region memory never reallocated, every step during a migration
  gives each layer exactly BF16 or exactly FP8 output, at least one step had
  mixed precisions; pinned host→device bandwidth printed.
- End-to-end in vLLM: generate in 16-bit, switch to FP8 over RPC while
  serving, generate, switch back, generate again; the last output must equal
  the first exactly.

### 8.3 Run 3 plan

- Tests above.
- Steady benchmark: stock BF16 (graphs on), stock BF16 with graphs off,
  swap fixed 16-bit, swap fixed FP8, swap auto.
- Phased load (new `bench/phase_load.py`): 4 users for 45 s → 96 users for
  60 s → 4 users for 60 s; random prompts (no prefix-cache hits); timeline of
  request latencies, vLLM queue/KV usage and the plugin's precision state
  every 0.5 s. Configs: stock BF16, dual auto, swap auto. The question: does
  the swap switch happen **without a latency spike**, and how long does it
  take?

## 9. Run 3 (2026-11-08): the swap format on a GPU

Environment: RunPod **Secure** Cloud RTX 4090, $0.89/h (Community full), driver
595.91, host RAM 503 GB (shared host), pod `rhvkyudwy4ukdx`. Same image, vLLM,
model.

### 9.1 Tests

- Unit tests: 17/17 pass. GPU tests: **15/15 pass** (12 op tests + 3 swap tests:
  round trip with bit-exact BF16 restore, consistency of every step during a
  migration, bandwidth).
- **Pinned CPU → GPU bandwidth: 22.5 GB/s** (PCIe 4.0 x16, measured with a
  512 MiB pinned copy).
- **End-to-end in vLLM** (`results/e2e/swap.json`), CUDA graphs off:

| Check | Result |
|---|---|
| FP8 output differs from 16-bit output | yes |
| After FP8 → 16-bit round trip, outputs identical to the first 16-bit run | **16 / 16 prompts, exactly** |
| Migration to FP8 | 144 layers, 3.38 GiB, **1.02 s**, 37 engine steps |
| Migration back to 16-bit | 144 layers, 6.77 GiB, **0.86 s**, 37 engine steps |

  So the weights really are restored bit for bit from CPU RAM, and the model
  keeps serving during the migration (37 steps ran while layers moved).
  Note: in this test the engine steps came from small "filler" requests with
  gaps between them, so the migration time here is bounded by step frequency
  (4 staging slots → up to 4 layers per step), not by PCIe: 6.77 GiB at
  22.5 GB/s would take ~0.3 s. Under real load (steps every ~10–40 ms) see 9.3.

### 9.2 Status when run 3 was stopped (2026-11-08 ~18:55 UTC)

Run 3 was stopped early (user shut down the laptop; pod terminated so it
wouldn't bill unattended). Pod time ~45 min at $0.89/h ≈ **$0.67**; total
spend so far ≈ **$2.05**.

Finished: swap tests (9.1); steady benchmark for **stock BF16 only** at 1, 8,
32, 64 users (101 / 692 / 1,991 / 2,674 tok/s; consistent with runs 1–2;
files in `results/bench/Qwen_Qwen3-4B/stock-bf16.*`).

**Not yet run** (next session, ~40 min): stock BF16 with CUDA graphs off,
swap fixed 16-bit, swap fixed FP8, swap auto; then the phased load test
(`bench/run_phases.sh`: stock BF16, dual auto, swap auto). Restart with:

    CONFIGS="stock-bf16+eager swap-high swap-fp8 swap-auto" bash bench/run_bench.sh
    CONFIGS="stock-bf16 dual-auto swap-auto" bash bench/run_phases.sh

### 9.3 Run 3b (2026-11-08 evening): the cost of turning CUDA graphs off

Pod `0hyfziu5kk9x1s`, RunPod Secure RTX 4090, $0.89/h, 52 min ≈ **$0.77**
(total so far ≈ **$2.82**). Files: `results/bench_run3/`, `results/ab_eager/`.

**Stock vLLM BF16, CUDA graphs on vs off** (same host, 512 in / 256 out):

| Users | Graphs on (tok/s) | Graphs off (tok/s) | Cost of no graphs | TPOT on → off (ms) |
|---|---|---|---|---|
| 1 | 101.7 | 70.4 | **-31%** | 9.7 → 14.1 |
| 8 | 690.5 | 542.0 | -22% | 11.1 → 14.3 |
| 32 | 1,994.1 | 1,681.0 | -16% | 15.0 → 18.1 |
| 64 | 2,677.9 | 2,622.9 | -2% | 22.2 → 22.7 |
| 128 | 2,966.9 | 2,945.8 | -1% | 38.4 → 38.8 |

Without graphs the CPU launches every kernel itself. With few users each
kernel is tiny, so launch overhead dominates; with many users the GPU work
per kernel hides it. KV cache without graphs: 91,728 tokens (no memory
reserved for recorded graphs) vs 88,592 with.

**Swap format memory: confirmed.** `swap-high` KV cache = **88,784 tokens**,
the same as stock BF16 (88,592), vs 57–62k for the dual format: keeping the
second copy in CPU RAM gives the whole KV cache back (the 4 × ~100 MB staging
buffers cost ~3k tokens vs stock without graphs).

**Swap format speed without graphs: the plugin adds Python overhead.**
Quick A/B on the same host (`bench/ab_eager.sh`, 512 in / 128 out, all
without graphs):

| Users | stock BF16 | swap, fixed 16-bit | Overhead |
|---|---|---|---|
| 1 | 70.9 tok/s (13.9 ms) | 57.5 tok/s (17.2 ms) | -19% |
| 8 | 496.8 tok/s (15.0 ms) | 356.2 tok/s (21.4 ms) | -28% |

Without graphs the plugin's custom op runs its Python on every one of the
144 layer calls per step; with graphs that Python is recorded away (which is
why the dual format showed ~0 overhead). So, at low load, the swap format
without graphs pays twice: no graphs (-31%) **and** plugin Python (-19%).

**Noisy-neighbour effect.** The first swap benchmark attempt on this shared
host measured 166 tok/s at 8 users (47.7 ms TPOT): vLLM's engine process sat
at 100% CPU while the GPU was only ~35% busy (stock without graphs: 88%),
and the host load average was ~38. Re-measured minutes later: 356 tok/s.
CPU-bound configurations (no graphs) are very sensitive to CPU contention on
shared cloud hosts. `py-spy` could not attach (container lacks
`SYS_PTRACE`).

**Bug found: vLLM's compile cache ignores plugin settings.** vLLM caches the
compiled model on disk (`~/.cache/vllm/torch_compile_cache`, AOT compile).
The cache key includes `--quantization precision_autoscaler` but not our
`VPA_FORMAT`, so after a `swap` run a `dual` run loaded the swap-format graph
and crashed (`KeyError: 'vpa_region'`). Runs 1–2 were not affected (one format
per fresh pod). Fix: set `VLLM_DISABLE_COMPILE_CACHE=1` when switching
formats on one machine (or use a separate `VLLM_CACHE_ROOT` per format).

**Conclusion so far for the swap format:** the memory side of the idea works
perfectly (full KV cache, ~1 s migrations, bit-exact restore), but running
without CUDA graphs makes it slower than stock at low load. Next step:
keep CUDA graphs on by recording **two sets of graphs** (one per precision)
from the plugin, without modifying vLLM or writing a kernel.

## 10. Swap format + CUDA graphs: two graph sets, from the plugin only

Constraint (user decision): no new kernel, and **no change to vLLM itself**;
everything lives in the plugin.

Key insight: **a recorded CUDA graph stores which kernels run on which memory
addresses, not the data.** So the plugin records every decode graph **twice**
at startup, once with the 16-bit kernels and once with the FP8 kernels, on the
same weight addresses (the memory may still hold BF16 while the FP8 set is
recorded; recorded outputs are discarded). At runtime it replays the set that
matches what the memory currently holds.

Implementation (`plugin/vllm_precision_autoscaler/graphs.py`), for vLLM
v0.32.0's model runner V2 (`vllm/v1/worker/gpu/`):

| vLLM piece | What the plugin does |
|---|---|
| `CudaGraphManager.graphs` (dict: batch shape → graph) | replaced by `PrecisionGraphs`: one dict per precision, the current precision's one is visible |
| `ModelCudaGraphManager.capture` | runs twice, with the op told which precision to record (skipped for vLLM's memory-profiling pass) |
| `CudaGraphManager.dispatch` | during a migration (mixed layers) takes vLLM's own "no graph" path, so those steps run eagerly |
| `GPUModelRunner.execute_model` | calls the swap manager's `on_step` **before** each real step (policy, background copies, flips), because a replay never runs the plugin's Python |

Requires `cudagraph_mode` `FULL_DECODE_ONLY` (or `FULL`): full graphs for
decode steps, prefill steps eager. Piecewise graphs are rejected: they live
inside torch.compile'd pieces and would freeze one precision. Honest caveat:
this is **runtime patching** (monkeypatching) of vLLM classes from the plugin;
vLLM's source is untouched, but the plugin is tied to this vLLM version.

Expected effect: low-load decode runs from graphs again (no -31% "no graphs"
cost and no per-layer plugin Python), while the memory benefit stays (one copy
in VRAM). Extra cost: a second graph set (~0.2–0.4 GB, outside vLLM's
memory-profiling estimate, taken from the ~10% headroom) and a longer startup.

### 10.1 Run 4 tests (2026-11-08, pod `f33305b98xe202`, Secure RTX 4090, $0.89/h)

Unit 17/17, GPU 15/15 pass. Pinned CPU→GPU bandwidth on this host:
**13.3 GB/s** (vs 22.5 GB/s on run 3's host: PCIe bandwidth varies by host).

End-to-end swap round trip in vLLM, two graph sets vs no graphs
(`results/e2e/swap_FULL_DECODE_ONLY.json`, `swap_NONE.json`; offline engine,
1-request decode speed, 256 tokens):

| | Two graph sets (`FULL_DECODE_ONLY`) | No graphs (`NONE`) |
|---|---|---|
| Graphs recorded | **35 (16-bit) + 35 (FP8)** | – |
| Graph memory | 0.73 GiB (both sets) | – |
| Decode, 16-bit | **103.4 tok/s** (stock BF16 ≈ 102) | 93.3 tok/s |
| Decode, FP8 | **135.4 tok/s** (stock FP8 ≈ 132) | 55.2 tok/s |
| Round trip outputs identical | 16/16 | 16/16 |
| Migration to FP8 (3.38 GiB) | 0.76 s, 37 steps | 0.59 s, 37 steps |
| Migration to 16-bit (6.77 GiB) | 0.89 s, 40 steps | 0.63 s, 49 steps |

**With two graph sets the swap format decodes at stock speed in both
precisions**, while keeping only one weight copy in VRAM. Without graphs the
FP8 path was especially slow (55 tok/s): two extra small kernels per layer
(activation quantization + scaled GEMM) cost the most when every kernel is
launched by the CPU.

### 10.2 Run 4 benchmark: swap format with two graph sets

Same pod (`f33305b98xe202`), 65 min ≈ **$0.97**; total spend ≈ **$3.79**.
Files: `results/bench_run4/` (steady), `results/phases/` (phased timeline).
`+fdo` = stock vLLM with the same `FULL_DECODE_ONLY` graph mode as the swap
format (fair baseline).

**Memory** (KV cache tokens): stock BF16 88,592 · stock BF16 `+fdo` 90,864 ·
swap 16-bit 87,920 · swap FP8 / auto 92,080. Swap keeps the full KV cache
(dual format: ~58–62k).

**Steady benchmark, output tokens/s** (512 in / 256 out):

| Users | stock BF16 | stock BF16 +fdo | swap 16-bit | swap FP8 | **swap auto** |
|---|---|---|---|---|---|
| 1 | 102.2 | 102.2 | 102.2 | 134.7 | 106.4 |
| 8 | 694.0 | 692.2 | 694.6 | 910.3 | 774.0 |
| 32 | 2,003.0 | 1,995.6 | 1,997.6 | 2,405.7 | 2,225.5 |
| 64 | 2,685.9 | 2,680.1 | 2,687.7 | 3,127.4 | **3,132.6** |
| 128 | 2,989.4 | 3,014.6 | 2,975.3 | 3,425.7 | **3,425.3** |

p95 TTFT at 128 users: stock BF16 2,294 ms · swap auto **1,748 ms**.
Goodput at 128 users: stock BF16 11.0 · swap auto **12.8** req/s.

- **Swap 16-bit = stock BF16 at every user count** (±1%), with the full KV
  cache. With two graph sets, the no-graphs and plugin-Python penalties are
  gone for decode.
- **Swap FP8 = full FP8 speed** from a single weight copy (134.7 tok/s at 1 user
  vs stock FP8 ~132 in run 1).
- **Swap auto at 64 and 128 users = swap FP8** (+17% / +15% vs stock BF16).
- At 1–32 users `auto` lands between the two (the steady benchmark's prompt
  bursts push the smoothed load around the thresholds, and the startup bug
  below also put it in FP8 for the first seconds).
- Swap 16-bit p95 TTFT at 1 user was higher (63 ms vs ~39 ms): prefill steps
  still run without graphs and go through the plugin's per-layer Python.

### 10.3 Run 4 phased load: quiet → busy → quiet, live switching

`bench/phase_load.py`: 4 users for 45 s → **96 users for 60 s** → 4 users for
60 s; random prompts (no prefix-cache hits), 512 in / 256 out, streaming.
"Token gap" = time between two streamed tokens of one request (p50 / p99 /
**max**; the max shows any pause).

| Config | Phase | tok/s | p95 TTFT | Token gap p50 / p99 / **max** |
|---|---|---|---|---|
| stock BF16 | quiet (4) | 371 | 112 ms | 10.4 / 22.4 / 134 ms |
| | busy (96) | 2,960 | 1,702 ms | 21.4 / 125.7 / 140 ms |
| | quiet (4) | 683* | 120 ms | 10.4 / 23.8 / 35 ms |
| dual auto | quiet | 376 | 98 ms | 10.3 / 20.8 / 102 ms |
| | busy | 3,010 | 2,234 ms | 19.4 / 97.1 / **4,524 ms** |
| | quiet | 717* | 55 ms | 10.3 / 22.1 / **3,569 ms** |
| **swap auto** | quiet | 387 | 129 ms | 10.3 / 27.2 / 120 ms |
| | busy | **3,307** | **1,655 ms** | 20.3 / 100.6 / **120 ms** |
| | quiet | 743* | 126 ms | 10.3 / 21.3 / 58 ms |

\*includes requests from the busy phase still finishing.

Live telemetry during the busy phase (sampled every 0.5 s):
- dual auto: 81–89 requests running, **up to 15 waiting, KV cache 99–100%**
  (two weight copies leave too little KV cache for 96 users; the 3.6–4.5 s
  token gaps are requests stalled behind the full KV cache).
- swap auto: **96 running, 0 waiting, KV cache 68–70%**, 144/144 layers FP8.

**Swap-auto switch timeline** (`results/phases/swap-auto.json`, t = seconds
since the test started; busy phase from t = 45 s to t = 105 s):

| t (s) | Event | Layers in FP8 | Smoothed load (tokens/step) | Running / waiting |
|---|---|---|---|---|
| 0.0–1.1 | startup glitch: to FP8 (see bug below) | 16 → 144 | 512 → 317 | 0–4 / 0 |
| 8.2–9.8 | back to 16-bit | 136 → 0 | 31 | 4 / 0 |
| **45.3** | **busy phase starts → target FP8** (0.3 s after the jump) | 6 | 199 | 14 / 86 |
| 45.3–49.1 | migration while 96 requests arrive and prefill | 6 → 144 | 637 → 1,441 → 745 | 14 → 100 / 86 → 0 |
| **49.1** | **fully FP8** (3.8 s migration) | 144 | 745 | 100 / 0 |
| **116.1** | **quiet again → target 16-bit** | 101 | 28 | 4 / 0 |
| **117.2** | **fully 16-bit** (1.1 s migration) | 0 | 18 | 4 / 0 |

(The switch back came ~11 s after the busy phase ended: in-flight busy
requests kept the load up for a while, then the smoothing and the 5 s
minimum dwell time applied.)

**Largest token gap in the whole swap-auto run: 120 ms**, at t = 47.4 s, in
the middle of the to-FP8 migration, during the big prefill steps of 96 new
requests. That's below stock BF16's own max (134–140 ms). **No pause was
caused by switching precision.**

Switch telemetry from the plugin (`vpa_stats` → `migrations`): to FP8 3.38 GiB
in 3.69 s (66 steps); to 16-bit 6.77 GiB in 1.22 s (65–73 steps). The to-FP8
switch under a burst is slower because the migration advances at most 4 layers
per engine step, and steps at the start of a burst are big prefill steps
(~50–300 ms each, run without graphs).

### 10.4 Bugs found in run 4 (fixed after the run)

1. **Startup switch.** The policy's smoothed load started at the first step's
   value; the first step was a 2,048-token prefill, so swap auto switched to
   FP8 at startup and back after the 5 s dwell. Fix: start the average at 0
   (`swap_policy.py`, new unit test).
2. **Plugin log lines missing.** The logger was named `vllm_precision_autoscaler`;
   vLLM only configures handlers under `vllm.`, so INFO lines (switch events)
   were dropped. Fix: `vllm.precision_autoscaler`.

### 10.5 Conclusion: the original idea, end to end

The user's original design (one precision in VRAM, both in CPU RAM, switch on
the fly with background prefetch) works in stock vLLM v0.32.0 as a plugin,
with no custom kernel and no change to vLLM's source:

- **Memory:** full KV cache (~88–92k tokens, like stock BF16), unlike keeping
  both copies in VRAM (~58–62k).
- **Speed:** 16-bit mode = stock BF16; FP8 mode = stock FP8 speed; with the
  two graph sets there is no low-load penalty.
- **Switching:** automatic under a real burst; to FP8 in 3.8 s, back in 1.1 s;
  **no pause** (max token gap 120 ms).
- **Busy phase vs stock BF16:** +12% throughput (3,307 vs 2,960 tok/s), lower
  p95 TTFT (1,655 vs 1,702 ms), with 0 queued requests.
- **Costs and caveats:** ~11.5 GB pinned CPU RAM; second graph set (~0.7 GB
  total graph memory, both sets); startup ~2.5 min (compile cache disabled +
  two captures); runtime patching of vLLM's graph manager, so the plugin is
  tied to vLLM v0.32.0; prefill steps still run without graphs.

## 11. H100 NVL + Qwen3-32B (run 5, 2026-11-08/09)

Pod `q8ngw7sfngt7j7`, RunPod **Community H100 NVL (94 GB)**, $2.59/h, 40 min ≈
**$1.73**; total spend ≈ **$5.6**. (H100 PCIe was out of stock.) Driver 570, 1.5 TB
host RAM. **No SSH copying:** the pod cloned the public repo (`main`) at boot and
installed the plugin with `pip install git+…@main#subdirectory=plugin`; later the
v0.3.0 control API was pushed to GitHub and the pod re-installed from `main`.
Files: `results/bench_h100/`, `results/manual/`.

### 11.1 Memory: the dual format can't load a 32B model, swap can

| Config | Weights in VRAM | KV cache (tokens) |
|---|---|---|
| dual (BF16 + FP8 in VRAM) | ~91.5 GiB | **fails: CUDA out of memory** during torch.compile (31 MiB free of 93 GiB) |
| stock BF16 (`FULL_DECODE_ONLY`) | 61.0 GiB | 81,840 |
| stock FP8 | 33.6 GiB | 193,296 |
| **swap** | **62.0 GiB** (one copy) | **77,008** (94% of stock BF16; staging 2 × 500 MiB) |

Swap pinned **87.2 GiB** of CPU RAM (both copies, 256 linear layers). Load time 75 s
(stock: 11 s).

### 11.2 Steady benchmark (512 in / 256 out, tok/s)

| Users | stock BF16 | stock FP8 | swap, built-in policy (default 5 s dwell) |
|---|---|---|---|
| 1 | 37.1 | 63.7 | 38.5 |
| 8 | 257.8 | 436.6 | 248.0 |
| 16 | 478.5 | 820.0 | 479.2 |
| 32 | 692.2 | 1,181.4 | 941.9 |

p95 TTFT at 32 users: BF16 3,232 ms · FP8 1,642 ms · swap 3,343 ms.

- **FP8 matters far more on 32B: +69–72% at every user count** (4B: +16–31%).
- Swap in 16-bit = stock BF16 (38.5 vs 37.1 tok/s at 1 user).
- **The built-in policy failed on 32B: it flapped.** With 8 and 16 users it switched
  every ~5 s (its minimum dwell), several migrations were reversed half-way, and
  each migration took 4–8 s (256 layers, at most 2 per engine step with 2 staging
  slots). Steps during a migration run without CUDA graphs, so swap got *neither*
  speed: 248 vs 258 (BF16) vs 437 (FP8) tok/s at 8 users. At 32 users it stayed in
  FP8 (942 tok/s, +36% vs BF16).
- **Idle stall:** a migration only advances on engine steps; one started right
  before the engine went idle took **43.9 s** (145 steps). With traffic: 4–5 s.

**Design decision (user): vLLM must not decide when to switch.** v0.3.0 makes
external control the default: the plugin exposes `GET/POST /v1/precision`
(a vLLM endpoint plugin, no dev mode), the built-in policy is opt-in
(`VPA_SWAP_POLICY=hysteresis`), and `GET` also advances a pending migration
while the engine is idle (fixes the idle stall for any polling controller).

### 11.3 Manually controlled switching through the API (v0.3.0)

Server: `VLLM_PLUGINS=precision_autoscaler,precision_autoscaler_api VPA_FORMAT=swap
VPA_MODE=high` (no dev mode, no worker-extension flag, no built-in policy). Load:
4 users 45 s → **32 users 90 s** → 4 users 75 s; switches sent by us with `curl` at
the phase boundaries (`results/manual/switches.log`):

| UTC | Action / state |
|---|---|
| 23:34:15 | server ready: `mode=high, fp8_layers 0/256` |
| **23:35:00** | **`POST {"mode":"fp8"}`** as the 32 users arrive → migrating |
| 23:35:09 | **fully FP8: 29.06 GiB in 8.0 s (143 steps)**; 36 running, 0 waiting, KV 31% |
| 23:35:09–23:36:30 | stays FP8 (no flapping): 31–32 running, 0 waiting, KV 29–31% |
| **23:36:30** | **`POST {"mode":"high"}`** at the end of the busy phase |
| 23:36:36 | mid-migration: 52/256 layers still FP8, 36 running |
| 23:36:38 | **fully 16-bit: 58.12 GiB from CPU RAM in 7.2 s (218 steps)** |

| Phase | tok/s | p50 / p95 TTFT | token gap p50 / p99 / max |
|---|---|---|---|
| quiet (4) | 138 | 438 / 446 ms | 27.0 / 39.3 / 1,639 ms |
| busy (32), FP8 | 937 | 2,284 / 3,329 ms | 22.8 / 44.2 / 1,639 ms |
| quiet (4) | 225* | 445 / 450 ms | 27.6 / 40.2 / 328 ms |

\*includes busy requests finishing.

- The API works end to end: a controller (here: us) decides, the plugin executes,
  `GET` shows the live state; both directions migrate while serving with 0 queued
  requests.
- **One 1.6 s stall** at t = 45.1 s, the moment 32 requests (16k prompt tokens) arrived
  *and* the FP8 migration started. The stock-BF16 phased baseline on this host was
  cancelled to save money, so this run can't separate burst-prefill stall from
  migration stall. The switch back (7.2 s) caused no gap above 328 ms.
- Busy-phase FP8 throughput 937 tok/s matches the steady swap result (942) but is
  below stock FP8 (1,181), and TTFT is high (p50 2.3 s). Likely cause: in the swap
  format, prefill steps run without CUDA graphs (`FULL_DECODE_ONLY`) and through the
  plugin's per-layer Python; on 32B with long prompts prefill is a large share of the
  work. Quiet-phase TTFT (438 ms for a 512-token prompt) points the same way.

### 11.4 Open items after run 5

1. **Prefill performance in the swap format** (TTFT 438 ms quiet, 2.3 s busy): reduce
   per-layer Python in eager steps, or allow piecewise graphs for the current precision.
2. **Migration speed:** copy more than 2 layers per step / continue copying between
   steps; 256 layers at 2 per step is the bottleneck on 32B.
3. Stock-BF16 phased baseline on the H100 (to attribute the 1.6 s stall).
4. External controller in C#/.NET using `GET /metrics` + `/v1/precision`.

## 12. Prefill investigation (run 6, 2026-11-09): the swap format is *not* slower

Pod `m8o4ds114pak1k`, RunPod Secure RTX 4090 (got one after ~1 min of retrying every
30 s; 4090/L40S/6000 Ada/5090 were all out of stock earlier), 16 min ≈ **$0.23**;
total spend ≈ **$5.85**. Qwen3-4B, vLLM v0.32.0, all configs with `cudagraph_mode`
`FULL_DECODE_ONLY` (prefill without graphs) unless noted. Files: `results/prefill/`,
`results/ttft_check/`, scripts `bench/prefill_probe.py`, `bench/analyze_trace.py`,
`bench/ttft_server_check.sh`.

**Question:** §11.3 reported TTFT 438 ms (swap) vs ~122 ms (stock) on Qwen3-32B, and
63 vs 40 ms p95 on Qwen3-4B, suggesting the plugin slows prefill.

**1. Prefill time in isolation** (offline engine, one request at a time,
`max_tokens=1`, distinct prompts, median of 10):

| Config | 128 tokens | 512 tokens | 2,048 tokens |
|---|---|---|---|
| stock | 14.0 ms | 30.7 ms | 114.6 ms |
| dual, 16-bit | 14.2 ms | 30.9 ms | 114.8 ms |
| swap, 16-bit (two graph sets) | 14.0 ms | 30.8 ms | 114.5 ms |
| swap, 16-bit, no graphs at all | 14.2 ms | 30.4 ms | 114.8 ms |

**2. torch profiler, one 512-token prefill ×3** (`results/prefill/trace_summary.txt`):

| | stock | swap 16-bit |
|---|---|---|
| GPU kernels | 1,365 | 1,365 |
| kernel launches | 678 | 678 |
| GPU kernel time | 83.3 ms | 83.6 ms |
| GPU busy | 85.4% | 86.5% |
| memcpys / syncs | 21 / 25 | 21 / 25 |

**3. Server mode** (`vllm serve`, 512 in / 64 out, `vllm bench serve`):

| | 1 user: TTFT p50 / p95 | 4 users: TTFT p50 / p95 | TPOT p50 |
|---|---|---|---|
| stock | 34.8 / 35.5 ms | 113.1 / 120.5 ms | 9.71 ms |
| swap 16-bit (API plugin on) | 35.7 / 36.8 ms | 113.5 / 120.3 ms | 9.71 ms |
| swap + `GET /v1/precision` every 0.5 s | 17.7 / 19.6 ms* | 26.2 / 33.4 ms* | 9.70 ms |

\*Not comparable: this run reused the previous run's random seed, so vLLM's prefix
cache already held the prompts. TPOT is what shows the polling cost: none.

**Conclusion.** The swap format adds **no prefill cost**: same kernels, same GPU
time, same TTFT (within ~1 ms) as stock at 1 and 4 users. The earlier "gap" was a
**measurement mistake (mine)**:
- the H100 swap TTFT (438 ms) came from a phase test with **4 concurrent users** and
  telemetry polling, the stock number (122 ms) from a **1-user** benchmark; the
  matching stock phase run had been cancelled. Here, 4 users alone raise stock TTFT
  from 35 to 113 ms (3.2×);
- the 4B p95 difference (63 vs 40 ms) in run 4 was a single-run tail (p50: 41.6 vs
  33.5 ms) and does not reproduce.

**Corrections to §11:** §11.3's "prefill performance of the swap format" and §11.4
item 1 are withdrawn. The busy-phase difference to stock FP8 on the H100 (937 vs
1,181 tok/s) remains unexplained by prefill; the remaining candidates are the
smaller KV-cache-independent costs measured there (e.g. the 1.6 s stall at the burst
start) and the absence of a same-host stock phase baseline. Lesson for the article:
**compare like with like** (same load shape, same concurrency, same host).

## 13. Faster migrations (run 7, 2026-11-09): packing + idle drain

Pod `tfgxfgwmhsqequ`, RunPod Secure RTX 4090, $0.89/h, 21 min ≈ **$0.32**; total
spend ≈ **$6.15**. Qwen3-4B, swap format with the HTTP API, `FULL_DECODE_ONLY`.
`bench/migration_bench.sh`: switch via `POST /v1/precision`, client polls
`GET /v1/precision` every 0.25 s; "loaded" = 8 users generating continuously,
"idle" = no traffic. Files: `results/migration_v031/`, `results/migration/`.

**Two problems found on the H100 (§11):** (1) each staging buffer carried exactly one
layer per engine step, although most layers are much smaller than the buffer (sized
for the largest layer); (2) with no traffic there are no engine steps, so the
migration only advanced once per API poll.

**Changes:**
- **v0.3.1, packing** (`VPA_SWAP_PACK=1`, default): fill each staging buffer with as
  many whole layers as fit; plus `VPA_SWAP_STAGING_MIB`, a staging budget in MiB
  instead of a buffer count.
- **v0.3.2, idle drain:** when no engine step ran for 0.2 s, a `GET`/`POST` on
  `/v1/precision` finishes the whole migration in one call (waiting for each copy).
  Under load nothing changes: steps keep driving the migration, so serving is never
  blocked.

**Results** (Qwen3-4B: 144 layers; to FP8 = 3.38 GiB, to 16-bit = 6.77 GiB):

| Variant | Staging | Loaded → FP8 | Loaded → 16-bit | Idle → FP8 | Idle → 16-bit |
|---|---|---|---|---|---|
| no packing (v0.3.0 behaviour) | 4 × 95 MiB | 0.88 s · 144 transfers | 0.90 s · 144 | 9.18 s | 9.20 s |
| packing (v0.3.1) | 4 × 95 MiB | 0.34 s · 40 | 0.63 s · 80 | 2.56 s | 5.11 s |
| packing, 1 GiB budget (v0.3.1) | 10 × 95 MiB | 0.35 s · 40 | 0.70 s · 80 | 1.03 s | 2.05 s |
| **packing + idle drain (v0.3.2)** | 4 × 95 MiB | **0.31 s** · 40 | **0.77 s** · 80 | **0.28 s** | **0.55 s** |

- **Packing: 2.6× faster to FP8 under load** (0.88 → 0.34 s), 3.6× fewer transfers,
  **with no extra memory**.
- **Under load, more staging memory doesn't help** (1 GiB ≈ same as 380 MiB): copies
  already run at ~10 GB/s, close to what this host's PCIe path delivers.
- **Idle drain: 33× / 17× faster idle switches** (9.2 s → 0.28 / 0.55 s, ~12 GB/s),
  independent of the controller's polling interval.
- GPU tests on v0.3.2: 4/4 swap tests pass. One test had to change: with packing, all
  three small test layers fit in the two staging buffers launched in one step, so the
  "some step must have mixed precisions" check never triggered; the fixture now uses
  one staging buffer (the per-step consistency check itself passed throughout).

**Expected on Qwen3-32B** (not yet measured): packing turns the ~4 layers per decoder
block into ~2 transfers, so the H100 switch (7–8 s in §11) should approach the copy
bandwidth limit (~2–3 s for 29–58 GiB at 13–22 GB/s), and idle switches no longer
stall (the 43.9 s case in §11.2).

## 14. N precision versions (v0.4.0, run 8, 2026-11-09)

Pod `y5al866mz6c280`, RunPod Secure RTX 4090, $0.89/h, ~18 min ≈ **$0.27**; total
spend ≈ **$6.40**. Qwen3-4B. Script: `bench/phase2_check.sh`; files: `results/phase2/`.

**What changed (design phase 2):** "16-bit vs FP8" is no longer hard-coded. A
*version kind* (`versions.py`) builds a precision's kernel-ready bytes from the 16-bit
weight, knows its size, and runs its GEMM. Today: `high` (16-bit, PyTorch/cuBLAS) and
`fp8` (per-channel W8A8, vLLM CUTLASS). The swap manager, the custom op and the CUDA
graph sets work with version names; `VPA_VERSIONS` (default `high,fp8`) selects
what is built; the API lists versions and takes `{"version": ...}`.

**Verification (must behave exactly like v0.3.2):**

| Check | Result |
|---|---|
| GPU tests (op + swap, incl. "unknown version rejected") | **17 / 17 pass** |
| Graph sets | recorded per version: `high: 35`, `fp8: 35` |
| Decode speed, 1 request | 16-bit 100.9 tok/s, FP8 129.4 tok/s (as before) |
| Round trip 16-bit → FP8 → 16-bit | **16 / 16 outputs identical** |
| `GET /v1/precision` | lists both versions with bits, method, RAM (6.77 / 3.38 GiB), active layers |
| `POST {"version":"int3"}` | **HTTP 400** "unknown version 'int3'; available: ['high', 'fp8']" |
| `POST {"version":"fp8"}` / `{"version":"bf16"}` (alias) | 0.28 s / 0.55 s (idle); completion after round trip: "The capital of France is" → " Paris. The capital of Germany is Berlin" |

**Migration speed vs v0.3.2 (§13)**: loaded → FP8 0.37 s (0.31), → 16-bit 0.76 s
(0.77); idle → FP8 0.28 s (0.28), → 16-bit 0.55 s (0.55). Same within run-to-run
noise.

Note: engine startup took 209 s on this host (compile cache disabled for plugin
runs + host variance); the SSH connection to this host dropped intermittently,
which interrupted the live log trail but not the run.

**Next (design phase 3–4):** a third version kind, INT4 weight-only (Marlin), and
the config file. Adding a kind is now one class in `versions.py` plus its name in
`versions_names.py`.

## 15. Third version: INT4 weight-only (v0.5.0, run 9, 2026-11-09)

Pod `gheu2vvnab01jx`, RunPod Secure RTX 4090, $0.89/h, ~35 min (incl. two failed
attempts) ≈ **$0.52**; total spend ≈ **$6.90**. Qwen3-4B. Script: `bench/int4_check.sh`;
files: `results/int4/` (+ `results/int4_try1`, `int4_try2` from the failed attempts).

**What was added:** one version kind, `Int4Kind` in `versions.py` (`VPA_VERSIONS=high,fp8,int4`):
W4A16, round-to-nearest, symmetric, group size 128, built online from the 16-bit
weights at load time (no calibration, no separate checkpoint), repacked for vLLM's
**Marlin** kernel (`gptq_marlin_repack`, `apply_gptq_marlin_linear`). Stored in pinned
RAM like the other versions; in VRAM it occupies the front quarter of the 16-bit region.

**Bugs found on the way (both fixed before the numbers below):**

| Attempt | Problem | Fix |
|---|---|---|
| 1 | Layers with bias gave wrong outputs: Marlin permutes output columns, the bias must be permuted the same way | `marlin_permute_bias`, cached per layer |
| 2 | Engine crash: "Workspace is locked but marlin_locks ... not allocated during warmup" (vLLM's shared workspace manager is locked after warmup, but INT4 graphs are captured later) | plugin owns its Marlin workspace, allocated when the version is built |

**Verification:**

| Check | Result |
|---|---|
| GPU tests (INT4 vs dequantized reference < 1e-2; bias; 3-version round trip) | **31 / 31 pass** |
| Graph sets | `high: 35`, `fp8: 35`, `int4: 35` |
| Decode speed, 1 request | 16-bit **101.6**, FP8 **130.6**, INT4 **226.1** tok/s |
| Round trip high → fp8 → int4 → high | **16 / 16 outputs identical** to the first 16-bit run |
| Migrations (under filler load) | → FP8 0.455 s (3.38 GiB, 40 transfers), → INT4 **0.252 s** (1.69 GiB, 19), → 16-bit 0.905 s (6.77 GiB, 80) |
| KV cache | 91,696 tokens in all three versions (region sized for 16-bit) |

**Load sweep** (`bench/run_bench.sh`, configs `swap-high`, `swap-fp8`, `swap-int4`, each pinned
to one version, CUDA graphs FULL_DECODE_ONLY):

| Users | 16-bit tok/s | FP8 tok/s | INT4 tok/s | 16-bit TPOT | FP8 TPOT | INT4 TPOT | P95 TTFT 16/8/4 (ms) |
|---|---|---|---|---|---|---|---|
| 1 | 100.8 | 127.5 (+26%) | **222.5 (+121%)** | 9.76 | 7.63 | **4.30** | 62 / 67 / 57 |
| 8 | 677.8 | 862.8 (+27%) | **1357.1 (+100%)** | 11.24 | 8.70 | **5.25** | 249 / 217 / 252 |
| 32 | 1959.6 | 2318.5 (+18%) | **2948.8 (+50%)** | 15.28 | 12.87 | **9.52** | 515 / 454 / 666 |
| 64 | 2622.9 | 3043.0 (+16%) | **3139.7 (+20%)** | 22.53 | 19.28 | **18.31** | 1205 / 965 / 1209 |
| 128 | 2939.8 | **3305.4 (+12%)** | 3105.8 (+6%) | 39.20 | **35.30** | 37.55 | 2409 / **1895** / 2395 |

**Reading it:**

- **INT4 wins big at low and medium load** (2.2× at 1 user, 2.0× at 8, 1.5× at 32):
  decode is memory-bound and INT4 reads a quarter of the bytes.
- **FP8 overtakes INT4 between 64 and 128 users.** Marlin W4A16 still does its math in
  16-bit and has to dequantize; FP8 W8A8 uses the FP8 tensor cores. Once the batch is
  big enough to be compute-bound, that matters more than the weight bytes. This
  matches the literature (W4A16 for latency, W8A8 for throughput, see the DESIGN doc).
  INT4 also has the worst P95 TTFT at 32 users (prefill is compute-bound too).
- So the precision ladder is **not monotonic in load**: a controller picking "lowest
  precision = fastest" would be wrong at high load. The useful policy on this GPU/model
  is roughly *INT4 up to ~64 concurrent users, FP8 beyond*, with 16-bit kept for when
  quality matters most. This is exactly why the decision lives in an external
  controller: the crossover depends on GPU, model and traffic shape.
- **Quality is not measured yet.** This INT4 is plain RTN (no AWQ/GPTQ calibration);
  outputs differ from 16-bit (expected). GSM8K per version is still to do before any
  quality claim in the article.

**Next:** quality per version (GSM8K incl. INT4), config file (`VPA_CONFIG`),
H100 + Qwen3-32B confirmation, C# controller + Prometheus telemetry.
