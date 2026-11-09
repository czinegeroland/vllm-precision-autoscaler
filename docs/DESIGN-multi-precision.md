# Design: multiple precision versions with tiered storage

Status: **proposal** (2026-11-09). Builds on the swap format (v0.3.0), which already
switches one model between BF16 and FP8 at runtime with both copies in pinned CPU RAM.

## 1. The idea

Serve **several precision versions of the same model** (for example BF16, FP8, INT4)
from one vLLM process. A **config file** given at startup says which versions exist
and **where each one lives**: GPU memory (VRAM), system memory (pinned RAM) or disk.
A **residency manager** moves versions between these tiers the way a CPU moves data
between its cache levels. An **HTTP API** lists the versions, their location and
switch cost, and activates or prepares a version. vLLM never decides *when* to switch;
an external controller does (decision recorded in [FINDINGS.md §11.2](FINDINGS.md)).

```text
            ┌──────────── fastest, smallest ────────────┐
  L1  VRAM   │ active version (kernel-ready weights)     │  switch: instant
  L2  RAM    │ pinned copies, ready to copy in           │  switch: seconds (PCIe)
  L3  disk   │ kernel-ready files, loaded on demand      │  switch: seconds–minutes
            └──────────── slowest, largest ─────────────┘
```

## 2. Prior work (what exists, what is new)

| Work | What it does | Relation |
|---|---|---|
| **vLLM sleep mode** ([docs](https://docs.vllm.ai/en/stable/features/sleep_mode/)) | Level 1: weights offloaded to CPU RAM, KV cache discarded; level 2: both discarded, reload from disk | Whole-model sleep/wake of *one* precision; serving stops while asleep |
| **vLLM CPU offloading** ([offload config](https://docs.vllm.ai/en/v0.32.0/api/vllm/config/offload/)) | UVA (GPU reads pinned CPU memory over PCIe) or prefetch (async, group-wise layer offload) | Extends capacity for *one* precision; every step pays PCIe traffic |
| **vLLM online quantization** (v0.32 `--quantization fp8_per_channel`, `int8_per_channel_weight_only`, `nvfp4_per_token`, `mxfp8`) | Quantizes BF16 weights to another format at load time | **Building block**: produces our FP8/INT8/NVFP4 versions from one BF16 checkpoint |
| **NestedFP** ([arXiv:2506.02024](https://arxiv.org/abs/2506.02024)) | FP16 and FP8 nested in one 16-bit footprint, per-step switching, custom CUTLASS kernel | Two precisions, VRAM only, kernel required |
| **Any-Precision LLM** ([ICML 2024](https://arxiv.org/abs/2402.10517), [code](https://github.com/SNU-ARC/any-precision-llm)) | 3…n-bit models overlaid as bitplanes in the footprint of one n-bit model; loads only the needed bitplanes | Many precisions in one footprint, but custom quantization (SqueezeLLM-style) and custom kernel; no tiers |
| **MatQuant / MatGPTQ** (Matryoshka quantization) | One integer checkpoint sliceable to int8/int4/int2 | Training or calibration needed; no serving tiers |
| **M2Cache** ([arXiv:2410.14740](https://arxiv.org/html/2410.14740)) | Mixed-precision neurons with a GPU–DRAM–SSD multi-level cache | Closest to the tier idea, but neuron-level sparsity on older servers; not a serving-engine plugin |
| **FlexGen / DeepSpeed ZeRO-Inference** ([FlexGen](https://arxiv.org/pdf/2303.06865)) | Weights spread over GPU, CPU and disk; offline, throughput-oriented, streams weights every step | Tiers for capacity, not for switching precision |
| **ServerlessLLM** ([arXiv:2401.14351](https://arxiv.org/pdf/2401.14351v2.pdf)) | GPU / DRAM / SSD checkpoint hierarchy, loading-optimized chunked format, pinned memory, 3.6–8.2× faster loading | **Building block idea** for our disk tier format |
| **NVIDIA Dynamo KVBM** ([design](https://docs.nvidia.com/dynamo/v1.2.0/design-docs/component-design/kvbm-design.md)) | KV cache blocks over G1 GPU / G2 CPU / G3 SSD / G4 remote, pinned memory, GPUDirect Storage, frequency-filtered SSD offload | Same tier concept, applied to **KV cache**, not weights |
| **Run:ai Model Streamer** ([vLLM](https://docs.vllm.ai/en/latest/models/extensions/runai_model_streamer/), [NVIDIA](https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer)) | Concurrent safetensors reads into GPU; 15 GB model in ~7.5 s from a 4 GB/s volume | Building block for disk → RAM speed |

**What is new here:** a *precision ladder* (several quantized versions of one model)
combined with a *storage hierarchy* (VRAM / RAM / disk) and *live switching while
serving*, inside stock vLLM as a plugin, with the decision left to an external
controller. Each part exists somewhere; the combination does not (searches on
2026-11-08/09).

## 3. Concepts

- **Version**: one precision of the model, *already in its kernel's format*
  (e.g. INT4 Marlin weights are repacked; FP8 has per-channel scales). Defined by a
  vLLM quantization method + its processed tensors for every linear layer.
- **Tier**: where a version's bytes live: `vram`, `ram` (pinned), `disk`.
- **Active version**: the one the model runs on. Its bytes occupy the per-layer
  **VRAM region**. Exactly one version is active (except during a migration, when
  layers are mixed).
- **Residency**: the set of tiers holding a copy of a version. A version can be in
  RAM *and* on disk.
- **Promote / demote**: copy a version up (disk → RAM, RAM → VRAM) or drop a copy.

## 4. Configuration

Given at startup with `VPA_CONFIG=/path/precision_autoscaler.yaml`.

```yaml
model: Qwen/Qwen3-32B          # informational; vLLM's --model is authoritative
versions:
  bf16:
    method: unquantized        # from the checkpoint vLLM loads
    tier: vram                 # starts active
  fp8:
    method: fp8_per_channel    # vLLM online quantization from bf16
    tier: ram
  int8:
    method: int8_per_channel_weight_only
    tier: disk
  int4:
    method: awq_marlin         # needs a separate quantized checkpoint
    source: Qwen/Qwen3-32B-AWQ
    tier: disk
active: bf16
tiers:
  ram:  { budget_gib: 64 }     # pinned CPU memory for versions + staging
  disk: { path: /workspace/vpa-cache, budget_gib: 200 }
migration:
  staging_mib: 512             # staging buffer size (chunked; see §7)
  streams: 2
```

Validation at startup (fail fast with a clear message):
- `method` must be a quantization method vLLM supports on this GPU
  (`nvfp4*` needs Blackwell; `fp8*` needs compute capability ≥ 8.9).
- Every non-`unquantized` method gets its **layer-by-layer processed tensors** from the
  BF16 weights (online) or from `source` (checkpoint) whose layer structure must match
  vLLM's model (fused `qkv_proj`, `gate_up_proj`); mismatch → error.
- The sum of RAM-tier versions + staging must fit `budget_gib` and the machine;
  the VRAM region size = the largest version's per-layer bytes.

## 5. Components

### 5.1 Version builder (startup)

For every configured version and every linear layer:

1. Create the layer's parameters with that method's `create_weights`.
2. Fill them from the BF16 weights (online methods) or from `source`.
3. Run the method's `process_weights_after_loading` (repacking, scale computation).
4. Serialize the resulting tensors (raw bytes + small metadata: dtype, shape, the
   method's attribute names) into the version's tier:
   - `ram`: pinned host buffers;
   - `disk`: one file per version in a **loading-optimized layout** (contiguous
     per-layer chunks, aligned, like ServerlessLLM), plus a manifest;
   - `vram`: written into the per-layer region (active version).

Building on the fly costs startup time once; disk files are reused on later starts
(keyed by model revision + method + vLLM version).

### 5.2 VRAM region and kernels

Each layer owns a region of `max_version_bytes(layer)`. Activating a version means its
bytes are in the region and the layer's `apply()` dispatches to **that version's vLLM
LinearMethod**, with its parameter tensors as views into the region. All methods are
vLLM's own kernels (cuBLAS, CUTLASS FP8, Marlin INT4/INT8, FP4 kernels on Blackwell):
**no custom kernel**.

### 5.3 Residency manager (the cache hierarchy)

State per version: `{vram|ram|disk} ⊆ residency`, plus `loading` flags.

| Operation | Path | Notes |
|---|---|---|
| `prepare(v, ram)` | disk → pinned RAM | background threads, chunked reads (Run:ai-streamer style); evicts other RAM versions LRU-first if over budget (never the active one, never one being migrated) |
| `activate(v)` | RAM → VRAM region | today's swap migration, generalized; requires `v` in RAM (auto-`prepare` first if only on disk, reported as a slower switch) |
| `evict(v, ram)` | drop pinned copy | only if another tier still holds it |

Policy inside the plugin is limited to **mechanics** (budgets, LRU eviction, chunking);
*which* version to use and *when* stays with the controller.

### 5.4 Migration engine (generalized from v0.3.0)

Improvements needed anyway (FINDINGS §11.4):
- **Chunked staging**: copy fixed-size chunks (e.g. 64 MiB) instead of whole layers, so
  more transfers fit in flight and big layers (500 MiB on 32B) don't limit slots.
- **Progress between steps**: a background thread keeps H2D copies going on a side
  stream; flips (staging → region + method switch) still happen at step boundaries.
- Target: migrations near PCIe bandwidth (32B FP8: 29 GiB / ~20 GB/s ≈ 1.5 s).

### 5.5 CUDA graphs

Generalize today's two graph sets to **one set per graph-ready version**: versions in
VRAM or RAM get their graphs recorded at startup; disk-tier versions get theirs when
first prepared (recording takes a few seconds and must not overlap a step: done from
an RPC between steps, with the engine paused briefly), or run without graphs.
Memory: ~0.3–0.7 GiB per set (measured: 0.73 GiB for two sets on Qwen3-4B).

### 5.6 HTTP API (extends `/v1/precision`)

```text
GET  /v1/precision
  -> { "active": "bf16", "migrating": false,
       "versions": [
         {"name":"bf16","method":"unquantized","bits":16,"residency":["vram","disk"],
          "bytes":65.6e9, "graphs":true},
         {"name":"fp8","method":"fp8_per_channel","bits":8,"residency":["ram"],
          "bytes":32.8e9, "graphs":true, "est_activate_s":1.6},
         {"name":"int4","method":"awq_marlin","bits":4,"residency":["disk"],
          "bytes":18.1e9, "graphs":false, "est_activate_s":9.5}
       ],
       "tiers": {"ram":{"used_gib":40.2,"budget_gib":64}, "disk":{...}} }

POST /v1/precision          {"version":"fp8"}               activate (async)
POST /v1/precision/prepare  {"version":"int4","tier":"ram"}  promote ahead of time
POST /v1/precision/evict    {"version":"int8","tier":"ram"}  free RAM
```

`est_activate_s` comes from measured tier bandwidths (PCIe, disk) and the version size,
so a controller can plan: "prepare INT4 now, activate it in 30 s if the queue keeps
growing".

### 5.7 Telemetry

`GET /v1/precision` as above, plus Prometheus metrics (via a vLLM stat-logger plugin):
active version, residency per version, migration duration histogram, bytes moved per
tier, RAM-tier usage. This feeds the observability part of the project.

## 6. Which versions make sense (measure, don't assume)

| Version | Why | Caveat |
|---|---|---|
| BF16 | reference quality; how checkpoints are trained and stored | largest, slowest |
| FP8 (W8A8) | ~2× tensor-core throughput, half the bytes; +70% on Qwen3-32B (FINDINGS §11) | Ada/Hopper/Blackwell only |
| INT8 weight-only (W8A16) | works on Ampere too; half the bytes | no compute speedup |
| INT4 weight-only (W4A16, Marlin) | quarter of the bytes: **fastest at low batch** | **slower than W8A8 at large batch** (dequantization cost); roofline crossover ≈ batch 78 on A100 ([QServe](https://arxiv.org/pdf/2405.04532)); W4A16 best for low-latency/synchronous, W8A8 best for high-throughput on high-end GPUs ([arXiv:2411.02355](https://arxiv.org/pdf/2411.02355)); quality needs a calibrated checkpoint (AWQ/GPTQ) |
| NVFP4 | Blackwell native FP4 | not on Hopper/Ada |
| FP32 | — | not useful: no quality over BF16-trained weights, 2× bytes |

So the ladder is **not monotonic in load**: INT4 can be the best choice at *low* load
(latency) and the worst at *high* load. The plugin's job is to make every version
available and report its characteristics; picking one is the controller's job. A
startup micro-benchmark per version (GEMM time at a few batch sizes) can be exposed in
the API to help the controller.

## 7. Hard limits and risks

1. **KV cache is sized once**, for the largest version that can be active. Smaller
   versions bring speed, not capacity. (Runtime KV resizing is out of scope.)
2. **Checkpoint compatibility** for `source` versions: AWQ/GPTQ checkpoints must match
   vLLM's fused layer structure and packed-modules mapping; some exclude layers from
   quantization (`ignore` lists). Needs per-model testing.
3. **Graph memory** grows with the number of graph-ready versions.
4. **Pinned RAM**: large pinned allocations take time and can fail on hosts with
   memory limits; the RAM budget must be explicit.
5. **Disk speed** varies by host (RunPod container disk vs network volume); report the
   measured value instead of assuming.
6. ~~Prefill performance of the swap format~~: investigated in FINDINGS §12, the
   swap format adds no prefill cost (same kernels and TTFT as stock); the earlier gap
   was a like-for-unlike comparison.
7. **vLLM version coupling**: the plugin patches vLLM's graph manager (graphs.py) and
   relies on quantization method internals; pinned to v0.32.0.
8. **Tensor parallelism**: every rank must migrate the same layers in lockstep; not yet
   tested.

## 8. Plan

| Phase | Work | Done when | GPU |
|---|---|---|---|
| 0 | ~~Profile swap prefill~~ **done** (FINDINGS §12): no prefill cost, nothing to fix | — | 4090, $0.23 |
| 1 | ~~Faster migrations~~ **done** (FINDINGS §13): packed staging + idle drain; 4B: 0.31 s / 0.77 s loaded, 0.28 s / 0.55 s idle | H100 confirmation pending | 4090, $0.32 |
| 2 | Generalize "precision" into versions using vLLM quantization methods (BF16 + FP8 first, behaviour identical to v0.3.0) | existing tests pass; API lists versions | 4090 |
| 3 | Config file + RAM budget + `prepare`/`evict` | BF16 + FP8 + INT8 (online) from config | 4090 |
| 4 | INT4 from an AWQ checkpoint (Marlin) | INT4 activates while serving; batch-size sweep BF16/FP8/INT4 | 4090 (Qwen3-4B-AWQ) |
| 5 | Disk tier (loading-optimized files, threaded reads) | disk → RAM → VRAM activation measured | 4090 |
| 6 | Telemetry (Prometheus) + controller update | controller plans with `est_activate_s` | — |
| 7 | H100 + Qwen3-32B end-to-end, quality per version (GSM8K) | article data | H100, ~$3–5 |

Each phase is shippable and keeps the current API working.

## 9. Open questions for the author

- Which versions matter most for the article: BF16/FP8/INT4, or include INT8 and NVFP4?
- Should `activate` of a disk-only version be allowed directly (slow), or require an
  explicit `prepare` first?
- RAM eviction: LRU is simple; would you rather have explicit-only eviction (the
  controller owns all residency decisions)?
