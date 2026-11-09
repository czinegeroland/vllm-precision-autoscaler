# vllm-precision-autoscaler

A [vLLM](https://github.com/vllm-project/vllm) plugin that lets one GPU change the
precision of a model's weights **on the fly, based on load**: 16-bit (BF16) when the
server is quiet, FP8 when it is busy. No restart, no second replica, no custom kernel,
and no change to vLLM's source.

Measured on an RTX 4090 with Qwen3-4B (vLLM v0.32.0), switching formats in a live
quiet → busy (96 users) → quiet test:

| | stock BF16 | **swap format, auto** |
|---|---|---|
| Busy phase throughput | 2,960 tok/s | **3,307 tok/s** (+12%) |
| Busy phase p95 time to first token | 1,702 ms | **1,655 ms** |
| Requests queued while busy | 0 | **0** |
| Switch to FP8 / back to 16-bit | – | 3.8 s / 1.1 s, while serving |
| Largest gap between two streamed tokens | 140 ms | **120 ms** (no pause from switching) |
| Quiet-phase output | BF16 | bit-identical to BF16 |

Full measurements, methods, costs and caveats: [`docs/FINDINGS.md`](docs/FINDINGS.md). Next design (multiple precision versions on VRAM / RAM / disk tiers): [`docs/DESIGN-multi-precision.md`](docs/DESIGN-multi-precision.md).

## How it works

`--quantization precision_autoscaler` routes every linear layer of the model through
the plugin. Three storage formats (`VPA_FORMAT`):

| Format | VRAM | When it switches | CUDA graphs |
|---|---|---|---|
| `swap` (recommended) | **one** precision; both copies kept in pinned CPU RAM | per load period: layers migrate CPU→GPU in the background while serving | yes: two graph sets, one per precision |
| `dual` | both BF16 and FP8 (~+50% weight memory) | every engine step, by tokens in the step | yes |
| `nested` | experimental, needs a Triton kernel, not GPU-validated | – | – |

**swap** in more detail:

- Each layer's GPU memory is sized for BF16 and holds either the BF16 weight or (in its
  first half) the FP8 weight. Both versions live in pinned CPU memory.
- When the target precision changes, layers are copied on a side CUDA stream into small
  staging buffers, then moved into place between engine steps. The model keeps
  serving; during the ~1–4 s migration some layers are FP8 and some BF16.
- An external controller sets the target over the plugin's HTTP API
  (`POST /v1/precision`); vLLM itself never decides when to switch.
- **Two CUDA graph sets.** A recorded CUDA graph stores *which kernels run on which
  addresses*, not the data, so the plugin records every decode graph twice (16-bit
  kernels and FP8 kernels on the same addresses) and replays the set that matches the
  memory. This is done by patching vLLM's model runner V2 graph manager at runtime
  from the plugin, so **the plugin is tied to vLLM v0.32.0**.

## Requirements

- vLLM **v0.32.0** (tested with the `vllm/vllm-openai:v0.32.0-cu129` image)
- an FP8-capable NVIDIA GPU (compute capability ≥ 8.9: RTX 4090, L40S, H100, RTX 5090…)
- a BF16 model; for `swap`: pinned CPU RAM ≈ 1.5 × the model's BF16 weight size

## Install

```bash
pip install --no-deps "git+https://github.com/czinegeroland/vllm-precision-autoscaler@main#subdirectory=plugin"
```

`--no-deps` keeps your existing vLLM install untouched (the plugin declares no
dependencies anyway). vLLM discovers the plugin through its `vllm.general_plugins` entry
point; nothing else to configure.

## Run

The plugin **executes** precision switches; deciding **when** to switch is the job of
an external controller. Start vLLM with the plugin and its HTTP control API:

```bash
export VLLM_PLUGINS=precision_autoscaler,precision_autoscaler_api
export VPA_FORMAT=swap VPA_MODE=high
vllm serve Qwen/Qwen3-4B \
  --quantization precision_autoscaler \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

(`VLLM_PLUGINS` is needed because vLLM loads endpoint plugins only when they are
allowlisted, and the allowlist then applies to all plugins, so both names are listed.)

### Control API

| Request | Effect |
|---|---|
| `GET /v1/precision` | current state: `mode`, `fp8_layers`/`layers`, `migrating`, recent `migrations` (also advances a pending migration while the engine is idle) |
| `POST /v1/precision` `{"mode": "fp8"}` | start switching to FP8 (`"high"` = back to BF16); returns immediately, layers migrate in the background while serving |

```bash
curl -s localhost:8000/v1/precision
curl -s localhost:8000/v1/precision -H 'Content-Type: application/json' -d '{"mode":"fp8"}'
```

### Reference controller

`controller/precision_controller.py` (Python, stdlib only) is a minimal external
controller: it reads `vllm:num_requests_running + waiting` from `/metrics`, applies
its own thresholds, hold times and dwell time, and calls `POST /v1/precision`. Every
decision is logged as a JSON line. Any other service (e.g. a C#/.NET controller,
or a KEDA-style autoscaler) can use the same three endpoints instead.

```bash
python3 controller/precision_controller.py --url http://localhost:8000 --up 16 --down 6 --dwell 20
```

An optional built-in policy (`VPA_SWAP_POLICY=hysteresis VPA_MODE=auto`) exists for
experiments, but in tests on Qwen3-32B its 5 s dwell made it flap while each
migration took ~5 s; external control is the default.

### Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `VPA_FORMAT` | `dual` | `swap`, `dual` or `nested` |
| `VPA_MODE` | `auto` | starting precision: `high` (16-bit) or `fp8`; `auto`/`dial` for `dual` (and for `swap` only with `VPA_SWAP_POLICY=hysteresis`) |
| `VPA_SWAP_POLICY` | `external` | swap: `external` (switch only via the API) or `hysteresis` (built-in, opt-in) |
| `VPA_SWAP_UP` / `VPA_SWAP_DOWN` | 64 / 32 | built-in policy only: tokens per step (smoothed) to go to FP8 / back |
| `VPA_SWAP_DWELL_S` | 5 | swap+auto: minimum seconds between precision changes |
| `VPA_SWAP_EMA_S` | 2 | swap+auto: smoothing time constant |
| `VPA_SWAP_SLOTS` | 4 | swap: staging buffers (each the size of the largest layer) |
| `VPA_SWAP_GRAPHS` | 1 | swap: record two CUDA graph sets (needs `FULL_DECODE_ONLY` or `FULL`) |
| `VPA_THRESHOLD` | 256 | dual+auto: tokens per step above which FP8 is used |
| `VPA_COUNT` | 0 | 1 = exact GPU-side counters of 16-bit/FP8 calls and tokens |

### Known caveats

- vLLM's on-disk compile cache key ignores `VPA_*` settings: set
  `VLLM_DISABLE_COMPILE_CACHE=1` when switching `VPA_FORMAT` on one machine.
- `swap` needs `cudagraph_mode` `FULL_DECODE_ONLY`, `FULL` or `NONE` (piecewise graphs
  would freeze one precision). Prefill steps run without graphs.
- The FP8 KV cache (`--kv-cache-dtype fp8`) is independent of the plugin and changes
  attention precision in every mode.

## Repository layout

```text
plugin/          the pip-installable plugin (vllm_precision_autoscaler) + tests
controller/      reference external controller (decides when to switch)
bench/           benchmark, phased-load, quality (GSM8K) and summary scripts
deploy/runpod/   rp.py (RunPod API: prices, create, wait, terminate) and pod.sh
deploy/pod/      scripts that run on a GPU pod (setup, tests)
docs/            FINDINGS.md: every experiment, number and lesson learned
```

Tests: `python -m unittest discover -s plugin/tests -p test_policy.py` runs anywhere;
`plugin/tests/gpu/` needs vLLM and an FP8 GPU.

## Background

Inspired by NestedFP (Lee et al., NeurIPS 2025, [arXiv:2506.02024](https://arxiv.org/abs/2506.02024)),
which switches FP16/FP8 per step in a vLLM fork with a custom kernel. This project is
an independent plugin implementation with a different storage approach.

License: Apache-2.0.
