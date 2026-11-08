# vllm-precision-autoscaler — rules for Claude Code

Medium article project: "Autoscale LLMs by lowering precision with vLLM".
Plugin switches linear layers between 16-bit and FP8 per engine step.

## Stack
- `plugin/` Python vLLM plugin (thin glue required by vLLM). Pinned to vLLM **v0.32.0**
  (image `vllm/vllm-openai:v0.32.0-cu129`). Check APIs against that tag, not `main`.
- `kernels/` (later, v2) CUDA C++ or Triton kernel exposed as a torch op.
- `controller/` (later) C#/.NET controller + load generator, talks to vLLM over HTTP.

## RunPod rules (budget is small)
- State GPU type and $/h before creating a pod; record every pod in `deploy/runpod/ledger.csv` (rp.py does this).
- Terminate pods as soon as a run finishes; never leave one idle. Pull results first (`pod.sh pull`).
- Develop/debug on the cheapest FP8-capable GPU (RTX 4090, Ada sm_89).
- Never commit or print the API key (`deploy/runpod/.runpod_api_key`, gitignored).

## Design facts that are easy to get wrong
- The precision branch lives inside a custom op (`torch.ops.precision_autoscaler.linear`).
  A plain Python `if` in `apply()` can be frozen by vLLM's torch.compile.
- Decisions depend only on tokens-per-step, so each CUDA graph size maps to one precision.
  Runtime `vpa_set_mode` only affects steps that run eagerly (or all steps with cudagraphs off).
- v0 keeps BF16 + FP8 copies (~+50% weight memory). Compare KV cache size in server logs.
- vLLM builds custom quant configs with `cls()` (no checkpoint config), so settings come from `VPA_*` env vars.
