# vllm-precision-autoscaler (plugin package)

vLLM plugin that switches a model between 16-bit and FP8 weights on the fly, based on
load. See the main README and docs/FINDINGS.md at
https://github.com/czinegeroland/vllm-precision-autoscaler

    pip install --no-deps "git+https://github.com/czinegeroland/vllm-precision-autoscaler@v0.2.0#subdirectory=plugin"
    VPA_FORMAT=swap VPA_MODE=auto vllm serve Qwen/Qwen3-4B --quantization precision_autoscaler \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'

Requires vLLM v0.32.0 and an FP8-capable NVIDIA GPU (compute capability >= 8.9).
