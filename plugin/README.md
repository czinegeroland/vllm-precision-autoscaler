# vllm-precision-autoscaler (plugin package)

vLLM plugin that switches a running model between precisions (BF16, FP8, INT4) while it
keeps serving requests. An outside controller decides when, through `POST /v1/precision`.
Documentation: https://github.com/czinegeroland/vllm-precision-autoscaler

    pip install --no-deps "git+https://github.com/czinegeroland/vllm-precision-autoscaler@main#subdirectory=plugin"

    export VLLM_PLUGINS=precision_autoscaler,precision_autoscaler_api
    export VPA_FORMAT=swap VPA_VERSIONS=high,fp8,int4a8 VPA_MODE=high
    vllm serve Qwen/Qwen3-4B --quantization precision_autoscaler \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'

Requires vLLM v0.32.0 and an FP8-capable NVIDIA GPU (compute capability >= 8.9).
