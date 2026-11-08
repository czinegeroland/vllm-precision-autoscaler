"""The quantization config vLLM sees: `--quantization precision_autoscaler`."""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

from . import state

NAME = "precision_autoscaler"


def _num_hidden_layers() -> int:
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config().model_config.hf_text_config
        return int(getattr(cfg, "num_hidden_layers", 0) or 0)
    except Exception:  # noqa: BLE001 - only used for the dial policy
        return 0


def _setup_swap_cudagraphs() -> None:
    """The swap format rewrites weight memory in place, so a recorded CUDA
    graph must match what the memory holds. Either no graphs at all, or full
    graphs recorded once per precision (graphs.py). Piecewise graphs live
    inside torch.compile'd pieces and can't be split per precision."""
    from vllm.config import get_current_vllm_config

    cfg = get_current_vllm_config()
    mode = getattr(cfg.compilation_config, "cudagraph_mode", None)
    name = getattr(mode, "name", str(mode))
    if name == "NONE":
        return
    hint = ("use --compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}' "
            "(two graph sets) or '{\"cudagraph_mode\": \"NONE\"}' (no graphs)")
    if not state.settings.swap_graphs:
        raise ValueError(f"VPA_FORMAT=swap with VPA_SWAP_GRAPHS=0 needs CUDA graphs off; {hint}")
    if name not in ("FULL_DECODE_ONLY", "FULL"):
        raise ValueError(f"VPA_FORMAT=swap can't use cudagraph_mode {name} (piecewise graphs); {hint}")
    from . import graphs

    graphs.install()


@register_quantization_config(NAME)
class PrecisionAutoscalerConfig(QuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

    def get_name(self) -> str:
        return NAME

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # FP8 tensor cores: Ada (8.9), Hopper (9.0), Blackwell (10.x/12.x).
        return 89

    @staticmethod
    def get_config_filenames() -> list[str]:
        # No config file in the checkpoint: vLLM then calls cls() directly.
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PrecisionAutoscalerConfig":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if not state.num_layers_hint:
                state.num_layers_hint = _num_hidden_layers()
                if state.settings.format == "swap":
                    _setup_swap_cudagraphs()
            from .linear import PrecisionAutoscalerLinearMethod

            return PrecisionAutoscalerLinearMethod(prefix)
        # Embeddings, LM head, attention, MoE: vLLM's unquantized defaults.
        return None
