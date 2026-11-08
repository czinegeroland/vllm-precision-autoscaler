"""Worker extension: methods callable with `collective_rpc`.

Offline:  LLM(..., worker_extension_cls=EXTENSION) then
          llm.collective_rpc("vpa_stats")
Server:   VLLM_SERVER_DEV_MODE=1 vllm serve ... --worker-extension-cls EXTENSION
          POST /collective_rpc {"method": "vpa_stats"}

The HTTP endpoint only passes string arguments, so every argument is a str.
"""

from __future__ import annotations

from . import state

EXTENSION = "vllm_precision_autoscaler.worker_ext.PrecisionAutoscalerWorkerExtension"


class PrecisionAutoscalerWorkerExtension:
    def vpa_stats(self, per_layer: str = "0") -> dict:
        return state.stats(per_layer=per_layer in ("1", "true", "True"))

    def vpa_reset(self) -> dict:
        state.reset_counters()
        return state.stats()

    def vpa_set_mode(self, mode: str, threshold: str = "") -> dict:
        state.set_mode(mode, int(threshold) if threshold else None)
        return state.stats()
