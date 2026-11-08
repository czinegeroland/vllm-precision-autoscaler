"""Worker-side control methods, called through `collective_rpc`.

The plugin attaches these to vLLM's GPU worker class automatically
(`attach_to_worker`, called while the model is built), so neither
`--worker-extension-cls` nor `VLLM_SERVER_DEV_MODE` is needed. They are used by:
  * the HTTP API (api.py):         GET/POST /v1/precision
  * offline engines:               llm.collective_rpc("vpa_stats")
  * vLLM's dev endpoint, if enabled: POST /collective_rpc {"method": "vpa_stats"}

`--worker-extension-cls vllm_precision_autoscaler.worker_ext.PrecisionAutoscalerWorkerExtension`
still works and does the same thing.

The HTTP dev endpoint only passes string arguments, so every argument is a str.
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
        if state.swap_manager is not None:
            state.swap_manager.pump()  # start the copies right away
        return state.stats()

    def vpa_pump(self) -> dict:
        """Advance a pending swap migration without an engine step (copies
        only progress at step boundaries otherwise, so an idle engine would
        never finish). Cheap no-op when nothing is pending."""
        if state.swap_manager is not None:
            state.swap_manager.pump()
        return state.stats()


_METHODS = ("vpa_stats", "vpa_reset", "vpa_set_mode", "vpa_pump")


def attach_to_worker() -> None:
    """Add the vpa_* methods to vLLM's GPU worker class (idempotent)."""
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except Exception:  # noqa: BLE001 - other platforms: use --worker-extension-cls
        return
    for name in _METHODS:
        if not hasattr(Worker, name):
            setattr(Worker, name, getattr(PrecisionAutoscalerWorkerExtension, name))
