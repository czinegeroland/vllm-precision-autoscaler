"""Two sets of CUDA graphs for the swap format, set up entirely from the plugin.

Problem: a recorded CUDA graph replays fixed kernels on fixed addresses. The
swap format rewrites each layer's weight memory in place (BF16 <-> FP8), so a
graph recorded with the BF16 kernel would run that kernel on FP8 bytes.

Idea: recording a graph stores *which kernels run on which addresses*, not the
data. So at startup we record every decode graph twice, once with the 16-bit
kernels and once with the FP8 kernels, on the same addresses, and at runtime
replay the set that matches what the memory currently holds:

  * `CudaGraphManager.graphs` (vLLM's dict: batch shape -> graph) is replaced
    by `PrecisionGraphs`, which keeps one dict per precision and exposes the
    one for the current precision.
  * `ModelCudaGraphManager.capture` runs twice, once per precision.
  * `CudaGraphManager.dispatch` reports "no graph" while a migration is in
    progress (layers are mixed then), so those few steps run eagerly.
  * `GPUModelRunner.execute_model` calls the swap manager's `on_step` *before*
    each real step: background copies and precision flips happen outside any
    graph, because a replay never runs the plugin's Python.

Supported vLLM: v0.32.0 "model runner V2" (`vllm/v1/worker/gpu/`), with
cudagraph_mode FULL_DECODE_ONLY or FULL. Piecewise graphs are rejected: they
live inside torch.compile'd pieces and would freeze one precision. This module
patches vLLM classes at runtime (monkeypatching); vLLM's source is not changed,
but the patch is tied to this vLLM version.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from . import state

def _versions() -> tuple:
    mgr = state.swap_manager
    return mgr.version_names if mgr is not None else ("high", "fp8")
_installed = False


def current_precision() -> str:
    if state.capture_precision is not None:
        return state.capture_precision
    mgr = state.swap_manager
    return mgr.target if mgr is not None else "high"


class PrecisionGraphs(MutableMapping):
    """Drop-in for vLLM's `dict[BatchExecutionDescriptor, CUDAGraph]`."""

    def __init__(self, initial: dict | None = None) -> None:
        self.sets: dict[str, dict] = {}
        if initial:
            self.sets.setdefault("high", {}).update(initial)

    def _cur(self) -> dict:
        return self.sets.setdefault(current_precision(), {})

    def __getitem__(self, key: Any) -> Any:
        return self._cur()[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._cur()[key] = value

    def __delitem__(self, key: Any) -> None:
        del self._cur()[key]

    def __iter__(self):
        return iter(self._cur())

    def __len__(self) -> int:
        return len(self._cur())

    def __contains__(self, key: object) -> bool:
        return key in self._cur()

    def clear(self) -> None:
        for s in self.sets.values():
            s.clear()

    def counts(self) -> dict[str, int]:
        return {p: len(s) for p, s in self.sets.items()}


def install() -> None:
    """Patch vLLM's graph manager and model runner (idempotent)."""
    global _installed
    if _installed:
        return
    from vllm.v1.worker.gpu import cudagraph_utils as cgu
    from vllm.v1.worker.gpu import model_runner as mr

    base = cgu.CudaGraphManager
    model_mgr = cgu.ModelCudaGraphManager
    runner = mr.GPUModelRunner

    orig_capture = model_mgr.capture
    orig_dispatch = base.dispatch
    orig_execute = runner.execute_model

    def capture(self, *args, **kwargs):
        if not isinstance(self.graphs, PrecisionGraphs):
            self.graphs = PrecisionGraphs(self.graphs)
        if getattr(self, "_capture_mem_samples", None) is not None:
            # vLLM's memory-profiling pass (throwaway graphs): one set is enough.
            return orig_capture(self, *args, **kwargs)
        try:
            for prec in _versions():
                state.capture_precision = prec
                orig_capture(self, *args, **kwargs)
        finally:
            state.capture_precision = None
        state.graph_set_counts = self.graphs.counts()
        state.logger.info(
            "precision_autoscaler: recorded CUDA graphs per precision: %s",
            state.graph_set_counts,
        )

    def dispatch(self, *args, **kwargs):
        mgr = state.swap_manager
        if mgr is not None and mgr.current is not None:
            # Mid-migration: layers hold mixed precisions, no recorded graph
            # matches. Take vLLM's own "no graph" path for this step.
            saved = self._graphs_captured
            self._graphs_captured = False
            try:
                return orig_dispatch(self, *args, **kwargs)
            finally:
                self._graphs_captured = saved
        return orig_dispatch(self, *args, **kwargs)

    def execute_model(self, scheduler_output, *args, **kwargs):
        dummy = kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
        mgr = state.swap_manager
        if mgr is not None and not dummy:
            n = scheduler_output.total_num_scheduled_tokens
            if n:
                mgr.on_step(n)
        return orig_execute(self, scheduler_output, *args, **kwargs)

    model_mgr.capture = capture
    base.dispatch = dispatch
    runner.execute_model = execute_model
    state.graphs_enabled = True
    _installed = True
    state.logger.info("precision_autoscaler: two-graph-set patches installed (vLLM model runner V2)")
