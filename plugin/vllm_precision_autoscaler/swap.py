"""Swap format: one precision in VRAM, both in pinned CPU RAM.

Each linear layer owns a GPU "region" the size of its BF16 weight. The region
holds either the BF16 weight or (in its first half) the FP8 weight. Both
versions live in pinned CPU memory. When the target precision changes, layers
migrate one by one *while the model keeps serving*:

  1. side CUDA stream: copy the layer's new weights CPU -> staging buffer
     (async over PCIe, overlaps with the model's kernels)
  2. at the start of a later engine step, once that copy has finished:
     staging -> region (fast GPU-to-GPU copy, on the main stream) and flip
     that layer's precision.

Because step 2 runs on the main stream before any layer of that step uses its
weights, every step sees a consistent set of weights; during a migration some
layers are FP8 and some are 16-bit. Staging is a few layer-sized buffers, not
a second model copy.

Requires CUDA graphs off: a recorded graph would keep calling the kernel it was
recorded with, on bytes that now hold the other format.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from .swap_policy import HysteresisPolicy


@dataclass
class SwapLayer:
    layer_id: int
    n: int
    k: int
    region: torch.Tensor        # GPU uint8 [2*n*k]
    scale: torch.Tensor         # GPU fp32 [n, 1] (FP8 per-channel scales, always resident)
    cpu_high: torch.Tensor      # pinned uint8 [2*n*k] (BF16/FP16 bytes)
    cpu_fp8: torch.Tensor       # pinned uint8 [n*k]
    mode: str = "high"

    def nbytes(self, mode: str) -> int:
        return self.n * self.k * (2 if mode == "high" else 1)


@dataclass
class Migration:
    target: str
    started: float
    finished: float | None = None
    layers: int = 0
    bytes: int = 0
    steps: int = 0


@dataclass
class _Transfer:
    slot: int
    layer_id: int
    mode: str
    done: torch.cuda.Event


class SwapManager:
    def __init__(self, settings: Any, logger: Any) -> None:
        self.settings = settings
        self.logger = logger
        self.layers: dict[int, SwapLayer] = {}
        self.order: list[int] = []
        self.target = "fp8" if settings.mode == "fp8" else "high"
        self.policy = (
            HysteresisPolicy(settings.swap_up, settings.swap_down, settings.swap_dwell_s,
                             settings.swap_ema_s, start=self.target)
            if settings.mode == "auto" else None
        )
        self.staging: list[torch.Tensor] = []
        self.slot_free: list[torch.cuda.Event | None] = []
        self.inflight: list[_Transfer] = []
        self.side: torch.cuda.Stream | None = None
        self.first_id: int | None = None
        self.migrations: list[Migration] = []
        self.current: Migration | None = None
        self.steps = 0

    # -- setup ---------------------------------------------------------------
    def register(self, layer: SwapLayer) -> None:
        self.layers[layer.layer_id] = layer
        self.order.append(layer.layer_id)
        if self.first_id is None or layer.layer_id < self.first_id:
            self.first_id = layer.layer_id

    def finalize(self, device: torch.device) -> None:
        """Allocate staging buffers (counted in model memory, before vLLM
        sizes the KV cache)."""
        if self.staging:
            return
        biggest = max(l.nbytes("high") for l in self.layers.values())
        for _ in range(self.settings.swap_slots):
            self.staging.append(torch.empty(biggest, dtype=torch.uint8, device=device))
            self.slot_free.append(None)
        self.side = torch.cuda.Stream(device=device)
        pinned = sum(l.cpu_high.numel() + l.cpu_fp8.numel() for l in self.layers.values())
        self.logger.info(
            "precision_autoscaler swap: %d layers, pinned CPU copies %.2f GiB, "
            "staging %d x %.0f MiB, start=%s, policy=%s",
            len(self.layers), pinned / 2**30, len(self.staging), biggest / 2**20, self.target,
            "hysteresis" if self.policy else "manual",
        )
        if self.target == "fp8":  # start in FP8: migrate before serving
            self.set_target("fp8")

    # -- control -------------------------------------------------------------
    def set_target(self, mode: str) -> None:
        mode = "fp8" if mode == "fp8" else "high"
        if mode != self.target or self._pending():
            self.target = mode
            if self.current is None or self.current.target != mode:
                self.current = Migration(target=mode, started=time.monotonic())
                self.logger.info("precision_autoscaler swap: migrating to %s", mode)

    def _pending(self) -> list[int]:
        busy = {t.layer_id for t in self.inflight}
        return [i for i in self.order if self.layers[i].mode != self.target and i not in busy]

    # -- called once per engine step (from the first swap layer's op) ---------
    def on_step(self, num_tokens: int) -> None:
        self.steps += 1
        if self.policy is not None:
            new = self.policy.update(time.monotonic(), num_tokens)
            if new != self.target:
                self.set_target(new)
        if self.current is not None:
            self.current.steps += 1
        self._complete()
        self._launch()
        if self.current is not None and not self.inflight and not self._pending():
            m = self.current
            m.finished = time.monotonic()
            self.migrations.append(m)
            self.current = None
            self.logger.info(
                "precision_autoscaler swap: now %s; %d layers, %.2f GiB in %.3f s over %d steps",
                m.target, m.layers, m.bytes / 2**30, m.finished - m.started, m.steps,
            )

    def _complete(self) -> None:
        main = torch.cuda.current_stream()
        still = []
        for t in self.inflight:
            if not t.done.query():
                still.append(t)
                continue
            layer = self.layers[t.layer_id]
            nb = layer.nbytes(t.mode)
            layer.region[:nb].copy_(self.staging[t.slot][:nb], non_blocking=True)
            layer.mode = t.mode
            free = torch.cuda.Event()
            free.record(main)
            self.slot_free[t.slot] = free
            if self.current is not None and self.current.target == t.mode:
                self.current.layers += 1
                self.current.bytes += nb
        self.inflight = still

    def _launch(self) -> None:
        pending = self._pending()
        if not pending:
            return
        used = {t.slot for t in self.inflight}
        for slot in range(len(self.staging)):
            if slot in used or not pending:
                continue
            layer = self.layers[pending.pop(0)]
            src = layer.cpu_high if self.target == "high" else layer.cpu_fp8
            with torch.cuda.stream(self.side):
                if self.slot_free[slot] is not None:
                    self.side.wait_event(self.slot_free[slot])
                self.staging[slot][: src.numel()].copy_(src, non_blocking=True)
                done = torch.cuda.Event()
                done.record(self.side)
            self.inflight.append(_Transfer(slot, layer.layer_id, self.target, done))

    # -- telemetry -----------------------------------------------------------
    def stats(self) -> dict:
        fp8 = sum(1 for l in self.layers.values() if l.mode == "fp8")
        recent = self.migrations[-10:]
        return {
            "target": self.target,
            "layers": len(self.layers),
            "fp8_layers": fp8,
            "migrating": self.current is not None,
            "ema_tokens_per_step": self.policy.ema if self.policy else None,
            "steps": self.steps,
            "migrations": [
                {"target": m.target, "seconds": (m.finished or 0) - m.started, "layers": m.layers,
                 "gib": m.bytes / 2**30, "steps": m.steps}
                for m in recent
            ],
        }
