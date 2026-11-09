"""Swap format: one precision version in VRAM, all versions in pinned CPU RAM.

Each linear layer owns a GPU "region" the size of its largest version. The
region holds the bytes of the *active* version; every configured version
(versions.py: 16-bit, FP8, ...) lives in pinned CPU memory in its kernel's
format. When the target version changes, layers migrate *while the model keeps
serving*:

  1. side CUDA stream: copy layers' new bytes CPU -> staging buffer (several
     whole layers packed per buffer; async over PCIe, overlaps with compute)
  2. at the start of a later engine step, once a copy has finished:
     staging -> region (fast GPU-to-GPU copy, main stream) and flip those
     layers to the new version.

Step 2 runs on the main stream before any layer of that step uses its weights,
so every step sees consistent weights; during a migration some layers run one
version and some another. With no traffic, pump() (called from the HTTP API)
finishes the migration in one go.

CUDA graphs: either off, or one recorded graph set per version (graphs.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from .swap_policy import HysteresisPolicy
from .versions import KINDS, LayerVersion, canonical

# pump() treats the engine as idle when no step ran for this long, and then
# finishes a pending migration in one go instead of one batch per call.
IDLE_AFTER_S = 0.2


@dataclass
class SwapLayer:
    layer_id: int
    n: int
    k: int
    region: torch.Tensor                  # GPU uint8, sized for the largest version
    versions: dict[str, LayerVersion]
    active: str = "high"

    def nbytes(self, version: str) -> int:
        return self.versions[version].nbytes


def build_swap_layer(layer_id: int, weight: torch.Tensor, region: torch.Tensor,
                     version_names: tuple[str, ...], active: str = "high") -> SwapLayer:
    """Build every configured version of one layer from its 16-bit weight."""
    n, k = weight.shape
    versions = {name: KINDS[name].build(weight) for name in version_names}
    biggest = max(v.nbytes for v in versions.values())
    assert region.numel() >= biggest, (region.numel(), biggest)
    return SwapLayer(layer_id, n, k, region, versions, active)


@dataclass
class Migration:
    target: str
    started: float
    finished: float | None = None
    layers: int = 0
    bytes: int = 0
    steps: int = 0
    transfers: int = 0


@dataclass
class _Transfer:
    """One staging buffer's worth of copies: several whole layers packed
    back to back (layer_id, offset in the staging buffer, bytes)."""
    slot: int
    items: list[tuple[int, int, int]]
    version: str
    done: torch.cuda.Event


class SwapManager:
    def __init__(self, settings: Any, logger: Any) -> None:
        self.settings = settings
        self.logger = logger
        self.version_names: tuple[str, ...] = tuple(getattr(settings, "swap_versions", ("high", "fp8")))
        self.layers: dict[int, SwapLayer] = {}
        self.order: list[int] = []
        start = canonical(settings.mode)
        self.target = start if start in self.version_names else "high"
        # Deciding *when* to switch is an external controller's job by default
        # (POST /v1/precision). The built-in policy is an explicit opt-in.
        self.policy = (
            HysteresisPolicy(settings.swap_up, settings.swap_down, settings.swap_dwell_s,
                             settings.swap_ema_s, start="high")
            if settings.mode == "auto" and settings.swap_policy == "hysteresis"
            and "fp8" in self.version_names else None
        )
        self.staging: list[torch.Tensor] = []
        self.slot_free: list[torch.cuda.Event | None] = []
        self.slot_bytes = 0
        self.inflight: list[_Transfer] = []
        self.side: torch.cuda.Stream | None = None
        self.first_id: int | None = None
        self.migrations: list[Migration] = []
        self.current: Migration | None = None
        self.steps = 0
        self.last_step_t = 0.0  # monotonic time of the last engine step

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
        biggest = max(max(v.nbytes for v in l.versions.values()) for l in self.layers.values())
        n_slots = self.settings.swap_slots
        budget = getattr(self.settings, "swap_staging_mib", 0)
        if budget:
            # Memory budget instead of a count: as many largest-layer-sized
            # buffers as fit (at least one).
            n_slots = max(1, int(budget * 2**20 // biggest))
        self.slot_bytes = biggest
        for _ in range(n_slots):
            self.staging.append(torch.empty(biggest, dtype=torch.uint8, device=device))
            self.slot_free.append(None)
        self.side = torch.cuda.Stream(device=device)
        self.logger.info(
            "precision_autoscaler swap: %d layers, versions %s, pinned CPU copies %.2f GiB, "
            "staging %d x %.0f MiB, start=%s, policy=%s",
            len(self.layers), ",".join(self.version_names), self.pinned_bytes() / 2**30,
            len(self.staging), biggest / 2**20, self.target,
            "hysteresis" if self.policy else "manual",
        )
        if self.target != "high":  # start in another version: migrate before serving
            target, self.target = self.target, "high"
            self.set_target(target)

    def pinned_bytes(self, version: str | None = None) -> int:
        return sum(v.cpu.numel() for l in self.layers.values() for name, v in l.versions.items()
                   if version is None or name == version)

    # -- control -------------------------------------------------------------
    def set_target(self, version: str) -> None:
        version = canonical(version)
        if version not in self.version_names:
            raise ValueError(f"unknown version {version!r}; available: {list(self.version_names)}")
        if version != self.target or self._pending():
            self.target = version
            if self.current is None or self.current.target != version:
                self.current = Migration(target=version, started=time.monotonic())
                self.logger.info("precision_autoscaler swap: migrating to %s", version)

    def _pending(self) -> list[int]:
        busy = {lid for t in self.inflight for lid, _, _ in t.items}
        return [i for i in self.order if self.layers[i].active != self.target and i not in busy]

    def _maybe_finish(self) -> None:
        if self.current is not None and not self.inflight and not self._pending():
            m = self.current
            m.finished = time.monotonic()
            self.migrations.append(m)
            self.current = None
            self.logger.info(
                "precision_autoscaler swap: now %s; %d layers, %.2f GiB in %.3f s "
                "over %d steps (%d transfers)",
                m.target, m.layers, m.bytes / 2**30, m.finished - m.started, m.steps, m.transfers,
            )

    # -- called once per engine step ------------------------------------------
    def on_step(self, num_tokens: int) -> None:
        self.steps += 1
        self.last_step_t = time.monotonic()
        if self.policy is not None:
            new = self.policy.update(time.monotonic(), num_tokens)
            if new != self.target:
                self.set_target(new)
        if self.current is not None:
            self.current.steps += 1
        self._complete()
        self._launch()
        self._maybe_finish()

    def pump(self) -> None:
        """Advance a pending migration outside an engine step (from an RPC,
        which vLLM runs between steps). Lets a migration finish while idle."""
        if self.current is None and not self.inflight and not self._pending():
            return
        if time.monotonic() - self.last_step_t > IDLE_AFTER_S:
            self._drain()  # engine idle: nothing to protect, finish now
            return
        self._complete()
        self._launch()
        self._maybe_finish()

    def _drain(self, budget_s: float = 30.0) -> None:
        """Finish the whole migration now, waiting for each copy. Only used
        when no engine step ran recently (pump() from an RPC while idle):
        vLLM runs RPCs between steps, so blocking here delays no request
        that is already running; a new one waits at most until this returns."""
        deadline = time.monotonic() + budget_s
        while (self.inflight or self._pending()) and time.monotonic() < deadline:
            self._launch()
            if self.side is not None:
                self.side.synchronize()
            self._complete()
        torch.cuda.current_stream().synchronize()
        self._maybe_finish()

    def _complete(self) -> None:
        main = torch.cuda.current_stream()
        still = []
        for t in self.inflight:
            if not t.done.query():
                still.append(t)
                continue
            for layer_id, off, nb in t.items:
                layer = self.layers[layer_id]
                layer.region[:nb].copy_(self.staging[t.slot][off : off + nb], non_blocking=True)
                layer.active = t.version
                if self.current is not None and self.current.target == t.version:
                    self.current.layers += 1
                    self.current.bytes += nb
            free = torch.cuda.Event()
            free.record(main)
            self.slot_free[t.slot] = free
        self.inflight = still

    def _launch(self) -> None:
        pending = self._pending()
        if not pending:
            return
        used = {t.slot for t in self.inflight}
        for slot in range(len(self.staging)):
            if slot in used or not pending:
                continue
            # Pack as many whole pending layers as fit into this buffer (in
            # order, skipping ones that don't fit so smaller ones can fill gaps).
            items, off, rest = [], 0, []
            for lid in pending:
                nb = self.layers[lid].nbytes(self.target)
                pack_ok = getattr(self.settings, "swap_pack", True) or not items
                if pack_ok and off + nb <= self.slot_bytes:
                    items.append((lid, off, nb))
                    off += nb
                else:
                    rest.append(lid)
            pending = rest
            if not items:
                continue
            with torch.cuda.stream(self.side):
                if self.slot_free[slot] is not None:
                    self.side.wait_event(self.slot_free[slot])
                for lid, o, nb in items:
                    src = self.layers[lid].versions[self.target].cpu
                    self.staging[slot][o : o + nb].copy_(src, non_blocking=True)
                done = torch.cuda.Event()
                done.record(self.side)
            self.inflight.append(_Transfer(slot, items, self.target, done))
            if self.current is not None:
                self.current.transfers += 1

    # -- telemetry -----------------------------------------------------------
    def stats(self) -> dict:
        by_version = {name: 0 for name in self.version_names}
        for l in self.layers.values():
            by_version[l.active] = by_version.get(l.active, 0) + 1
        recent = self.migrations[-10:]
        return {
            "target": self.target,
            "versions": [
                {"name": name, "bits": KINDS[name].bits, "method": KINDS[name].method,
                 "ram_gib": round(self.pinned_bytes(name) / 2**30, 3),
                 "active_layers": by_version.get(name, 0)}
                for name in self.version_names
            ],
            "staging": {"buffers": len(self.staging), "mib_each": round(self.slot_bytes / 2**20)},
            "policy": "hysteresis" if self.policy else "external",
            "layers": len(self.layers),
            "fp8_layers": by_version.get("fp8", 0),
            "migrating": self.current is not None,
            "ema_tokens_per_step": self.policy.ema if self.policy else None,
            "steps": self.steps,
            "migrations": [
                {"target": m.target, "seconds": (m.finished or 0) - m.started, "layers": m.layers,
                 "gib": m.bytes / 2**30, "steps": m.steps, "transfers": m.transfers}
                for m in recent
            ],
        }
