"""HTTP control API: the plugin switches precision, an external controller decides when.

Registered under `vllm.endpoint_plugins` (loaded in vLLM's API server process).
vLLM loads endpoint plugins only when they are named in `VLLM_PLUGINS`, and that
variable then also allowlists general plugins, so name both:

    VLLM_PLUGINS=precision_autoscaler,precision_autoscaler_api vllm serve ...

Routes:
    GET  /v1/precision                    -> active version, all configured versions,
                                             migration state (also finishes a pending
                                             migration while the engine is idle)
    GET  /v1/precision/metrics            -> the same state in Prometheus text format
                                             (scrape it next to vLLM's /metrics)
    POST /v1/precision {"version": "fp8"} -> start switching to that version
                                             ("mode" is accepted as an alias);
                                             400 for a version that isn't configured

The routes reach the engine through `engine_client.collective_rpc`, calling the
`vpa_*` worker methods that the general plugin attaches (worker_ext.py).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse


class PrecisionAPIPlugin:
    name = "precision_autoscaler_api"
    required_tasks: tuple[str, ...] | None = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        @app.get("/v1/precision")
        async def get_precision(raw_request: Request):
            client = _client(raw_request)
            results = await client.collective_rpc("vpa_pump")
            return _summary(results)

        @app.get("/v1/precision/metrics")
        async def precision_metrics(raw_request: Request):
            client = _client(raw_request)
            results = await client.collective_rpc("vpa_pump")
            return PlainTextResponse(prometheus_text(results[0] if results else {}),
                                     media_type="text/plain; version=0.0.4")

        @app.post("/v1/precision")
        async def set_precision(raw_request: Request):
            client = _client(raw_request)
            try:
                body = await raw_request.json()
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f"invalid JSON: {e}") from e
            body = body or {}
            version = str(body.get("version", body.get("mode", ""))).strip()
            if not version:
                raise HTTPException(status_code=400, detail='body must be {"version": "<name>"}')
            results = await client.collective_rpc("vpa_set_mode", args=(version,))
            if results and isinstance(results[0], dict) and results[0].get("error"):
                raise HTTPException(status_code=400, detail=results[0]["error"])
            return _summary(results)

    async def init_state(self, engine_client, state, args) -> None:
        state.vpa_engine_client = engine_client


def _client(raw_request: Request):
    client = getattr(raw_request.app.state, "vpa_engine_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="no engine on this server")
    return client


def _summary(results) -> dict:
    """One worker's view (all workers hold the same state with TP=1; with
    tensor parallelism every rank reports, and they should agree)."""
    if not results:
        raise HTTPException(status_code=500, detail="no worker answered")
    first = results[0]
    if not isinstance(first, dict):
        raise HTTPException(status_code=500, detail=f"unexpected worker reply: {first!r}")
    swap = first.get("swap") or {}
    return {
        "format": first.get("format"),
        "version": swap.get("target", first.get("mode")),
        "mode": swap.get("target", first.get("mode")),  # alias (v0.3 clients)
        "versions": swap.get("versions", []),
        "fp8_layers": swap.get("fp8_layers"),
        "layers": swap.get("layers", first.get("layers")),
        "migrating": swap.get("migrating", False),
        "migrations": swap.get("migrations", []),
        "workers": len(results),
    }


def prometheus_text(stats: dict) -> str:
    """Plugin state as Prometheus metrics (prefix vpa_)."""
    swap = (stats or {}).get("swap") or {}
    versions = swap.get("versions") or []
    layers = swap.get("layers") or 0
    out: list[str] = []

    def metric(name, kind, help_, samples):
        out.append(f"# HELP {name} {help_}")
        out.append(f"# TYPE {name} {kind}")
        for labels, value in samples:
            lab = ",".join(f'{k}="{v}"' for k, v in labels.items())
            out.append(f"{name}{{{lab}}} {value}" if lab else f"{name} {value}")

    target = swap.get("target")
    bits = {v["name"]: v.get("bits", 0) for v in versions}
    active = max(versions, key=lambda v: v["active_layers"])["name"] if versions else None
    metric("vpa_target_bits", "gauge", "Bits of the requested precision version (16, 8, 4).",
           [({}, bits.get(target, 0))])
    metric("vpa_active_bits", "gauge", "Bits of the version most layers currently hold.",
           [({}, bits.get(active, 0))])
    metric("vpa_version_target", "gauge", "1 for the requested version.",
           [({"version": v["name"]}, int(v["name"] == target)) for v in versions])
    metric("vpa_version_active_layers", "gauge", "Layers currently holding each version.",
           [({"version": v["name"]}, v["active_layers"]) for v in versions])
    metric("vpa_version_ram_bytes", "gauge", "Pinned host RAM held by each version.",
           [({"version": v["name"]}, int(v["ram_gib"] * 2**30)) for v in versions])
    weights = sum(v["ram_gib"] * v["active_layers"] / layers for v in versions) if layers else 0
    metric("vpa_weights_vram_bytes", "gauge", "Bytes of weights of the active precision in VRAM.",
           [({}, int(weights * 2**30))])
    metric("vpa_migrating", "gauge", "1 while layers are being migrated.",
           [({}, int(bool(swap.get("migrating"))))])
    metric("vpa_migrations_total", "counter", "Finished and started precision migrations.",
           [({}, swap.get("migrations_total", len(swap.get("migrations") or [])))])
    metric("vpa_migration_seconds_total", "counter", "Total time spent migrating.",
           [({}, round(swap.get("migration_seconds_total", 0.0), 4))])
    last = (swap.get("migrations") or [None])[-1]
    if last:
        metric("vpa_last_migration_seconds", "gauge", "Duration of the last migration.",
               [({"version": last["target"]}, round(last["seconds"], 4))])
    return "\n".join(out) + "\n"
