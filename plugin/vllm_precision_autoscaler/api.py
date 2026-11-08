"""HTTP control API: the plugin switches precision, an external controller decides when.

Registered under `vllm.endpoint_plugins` (loaded in vLLM's API server process).
vLLM loads endpoint plugins only when they are named in `VLLM_PLUGINS`, and that
variable then also allowlists general plugins, so name both:

    VLLM_PLUGINS=precision_autoscaler,precision_autoscaler_api vllm serve ...

Routes:
    GET  /v1/precision                 -> current state (also advances a pending
                                          migration while the engine is idle)
    POST /v1/precision {"mode": "fp8"} -> start switching to fp8 | high; returns state

The routes reach the engine through `engine_client.collective_rpc`, calling the
`vpa_*` worker methods that the general plugin attaches (worker_ext.py).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

MODES = ("high", "fp8")


class PrecisionAPIPlugin:
    name = "precision_autoscaler_api"
    required_tasks: tuple[str, ...] | None = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        @app.get("/v1/precision")
        async def get_precision(raw_request: Request):
            client = _client(raw_request)
            results = await client.collective_rpc("vpa_pump")
            return _summary(results)

        @app.post("/v1/precision")
        async def set_precision(raw_request: Request):
            client = _client(raw_request)
            try:
                body = await raw_request.json()
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f"invalid JSON: {e}") from e
            mode = str((body or {}).get("mode", "")).strip().lower()
            mode = {"bf16": "high", "fp16": "high", "16": "high"}.get(mode, mode)
            if mode not in MODES:
                raise HTTPException(status_code=400, detail=f"mode must be one of {MODES}")
            results = await client.collective_rpc("vpa_set_mode", args=(mode,))
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
        "mode": swap.get("target", first.get("mode")),
        "fp8_layers": swap.get("fp8_layers"),
        "layers": swap.get("layers", first.get("layers")),
        "migrating": swap.get("migrating", False),
        "migrations": swap.get("migrations", []),
        "workers": len(results),
    }
