"""A stand-in for one vLLM server with the plugin, for testing the Kubernetes setup locally.

It speaks the parts of vLLM's HTTP API that the experiment uses (streaming /v1/completions,
/health, /metrics with the same metric names, and the plugin's /v1/precision routes), and
generates tokens at the per-user speed measured on a real RTX 4090 with Qwen3-4B
(measured total tokens/s at 1/8/32/64/128 users). No GPU, no model.

    python fake_vllm.py --port 8001 --cold-start 40            # plugin (BF16/FP8/INT4 W4A8)
    python fake_vllm.py --port 8001 --cold-start 30 --stock    # stock vLLM, BF16 only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from aiohttp import web

USERS = (1, 8, 32, 64, 128)
TOTAL_TOK_S = {  # measured, RTX 4090 24 GB, Qwen3-4B, 512-token prompts, 256 output tokens
    "high": (101, 675, 1946, 2635, 2890),
    "fp8": (127, 858, 2301, 3051, 3322),
    "int4": (222, 1348, 2922, 3145, 3131),
    "int4a8": (203, 1276, 3060, 3730, 3715),
}
BITS = {"high": 16, "fp8": 8, "int4": 4, "int4a8": 4}
KV_TOKENS = 87_488          # KV cache of the 24 GB card
TOKENS_PER_REQUEST = 768    # 512 prompt + 256 output


def per_user_tok_s(version: str, users: int) -> float:
    """Linear interpolation of total tokens/s over the measured user counts, per user."""
    n = max(1, users)
    xs, ys = USERS, TOTAL_TOK_S[version]
    if n >= xs[-1]:
        total = ys[-1]
    else:
        i = next(k for k in range(1, len(xs)) if n <= xs[k])
        f = (n - xs[i - 1]) / (xs[i] - xs[i - 1])
        total = ys[i - 1] + f * (ys[i] - ys[i - 1])
    return total / n


class Server:
    def __init__(self, args):
        self.stock = args.stock
        self.versions = ["high"] if args.stock else args.versions.split(",")
        self.version = self.versions[0]
        self.target = self.version
        self.migrating = False
        self.migrations: list[dict] = []
        self.model = args.model
        self.running = 0
        self.waiting = 0
        self.slots = asyncio.Semaphore(KV_TOKENS // TOKENS_PER_REQUEST)
        self.gen_tokens = 0
        self.prompt_tokens = 0
        self.success = 0

    async def completions(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        max_tokens = int(body.get("max_tokens", 16))
        prompt = body.get("prompt", "")
        self.waiting += 1
        async with self.slots:
            self.waiting -= 1
            self.running += 1
            try:
                resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await resp.prepare(request)
                await asyncio.sleep(0.03 + 0.002 * self.running)  # prefill
                self.prompt_tokens += len(prompt) if isinstance(prompt, list) else 1
                for i in range(max_tokens):
                    if i:
                        await asyncio.sleep(1.0 / per_user_tok_s(self.version, self.running))
                    self.gen_tokens += 1
                    chunk = {"id": "fake", "object": "text_completion", "model": self.model,
                             "choices": [{"index": 0, "text": " tok", "finish_reason": None}]}
                    await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
                await resp.write(b"data: [DONE]\n\n")
                self.success += 1
                return resp
            finally:
                self.running -= 1

    async def health(self, _request):
        return web.Response(text="")

    async def metrics(self, _request):
        lab = f'{{engine="0",model_name="{self.model}"}}'
        kv = min(1.0, self.running * TOKENS_PER_REQUEST / KV_TOKENS)
        lines = [
            "# TYPE vllm:num_requests_running gauge", f"vllm:num_requests_running{lab} {self.running}",
            "# TYPE vllm:num_requests_waiting gauge", f"vllm:num_requests_waiting{lab} {self.waiting}",
            "# TYPE vllm:kv_cache_usage_perc gauge", f"vllm:kv_cache_usage_perc{lab} {kv}",
            "# TYPE vllm:generation_tokens_total counter",
            f"vllm:generation_tokens_total{lab} {self.gen_tokens}",
            "# TYPE vllm:prompt_tokens_total counter", f"vllm:prompt_tokens_total{lab} {self.prompt_tokens}",
            "# TYPE vllm:request_success_total counter",
            f'vllm:request_success_total{{engine="0",finished_reason="length",model_name="{self.model}"}} {self.success}',
        ]
        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")

    def state(self) -> dict:
        return {"format": "swap", "version": self.target, "mode": self.target,
                "versions": self.versions, "layers": 144, "migrating": self.migrating,
                "migrations": self.migrations[-20:], "workers": 1}

    async def get_precision(self, _request):
        return web.json_response(self.state())

    async def set_precision(self, request):
        body = await request.json()
        version = str(body.get("version", body.get("mode", ""))).strip()
        if version not in self.versions:
            raise web.HTTPBadRequest(text=json.dumps({"detail": f"unknown version {version!r}"}))
        if version != self.target:
            self.target, self.migrating = version, True
            asyncio.get_running_loop().create_task(self._migrate(version))
        return web.json_response(self.state())

    async def _migrate(self, version):
        t = time.monotonic()
        await asyncio.sleep(0.5)  # measured migrations: 0.25-1.2 s
        if self.target == version:
            self.version, self.migrating = version, False
            self.migrations.append({"target": version, "seconds": time.monotonic() - t, "layers": 144})

    async def precision_metrics(self, _request):
        bits = BITS.get(self.version, 0)
        lines = ["# TYPE vpa_active_bits gauge", f"vpa_active_bits {bits}",
                 "# TYPE vpa_target_bits gauge", f"vpa_target_bits {BITS.get(self.target, 0)}",
                 "# TYPE vpa_migrating gauge", f"vpa_migrating {int(self.migrating)}",
                 "# TYPE vpa_migrations_total counter", f"vpa_migrations_total {len(self.migrations)}"]
        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--cold-start", type=float, default=40.0, help="seconds before the server answers")
    ap.add_argument("--stock", action="store_true", help="stock vLLM: BF16 only, no /v1/precision")
    ap.add_argument("--versions", default="high,fp8,int4a8")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    args = ap.parse_args()
    time.sleep(args.cold_start)  # model load + CUDA graph capture
    srv = Server(args)
    app = web.Application()
    app.router.add_post("/v1/completions", srv.completions)
    app.router.add_get("/health", srv.health)
    app.router.add_get("/metrics", srv.metrics)
    if not args.stock:
        app.router.add_get("/v1/precision", srv.get_precision)
        app.router.add_post("/v1/precision", srv.set_precision)
        app.router.add_get("/v1/precision/metrics", srv.precision_metrics)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
