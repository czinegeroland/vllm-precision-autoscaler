"""Runs ON each warm RunPod pod: starts and stops vLLM on demand and forwards traffic to it.

The GPU machine stays up the whole time (image pulled, model on disk); vLLM is started only
when Kubernetes schedules the replica that owns this machine. That makes the cold start
real (model load, plugin quantization, CUDA graph capture) but repeatable.

    python3 poc/k8s-keda/agent.py --port 8000                      # real vLLM on this GPU
    python3 poc/k8s-keda/agent.py --port 8000 --fake --cold-start 40   # local test, no GPU

Routes (all need the header X-Agent-Token when AGENT_TOKEN is set):
    POST /agent/start {"mode": "stock"|"plugin"}  start vLLM (no-op if already up in that mode)
    POST /agent/stop                               stop vLLM (and the precision controller)
    GET  /agent/status                             state, mode, cold start seconds
    GET  /agent/ready                              200 when vLLM answers, else 503
    GET  /metrics                                  agent + vLLM + plugin + controller metrics
    anything else                                  streamed to vLLM (/v1/completions, /health, ...)

Modes:
    stock   plain vLLM, BF16, default compilation (what you would run without the plugin)
    plugin  vLLM + precision_autoscaler (swap format, BF16/FP8/INT4 W4A8) + the precision
            controller (controller/precision_controller.py, config poc/k8s-keda/controller.yaml)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

import aiohttp
from aiohttp import web

ROOT = pathlib.Path(__file__).resolve().parents[2]
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host",
       "content-encoding", "x-agent-token"}


class Agent:
    def __init__(self, args):
        self.args = args
        self.token = os.environ.get("AGENT_TOKEN", "")
        self.vllm_url = f"http://127.0.0.1:{args.vllm_port}"
        self.logs = pathlib.Path(args.log_dir)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.state = "idle"          # idle | starting | ready | stopping | failed
        self.mode: str | None = None
        self.proc: subprocess.Popen | None = None
        self.ctl: subprocess.Popen | None = None
        self.started_at = self.ready_at = None
        self.cold_starts: list[dict] = []
        self.error = ""
        self.lock = asyncio.Lock()
        self.watch: asyncio.Task | None = None
        self.session: aiohttp.ClientSession | None = None

    # -- process control ---------------------------------------------------------------
    def vllm_command(self, mode: str) -> tuple[list[str], dict]:
        env = dict(os.environ)
        if self.args.fake:
            cmd = [sys.executable, str(ROOT / "poc/k8s-keda/fake_vllm.py"), "--port",
                   str(self.args.vllm_port), "--cold-start",
                   str(self.args.cold_start * (1.0 if mode == "plugin" else self.args.stock_factor))]
            return cmd + (["--stock"] if mode == "stock" else []), env
        env.update({"HF_HOME": self.args.hf_home,
                    # one compile cache per mode: vLLM's cache key ignores VPA_* settings
                    "VLLM_CACHE_ROOT": f"{self.args.cache_root}/{mode}"})
        cmd = ["vllm", "serve", self.args.model, "--port", str(self.args.vllm_port),
               "--host", "127.0.0.1", "--max-model-len", "4096",
               "--gpu-memory-utilization", "0.90", "--seed", "0"]
        if mode == "plugin":
            env.update({"VPA_FORMAT": "swap", "VPA_VERSIONS": self.args.versions, "VPA_MODE": "high",
                        "VLLM_PLUGINS": "precision_autoscaler,precision_autoscaler_api"})
            cmd += ["--quantization", "precision_autoscaler",
                    "--compilation-config", '{"cudagraph_mode": "FULL_DECODE_ONLY"}']
        return cmd, env

    async def start(self, mode: str) -> None:
        async with self.lock:
            if self.mode == mode and self.state in ("starting", "ready"):
                return
            await self._stop_locked()
            cmd, env = self.vllm_command(mode)
            log = open(self.logs / f"vllm-{mode}.log", "ab")
            log.write(f"\n==== start {time.strftime('%H:%M:%S')} {' '.join(cmd)}\n".encode())
            self.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, cwd=str(ROOT))
            self.mode, self.state, self.error = mode, "starting", ""
            self.started_at, self.ready_at = time.time(), None
            self.watch = asyncio.get_running_loop().create_task(self._wait_ready())
            print(f"[agent] starting vLLM ({mode}) pid {self.proc.pid}", flush=True)

    async def _wait_ready(self) -> None:
        while self.proc and self.proc.poll() is None:
            try:
                async with self.session.get(f"{self.vllm_url}/health",
                                            timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(0.5)
        else:
            self.state = "failed"
            self.error = f"vLLM exited with code {self.proc.returncode if self.proc else '?'}"
            print(f"[agent] {self.error}", flush=True)
            return
        if self.mode == "plugin" and self.args.controller_config:
            self.ctl = subprocess.Popen(
                [sys.executable, str(ROOT / "controller/precision_controller.py"),
                 "--config", self.args.controller_config, "--url", self.vllm_url,
                 "--metrics-port", str(self.args.controller_metrics_port),
                 "--log", str(self.logs / "controller.jsonl")],
                stdout=open(self.logs / "controller.out", "ab"), stderr=subprocess.STDOUT,
                start_new_session=True, cwd=str(ROOT))
        self.ready_at = time.time()
        self.state = "ready"
        cs = {"mode": self.mode, "seconds": round(self.ready_at - self.started_at, 2),
              "at": self.ready_at}
        self.cold_starts.append(cs)
        with open(self.logs / "cold_starts.jsonl", "a") as f:
            f.write(json.dumps(cs) + "\n")
        print(f"[agent] vLLM ready ({self.mode}) after {cs['seconds']} s", flush=True)

    async def stop(self) -> None:
        async with self.lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        if self.watch and not self.watch.done():
            self.watch.cancel()
        procs = [p for p in (self.ctl, self.proc) if p and p.poll() is None]
        if procs:
            self.state = "stopping"
            for p in procs:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except (ProcessLookupError, AttributeError):
                    p.terminate()
            deadline = time.time() + 30
            while time.time() < deadline and any(p.poll() is None for p in procs):
                await asyncio.sleep(0.25)
            for p in procs:
                if p.poll() is None:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except (ProcessLookupError, AttributeError):
                        p.kill()
            if not self.args.fake:
                await self._wait_gpu_free()
            print("[agent] vLLM stopped", flush=True)
        self.proc = self.ctl = None
        self.state, self.mode = "idle", None

    async def _wait_gpu_free(self) -> None:
        """The next start must see the whole GPU, so wait until the old process' memory is gone."""
        for _ in range(60):
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout
                if int(out.split()[0]) < 1500:
                    return
            except Exception:  # noqa: BLE001
                return
            await asyncio.sleep(0.5)

    def status(self) -> dict:
        return {"state": self.state, "mode": self.mode, "started_at": self.started_at,
                "ready_at": self.ready_at, "error": self.error,
                "cold_start_s": self.cold_starts[-1]["seconds"] if self.cold_starts else None,
                "cold_starts": self.cold_starts[-10:]}

    # -- HTTP --------------------------------------------------------------------------
    @web.middleware
    async def auth(self, request, handler):
        if self.token and request.headers.get("X-Agent-Token") != self.token:
            raise web.HTTPUnauthorized(text="missing or wrong X-Agent-Token")
        return await handler(request)

    async def h_start(self, request):
        body = await request.json() if request.can_read_body else {}
        mode = body.get("mode", "plugin")
        if mode not in ("stock", "plugin"):
            raise web.HTTPBadRequest(text="mode must be stock or plugin")
        await self.start(mode)
        return web.json_response(self.status())

    async def h_stop(self, _request):
        await self.stop()
        return web.json_response(self.status())

    async def h_status(self, _request):
        return web.json_response(self.status())

    async def h_ready(self, _request):
        return web.json_response(self.status(), status=200 if self.state == "ready" else 503)

    async def h_metrics(self, _request):
        ready = int(self.state == "ready")
        last = self.cold_starts[-1]["seconds"] if self.cold_starts else 0
        out = ["# TYPE vpa_agent_ready gauge", f"vpa_agent_ready {ready}",
               "# TYPE vpa_agent_starting gauge", f"vpa_agent_starting {int(self.state == 'starting')}",
               "# TYPE vpa_agent_plugin gauge", f"vpa_agent_plugin {int(self.mode == 'plugin')}",
               "# TYPE vpa_agent_last_cold_start_seconds gauge",
               f"vpa_agent_last_cold_start_seconds {last}"]
        if ready:
            urls = [f"{self.vllm_url}/metrics"]
            if self.mode == "plugin":
                urls += [f"{self.vllm_url}/v1/precision/metrics",
                         f"http://127.0.0.1:{self.args.controller_metrics_port}/metrics"]
            for url in urls:
                try:
                    async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as r:
                        if r.status == 200:
                            out.append(await r.text())
                except Exception:  # noqa: BLE001
                    pass
        return web.Response(text="\n".join(out) + "\n", content_type="text/plain")

    async def h_proxy(self, request: web.Request) -> web.StreamResponse:
        if self.state != "ready":
            raise web.HTTPServiceUnavailable(text=f"vLLM is {self.state}")
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        data = await request.read()
        try:
            async with self.session.request(request.method, f"{self.vllm_url}{request.path_qs}",
                                            headers=headers, data=data) as up:
                resp = web.StreamResponse(status=up.status, headers={
                    k: v for k, v in up.headers.items() if k.lower() not in HOP})
                await resp.prepare(request)
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        except aiohttp.ClientError as e:
            raise web.HTTPBadGateway(text=f"vLLM: {e}") from e

    async def on_startup(self, _app):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_read=600),
            connector=aiohttp.TCPConnector(limit=0))
        if self.args.start_mode:
            await self.start(self.args.start_mode)

    async def on_cleanup(self, _app):
        await self.stop()
        await self.session.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--vllm-port", type=int, default=8001)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--versions", default="high,fp8,int4a8")
    ap.add_argument("--controller-config", default=str(ROOT / "poc/k8s-keda/controller.yaml"))
    ap.add_argument("--controller-metrics-port", type=int, default=9101)
    ap.add_argument("--hf-home", default="/workspace/hf")
    ap.add_argument("--cache-root", default="/workspace/vllm-cache")
    ap.add_argument("--log-dir", default="/workspace/agent-logs")
    ap.add_argument("--start-mode", choices=["stock", "plugin"], help="start vLLM right away")
    ap.add_argument("--fake", action="store_true", help="fake_vllm.py instead of vLLM (no GPU)")
    ap.add_argument("--cold-start", type=float, default=40.0, help="fake: plugin cold start (s)")
    ap.add_argument("--stock-factor", type=float, default=0.75,
                    help="fake: stock cold start = this x plugin cold start")
    args = ap.parse_args()
    agent = Agent(args)
    app = web.Application(middlewares=[agent.auth], client_max_size=64 * 2**20)
    app.router.add_post("/agent/start", agent.h_start)
    app.router.add_post("/agent/stop", agent.h_stop)
    app.router.add_get("/agent/status", agent.h_status)
    app.router.add_get("/agent/ready", agent.h_ready)
    app.router.add_get("/metrics", agent.h_metrics)
    app.router.add_route("*", "/{tail:.*}", agent.h_proxy)
    app.on_startup.append(agent.on_startup)
    app.on_cleanup.append(agent.on_cleanup)
    web.run_app(app, host="0.0.0.0", port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
