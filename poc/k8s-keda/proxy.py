"""Runs IN Kubernetes as one vLLM replica (StatefulSet pod vllm-N).

Replica N owns GPU machine N (a warm RunPod pod running poc/k8s-keda/agent.py). When the pod
starts, it asks the agent to start vLLM; it reports Ready only once vLLM answers, so the
Service sends traffic to it exactly when a real vLLM pod would be ready. When Kubernetes
removes the pod (scale-down), it drains in-flight requests and stops vLLM on the machine.

Environment:
    BACKENDS     comma-separated agent URLs, index = StatefulSet ordinal
    MODE         stock | plugin (what the agent starts)
    POD_NAME     from the downward API (vllm-0, vllm-1, ...)
    AGENT_TOKEN  shared secret for the agents
"""

from __future__ import annotations

import asyncio
import os
import time

import aiohttp
from aiohttp import web

HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host",
       "content-encoding"}


class Replica:
    def __init__(self):
        self.pod = os.environ.get("POD_NAME", "vllm-0")
        self.ordinal = int(self.pod.rsplit("-", 1)[-1])
        backends = [b.strip().rstrip("/") for b in os.environ["BACKENDS"].split(",") if b.strip()]
        if self.ordinal >= len(backends):
            raise SystemExit(f"{self.pod}: no backend for ordinal {self.ordinal} ({len(backends)} configured)")
        self.backend = backends[self.ordinal]
        self.mode = os.environ.get("MODE", "plugin")
        self.headers = {"X-Agent-Token": os.environ.get("AGENT_TOKEN", "")}
        self.ready = False
        self.backend_state = "unknown"
        self.inflight = 0
        self.requests = 0
        self.failures = 0
        self.closing = False
        self.created = time.time()
        self.backend_metrics: tuple[float, str] = (0.0, "")
        self.session: aiohttp.ClientSession | None = None

    async def keep_started(self):
        """Ask the agent to start vLLM, then follow its state (re-ask if it was stopped)."""
        while not self.closing:
            try:
                async with self.session.get(f"{self.backend}/agent/status", headers=self.headers,
                                            timeout=aiohttp.ClientTimeout(total=5)) as r:
                    st = await r.json()
                self.backend_state = st["state"]
                if st["state"] in ("idle", "failed") or st.get("mode") != self.mode and st["state"] != "stopping":
                    if st["state"] == "failed":
                        print(f"[{self.pod}] backend failed: {st.get('error')}; restarting", flush=True)
                    async with self.session.post(f"{self.backend}/agent/start", headers=self.headers,
                                                 json={"mode": self.mode},
                                                 timeout=aiohttp.ClientTimeout(total=60)) as r:
                        st = await r.json()
                        print(f"[{self.pod}] asked {self.backend} to start vLLM ({self.mode}): {st['state']}",
                              flush=True)
                was = self.ready
                self.ready = st["state"] == "ready" and st.get("mode") == self.mode
                if self.ready and not was:
                    print(f"[{self.pod}] backend ready, cold start {st.get('cold_start_s')} s "
                          f"(pod age {time.time() - self.created:.1f} s)", flush=True)
            except Exception as e:  # noqa: BLE001
                self.ready = False
                print(f"[{self.pod}] backend unreachable: {e!r}", flush=True)
            await asyncio.sleep(1.0)

    async def h_ready(self, _request):
        return web.Response(status=200 if self.ready and not self.closing else 503,
                            text=self.backend_state)

    async def h_healthz(self, _request):
        return web.Response(text="ok")

    async def poll_metrics(self):
        """Fetch the backend's metrics every second, so a Prometheus scrape (2 s timeout) never
        waits on the internet round trip to the GPU machine."""
        while True:
            try:
                async with self.session.get(f"{self.backend}/metrics", headers=self.headers,
                                            timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        self.backend_metrics = (time.time(), await r.text())
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(1.0)

    async def h_metrics(self, _request):
        out = ["# TYPE vpa_proxy_ready gauge", f"vpa_proxy_ready {int(self.ready)}",
               "# TYPE vpa_proxy_inflight gauge", f"vpa_proxy_inflight {self.inflight}",
               "# TYPE vpa_proxy_requests_total counter", f"vpa_proxy_requests_total {self.requests}",
               "# TYPE vpa_proxy_failures_total counter", f"vpa_proxy_failures_total {self.failures}"]
        at, text = self.backend_metrics
        age = time.time() - at
        out += ["# TYPE vpa_proxy_backend_metrics_age_seconds gauge",
                f"vpa_proxy_backend_metrics_age_seconds {age:.2f}"]
        if age < 5:
            out.append(text)
        return web.Response(text="\n".join(out) + "\n", content_type="text/plain")

    async def h_proxy(self, request: web.Request) -> web.StreamResponse:
        self.inflight += 1
        self.requests += 1
        try:
            headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
            headers.update(self.headers)
            data = await request.read()
            async with self.session.request(request.method, f"{self.backend}{request.path_qs}",
                                            headers=headers, data=data) as up:
                out_headers = {k: v for k, v in up.headers.items() if k.lower() not in HOP}
                out_headers["X-Replica"] = self.pod
                resp = web.StreamResponse(status=up.status, headers=out_headers)
                await resp.prepare(request)
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                if up.status >= 400:
                    self.failures += 1
                return resp
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            self.failures += 1
            raise web.HTTPBadGateway(text=f"{self.pod}: {e!r}") from e
        finally:
            self.inflight -= 1

    async def on_startup(self, app):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_read=600),
            connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=30))
        app["keeper"] = asyncio.get_running_loop().create_task(self.keep_started())
        app["metrics"] = asyncio.get_running_loop().create_task(self.poll_metrics())
        print(f"[{self.pod}] backend {self.backend}, mode {self.mode}", flush=True)

    async def on_shutdown(self, app):
        """SIGTERM (scale-down): stop being Ready, let in-flight requests finish, stop vLLM."""
        self.closing = True
        app["keeper"].cancel()  # must not start vLLM again after we stop it
        deadline = time.time() + 45
        while self.inflight and time.time() < deadline:
            await asyncio.sleep(0.25)
        try:
            async with self.session.post(f"{self.backend}/agent/stop", headers=self.headers,
                                         timeout=aiohttp.ClientTimeout(total=60)) as r:
                print(f"[{self.pod}] stopped vLLM on {self.backend}: {r.status}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{self.pod}] stop failed: {e!r}", flush=True)

    async def on_cleanup(self, app):
        app["metrics"].cancel()
        await self.session.close()


def main() -> None:
    rep = Replica()
    app = web.Application(client_max_size=64 * 2**20)
    app.router.add_get("/ready", rep.h_ready)
    app.router.add_get("/healthz", rep.h_healthz)
    app.router.add_get("/metrics", rep.h_metrics)
    app.router.add_route("*", "/{tail:.*}", rep.h_proxy)
    app.on_startup.append(rep.on_startup)
    app.on_shutdown.append(rep.on_shutdown)
    app.on_cleanup.append(rep.on_cleanup)
    web.run_app(app, host="0.0.0.0", port=8080, print=None, access_log=None, shutdown_timeout=50)


if __name__ == "__main__":
    main()
