"""Simulated unreliable HTTP service (provided, no TODOs).

Three configurable failure modes switchable at runtime:
  - healthy:  100% success, 10-50ms latency
  - degraded: 50% errors, 500-2000ms latency
  - down:     100% errors (503)

Run:
    uv run python -m src._00_unreliable_service
"""

import asyncio
import random
import time
from enum import Enum

from aiohttp import web


class ServiceMode(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class ServiceState:
    def __init__(self) -> None:
        self.mode: ServiceMode = ServiceMode.HEALTHY
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0
        self.start_time: float = time.time()


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def handle_api_data(request: web.Request) -> web.Response:
    state: ServiceState = request.app["state"]
    state.total_requests += 1

    if state.mode == ServiceMode.HEALTHY:
        await asyncio.sleep(random.uniform(0.01, 0.05))
        state.successful_requests += 1
        return web.json_response({"data": random.random(), "mode": state.mode.value})

    if state.mode == ServiceMode.DEGRADED:
        if random.random() < 0.5:
            await asyncio.sleep(random.uniform(0.5, 2.0))
            state.successful_requests += 1
            return web.json_response({"data": random.random(), "mode": state.mode.value})
        state.failed_requests += 1
        return web.json_response({"error": "degraded"}, status=503)

    state.failed_requests += 1
    return web.json_response({"error": "down"}, status=503)


async def handle_set_mode(request: web.Request) -> web.Response:
    state: ServiceState = request.app["state"]
    try:
        new_mode = ServiceMode(request.match_info["mode"])
    except ValueError:
        return web.json_response({"error": "unknown mode"}, status=400)
    old_mode = state.mode
    state.mode = new_mode
    print(f"  [service] {old_mode.value} -> {new_mode.value}")
    return web.json_response({"previous_mode": old_mode.value, "current_mode": new_mode.value})


async def handle_status(request: web.Request) -> web.Response:
    state: ServiceState = request.app["state"]
    return web.json_response({
        "mode": state.mode.value,
        "uptime_secs": round(time.time() - state.start_time, 1),
        "total_requests": state.total_requests,
        "successful_requests": state.successful_requests,
        "failed_requests": state.failed_requests,
    })


def create_app() -> web.Application:
    app = web.Application()
    app["state"] = ServiceState()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/api/data", handle_api_data)
    app.router.add_post("/admin/mode/{mode}", handle_set_mode)
    app.router.add_get("/admin/status", handle_status)
    return app


def main() -> None:
    print("Unreliable service on http://localhost:8089 (mode: healthy)")
    print("  POST /admin/mode/{healthy|degraded|down} to switch")
    web.run_app(create_app(), host="0.0.0.0", port=8089, print=lambda _: None)


if __name__ == "__main__":
    main()
