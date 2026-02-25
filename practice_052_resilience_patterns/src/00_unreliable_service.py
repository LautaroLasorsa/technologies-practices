"""Simulated unreliable HTTP service for testing resilience patterns.

This service has three configurable failure modes:
  - healthy:  100% success, fast responses (10-50ms)
  - degraded: 50% errors, slow responses (500-2000ms)
  - down:     100% errors (503 Service Unavailable)

The failure mode can be switched at runtime via the /admin/mode/{mode}
endpoint, allowing resilience tests to simulate service degradation
and recovery without restarting the server.

Endpoints:
  GET  /health           - Always returns 200 (health check, not affected by mode)
  GET  /api/data         - Returns data or errors depending on current mode
  POST /admin/mode/{mode} - Switch failure mode (healthy|degraded|down)
  GET  /admin/status     - Show current mode and request statistics

Run:
    uv run python src/00_unreliable_service.py
"""

import asyncio
import random
import time
from enum import Enum

from aiohttp import web


# -- Failure modes --------------------------------------------------------


class ServiceMode(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


# -- Service state --------------------------------------------------------


class ServiceState:
    """Mutable state shared across all request handlers."""

    def __init__(self) -> None:
        self.mode: ServiceMode = ServiceMode.HEALTHY
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0
        self.start_time: float = time.time()

    def record_success(self) -> None:
        self.total_requests += 1
        self.successful_requests += 1

    def record_failure(self) -> None:
        self.total_requests += 1
        self.failed_requests += 1


# -- Request handlers -----------------------------------------------------


async def handle_health(request: web.Request) -> web.Response:
    """Health check endpoint -- always returns 200 regardless of mode."""
    return web.json_response({"status": "ok"})


async def handle_api_data(request: web.Request) -> web.Response:
    """Data endpoint -- behavior depends on current failure mode.

    - healthy:  immediate 200 with payload
    - degraded: 50% chance of 503; successful responses delayed 500-2000ms
    - down:     always 503
    """
    state: ServiceState = request.app["state"]

    if state.mode == ServiceMode.HEALTHY:
        # Fast, reliable response
        delay = random.uniform(0.01, 0.05)
        await asyncio.sleep(delay)
        state.record_success()
        return web.json_response({
            "data": {"id": random.randint(1, 1000), "value": random.random()},
            "latency_ms": round(delay * 1000, 1),
            "mode": state.mode.value,
        })

    if state.mode == ServiceMode.DEGRADED:
        # 50% chance of failure; successful responses are slow
        if random.random() < 0.5:
            delay = random.uniform(0.5, 2.0)
            await asyncio.sleep(delay)
            state.record_success()
            return web.json_response({
                "data": {"id": random.randint(1, 1000), "value": random.random()},
                "latency_ms": round(delay * 1000, 1),
                "mode": state.mode.value,
            })
        else:
            state.record_failure()
            return web.json_response(
                {"error": "Service degraded", "mode": state.mode.value},
                status=503,
            )

    # DOWN mode -- always fail
    state.record_failure()
    return web.json_response(
        {"error": "Service unavailable", "mode": state.mode.value},
        status=503,
    )


async def handle_set_mode(request: web.Request) -> web.Response:
    """Switch the service failure mode at runtime."""
    mode_str = request.match_info["mode"]
    state: ServiceState = request.app["state"]

    try:
        new_mode = ServiceMode(mode_str)
    except ValueError:
        return web.json_response(
            {"error": f"Unknown mode '{mode_str}'. Use: healthy, degraded, down"},
            status=400,
        )

    old_mode = state.mode
    state.mode = new_mode
    print(f"[MODE CHANGE] {old_mode.value} -> {new_mode.value}")
    return web.json_response({
        "previous_mode": old_mode.value,
        "current_mode": new_mode.value,
    })


async def handle_status(request: web.Request) -> web.Response:
    """Return current mode and request statistics."""
    state: ServiceState = request.app["state"]
    uptime = time.time() - state.start_time
    return web.json_response({
        "mode": state.mode.value,
        "uptime_secs": round(uptime, 1),
        "total_requests": state.total_requests,
        "successful_requests": state.successful_requests,
        "failed_requests": state.failed_requests,
    })


# -- Application factory --------------------------------------------------


def create_app() -> web.Application:
    """Create and configure the aiohttp application."""
    app = web.Application()
    app["state"] = ServiceState()

    app.router.add_get("/health", handle_health)
    app.router.add_get("/api/data", handle_api_data)
    app.router.add_post("/admin/mode/{mode}", handle_set_mode)
    app.router.add_get("/admin/status", handle_status)

    return app


# -- Entry point -----------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Unreliable Service -- Resilience Pattern Testing Target")
    print("=" * 60)
    print()
    print("Endpoints:")
    print("  GET  http://localhost:8089/health          - Health check")
    print("  GET  http://localhost:8089/api/data         - Data (unreliable)")
    print("  POST http://localhost:8089/admin/mode/{m}   - Set mode")
    print("  GET  http://localhost:8089/admin/status      - Statistics")
    print()
    print("Modes: healthy | degraded | down")
    print("Starting in HEALTHY mode...")
    print()

    app = create_app()
    web.run_app(app, host="0.0.0.0", port=8089, print=lambda _: None)


if __name__ == "__main__":
    main()
