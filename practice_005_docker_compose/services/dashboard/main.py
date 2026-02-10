"""Dashboard -- Read-only FastAPI service showing pipeline status from Redis."""

from __future__ import annotations

import os

import redis
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
TASK_QUEUE = os.getenv("TASK_QUEUE", "tasks:pending")

# ---------------------------------------------------------------------------
# Redis client
# ---------------------------------------------------------------------------

_redis: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=5,
        )
    return _redis


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Task Pipeline -- Dashboard",
    root_path="/dashboard",
)


@app.get("/health")
def health_check() -> dict:
    """Liveness probe."""
    try:
        get_redis().ping()
        return {"service": "dashboard", "status": "healthy", "redis_connected": True}
    except redis.ConnectionError:
        return {"service": "dashboard", "status": "degraded", "redis_connected": False}


@app.get("/stats")
def get_stats() -> dict:
    """JSON stats endpoint consumed by the HTML dashboard."""
    r = get_redis()
    active_workers = r.smembers("workers:active")
    return {
        "queue_depth": r.llen(TASK_QUEUE),
        "total_submitted": int(r.get("stats:total_submitted") or 0),
        "total_completed": int(r.get("stats:total_completed") or 0),
        "total_failed": int(r.get("stats:total_failed") or 0),
        "active_workers": sorted(active_workers),
        "worker_count": len(active_workers),
    }


@app.get("/", response_class=HTMLResponse)
def dashboard_page() -> str:
    """Minimal HTML dashboard that auto-refreshes stats every 2 seconds."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Task Pipeline Dashboard</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; background: #0d1117; color: #c9d1d9; }
        h1 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }
        .stat { display: inline-block; text-align: center; margin: 0 2rem 1rem 0; }
        .stat .value { font-size: 2rem; font-weight: bold; color: #58a6ff; }
        .stat .label { font-size: 0.85rem; color: #8b949e; }
        .workers { color: #3fb950; }
        .failed { color: #f85149; }
        #error { color: #f85149; display: none; }
    </style>
</head>
<body>
    <h1>Task Pipeline Dashboard</h1>
    <p id="error">Could not reach the stats endpoint.</p>

    <div class="card">
        <div class="stat">
            <div class="value" id="queue_depth">-</div>
            <div class="label">Queue Depth</div>
        </div>
        <div class="stat">
            <div class="value" id="total_submitted">-</div>
            <div class="label">Submitted</div>
        </div>
        <div class="stat">
            <div class="value" id="total_completed">-</div>
            <div class="label">Completed</div>
        </div>
        <div class="stat failed">
            <div class="value" id="total_failed">-</div>
            <div class="label">Failed</div>
        </div>
    </div>

    <div class="card">
        <h3 class="workers">Active Workers</h3>
        <div id="workers">-</div>
    </div>

    <script>
        async function refresh() {
            try {
                const resp = await fetch("/dashboard/stats");
                const data = await resp.json();
                document.getElementById("queue_depth").textContent = data.queue_depth;
                document.getElementById("total_submitted").textContent = data.total_submitted;
                document.getElementById("total_completed").textContent = data.total_completed;
                document.getElementById("total_failed").textContent = data.total_failed;
                document.getElementById("workers").textContent = data.active_workers.length
                    ? data.active_workers.join(", ")
                    : "No active workers";
                document.getElementById("error").style.display = "none";
            } catch (e) {
                document.getElementById("error").style.display = "block";
            }
        }
        refresh();
        setInterval(refresh, 2000);
    </script>
</body>
</html>"""
