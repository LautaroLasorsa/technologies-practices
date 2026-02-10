"""Backend API service.

Provides item data, health/readiness endpoints for K8s probes,
and a CPU-burn endpoint for HPA autoscaling demonstrations.
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

# ---------------------------------------------------------------------------
# Simulated data store
# ---------------------------------------------------------------------------

ITEMS: list[dict[str, str | float]] = [
    {"id": "1", "name": "Keyboard", "price": 49.99},
    {"id": "2", "name": "Mouse", "price": 29.99},
    {"id": "3", "name": "Monitor", "price": 349.99},
    {"id": "4", "name": "Headset", "price": 79.99},
    {"id": "5", "name": "Webcam", "price": 59.99},
]

# ---------------------------------------------------------------------------
# Readiness gate — simulates slow startup (DB connection, cache warm-up, etc.)
# ---------------------------------------------------------------------------

_ready = False
STARTUP_DELAY_SECONDS = int(os.getenv("STARTUP_DELAY", "3"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Simulate a slow startup, then mark the service as ready."""
    global _ready
    time.sleep(STARTUP_DELAY_SECONDS)
    _ready = True
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Backend API",
    description="Practice 006b — Backend microservice for K8s networking exercises",
    lifespan=lifespan,
)

VERSION = os.getenv("APP_VERSION", "v1")


@app.get("/health")
def health():
    """Liveness probe endpoint.

    Returns 200 if the process is alive. Does NOT check dependencies — that
    is intentional: liveness should only confirm the process hasn't deadlocked.
    """
    return {"status": "alive", "version": VERSION}


@app.get("/ready")
def ready(response: Response):
    """Readiness probe endpoint.

    Returns 200 only after startup is complete. K8s will not route traffic
    to this pod until this returns a success status.
    """
    if not _ready:
        response.status_code = 503
        return {"status": "not ready"}
    return {"status": "ready", "version": VERSION}


@app.get("/api/items")
def list_items():
    """Return all items in the catalogue."""
    return {"items": ITEMS, "source": f"backend-{VERSION}", "pod": os.getenv("HOSTNAME", "unknown")}


@app.get("/api/items/{item_id}")
def get_item(item_id: str, response: Response):
    """Return a single item by ID."""
    for item in ITEMS:
        if item["id"] == item_id:
            return {"item": item, "source": f"backend-{VERSION}", "pod": os.getenv("HOSTNAME", "unknown")}
    response.status_code = 404
    return {"error": f"Item '{item_id}' not found"}


@app.get("/api/cpu-burn")
def cpu_burn():
    """Burn CPU for ~200ms to drive up utilization for HPA testing.

    This endpoint intentionally wastes CPU so the metrics-server reports
    high utilization, triggering the HorizontalPodAutoscaler to scale up.
    """
    total = 0.0
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:
        total += sum(i * i for i in range(1000))
    return {
        "burned": True,
        "pod": os.getenv("HOSTNAME", "unknown"),
        "version": VERSION,
    }
