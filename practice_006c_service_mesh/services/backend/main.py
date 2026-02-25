"""Backend service -- processes requests received through its Envoy sidecar.

This service provides:
  - Data and items endpoints (business logic)
  - Configurable failure modes for testing mesh resilience (healthy/degraded/down)
  - Simulated latency via query parameters

The backend never knows about the mesh -- it just listens on port 8000.
Its Envoy sidecar (backend-envoy) receives requests on port 10000 and
forwards them to localhost:8000.
"""

import asyncio
import random
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Backend Service", version="1.0.0")

# ---------------------------------------------------------------------------
# State -- failure simulation
# ---------------------------------------------------------------------------

# Failure mode: "healthy" (default), "degraded" (50% errors, slow), "down" (100% errors)
current_mode = "healthy"

# Simple in-memory store
items: list[dict] = [
    {"id": str(uuid.uuid4())[:8], "name": "Widget A", "price": 19.99},
    {"id": str(uuid.uuid4())[:8], "name": "Widget B", "price": 29.99},
    {"id": str(uuid.uuid4())[:8], "name": "Gadget C", "price": 49.99},
]

request_count = 0
error_count = 0
start_time = time.time()


# ---------------------------------------------------------------------------
# Failure simulation middleware
# ---------------------------------------------------------------------------

async def maybe_fail():
    """Simulate failures based on current mode."""
    global request_count, error_count
    request_count += 1

    if current_mode == "down":
        error_count += 1
        raise HTTPException(status_code=503, detail="Service is down (simulated)")

    if current_mode == "degraded":
        # 50% chance of error, add latency on success
        if random.random() < 0.5:
            error_count += 1
            raise HTTPException(status_code=503, detail="Service degraded (simulated)")
        # Simulate slow response (0.5-2 seconds)
        await asyncio.sleep(random.uniform(0.5, 2.0))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check -- always responds (even in degraded/down mode).

    Envoy uses this to check if the backend process is running.
    The /health endpoint bypasses failure simulation on purpose:
    the process is alive, even if business endpoints are failing.
    """
    uptime = round(time.time() - start_time, 1)
    return {
        "status": "ok",
        "service": "backend",
        "mode": current_mode,
        "uptime_seconds": uptime,
        "total_requests": request_count,
        "total_errors": error_count,
    }


@app.get("/data")
async def get_data(delay: float = 0):
    """Return sample data. Subject to failure simulation.

    Query params:
      - delay: Additional delay in seconds (for testing timeouts)
    """
    await maybe_fail()

    if delay > 0:
        await asyncio.sleep(delay)

    return {
        "service": "backend",
        "timestamp": time.time(),
        "mode": current_mode,
        "data": {
            "message": "Hello from backend through the service mesh!",
            "items_count": len(items),
            "request_number": request_count,
        },
    }


@app.get("/items")
async def list_items():
    """List all items. Subject to failure simulation."""
    await maybe_fail()
    return items


@app.post("/items")
async def create_item(request: Request):
    """Create a new item. Subject to failure simulation."""
    await maybe_fail()
    body = await request.json()

    name = body.get("name")
    price = body.get("price")
    if not name or price is None:
        raise HTTPException(status_code=400, detail="name and price are required")

    item = {"id": str(uuid.uuid4())[:8], "name": name, "price": float(price)}
    items.append(item)
    return item


@app.post("/admin/mode/{mode}")
async def set_mode(mode: str):
    """Change the failure simulation mode.

    Modes:
      - healthy:  100% success, no added latency
      - degraded: 50% chance of 503, 0.5-2s latency on success
      - down:     100% 503 errors
    """
    global current_mode
    if mode not in ("healthy", "degraded", "down"):
        raise HTTPException(status_code=400, detail=f"Unknown mode '{mode}'. Use: healthy, degraded, down")
    current_mode = mode
    return {"mode": current_mode, "message": f"Backend mode set to '{mode}'"}
