"""Frontend service -- receives client requests and calls backend through Envoy sidecar.

This service demonstrates the sidecar communication pattern:
  - Receives requests from its Envoy sidecar (inbound listener, port 10000)
  - Calls the backend by sending requests to its OWN sidecar (outbound listener, port 10001)
  - NEVER calls the backend directly -- all traffic flows through the mesh

The BACKEND_URL points to the frontend's own Envoy sidecar outbound port,
NOT to the backend service. Envoy handles routing to the correct upstream.
"""

import os
import time

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Frontend Service", version="1.0.0")

# The frontend calls its OWN Envoy sidecar's outbound listener.
# Envoy routes this to the backend cluster based on the route config.
# In Docker Compose with network_mode sharing, "localhost" = the sidecar.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:10001")

# Shared HTTP client (reuses connections)
client = httpx.AsyncClient(base_url=BACKEND_URL, timeout=10.0)


@app.get("/health")
async def health():
    """Health check endpoint -- called by Docker/Envoy health checks."""
    return {"status": "ok", "service": "frontend"}


@app.get("/")
async def root():
    """Root endpoint with service mesh info."""
    return {
        "service": "frontend",
        "mesh_info": {
            "sidecar_outbound": BACKEND_URL,
            "note": "All backend calls go through the Envoy sidecar outbound listener",
        },
    }


@app.get("/api/data")
async def get_data(delay: float = 0):
    """Fetch data from backend through the service mesh.

    The request path:
      1. Client -> frontend-envoy:10000 (inbound)
      2. frontend-envoy -> frontend-app:8000 (this handler)
      3. frontend-app -> frontend-envoy:10001 (outbound, via BACKEND_URL)
      4. frontend-envoy -> backend-envoy:10000 (mesh routing)
      5. backend-envoy -> backend-app:8000
      6. Response travels the reverse path
    """
    start = time.monotonic()
    try:
        params = {"delay": delay} if delay > 0 else {}
        response = await client.get("/data", params=params)
        elapsed_ms = (time.monotonic() - start) * 1000

        if response.status_code == 200:
            backend_data = response.json()
            return {
                "source": "frontend",
                "backend_response": backend_data,
                "mesh_latency_ms": round(elapsed_ms, 2),
            }
        else:
            return JSONResponse(
                status_code=response.status_code,
                content={
                    "source": "frontend",
                    "error": f"Backend returned {response.status_code}",
                    "backend_body": response.text,
                    "mesh_latency_ms": round(elapsed_ms, 2),
                },
            )
    except httpx.RequestError as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        raise HTTPException(
            status_code=502,
            detail={
                "error": f"Failed to reach backend through mesh: {exc}",
                "mesh_latency_ms": round(elapsed_ms, 2),
            },
        )


@app.get("/api/items")
async def list_items():
    """List items from backend through the mesh."""
    try:
        response = await client.get("/items")
        response.raise_for_status()
        return {"source": "frontend", "items": response.json()}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Mesh error: {exc}")


@app.post("/api/items")
async def create_item(request: Request):
    """Create an item via backend through the mesh."""
    body = await request.json()
    try:
        response = await client.post("/items", json=body)
        response.raise_for_status()
        return {"source": "frontend", "created": response.json()}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Mesh error: {exc}")


@app.post("/api/admin/mode/{mode}")
async def set_backend_mode(mode: str):
    """Proxy the admin mode change to the backend through the mesh.

    Valid modes: healthy, degraded, down
    """
    try:
        response = await client.post(f"/admin/mode/{mode}")
        response.raise_for_status()
        return {"source": "frontend", "backend_response": response.json()}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Mesh error: {exc}")


@app.on_event("shutdown")
async def shutdown():
    await client.aclose()
