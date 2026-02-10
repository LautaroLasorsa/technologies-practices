"""Frontend gateway service.

Proxies requests to the backend API via its Kubernetes Service name,
demonstrating internal service discovery with ClusterIP + CoreDNS.
"""

import os

import httpx
from fastapi import FastAPI, Response

# ---------------------------------------------------------------------------
# Configuration via environment — BACKEND_URL will be the K8s service name
# ---------------------------------------------------------------------------

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend-svc:8000")
VERSION = os.getenv("APP_VERSION", "v1")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Frontend Gateway",
    description="Practice 006b — Frontend that proxies to backend via K8s service discovery",
)

client = httpx.Client(base_url=BACKEND_URL, timeout=5.0)


@app.get("/health")
def health():
    """Liveness probe — confirms this process is alive."""
    return {"status": "alive", "service": "frontend", "version": VERSION}


@app.get("/ready")
def ready(response: Response):
    """Readiness probe — checks that the backend is reachable."""
    try:
        r = client.get("/health")
        r.raise_for_status()
        return {"status": "ready", "backend": "reachable", "version": VERSION}
    except httpx.HTTPError:
        response.status_code = 503
        return {"status": "not ready", "backend": "unreachable"}


@app.get("/")
def root():
    """Landing page with service info."""
    return {
        "service": "frontend",
        "version": VERSION,
        "pod": os.getenv("HOSTNAME", "unknown"),
        "backend_url": BACKEND_URL,
    }


@app.get("/proxy/items")
def proxy_items(response: Response):
    """Fetch items from the backend API via internal service discovery."""
    try:
        r = client.get("/api/items")
        r.raise_for_status()
        data = r.json()
        data["proxied_by"] = f"frontend-{VERSION}"
        data["frontend_pod"] = os.getenv("HOSTNAME", "unknown")
        return data
    except httpx.HTTPError as exc:
        response.status_code = 502
        return {"error": "Backend unavailable", "detail": str(exc)}


@app.get("/proxy/items/{item_id}")
def proxy_item(item_id: str, response: Response):
    """Fetch a single item from the backend API."""
    try:
        r = client.get(f"/api/items/{item_id}")
        if r.status_code == 404:
            response.status_code = 404
        return r.json()
    except httpx.HTTPError as exc:
        response.status_code = 502
        return {"error": "Backend unavailable", "detail": str(exc)}
