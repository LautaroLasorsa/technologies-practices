"""Demo App -- minimal FastAPI service for GitOps practice."""

import os
from datetime import datetime, timezone

from fastapi import FastAPI

app = FastAPI(title="GitOps Demo App")

APP_VERSION = os.environ.get("APP_VERSION", "1.0.0")
APP_ENV = os.environ.get("APP_ENV", "development")
HOSTNAME = os.environ.get("HOSTNAME", "unknown")


@app.get("/health")
def health():
    """Liveness/readiness probe endpoint."""
    return {
        "status": "healthy",
        "version": APP_VERSION,
        "environment": APP_ENV,
        "hostname": HOSTNAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/")
def root():
    """Root endpoint with app info."""
    return {
        "app": "gitops-demo",
        "version": APP_VERSION,
        "environment": APP_ENV,
        "message": f"Hello from {HOSTNAME}!",
    }


@app.get("/config")
def config():
    """Show injected configuration (for verifying ConfigMap mounts)."""
    return {
        "APP_VERSION": APP_VERSION,
        "APP_ENV": APP_ENV,
        "HOSTNAME": HOSTNAME,
    }
