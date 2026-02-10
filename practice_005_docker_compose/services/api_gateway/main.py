"""API Gateway -- Accepts task submissions and enqueues them into Redis."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
TASK_QUEUE = os.getenv("TASK_QUEUE", "tasks:pending")

# ---------------------------------------------------------------------------
# Redis client (lazy connection)
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
# Models
# ---------------------------------------------------------------------------


class TaskSubmission(BaseModel):
    """Payload sent by the client to submit a new task."""

    name: str = Field(..., min_length=1, max_length=200, examples=["resize-image"])
    payload: dict = Field(default_factory=dict, examples=[{"width": 800}])
    priority: int = Field(default=0, ge=0, le=10)


class TaskResponse(BaseModel):
    """Returned to the client after a task is accepted."""

    task_id: str
    status: str
    submitted_at: str


class HealthResponse(BaseModel):
    service: str
    status: str
    redis_connected: bool


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Task Pipeline -- API Gateway",
    root_path="/api",
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Liveness/readiness probe used by Docker health checks."""
    try:
        get_redis().ping()
        redis_ok = True
    except redis.ConnectionError:
        redis_ok = False

    return HealthResponse(
        service="api_gateway",
        status="healthy" if redis_ok else "degraded",
        redis_connected=redis_ok,
    )


@app.post("/tasks", response_model=TaskResponse, status_code=201)
def submit_task(submission: TaskSubmission) -> TaskResponse:
    """Accept a task and push it onto the Redis pending queue."""
    r = get_redis()
    if not _redis_available(r):
        raise HTTPException(status_code=503, detail="Redis unavailable")

    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    task = {
        "task_id": task_id,
        "name": submission.name,
        "payload": json.dumps(submission.payload),
        "priority": submission.priority,
        "status": "pending",
        "submitted_at": now,
    }

    r.hset(f"task:{task_id}", mapping=task)
    r.lpush(TASK_QUEUE, task_id)
    r.incr("stats:total_submitted")

    return TaskResponse(task_id=task_id, status="pending", submitted_at=now)


@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    """Retrieve current state of a task by ID."""
    r = get_redis()
    data = r.hgetall(f"task:{task_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Task not found")
    return data


@app.get("/stats")
def get_stats() -> dict:
    """Return basic queue statistics."""
    r = get_redis()
    return {
        "queue_depth": r.llen(TASK_QUEUE),
        "total_submitted": int(r.get("stats:total_submitted") or 0),
        "total_completed": int(r.get("stats:total_completed") or 0),
        "total_failed": int(r.get("stats:total_failed") or 0),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redis_available(r: redis.Redis) -> bool:
    try:
        return r.ping()
    except redis.ConnectionError:
        return False
