"""FastAPI backend with optimistic concurrency control and simulated conflicts."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

TaskStatus = Literal["todo", "in_progress", "review", "done"]
TaskPriority = Literal["low", "medium", "high", "urgent"]


class Task(BaseModel):
    id: str
    title: str
    description: str
    status: TaskStatus
    priority: TaskPriority
    version: int
    assignee: str
    created_at: str
    updated_at: str


class TaskPatchRequest(BaseModel):
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    title: Optional[str] = None
    version: int = Field(
        ...,
        description="The client's cached version of the task for optimistic concurrency check.",
    )


class BatchUpdateRequest(BaseModel):
    task_ids: list[str]
    status: TaskStatus


class TaskListResponse(BaseModel):
    items: list[Task]
    next_cursor: Optional[str]
    total: int


INITIAL_TASKS: list[dict] = [
    {
        "id": "task-101",
        "title": "Migrate auth session cookies to SameSite=Strict",
        "description": "Prevent CSRF cross-origin leaks during tenant migration.",
        "status": "todo",
        "priority": "high",
        "version": 1,
        "assignee": "Alex Rivera",
        "created_at": "2026-11-01T08:00:00Z",
        "updated_at": "2026-11-01T08:00:00Z",
    },
    {
        "id": "task-102",
        "title": "Implement idempotency keys for billing webhook dispatcher",
        "description": "Guarantee at-most-once invoice settlement across retry bursts.",
        "status": "todo",
        "priority": "urgent",
        "version": 1,
        "assignee": "Sora Tanaka",
        "created_at": "2026-11-01T09:15:00Z",
        "updated_at": "2026-11-01T09:15:00Z",
    },
    {
        "id": "task-103",
        "title": "Optimize Redis connection pool in edge gateway",
        "description": "Eliminate TCP socket starvation under 5k concurrent WebSocket connections.",
        "status": "in_progress",
        "priority": "urgent",
        "version": 2,
        "assignee": "Elena Rostova",
        "created_at": "2026-11-01T10:30:00Z",
        "updated_at": "2026-11-02T14:10:00Z",
    },
    {
        "id": "task-104",
        "title": "Add OpenTelemetry trace context propagation to Kafka producer",
        "description": "Bridge W3C traceparent headers across asynchronous event streams.",
        "status": "in_progress",
        "priority": "medium",
        "version": 1,
        "assignee": "Marcus Vance",
        "created_at": "2026-11-01T11:00:00Z",
        "updated_at": "2026-11-01T11:00:00Z",
    },
    {
        "id": "task-105",
        "title": "Audit database indexes on order_events table",
        "description": "Composite index (tenant_id, created_at DESC) to speed up merchant dashboard.",
        "status": "review",
        "priority": "medium",
        "version": 3,
        "assignee": "Chloe Dupont",
        "created_at": "2026-11-01T13:45:00Z",
        "updated_at": "2026-11-03T09:20:00Z",
    },
    {
        "id": "task-106",
        "title": "Refactor TanStack Query cache invalidation boundaries",
        "description": "Prevent whole-board refetching when updating single card metadata.",
        "status": "review",
        "priority": "high",
        "version": 1,
        "assignee": "Alex Rivera",
        "created_at": "2026-11-02T08:30:00Z",
        "updated_at": "2026-11-02T08:30:00Z",
    },
    {
        "id": "task-107",
        "title": "Decommission legacy v1 REST user profiles endpoint",
        "description": "Route all mobile clients to v2 graph representation.",
        "status": "done",
        "priority": "low",
        "version": 4,
        "assignee": "Devin Cole",
        "created_at": "2026-10-28T16:00:00Z",
        "updated_at": "2026-11-01T17:00:00Z",
    },
    {
        "id": "task-108",
        "title": "Setup rate limiting tier for public catalog search",
        "description": "Sliding window rate limit of 60 req/min per IP with Redis script.",
        "status": "done",
        "priority": "medium",
        "version": 2,
        "assignee": "Marcus Vance",
        "created_at": "2026-10-30T10:00:00Z",
        "updated_at": "2026-11-02T11:30:00Z",
    },
    {
        "id": "task-109",
        "title": "Harden CORS preflight caching for mobile webview",
        "description": "Set Access-Control-Max-Age to 86400 to cut OPTIONS round-trips.",
        "status": "todo",
        "priority": "low",
        "version": 1,
        "assignee": "Sora Tanaka",
        "created_at": "2026-11-02T15:00:00Z",
        "updated_at": "2026-11-02T15:00:00Z",
    },
    {
        "id": "task-110",
        "title": "Add schema validation for Kafka dead-letter queue consumer",
        "description": "Reject malformed poison pills before triggering rebalance loop.",
        "status": "in_progress",
        "priority": "high",
        "version": 1,
        "assignee": "Elena Rostova",
        "created_at": "2026-11-03T07:15:00Z",
        "updated_at": "2026-11-03T07:15:00Z",
    },
]

# In-memory store
TASKS_DB: dict[str, dict] = {t["id"]: copy.deepcopy(t) for t in INITIAL_TASKS}

app = FastAPI(
    title="Operations Task Board API",
    description="State Taxonomy practice backend demonstrating server state cache and optimistic concurrency.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "task_count": len(TASKS_DB)}


@app.post("/api/reset")
def reset_tasks() -> dict:
    """Reset the in-memory tasks database to original seed state."""
    global TASKS_DB
    TASKS_DB = {t["id"]: copy.deepcopy(t) for t in INITIAL_TASKS}
    return {"status": "reset", "task_count": len(TASKS_DB)}


@app.get("/api/tasks", response_model=TaskListResponse)
def get_tasks(
    status: Optional[str] = Query(None, description="Filter by task status"),
    priority: Optional[str] = Query(None, description="Filter by task priority"),
    search: Optional[str] = Query(None, description="Search term in title/description"),
    cursor: Optional[str] = Query(None, description="Cursor for pagination (task ID)"),
    limit: int = Query(10, ge=1, le=50, description="Page size"),
) -> TaskListResponse:
    tasks = list(TASKS_DB.values())

    # Filtering
    if status and status != "all":
        tasks = [t for t in tasks if t["status"] == status]
    if priority and priority != "all":
        tasks = [t for t in tasks if t["priority"] == priority]
    if search:
        term = search.lower().strip()
        tasks = [
            t
            for t in tasks
            if term in t["title"].lower() or term in t["description"].lower() or term in t["assignee"].lower()
        ]

    # Sort deterministically by id
    tasks.sort(key=lambda t: t["id"])
    total = len(tasks)

    # Cursor pagination
    start_idx = 0
    if cursor:
        for idx, t in enumerate(tasks):
            if t["id"] == cursor:
                start_idx = idx + 1
                break

    page_items = tasks[start_idx : start_idx + limit]
    next_cursor = page_items[-1]["id"] if (start_idx + limit) < total and page_items else None

    return TaskListResponse(
        items=[Task(**t) for t in page_items],
        next_cursor=next_cursor,
        total=total,
    )


@app.get("/api/tasks/{task_id}", response_model=Task)
def get_task(task_id: str) -> Task:
    if task_id not in TASKS_DB:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return Task(**TASKS_DB[task_id])


@app.patch("/api/tasks/{task_id}", response_model=Task)
def patch_task(
    task_id: str,
    payload: TaskPatchRequest,
    simulate_conflict: bool = Query(
        False,
        description="Force a 409 Conflict to test frontend optimistic rollback.",
    ),
) -> Task:
    """Updates a task with optimistic concurrency control.

    If simulate_conflict is True or payload.version != current.version,
    HTTP 409 Conflict is raised to emulate concurrent user edits.
    """
    if task_id not in TASKS_DB:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

    current = TASKS_DB[task_id]

    # Concurrency conflict check
    if simulate_conflict or payload.version != current["version"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Optimistic Concurrency Conflict: Task was modified by another session.",
                "task_id": task_id,
                "client_expected_version": payload.version,
                "server_current_version": current["version"],
                "server_task": current,
            },
        )

    # Apply valid mutation
    if payload.status is not None:
        current["status"] = payload.status
    if payload.priority is not None:
        current["priority"] = payload.priority
    if payload.title is not None:
        current["title"] = payload.title

    current["version"] += 1
    current["updated_at"] = datetime.now(timezone.utc).isoformat()

    return Task(**current)


@app.post("/api/tasks/batch")
def batch_update_tasks(payload: BatchUpdateRequest) -> dict:
    """Batch update statuses for selected task IDs."""
    updated = []
    for tid in payload.task_ids:
        if tid in TASKS_DB:
            TASKS_DB[tid]["status"] = payload.status
            TASKS_DB[tid]["version"] += 1
            TASKS_DB[tid]["updated_at"] = datetime.now(timezone.utc).isoformat()
            updated.append(TASKS_DB[tid])

    return {"updated_count": len(updated), "items": [Task(**t) for t in updated]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8112, reload=True)
