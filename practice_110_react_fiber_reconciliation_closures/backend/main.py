import asyncio
import random
import time
from typing import List, Literal, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Fleet Telemetry & Task Dispatcher API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskItem(BaseModel):
    id: str
    title: str
    priority: Literal["low", "medium", "high", "critical"]
    status: Literal["queued", "in-progress", "completed"]
    assigned_worker: str
    battery_level: int


INITIAL_TASKS: List[TaskItem] = [
    TaskItem(
        id="tsk-001",
        title="Drone Alpha: Perimeter Surveillance",
        priority="high",
        status="queued",
        assigned_worker="unit-sky-01",
        battery_level=88,
    ),
    TaskItem(
        id="tsk-002",
        title="Autonomous Rover: Geological Core Sampling",
        priority="medium",
        status="in-progress",
        assigned_worker="unit-ground-04",
        battery_level=62,
    ),
    TaskItem(
        id="tsk-003",
        title="Vessel Gamma: Harbor Docking Calibration",
        priority="low",
        status="queued",
        assigned_worker="unit-sea-02",
        battery_level=95,
    ),
    TaskItem(
        id="tsk-004",
        title="Drone Beta: High-Voltage Grid Thermal Scan",
        priority="critical",
        status="queued",
        assigned_worker="unit-sky-02",
        battery_level=41,
    ),
    TaskItem(
        id="tsk-005",
        title="AGV Delta: Automated Battery Swap Sequence",
        priority="high",
        status="in-progress",
        assigned_worker="unit-depot-09",
        battery_level=19,
    ),
    TaskItem(
        id="tsk-006",
        title="Station Epsilon: Atmospheric Sensor Mesh Sync",
        priority="medium",
        status="completed",
        assigned_worker="unit-tower-01",
        battery_level=100,
    ),
]

tasks_db: List[TaskItem] = [task.model_copy() for task in INITIAL_TASKS]


@app.get("/api/tasks")
async def get_tasks() -> List[TaskItem]:
    """Returns the current ordered list of fleet dispatch tasks."""
    return tasks_db


class ReorderPayload(BaseModel):
    task_ids: List[str]


@app.post("/api/tasks/reorder")
async def reorder_tasks(payload: ReorderPayload):
    """Reorder in-memory tasks according to the supplied ID order."""
    global tasks_db
    id_map = {t.id: t for t in tasks_db}
    reordered = []
    for tid in payload.task_ids:
        if tid in id_map:
            reordered.append(id_map[tid])
    # Append any remaining not in payload
    for t in tasks_db:
        if t not in reordered:
            reordered.append(t)
    tasks_db = reordered
    return {"status": "reordered", "count": len(tasks_db)}


@app.get("/api/tasks/search")
async def search_tasks(
    q: str = Query(..., description="Search query string"),
    delay_mode: str = Query("jitter", description="Simulated latency mode: 'jitter' or 'instant'"),
):
    """
    Search tasks with intentional inverted latency to reproduce race conditions.

    When delay_mode is 'jitter':
    - Short queries (<= 2 chars) are delayed by 1200ms.
    - Medium/long queries (> 2 chars) resolve rapidly in 80ms.
    This guarantees that typing 'd' then 'drone' will resolve 'drone' BEFORE 'd',
    causing out-of-order state overwrites unless AbortController is used.
    """
    start_time = time.time()
    query_clean = q.strip().lower()

    if delay_mode == "jitter":
        if len(query_clean) <= 2:
            await asyncio.sleep(1.2)  # Intentional slow response for early keystrokes
        else:
            await asyncio.sleep(0.08)  # Fast response for refined query
    else:
        await asyncio.sleep(0.01)

    matched = [
        t
        for t in tasks_db
        if (
            query_clean in t.title.lower()
            or query_clean in t.assigned_worker.lower()
            or query_clean in t.priority.lower()
        )
    ]

    elapsed_ms = int((time.time() - start_time) * 1000)
    return {
        "query": q,
        "results": matched,
        "total": len(matched),
        "latency_ms": elapsed_ms,
        "server_timestamp": time.time(),
    }


@app.get("/api/telemetry/tick")
async def get_telemetry_tick():
    """Emits high-frequency telemetry metric snapshot representing active fleet load."""
    return {
        "timestamp": time.time(),
        "cpu_usage": round(random.uniform(42.0, 88.5), 1),
        "memory_usage": round(random.uniform(55.0, 82.0), 1),
        "active_workers": random.randint(12, 18),
        "throughput_req_sec": random.randint(120, 480),
    }


@app.post("/api/reset")
async def reset_tasks():
    """Resets the tasks back to the default state."""
    global tasks_db
    tasks_db = [task.model_copy() for task in INITIAL_TASKS]
    return {"status": "reset", "count": len(tasks_db)}
