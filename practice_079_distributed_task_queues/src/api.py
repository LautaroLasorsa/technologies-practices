"""
FastAPI HTTP API for dispatching tasks -- Phase 8.

This module provides HTTP endpoints that dispatch Celery and ARQ tasks.
It demonstrates the most common integration pattern: user sends HTTP request,
server dispatches a background task, returns the task ID immediately, and
the client polls for the result.

This is how production APIs handle long-running operations:
  1. POST /tasks -> 202 Accepted + task_id
  2. GET /tasks/{id} -> 200 + result (or 202 if still pending)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from celery.result import AsyncResult

from arq.connections import create_pool
from arq.jobs import Job

from src.celery_app import app as celery_app
from src.tasks_basic import add as celery_add
from src.arq_tasks import redis_settings as arq_redis_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ARQ Redis pool (initialized during startup)
# ---------------------------------------------------------------------------
arq_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize ARQ Redis pool on startup, close on shutdown."""
    global arq_pool
    arq_pool = await create_pool(arq_redis_settings)
    logger.info("FastAPI started, ARQ pool connected")
    yield
    arq_pool.close()
    await arq_pool.wait_closed()


app = FastAPI(
    title="Task Queue API",
    description="HTTP API for dispatching Celery and ARQ tasks",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AddTaskRequest(BaseModel):
    x: int
    y: int


class TaskCreatedResponse(BaseModel):
    task_id: str
    status: str
    queue: str  # "celery" or "arq"


class TaskResultResponse(BaseModel):
    task_id: str
    status: str
    result: int | dict | None = None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "celery_broker": celery_app.conf.broker_url}


# =============================================================================
# Exercise 14: Celery task dispatch endpoint
# =============================================================================

@app.post("/tasks/celery", response_model=TaskCreatedResponse, status_code=202)
async def dispatch_celery_task(request: AddTaskRequest) -> TaskCreatedResponse:
    """
    Dispatch a Celery add task and return the task ID.

    # -- Exercise Context -------------------------------------------------------
    # This is the standard pattern for integrating Celery with a web framework.
    # The HTTP handler does NOT wait for the task to complete -- it dispatches
    # the task and returns immediately with a 202 Accepted status code.
    #
    # The client receives a task_id that it can use to poll for the result
    # via GET /tasks/celery/{task_id}.
    #
    # This is called "fire-and-forget" dispatch. The alternative is using
    # WebSockets or webhooks for push-based notifications, but polling is
    # simpler and sufficient for most use cases.
    #
    # Two ways to dispatch Celery tasks:
    #   task.delay(arg1, arg2)
    #     -> shorthand, returns AsyncResult
    #   task.apply_async(args=(arg1, arg2), queue="...", countdown=5)
    #     -> full control: queue, countdown, eta, expires, priority
    #
    # .delay() is convenient but .apply_async() is preferred in production
    # because it makes the options explicit.
    # ---------------------------------------------------------------------------

    TODO(human): Implement the Celery task dispatch endpoint.

    Steps:
    1. Dispatch the celery_add task using either:
       - result = celery_add.delay(request.x, request.y)
       - result = celery_add.apply_async(args=(request.x, request.y))
    2. Return a TaskCreatedResponse with:
       - task_id = result.id (the unique task ID from Celery)
       - status = "dispatched"
       - queue = "celery"

    Hint: celery_add is imported at the top of this file from src.tasks_basic.
    .delay() returns an AsyncResult whose .id attribute is the task ID.
    """
    raise NotImplementedError("TODO(human): Implement dispatch_celery_task")


# =============================================================================
# Exercise 15: Celery task result polling endpoint
# =============================================================================

@app.get("/tasks/celery/{task_id}", response_model=TaskResultResponse)
async def get_celery_task_result(task_id: str) -> TaskResultResponse:
    """
    Poll for a Celery task result by task ID.

    # -- Exercise Context -------------------------------------------------------
    # AsyncResult wraps a Celery task result stored in the result backend.
    # It provides:
    #   .state -> "PENDING", "STARTED", "SUCCESS", "FAILURE", "RETRY"
    #   .result -> the return value (if SUCCESS) or exception (if FAILURE)
    #   .ready() -> True if task is done (success or failure)
    #   .successful() -> True if task completed without error
    #   .get(timeout=5) -> BLOCKS until result is ready (avoid in async code!)
    #
    # IMPORTANT: Never call .get() in an async endpoint! It blocks the event
    # loop. Instead, check .state and .result without blocking.
    #
    # State transitions:
    #   PENDING -> STARTED -> SUCCESS (happy path)
    #   PENDING -> STARTED -> RETRY -> STARTED -> SUCCESS (with retries)
    #   PENDING -> STARTED -> FAILURE (task raised an exception)
    #
    # "PENDING" is the default state for unknown task IDs too -- Celery can't
    # distinguish "task doesn't exist" from "task hasn't started yet".
    # ---------------------------------------------------------------------------

    TODO(human): Implement the Celery result polling endpoint.

    Steps:
    1. Create an AsyncResult for the given task_id:
       result = AsyncResult(task_id, app=celery_app)
    2. Check result.state:
       - If "SUCCESS": return TaskResultResponse with status="completed"
         and result=result.result
       - If "FAILURE": return TaskResultResponse with status="failed"
         and result=str(result.result)  (the exception message)
       - Otherwise (PENDING, STARTED, RETRY): return TaskResultResponse
         with status=result.state.lower() and result=None
    """
    raise NotImplementedError("TODO(human): Implement get_celery_task_result")


# =============================================================================
# Exercise 16: ARQ task dispatch endpoint
# =============================================================================

@app.post("/tasks/arq", response_model=TaskCreatedResponse, status_code=202)
async def dispatch_arq_task(request: AddTaskRequest) -> TaskCreatedResponse:
    """
    Dispatch an ARQ add task and return the job ID.

    # -- Exercise Context -------------------------------------------------------
    # ARQ job enqueuing is async (unlike Celery's synchronous dispatch):
    #   job = await arq_pool.enqueue_job("arq_add", x, y)
    #
    # The first argument is the FUNCTION NAME (string), not the function itself.
    # Additional arguments are passed positionally.
    #
    # arq_pool is an ArqRedis connection (initialized in lifespan above).
    # It's the equivalent of Celery's app -- the entry point for dispatching.
    #
    # ARQ job IDs are auto-generated UUIDs (like Celery task IDs).
    # You can also specify a custom job ID for deduplication:
    #   await arq_pool.enqueue_job("fn", _job_id="custom-id-123")
    #
    # If a job with the same _job_id is already queued or running,
    # ARQ silently skips the duplicate. This is built-in deduplication!
    # (Celery requires manual Redis-based dedup, as shown in tasks_retry.py.)
    # ---------------------------------------------------------------------------

    TODO(human): Implement the ARQ task dispatch endpoint.

    Steps:
    1. Dispatch the ARQ add task:
       job = await arq_pool.enqueue_job("arq_add", request.x, request.y)
    2. Return a TaskCreatedResponse with:
       - task_id = job.job_id
       - status = "dispatched"
       - queue = "arq"
    """
    raise NotImplementedError("TODO(human): Implement dispatch_arq_task")


# =============================================================================
# Exercise 17: ARQ task result polling endpoint
# =============================================================================

@app.get("/tasks/arq/{job_id}", response_model=TaskResultResponse)
async def get_arq_task_result(job_id: str) -> TaskResultResponse:
    """
    Poll for an ARQ job result by job ID.

    # -- Exercise Context -------------------------------------------------------
    # ARQ stores job results in Redis with a configurable TTL (keep_result
    # in WorkerSettings, default 3600 seconds).
    #
    # To check a job's status:
    #   job = Job(job_id, arq_pool)
    #   status = await job.status()  -> JobStatus enum
    #   info = await job.info()      -> JobResult or None
    #
    # JobStatus values:
    #   not_found  -> job ID doesn't exist in Redis
    #   queued     -> job is waiting to be picked up
    #   in_progress -> worker is executing the job
    #   complete   -> job finished (success or failure)
    #   deferred   -> job is scheduled for later execution
    #
    # The info() method returns a JobResult with:
    #   .result   -> return value (if successful)
    #   .success  -> True/False
    #   .function -> function name
    #   .args / .kwargs -> original arguments
    # ---------------------------------------------------------------------------

    TODO(human): Implement the ARQ result polling endpoint.

    Steps:
    1. Create a Job reference: job = Job(job_id, arq_pool)
    2. Get the job status: status = await job.status()
    3. If status is "complete":
       - Get job info: info = await job.info()
       - If info and info.success:
         return TaskResultResponse(task_id=job_id, status="completed",
                                   result=info.result)
       - Else (failure):
         return TaskResultResponse(task_id=job_id, status="failed",
                                   result=str(info.result) if info else None)
    4. If status is "not_found":
       raise HTTPException(status_code=404, detail="Job not found")
    5. Otherwise (queued, in_progress, deferred):
       return TaskResultResponse(task_id=job_id, status=str(status), result=None)

    Hint: Import JobStatus from arq.jobs for comparison, or compare with strings.
    """
    raise NotImplementedError("TODO(human): Implement get_arq_task_result")
