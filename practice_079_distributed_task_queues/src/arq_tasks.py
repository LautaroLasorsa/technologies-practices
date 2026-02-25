"""
ARQ async task definitions -- Phase 7.

ARQ is a lightweight async task queue built on asyncio + Redis. Unlike Celery,
ARQ is designed from the ground up for async Python: tasks are async functions,
workers use asyncio event loops, and job enqueuing is non-blocking.

Key differences from Celery:
- Tasks are plain async def functions (no decorator needed)
- First parameter is always `ctx` (a dict with job metadata + shared state)
- Workers are configured via a WorkerSettings class, not CLI flags
- No canvas primitives (chain/group/chord) -- compose manually with asyncio
- Built-in cron support (simpler than Celery Beat, but fewer features)

ARQ is ideal when your entire stack is async (FastAPI + httpx + asyncpg)
and your tasks are I/O-bound (API calls, DB queries, file downloads).
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone

from arq import cron
from arq.connections import RedisSettings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis connection settings (parsed from URL or defaults)
# ---------------------------------------------------------------------------
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Parse Redis URL into components for ARQ's RedisSettings
# ARQ doesn't accept a URL string directly -- it needs host/port/database
def _parse_redis_url(url: str) -> RedisSettings:
    """Parse redis://host:port/db into ARQ RedisSettings."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=int(parsed.path.lstrip("/") or "0"),
    )

redis_settings = _parse_redis_url(REDIS_URL)


# =============================================================================
# Exercise 11: ARQ add task (comparison with Celery add)
# =============================================================================

async def arq_add(ctx: dict, x: int, y: int) -> int:
    """
    Add two numbers using ARQ (async equivalent of Celery's add task).

    # -- Exercise Context -------------------------------------------------------
    # ARQ tasks are plain async functions. The first parameter is ALWAYS `ctx`,
    # a dict containing:
    #   - ctx["job_id"]: unique job identifier
    #   - ctx["job_try"]: current attempt number (1-based, increments on retry)
    #   - ctx["enqueue_time"]: when the job was enqueued
    #   - ctx["redis"]: the ArqRedis connection (for enqueuing sub-jobs)
    #   - Any custom state set in on_startup (e.g., DB connections, HTTP clients)
    #
    # Unlike Celery, there's no @task decorator. ARQ discovers tasks via the
    # `functions` list in WorkerSettings. This is simpler but less "magical".
    #
    # Comparison with Celery:
    #   Celery: @app.task def add(x, y): return x + y
    #   ARQ:    async def arq_add(ctx, x, y): return x + y
    #
    # Notice that ARQ tasks are async by default -- if your task does I/O
    # (HTTP requests, DB queries), you use `await` directly instead of
    # blocking the worker thread. This is ARQ's main advantage for I/O-bound
    # workloads.
    # ---------------------------------------------------------------------------

    TODO(human): Implement the ARQ add task.

    Steps:
    1. Log the job: include ctx["job_id"], x, y
    2. Simulate a small delay with await asyncio.sleep(0.5)
       (Note: asyncio.sleep, NOT time.sleep -- never block the event loop!)
    3. Return x + y

    IMPORTANT: In ARQ, NEVER use time.sleep() -- it blocks the entire
    asyncio event loop, freezing ALL concurrent tasks. Always use
    asyncio.sleep() for delays in async code.
    """
    raise NotImplementedError("TODO(human): Implement arq_add")


# =============================================================================
# Exercise 12: ARQ fetch task (comparison with Celery fetch_url_content)
# =============================================================================

async def arq_fetch_url(ctx: dict, url: str) -> dict:
    """
    Simulate fetching URL content using ARQ (async equivalent).

    # -- Exercise Context -------------------------------------------------------
    # This is the async version of Celery's fetch_url_content. In production,
    # you'd use httpx or aiohttp here instead of asyncio.sleep:
    #
    #   async with httpx.AsyncClient() as client:
    #       response = await client.get(url)
    #       return {"url": url, "status": response.status_code, ...}
    #
    # Because ARQ runs on asyncio, this HTTP request is non-blocking: while
    # waiting for the response, the worker can execute OTHER tasks. A single
    # ARQ worker can handle hundreds of concurrent I/O-bound tasks this way.
    #
    # In contrast, Celery's prefork pool dedicates an entire OS process to each
    # task. A Celery worker with concurrency=4 can run 4 tasks simultaneously.
    # An ARQ worker can run hundreds, as long as they're I/O-bound.
    #
    # The trade-off: ARQ can't parallelize CPU-bound work (everything runs in
    # one event loop thread). For CPU-heavy tasks, Celery's prefork pool wins.
    # ---------------------------------------------------------------------------

    TODO(human): Implement the ARQ fetch URL task.

    Steps:
    1. Log the start of the fetch (include ctx["job_id"] and url)
    2. Simulate async I/O delay: await asyncio.sleep(2)
    3. Create a result dict:
       - "job_id": ctx["job_id"]
       - "url": url
       - "content_length": len(url) * 100  (simulated)
       - "status": "fetched"
    4. Log completion
    5. Return the result dict
    """
    raise NotImplementedError("TODO(human): Implement arq_fetch_url")


# =============================================================================
# Exercise 13: ARQ cron task (comparison with Celery Beat)
# =============================================================================

async def arq_periodic_health_check(ctx: dict) -> dict:
    """
    A periodic health check using ARQ's built-in cron scheduling.

    # -- Exercise Context -------------------------------------------------------
    # ARQ has built-in cron support, unlike Celery which requires a separate
    # Beat process. ARQ cron jobs run inside the worker process itself.
    #
    # The cron schedule is configured in WorkerSettings:
    #   cron_jobs = [cron(arq_periodic_health_check, minute={0, 30})]
    #
    # ARQ cron uses a set of allowed values for each field:
    #   minute={0, 15, 30, 45}  -> runs at these minutes of each hour
    #   hour={6, 18}            -> runs at 6 AM and 6 PM
    #   second={0}              -> (ARQ-specific!) sub-minute precision
    #
    # Advantage over Celery Beat: no separate process to manage.
    # Disadvantage: no dynamic schedule changes at runtime (hardcoded in code).
    # ---------------------------------------------------------------------------

    TODO(human): Implement the ARQ periodic health check.

    Steps:
    1. Record the current UTC timestamp
    2. Log "Health check running" with the timestamp
    3. Simulate a health check: await asyncio.sleep(0.5)
    4. Return a dict: {"task": "health_check", "timestamp": timestamp,
       "status": "healthy"}
    """
    raise NotImplementedError("TODO(human): Implement arq_periodic_health_check")


# =============================================================================
# Startup / Shutdown hooks (resource initialization -- fully implemented)
# =============================================================================

async def on_startup(ctx: dict) -> None:
    """
    Called when the ARQ worker starts.

    Use this to initialize shared resources: DB connections, HTTP clients,
    configuration objects. These are stored in ctx and accessible in all tasks.
    """
    logger.info("ARQ worker starting up")
    ctx["start_time"] = datetime.now(timezone.utc).isoformat()


async def on_shutdown(ctx: dict) -> None:
    """
    Called when the ARQ worker shuts down.

    Use this to close connections and clean up resources.
    """
    logger.info("ARQ worker shutting down")


# =============================================================================
# ARQ WorkerSettings (the equivalent of Celery's app configuration)
# =============================================================================

class WorkerSettings:
    """
    ARQ worker configuration.

    This class is referenced by the CLI:
        arq src.arq_tasks.WorkerSettings

    It tells the worker which functions to register, how to connect to Redis,
    and how many concurrent jobs to allow.
    """
    # Task functions the worker can execute
    functions = [arq_add, arq_fetch_url]

    # Cron jobs (periodic tasks built into the worker)
    # Run health check every minute (for demo purposes)
    cron_jobs = [
        cron(arq_periodic_health_check, minute=set(range(60)), second={0}),
    ]

    # Redis connection
    redis_settings = redis_settings

    # Lifecycle hooks
    on_startup = on_startup
    on_shutdown = on_shutdown

    # Max concurrent jobs (ARQ uses asyncio tasks, not processes)
    max_jobs = 10

    # How long to keep job results in Redis (seconds)
    keep_result = 3600  # 1 hour

    # Job timeout (max execution time per job)
    job_timeout = 300  # 5 minutes
