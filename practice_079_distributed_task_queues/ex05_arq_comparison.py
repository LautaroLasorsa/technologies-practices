# /// script
# requires-python = ">=3.12"
# dependencies = ["arq>=0.26,<1"]
# ///
"""
Exercise 5: ARQ Async Task Comparison.

This script dispatches ARQ tasks and compares the experience with Celery:
- Enqueuing is async (await pool.enqueue_job)
- Results are polled via Job objects
- No canvas primitives -- but asyncio.gather provides parallel execution

Prerequisites:
  1. Redis running: docker compose up -d redis
  2. ARQ worker running:
     uv run arq src.arq_tasks.WorkerSettings
"""

from __future__ import annotations

import asyncio

from arq.connections import create_pool
from arq.jobs import Job

from src.arq_tasks import redis_settings


async def main() -> None:
    print("=" * 60)
    print("Exercise 5: ARQ Async Tasks")
    print("=" * 60)

    # Create ARQ Redis connection pool
    pool = await create_pool(redis_settings)

    # --- Test 1: Basic ARQ add task ---
    print("\n--- Test 1: arq_add(4, 5) ---")
    job = await pool.enqueue_job("arq_add", 4, 5)
    print(f"Job dispatched! ID: {job.job_id}")

    # Poll for result
    for i in range(10):
        status = await job.status()
        print(f"  [{i}s] Status: {status}")
        if str(status) == "complete":
            break
        await asyncio.sleep(1)

    info = await job.info()
    if info and info.success:
        print(f"Result: {info.result}")
        assert info.result == 9, f"Expected 9, got {info.result}"
        print("PASSED")
    else:
        print(f"Failed or not complete: {info}")

    # --- Test 2: ARQ fetch URL task ---
    print("\n--- Test 2: arq_fetch_url ---")
    job2 = await pool.enqueue_job("arq_fetch_url", "https://example.com/api")
    print(f"Job dispatched! ID: {job2.job_id}")

    for i in range(10):
        status = await job2.status()
        if str(status) == "complete":
            break
        await asyncio.sleep(1)

    info2 = await job2.info()
    if info2 and info2.success:
        print(f"Result: {info2.result}")
        print("PASSED")

    # --- Test 3: Multiple concurrent ARQ tasks (asyncio.gather equivalent) ---
    print("\n--- Test 3: Dispatch 5 ARQ add tasks concurrently ---")
    jobs = []
    for i in range(5):
        job = await pool.enqueue_job("arq_add", i, i * 10)
        jobs.append(job)
        print(f"  Dispatched: arq_add({i}, {i * 10}) -> ID: {job.job_id}")

    # Wait for all to complete
    await asyncio.sleep(5)

    print("Results:")
    for j in jobs:
        info = await j.info()
        if info and info.success:
            print(f"  Job {j.job_id}: result={info.result}")
        else:
            status = await j.status()
            print(f"  Job {j.job_id}: status={status}")

    # --- Test 4: Dedup with custom job_id ---
    print("\n--- Test 4: ARQ built-in dedup with _job_id ---")
    custom_id = "unique-add-operation-001"

    job4a = await pool.enqueue_job("arq_add", 100, 200, _job_id=custom_id)
    print(f"First dispatch with _job_id='{custom_id}': {job4a}")

    job4b = await pool.enqueue_job("arq_add", 100, 200, _job_id=custom_id)
    print(f"Duplicate dispatch with same _job_id: {job4b}")
    print("(ARQ silently skips duplicates -- job4b may be None)")

    print("\n" + "=" * 60)
    print("Exercise 5 complete!")
    print("Compare: Celery uses .delay() (sync), ARQ uses await enqueue_job (async)")
    print("=" * 60)

    pool.close()
    await pool.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
