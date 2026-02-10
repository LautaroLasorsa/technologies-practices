"""Worker -- Polls Redis for pending tasks, processes them, and stores results."""

from __future__ import annotations

import json
import os
import random
import signal
import socket
import sys
import time
from datetime import datetime, timezone

import redis

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
TASK_QUEUE = os.getenv("TASK_QUEUE", "tasks:pending")
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "1.0"))
WORKER_ID = os.getenv("HOSTNAME", socket.gethostname())

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_running = True


def _shutdown_handler(signum: int, _frame: object) -> None:
    global _running
    print(f"[{WORKER_ID}] Received signal {signum}, shutting down gracefully...")
    _running = False


signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)

# ---------------------------------------------------------------------------
# Redis client
# ---------------------------------------------------------------------------


def connect_redis() -> redis.Redis:
    """Connect to Redis with retry logic."""
    for attempt in range(1, 11):
        try:
            r = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            r.ping()
            print(f"[{WORKER_ID}] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            return r
        except redis.ConnectionError:
            wait = min(attempt * 2, 10)
            print(f"[{WORKER_ID}] Redis not ready, retrying in {wait}s (attempt {attempt}/10)")
            time.sleep(wait)

    print(f"[{WORKER_ID}] Could not connect to Redis after 10 attempts, exiting.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Task processing
# ---------------------------------------------------------------------------


def process_task(task_data: dict) -> dict:
    """Simulate task processing with a random delay.

    In a real system this would call an external service, run a computation, etc.
    Returns a result dict with the processing outcome.
    """
    name = task_data.get("name", "unknown")
    payload = json.loads(task_data.get("payload", "{}"))

    processing_time = random.uniform(0.5, 3.0)
    print(f"[{WORKER_ID}] Processing task '{name}' (simulated {processing_time:.1f}s)...")
    time.sleep(processing_time)

    # 10% simulated failure rate for realism
    if random.random() < 0.1:
        raise RuntimeError(f"Simulated failure for task '{name}'")

    return {
        "processed_by": WORKER_ID,
        "processing_time_s": round(processing_time, 2),
        "input_payload": payload,
        "output": f"Result for '{name}': OK ({len(payload)} payload keys processed)",
    }


def handle_task(r: redis.Redis, task_id: str) -> None:
    """Fetch task data, process it, and write the result back to Redis."""
    task_key = f"task:{task_id}"
    task_data = r.hgetall(task_key)

    if not task_data:
        print(f"[{WORKER_ID}] Task {task_id} not found in Redis, skipping.")
        return

    r.hset(task_key, "status", "processing")
    r.hset(task_key, "worker_id", WORKER_ID)
    r.hset(task_key, "started_at", datetime.now(timezone.utc).isoformat())

    try:
        result = process_task(task_data)
        r.hset(task_key, mapping={
            "status": "completed",
            "result": json.dumps(result),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        r.incr("stats:total_completed")
        print(f"[{WORKER_ID}] Task {task_id} completed successfully.")

    except Exception as exc:
        r.hset(task_key, mapping={
            "status": "failed",
            "error": str(exc),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        })
        r.incr("stats:total_failed")
        print(f"[{WORKER_ID}] Task {task_id} failed: {exc}")


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"[{WORKER_ID}] Worker starting...")
    r = connect_redis()

    r.sadd("workers:active", WORKER_ID)
    print(f"[{WORKER_ID}] Registered as active worker. Polling queue '{TASK_QUEUE}'...")

    try:
        while _running:
            # BRPOP blocks for up to POLL_INTERVAL seconds, then returns None
            result = r.brpop(TASK_QUEUE, timeout=int(POLL_INTERVAL))
            if result is None:
                continue

            _queue_name, task_id = result
            handle_task(r, task_id)

    finally:
        r.srem("workers:active", WORKER_ID)
        print(f"[{WORKER_ID}] Unregistered from active workers. Goodbye.")


if __name__ == "__main__":
    main()
