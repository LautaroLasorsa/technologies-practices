"""
Retry and idempotency tasks -- Phase 4.

This module covers two critical production patterns:

1. Automatic retry with exponential backoff -- for transient failures (network
   timeouts, rate limits, temporary service outages).

2. Idempotent task execution with dedup keys -- ensuring that re-delivery or
   retry doesn't cause duplicate side effects.

These patterns are not optional in production. Without retries, a single
network blip drops your task permanently. Without idempotency, retries
cause duplicate charges, emails, or database records.
"""

from __future__ import annotations

import logging
import os
import random
import time

import redis

from src.celery_app import app

logger = logging.getLogger(__name__)

# Redis client for idempotency keys (separate from Celery's broker connection)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL)


# =============================================================================
# Custom exception for simulating transient failures
# =============================================================================

class TransientError(Exception):
    """Simulates a transient failure (network timeout, rate limit, etc.)."""
    pass


class PermanentError(Exception):
    """Simulates a permanent failure (invalid input, auth error, etc.)."""
    pass


# =============================================================================
# Exercise 5: Task with automatic retry and exponential backoff
# =============================================================================

@app.task(
    name="src.tasks_retry.unreliable_api_call",
    bind=True,
    # =========================================================================
    # autoretry_for: list of exception types that trigger automatic retry.
    # Only TransientError retries -- PermanentError fails immediately.
    # This distinction is critical: retrying a 400 Bad Request is pointless,
    # but retrying a 503 Service Unavailable makes sense.
    # =========================================================================
    autoretry_for=(TransientError,),
    # =========================================================================
    # max_retries: maximum number of retry attempts before giving up.
    # After max_retries, the task enters FAILURE state and the exception
    # is stored in the result backend.
    # =========================================================================
    max_retries=4,
    # =========================================================================
    # retry_backoff: enables exponential backoff between retries.
    # True = use default base (1 second): 1s, 2s, 4s, 8s, ...
    # Integer = custom base: e.g., 3 means 3s, 6s, 12s, 24s, ...
    # =========================================================================
    retry_backoff=True,
    # =========================================================================
    # retry_backoff_max: cap the maximum delay between retries (seconds).
    # Without this, backoff grows unbounded: 1, 2, 4, 8, 16, 32, 64, ...
    # =========================================================================
    retry_backoff_max=30,
    # =========================================================================
    # retry_jitter: adds random jitter to backoff delay.
    # This prevents the "thundering herd" problem: if 100 tasks fail at the
    # same time, without jitter they ALL retry at the same instant, causing
    # another failure cascade. Jitter spreads retries over a time window.
    # =========================================================================
    retry_jitter=True,
)
def unreliable_api_call(self, endpoint: str, fail_probability: float = 0.7) -> dict:
    """
    Simulate an unreliable external API call with automatic retry.

    # -- Exercise Context -------------------------------------------------------
    # In production, external services fail: APIs return 503, databases timeout,
    # network connections drop. Celery's retry mechanism handles this gracefully:
    #
    # 1. Task raises TransientError (simulated API failure)
    # 2. Celery catches it (because of autoretry_for)
    # 3. Celery schedules a retry with exponential backoff + jitter
    # 4. After max_retries exhausted, task enters FAILURE state
    #
    # The retry delays follow: ~1s, ~2s, ~4s, ~8s (with jitter randomness).
    # You can see each retry attempt in Flower's task detail view.
    #
    # IMPORTANT: Because the task may execute multiple times (due to retries),
    # it should be idempotent -- or at minimum, its side effects should be
    # safe to repeat. For example, "set status to X" is idempotent, but
    # "increment counter by 1" is NOT.
    # ---------------------------------------------------------------------------

    TODO(human): Implement the unreliable API call simulation.

    Steps:
    1. Log the attempt: include endpoint, self.request.retries (current attempt),
       and self.max_retries
    2. Simulate unreliable behavior:
       - Generate a random number with random.random()
       - If random value < fail_probability, raise TransientError with a message
         like "API returned 503 for {endpoint}"
       - (Celery's autoretry_for will catch it and schedule a retry automatically)
    3. If we get past the failure check (success!):
       - Log success
       - Return a dict: {"endpoint": endpoint, "status": "success",
         "attempts": self.request.retries + 1}

    Hint: self.request.retries starts at 0 on the first attempt.
    The actual retry logic (backoff, jitter, scheduling) is handled by Celery
    automatically -- you just raise the exception and Celery does the rest.
    """
    raise NotImplementedError("TODO(human): Implement unreliable_api_call")


# =============================================================================
# Exercise 6: Idempotent task with Redis dedup key
# =============================================================================

@app.task(name="src.tasks_retry.idempotent_process_payment", bind=True)
def idempotent_process_payment(self, payment_id: str, amount: float) -> dict:
    """
    Process a payment idempotently using a Redis-based deduplication key.

    # -- Exercise Context -------------------------------------------------------
    # Idempotency means: executing the same operation multiple times produces
    # the same result and side effects as executing it once.
    #
    # Why is this critical for task queues?
    # 1. At-least-once delivery: if a worker crashes after completing a task
    #    but BEFORE acknowledging it (acks_late=True), the broker re-delivers
    #    the message to another worker. Without idempotency, the payment
    #    gets processed twice.
    # 2. Retries: if a task fails mid-execution and retries, partial side
    #    effects from the first attempt may already exist.
    # 3. Duplicate messages: network issues can cause the producer to publish
    #    the same task message twice.
    #
    # The dedup pattern:
    # 1. Before doing work, check Redis for key "dedup:{payment_id}"
    # 2. If key exists -> already processed, return cached result
    # 3. If key doesn't exist -> SET the key with NX (atomic), do the work
    # 4. Store the result in the dedup key with a TTL (auto-cleanup)
    #
    # Redis SET with NX (set-if-not-exists) is atomic, so even if two workers
    # receive the same message simultaneously, only one will "win" the SET.
    # The TTL ensures dedup keys don't accumulate forever.
    # ---------------------------------------------------------------------------

    TODO(human): Implement the idempotent payment processing task.

    Steps:
    1. Construct a dedup_key: f"dedup:payment:{payment_id}"
    2. Check if the dedup key already exists in Redis:
       - Use redis_client.get(dedup_key)
       - If it exists, log "Payment {payment_id} already processed (idempotent skip)"
       - Return the cached result (use json.loads on the stored value)
    3. If NOT already processed:
       - Log "Processing payment {payment_id} for ${amount}"
       - Simulate payment processing with time.sleep(1)
       - Create a result dict: {"payment_id": payment_id, "amount": amount,
         "status": "completed", "task_id": self.request.id}
       - Store the result in Redis with TTL:
         redis_client.setex(dedup_key, 300, json.dumps(result))
         (300 seconds = 5 minute dedup window)
       - Log completion
       - Return the result dict

    Hint: You'll need to import json at the top of this file.
    The redis_client is already initialized above.
    """
    raise NotImplementedError("TODO(human): Implement idempotent_process_payment")
