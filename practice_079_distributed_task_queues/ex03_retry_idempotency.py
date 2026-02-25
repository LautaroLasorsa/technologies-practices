# /// script
# requires-python = ">=3.12"
# dependencies = ["celery[redis]>=5.4,<6"]
# ///
"""
Exercise 3: Retry Policies and Idempotent Task Patterns.

This script demonstrates:
- Automatic retry with exponential backoff on transient failures
- Idempotent task execution using Redis-based dedup keys

Prerequisites:
  1. Redis running: docker compose up -d redis
  2. Celery worker running:
     uv run celery -A src.celery_app worker --loglevel=info --pool=solo -Q default
"""

from __future__ import annotations

import time

from src.tasks_retry import idempotent_process_payment, unreliable_api_call


def main() -> None:
    print("=" * 60)
    print("Exercise 3: Retry & Idempotency")
    print("=" * 60)

    # --- Test 1: Automatic retry with exponential backoff ---
    print("\n--- Test 1: unreliable_api_call (high failure probability) ---")
    print("Task retries up to 4 times with exponential backoff + jitter")
    print("Watch the worker logs for retry delays: ~1s, ~2s, ~4s, ~8s")

    result = unreliable_api_call.delay("/api/v1/external-service", 0.6)
    print(f"Task dispatched! ID: {result.id}")

    # Poll with longer timeout (retries take time)
    for i in range(30):
        state = result.state
        print(f"  [{i}s] State: {state}")
        if result.ready():
            break
        time.sleep(1)

    if result.successful():
        print(f"SUCCESS after retries: {result.result}")
    else:
        print(f"FAILED after max retries: {result.result}")
    print("(Either outcome is valid -- the task is probabilistic)")

    # --- Test 2: Low failure probability (should succeed quickly) ---
    print("\n--- Test 2: unreliable_api_call (low failure probability) ---")
    result2 = unreliable_api_call.delay("/api/v1/reliable-service", 0.1)
    print(f"Task dispatched! ID: {result2.id}")

    for _ in range(10):
        if result2.ready():
            break
        time.sleep(1)

    if result2.successful():
        print(f"Result: {result2.result}")
    print("PASSED")

    # --- Test 3: Idempotent payment processing ---
    print("\n--- Test 3: Idempotent payment (first call) ---")
    payment_id = "PAY-12345"

    result3 = idempotent_process_payment.delay(payment_id, 99.99)
    print(f"First dispatch! ID: {result3.id}")

    while not result3.ready():
        time.sleep(0.5)

    print(f"First call result: {result3.result}")

    # --- Test 4: Idempotent payment (duplicate call -- should skip) ---
    print("\n--- Test 4: Idempotent payment (duplicate call -- same payment_id) ---")
    result4 = idempotent_process_payment.delay(payment_id, 99.99)
    print(f"Duplicate dispatch! ID: {result4.id}")

    while not result4.ready():
        time.sleep(0.5)

    print(f"Duplicate call result: {result4.result}")
    print("(Should return cached result with original task_id, not process again)")

    # --- Test 5: Different payment_id (should process normally) ---
    print("\n--- Test 5: Idempotent payment (different payment_id) ---")
    result5 = idempotent_process_payment.delay("PAY-67890", 150.00)
    print(f"New payment dispatch! ID: {result5.id}")

    while not result5.ready():
        time.sleep(0.5)

    print(f"New payment result: {result5.result}")
    print("PASSED")

    print("\n" + "=" * 60)
    print("Exercise 3 complete! Check Flower for retry timeline")
    print("=" * 60)


if __name__ == "__main__":
    main()
