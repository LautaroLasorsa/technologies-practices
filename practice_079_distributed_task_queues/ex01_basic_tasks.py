# /// script
# requires-python = ">=3.12"
# dependencies = ["celery[redis]>=5.4,<6"]
# ///
"""
Exercise 1: Basic Celery Task Dispatch and Result Retrieval.

This script dispatches basic Celery tasks and demonstrates how to:
- Send tasks to workers with .delay() and .apply_async()
- Poll for results with AsyncResult
- Inspect task state transitions (PENDING -> STARTED -> SUCCESS)

Prerequisites:
  1. Redis running: docker compose up -d redis
  2. Celery worker running:
     uv run celery -A src.celery_app worker --loglevel=info --pool=solo -Q default
"""

from __future__ import annotations

import time

from src.tasks_basic import add, fetch_url_content


def main() -> None:
    print("=" * 60)
    print("Exercise 1: Basic Celery Tasks")
    print("=" * 60)

    # --- Test 1: Simple add task via .delay() ---
    print("\n--- Test 1: add.delay(4, 5) ---")
    result = add.delay(4, 5)
    print(f"Task dispatched! ID: {result.id}")
    print(f"State immediately after dispatch: {result.state}")

    # Poll until ready
    while not result.ready():
        print(f"  Waiting... state={result.state}")
        time.sleep(0.5)

    print(f"Result: {result.result}")
    print(f"Final state: {result.state}")
    assert result.result == 9, f"Expected 9, got {result.result}"
    print("PASSED")

    # --- Test 2: add via .apply_async() with countdown ---
    print("\n--- Test 2: add.apply_async(args=(10, 20), countdown=2) ---")
    result2 = add.apply_async(args=(10, 20), countdown=2)
    print(f"Task dispatched with 2s countdown! ID: {result2.id}")
    print(f"State: {result2.state} (should be PENDING for ~2 seconds)")
    time.sleep(3)
    print(f"After 3s -- State: {result2.state}, Result: {result2.result}")

    # --- Test 3: Bound task with self.request.id ---
    print("\n--- Test 3: fetch_url_content ---")
    result3 = fetch_url_content.delay("https://example.com/api/data")
    print(f"Task dispatched! ID: {result3.id}")

    # Poll with timeout
    for _ in range(10):
        if result3.ready():
            break
        print(f"  Waiting... state={result3.state}")
        time.sleep(1)

    if result3.successful():
        print(f"Result: {result3.result}")
        print("PASSED")
    else:
        print(f"Task failed: {result3.result}")

    # --- Test 4: Multiple tasks in parallel ---
    print("\n--- Test 4: Dispatch 5 add tasks in parallel ---")
    results = [add.delay(i, i * 2) for i in range(5)]
    print(f"Dispatched {len(results)} tasks")

    time.sleep(3)  # wait for all to complete
    for i, r in enumerate(results):
        print(f"  Task {i}: state={r.state}, result={r.result}")

    print("\n" + "=" * 60)
    print("Exercise 1 complete! Check Flower at http://localhost:5555")
    print("=" * 60)


if __name__ == "__main__":
    main()
