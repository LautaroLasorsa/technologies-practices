# /// script
# requires-python = ">=3.12"
# dependencies = ["celery[redis]>=5.4,<6"]
# ///
"""
Exercise 4: Task Routing to Dedicated Queues.

This script demonstrates routing tasks to different queues:
- high_priority_task -> "high_priority" queue
- low_priority_task -> "low_priority" queue
- add (default) -> "default" queue

Prerequisites:
  1. Redis running: docker compose up -d redis
  2. Celery worker consuming ALL queues:
     uv run celery -A src.celery_app worker --loglevel=info --pool=solo
       -Q default,high_priority,low_priority
"""

from __future__ import annotations

import time

from src.tasks_basic import add
from src.tasks_routing import high_priority_task, low_priority_task


def main() -> None:
    print("=" * 60)
    print("Exercise 4: Task Routing")
    print("=" * 60)

    # --- Test 1: Default queue task ---
    print("\n--- Test 1: add task -> default queue ---")
    result1 = add.delay(10, 20)
    print(f"Dispatched to 'default' queue. ID: {result1.id}")

    # --- Test 2: High priority task ---
    print("\n--- Test 2: high_priority_task -> high_priority queue ---")
    result2 = high_priority_task.delay("otp_code", "user@example.com")
    print(f"Dispatched to 'high_priority' queue. ID: {result2.id}")

    # --- Test 3: Low priority task ---
    print("\n--- Test 3: low_priority_task -> low_priority queue ---")
    result3 = low_priority_task.delay("monthly_summary", "2024-01")
    print(f"Dispatched to 'low_priority' queue. ID: {result3.id}")

    # --- Test 4: Override routing at call time ---
    print("\n--- Test 4: low_priority_task routed to 'default' (override) ---")
    result4 = low_priority_task.apply_async(
        args=("ad_hoc_report", "2024-02"),
        queue="default",  # override the configured route
    )
    print(f"Dispatched to 'default' queue (override). ID: {result4.id}")

    # Wait for all tasks
    print("\n--- Waiting for results ---")
    all_results = [
        ("add (default)", result1),
        ("high_priority", result2),
        ("low_priority", result3),
        ("low_priority (override)", result4),
    ]

    for _ in range(15):
        all_done = all(r.ready() for _, r in all_results)
        if all_done:
            break
        time.sleep(1)

    for name, r in all_results:
        state = r.state
        result = r.result if r.ready() else "still pending"
        print(f"  {name}: state={state}, result={result}")

    print("\n" + "=" * 60)
    print("Exercise 4 complete!")
    print("Check Redis queue lengths:")
    print("  docker compose exec redis redis-cli LLEN default")
    print("  docker compose exec redis redis-cli LLEN high_priority")
    print("  docker compose exec redis redis-cli LLEN low_priority")
    print("=" * 60)


if __name__ == "__main__":
    main()
