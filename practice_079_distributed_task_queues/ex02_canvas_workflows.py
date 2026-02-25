# /// script
# requires-python = ">=3.12"
# dependencies = ["celery[redis]>=5.4,<6"]
# ///
"""
Exercise 2: Canvas Workflows -- Chain, Group, Chord.

This script demonstrates Celery's canvas primitives for composing tasks
into workflows:
- chain: sequential pipeline (a -> b -> c)
- group: parallel fan-out (a, b, c simultaneously)
- chord: parallel + callback (fan-out, then aggregate)

Prerequisites:
  1. Redis running: docker compose up -d redis
  2. Celery worker running:
     uv run celery -A src.celery_app worker --loglevel=info --pool=solo -Q default
"""

from __future__ import annotations

import time

from celery import chain, chord, group

from src.tasks_basic import add
from src.tasks_canvas import (
    add_offset,
    aggregate_results,
    double,
    multiply,
    process_item,
    square,
)


def main() -> None:
    print("=" * 60)
    print("Exercise 2: Canvas Workflows")
    print("=" * 60)

    # --- Test 1: Chain (sequential pipeline) ---
    print("\n--- Test 1: chain(add.s(2, 3), double.s(), square.s()) ---")
    print("Expected: add(2,3)=5 -> double(5)=10 -> square(10)=100")

    # chain passes each result to the next task as first argument
    pipeline = chain(add.s(2, 3), double.s(), square.s())
    result = pipeline()
    print(f"Chain dispatched! ID: {result.id}")

    while not result.ready():
        time.sleep(0.5)

    print(f"Chain result: {result.result}")
    assert result.result == 100, f"Expected 100, got {result.result}"
    print("PASSED")

    # --- Test 2: Chain with partial application ---
    print("\n--- Test 2: chain with add_offset.s(10) ---")
    print("Expected: add(1,2)=3 -> double(3)=6 -> add_offset(6, 10)=16")

    pipeline2 = chain(add.s(1, 2), double.s(), add_offset.s(10))
    result2 = pipeline2()

    while not result2.ready():
        time.sleep(0.5)

    print(f"Chain result: {result2.result}")
    assert result2.result == 16, f"Expected 16, got {result2.result}"
    print("PASSED")

    # --- Test 3: Group (parallel fan-out) ---
    print("\n--- Test 3: group(process_item for items 1..4) ---")
    print("All 4 tasks execute in parallel on available workers")

    g = group(process_item.s(i) for i in range(1, 5))
    group_result = g()
    print(f"Group dispatched! {len(group_result)} tasks")

    # Wait for all tasks in the group to complete
    while not group_result.ready():
        completed = sum(1 for r in group_result if r.ready())
        print(f"  Progress: {completed}/{len(group_result)} complete")
        time.sleep(1)

    print("Group results:")
    for r in group_result:
        print(f"  {r.result}")
    print("PASSED")

    # --- Test 4: Chord (parallel + callback) ---
    print("\n--- Test 4: chord(group(process_item 1..3), aggregate_results) ---")
    print("Fan-out 3 tasks, then aggregate when ALL complete")

    c = chord(
        group(process_item.s(i) for i in range(1, 4)),
        aggregate_results.s(),
    )
    chord_result = c()
    print(f"Chord dispatched! ID: {chord_result.id}")

    while not chord_result.ready():
        print(f"  Waiting for chord callback... state={chord_result.state}")
        time.sleep(1)

    print(f"Chord callback result: {chord_result.result}")
    print("PASSED")

    print("\n" + "=" * 60)
    print("Exercise 2 complete! Check Flower for workflow execution timeline")
    print("=" * 60)


if __name__ == "__main__":
    main()
