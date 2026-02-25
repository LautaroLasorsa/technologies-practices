"""
Celery Canvas workflow tasks -- Phase 3.

Canvas is Celery's workflow composition system. It provides primitives for
building complex task pipelines from simple task building blocks.

Key primitives:
- Signature: task.s(args) creates an immutable task description (like a closure)
- Chain: chain(a.s(), b.s()) -- b receives a's result as first argument
- Group: group(a.s(1), a.s(2)) -- all execute in parallel
- Chord: chord(group(a.s(1), a.s(2)), callback.s()) -- parallel then callback

These primitives compose: you can put a chain inside a group, a group inside
a chord, etc. This enables DAG-like workflow execution without explicit
orchestration code.
"""

from __future__ import annotations

import logging
import time

from src.celery_app import app

logger = logging.getLogger(__name__)


# =============================================================================
# Building-block tasks (used by workflows)
# =============================================================================

@app.task(name="src.tasks_canvas.multiply")
def multiply(x: int, y: int) -> int:
    """Multiply two numbers. Used as a building block in canvas workflows."""
    logger.info(f"multiply({x}, {y})")
    time.sleep(0.5)  # simulate some work
    return x * y


@app.task(name="src.tasks_canvas.square")
def square(x: int) -> int:
    """Square a number. Demonstrates single-argument tasks in chains."""
    logger.info(f"square({x})")
    time.sleep(0.5)
    return x * x


# =============================================================================
# Exercise 3: Chain workflow building blocks
# =============================================================================

@app.task(name="src.tasks_canvas.double")
def double(x: int) -> int:
    """
    Double a number. Used as a step in chain workflows.

    # -- Exercise Context -------------------------------------------------------
    # In a chain, each task receives the RETURN VALUE of the previous task as
    # its first argument. So chain(add.s(2, 3), double.s(), square.s()) means:
    #   1. add(2, 3) -> 5
    #   2. double(5) -> 10  (5 is passed automatically from add's result)
    #   3. square(10) -> 100 (10 is passed automatically from double's result)
    #
    # This is why chain tasks must accept exactly one argument (the previous
    # result) -- except the first task in the chain, which can take any args.
    #
    # The .s() method creates an "immutable signature" that carries the task
    # name, args, and kwargs. It's like a closure that hasn't been called yet.
    # ---------------------------------------------------------------------------

    TODO(human): Implement the double task.

    Steps:
    1. Log a message showing the input value
    2. Simulate a small delay with time.sleep(0.5)
    3. Return x * 2
    """
    raise NotImplementedError("TODO(human): Implement the double task")


@app.task(name="src.tasks_canvas.add_offset")
def add_offset(x: int, offset: int) -> int:
    """
    Add an offset to a number. Demonstrates partial application in chains.

    # -- Exercise Context -------------------------------------------------------
    # Sometimes you need a chain step that takes MORE than just the previous
    # result. Celery handles this with "partial signatures":
    #   chain(get_value.s(), add_offset.s(10))
    #
    # Here, add_offset receives two arguments:
    #   - x = the result of get_value (passed automatically by the chain)
    #   - offset = 10 (provided when creating the signature)
    #
    # The .s(10) creates a partial: add_offset(?, 10) where ? is filled by
    # the previous task's result. This is analogous to functools.partial().
    #
    # For "immutable signatures" that IGNORE the previous result, use .si():
    #   chain(a.s(), b.si(42))  -- b always receives 42, not a's result
    # ---------------------------------------------------------------------------

    TODO(human): Implement the add_offset task.

    Steps:
    1. Log both x and offset
    2. Return x + offset
    """
    raise NotImplementedError("TODO(human): Implement the add_offset task")


# =============================================================================
# Exercise 4: Group and Chord building blocks
# =============================================================================

@app.task(name="src.tasks_canvas.process_item")
def process_item(item_id: int) -> dict:
    """
    Process a single item (fan-out step in group/chord).

    # -- Exercise Context -------------------------------------------------------
    # A group executes multiple tasks in PARALLEL. Each task in the group runs
    # independently on (potentially different) workers:
    #   group(process_item.s(1), process_item.s(2), process_item.s(3))()
    #
    # This dispatches 3 tasks simultaneously. The result is a GroupResult that
    # you can iterate to get each task's return value.
    #
    # A chord adds a CALLBACK that runs after ALL group tasks complete:
    #   chord(
    #       group(process_item.s(1), process_item.s(2)),
    #       aggregate_results.s()
    #   )()
    #
    # The callback receives a list of all group results as its first argument.
    # This is the classic fan-out / fan-in pattern: distribute work across
    # workers, then aggregate the results.
    # ---------------------------------------------------------------------------

    TODO(human): Implement the process_item task.

    Steps:
    1. Log which item_id is being processed
    2. Simulate variable processing time: time.sleep(1 + (item_id % 3))
       (this creates visible parallelism in Flower -- items finish at different times)
    3. Return a dict: {"item_id": item_id, "result": item_id * 10, "status": "processed"}
    """
    raise NotImplementedError("TODO(human): Implement the process_item task")


@app.task(name="src.tasks_canvas.aggregate_results")
def aggregate_results(results: list[dict]) -> dict:
    """
    Aggregate results from a group/chord (fan-in callback).

    # -- Exercise Context -------------------------------------------------------
    # This task is the CALLBACK of a chord. It receives a list of return values
    # from all tasks in the chord's header (group).
    #
    # For example, if the chord header had 3 process_item tasks:
    #   results = [
    #       {"item_id": 1, "result": 10, "status": "processed"},
    #       {"item_id": 2, "result": 20, "status": "processed"},
    #       {"item_id": 3, "result": 30, "status": "processed"},
    #   ]
    #
    # The chord guarantees that this callback runs ONLY after ALL header tasks
    # complete. If any header task fails, the chord enters an error state and
    # the callback does NOT run (by default).
    #
    # This pattern is common for: batch processing (process N items, then
    # generate summary), map-reduce (distribute computation, then combine),
    # parallel API calls (fetch N URLs, then merge responses).
    # ---------------------------------------------------------------------------

    TODO(human): Implement the aggregate_results task.

    Steps:
    1. Log how many results are being aggregated
    2. Compute total = sum of all result["result"] values
    3. Compute processed_count = len(results)
    4. Return a summary dict:
       {"total": total, "count": processed_count, "items": results}
    """
    raise NotImplementedError("TODO(human): Implement the aggregate_results task")
