"""
Basic Celery tasks -- Phase 2.

This module contains the fundamental Celery task patterns: defining tasks
with @app.task, dispatching them, and retrieving results.

Key concepts:
- @app.task decorator registers a function as a Celery task
- task.delay(args) is shorthand for task.apply_async(args=(args,))
- apply_async gives you more control: countdown, eta, queue, priority
- The return value is an AsyncResult that can be polled for state/result
"""

from __future__ import annotations

import logging
import time

from src.celery_app import app

logger = logging.getLogger(__name__)


# =============================================================================
# Exercise 1: Simple add task
# =============================================================================

@app.task(name="src.tasks_basic.add")
def add(x: int, y: int) -> int:
    """
    Add two numbers and return the result.

    # -- Exercise Context -------------------------------------------------------
    # This is the "hello world" of Celery. A task is just a Python function
    # decorated with @app.task. When you call add.delay(4, 5), Celery:
    #   1. Serializes the function name + args to JSON
    #   2. Publishes the JSON message to the Redis broker queue
    #   3. Returns an AsyncResult immediately (non-blocking)
    #   4. A worker picks up the message, deserializes it, calls add(4, 5)
    #   5. The worker stores the return value (9) in the Redis result backend
    #   6. The caller can poll AsyncResult.get() to retrieve the result
    #
    # This decoupling is the whole point: the producer (caller) doesn't wait
    # for the worker to finish. The work happens asynchronously.
    # ---------------------------------------------------------------------------

    TODO(human): Implement the add task.

    Steps:
    1. Log a message showing what's being added (use logger.info)
    2. Return the sum of x and y

    This is intentionally trivial -- the learning is in HOW the task is
    dispatched and executed, not WHAT it computes.
    """
    raise NotImplementedError("TODO(human): Implement the add task")


# =============================================================================
# Exercise 2: Simulated I/O-bound task
# =============================================================================

@app.task(name="src.tasks_basic.fetch_url_content", bind=True)
def fetch_url_content(self, url: str) -> dict:
    """
    Simulate fetching content from a URL (I/O-bound task).

    # -- Exercise Context -------------------------------------------------------
    # Real-world tasks are rarely as simple as "add two numbers". Most tasks
    # involve I/O: HTTP requests, database queries, file processing.
    #
    # bind=True makes the task instance available as `self`, giving access to:
    #   - self.request.id -- the unique task ID
    #   - self.request.retries -- current retry count
    #   - self.retry() -- manually trigger a retry
    #   - self.update_state() -- update custom state in the result backend
    #
    # The `self` parameter is essential for retry logic and progress tracking.
    # We simulate I/O with time.sleep() because real HTTP calls would require
    # network setup -- the pattern is identical either way.
    # ---------------------------------------------------------------------------

    TODO(human): Implement the fetch_url_content task.

    Steps:
    1. Log the start of the fetch (include self.request.id and url)
    2. Simulate I/O delay with time.sleep(2) (represents an HTTP request)
    3. Create a result dict with:
       - "task_id": self.request.id
       - "url": the url parameter
       - "content_length": a simulated length (e.g., len(url) * 100)
       - "status": "fetched"
    4. Log completion
    5. Return the result dict

    Hint: self.request.id gives you the unique task ID assigned by Celery.
    """
    raise NotImplementedError("TODO(human): Implement the fetch_url_content task")
