"""
Task routing -- Phase 5.

Task routing directs specific tasks to specific queues. Workers subscribe
to queues, so you can run dedicated workers for different workload types:

  Worker A: celery worker -Q high_priority     (fast machine, few tasks)
  Worker B: celery worker -Q low_priority      (cheap machine, batch tasks)
  Worker C: celery worker -Q default           (general purpose)

The routing is configured in celery_app.py via task_routes. Tasks can also
be routed at call time: task.apply_async(queue="high_priority").

This separation prevents low-priority batch jobs from starving
time-sensitive operations (e.g., sending OTP emails should not wait
behind 10,000 report-generation tasks).
"""

from __future__ import annotations

import logging
import time

from src.celery_app import app

logger = logging.getLogger(__name__)


# =============================================================================
# Exercise 7: High-priority task (routed to "high_priority" queue)
# =============================================================================

@app.task(name="src.tasks_routing.high_priority_task")
def high_priority_task(notification_type: str, recipient: str) -> dict:
    """
    A high-priority task that should be processed immediately.

    # -- Exercise Context -------------------------------------------------------
    # This task is routed to the "high_priority" queue via task_routes in
    # celery_app.py. Workers consuming this queue should be dedicated and
    # have low concurrency to ensure fast pickup.
    #
    # Use cases for high-priority queues:
    # - OTP / 2FA code delivery (time-sensitive)
    # - Real-time notifications (must arrive within seconds)
    # - Payment confirmations (user is waiting)
    # - Webhook delivery with SLA guarantees
    #
    # The task_routes config maps task names to queues:
    #   "src.tasks_routing.high_priority_task": {"queue": "high_priority"}
    #
    # You can also route at call time (overrides config):
    #   high_priority_task.apply_async(args=(...), queue="urgent")
    # ---------------------------------------------------------------------------

    TODO(human): Implement the high-priority notification task.

    Steps:
    1. Log that a high-priority task is being processed (include notification_type
       and recipient)
    2. Simulate quick processing with time.sleep(0.5)  (high-priority = fast)
    3. Return a dict: {"type": notification_type, "recipient": recipient,
       "status": "delivered", "priority": "high"}
    """
    raise NotImplementedError("TODO(human): Implement high_priority_task")


# =============================================================================
# Exercise 8: Low-priority task (routed to "low_priority" queue)
# =============================================================================

@app.task(name="src.tasks_routing.low_priority_task")
def low_priority_task(report_type: str, data_range: str) -> dict:
    """
    A low-priority batch task that can wait.

    # -- Exercise Context -------------------------------------------------------
    # This task is routed to the "low_priority" queue. Workers consuming this
    # queue can have higher concurrency and run on cheaper infrastructure.
    #
    # Use cases for low-priority queues:
    # - Nightly report generation
    # - Data migration / backfill
    # - Cache warming
    # - Non-urgent email campaigns
    #
    # Separating queues prevents "priority inversion": without routing, a burst
    # of 10,000 report tasks would fill the default queue, blocking all
    # high-priority notifications until the reports finish.
    #
    # You can also observe queue lengths in Flower or Redis:
    #   redis-cli LLEN high_priority  -> should be near 0 (fast processing)
    #   redis-cli LLEN low_priority   -> may accumulate (batch backlog)
    # ---------------------------------------------------------------------------

    TODO(human): Implement the low-priority report generation task.

    Steps:
    1. Log that a low-priority task is being processed (include report_type
       and data_range)
    2. Simulate slow processing with time.sleep(3)  (reports take time)
    3. Return a dict: {"report_type": report_type, "data_range": data_range,
       "status": "generated", "priority": "low"}
    """
    raise NotImplementedError("TODO(human): Implement low_priority_task")
