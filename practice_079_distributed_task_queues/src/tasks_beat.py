"""
Celery Beat periodic tasks -- Phase 6.

Celery Beat is a scheduler that kicks off tasks at regular intervals.
It's a separate process (celery beat) that reads the beat_schedule config
and publishes task messages to the broker at the specified times.

Beat itself does NOT execute tasks -- it only schedules them. Workers
consume and execute the tasks as usual. This separation means:
- Beat can run on a single node (no concurrency issues)
- Workers can scale independently
- If no worker is available, tasks queue up in the broker

IMPORTANT: Only run ONE Beat instance. Multiple Beat processes will
dispatch duplicate periodic tasks. This is a common production mistake.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from src.celery_app import app

logger = logging.getLogger(__name__)


# =============================================================================
# Exercise 9: Periodic cleanup task (runs every 30 seconds)
# =============================================================================

@app.task(name="src.tasks_beat.periodic_cleanup")
def periodic_cleanup() -> dict:
    """
    A periodic task that simulates cleaning up expired data.

    # -- Exercise Context -------------------------------------------------------
    # Periodic tasks are defined in the beat_schedule config (celery_app.py).
    # Beat dispatches this task every 30 seconds, regardless of whether the
    # previous invocation has completed.
    #
    # This is the Celery equivalent of a cron job, but with advantages:
    # - Runs in the same ecosystem as your other tasks (same monitoring, retry)
    # - No separate cron daemon to manage
    # - Schedule changes don't require server restarts (with django-celery-beat)
    # - Failures are visible in Flower, not hidden in cron logs
    #
    # Common periodic task use cases:
    # - Cache invalidation / cleanup
    # - Session expiration
    # - Health check pinging
    # - Metrics aggregation
    # - Temporary file cleanup
    #
    # WARNING: If the task takes longer than the schedule interval (e.g.,
    # cleanup takes 45 seconds but is scheduled every 30), tasks will overlap.
    # Use a Redis lock to prevent this (or Celery's `solo` task pattern).
    # ---------------------------------------------------------------------------

    TODO(human): Implement the periodic cleanup task.

    Steps:
    1. Record the current UTC timestamp with datetime.now(timezone.utc).isoformat()
    2. Log that cleanup is starting (include the timestamp)
    3. Simulate cleanup work with time.sleep(1)
    4. Generate a fake count of cleaned items (e.g., random or deterministic)
    5. Log completion with the cleaned count
    6. Return a dict: {"task": "cleanup", "timestamp": timestamp,
       "cleaned_items": count}
    """
    raise NotImplementedError("TODO(human): Implement periodic_cleanup")


# =============================================================================
# Exercise 10: Crontab-scheduled task
# =============================================================================

@app.task(name="src.tasks_beat.generate_daily_report")
def generate_daily_report() -> dict:
    """
    A crontab-scheduled task that simulates generating a daily report.

    # -- Exercise Context -------------------------------------------------------
    # The beat_schedule in celery_app.py uses crontab() for this task.
    # For demo purposes, it runs every minute instead of once daily.
    #
    # Celery's crontab() supports the same fields as Unix cron:
    #   crontab(minute=0, hour=6)                 -> 6:00 AM daily
    #   crontab(minute=0, hour=0, day_of_week=1)  -> Monday midnight
    #   crontab(minute='*/15')                     -> every 15 minutes
    #   crontab(minute=0, hour='*/2')              -> every 2 hours
    #
    # The schedule uses UTC by default (configurable via timezone setting).
    # Beat stores the last-run timestamp to avoid firing missed schedules
    # after restart (configurable via beat_schedule_filename).
    #
    # NOTE: Crontab precision is 1 minute (like Unix cron). For sub-minute
    # scheduling, use timedelta intervals instead.
    # ---------------------------------------------------------------------------

    TODO(human): Implement the daily report generation task.

    Steps:
    1. Record the current UTC timestamp
    2. Log that report generation is starting
    3. Simulate report generation with time.sleep(2)
    4. Log completion
    5. Return a dict: {"task": "daily_report", "timestamp": timestamp,
       "status": "generated", "rows_processed": <any number>}
    """
    raise NotImplementedError("TODO(human): Implement generate_daily_report")
