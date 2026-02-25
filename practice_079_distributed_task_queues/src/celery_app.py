"""
Celery application factory and configuration.

This module initializes the Celery app with Redis as both broker and result
backend. It also configures task routing, serialization, and the Beat schedule.

All configuration is centralized here rather than spread across task modules.
This follows Celery's recommended pattern: one app instance, imported everywhere.
"""

from __future__ import annotations

import os

from celery import Celery
from celery.schedules import crontab

# ---------------------------------------------------------------------------
# Redis URL from environment (Docker) or default (local dev)
# ---------------------------------------------------------------------------
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# ---------------------------------------------------------------------------
# Celery app instance
# ---------------------------------------------------------------------------
app = Celery("taskqueue")

# ---------------------------------------------------------------------------
# Celery configuration
# ---------------------------------------------------------------------------
app.conf.update(
    # Broker (message transport) -- where tasks are queued
    broker_url=REDIS_URL,

    # Result backend -- where task return values and state are stored
    result_backend=REDIS_URL,

    # Serialization: JSON is human-readable, debuggable, and cross-language safe.
    # Pickle is faster but allows arbitrary code execution (security risk).
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone -- use UTC to avoid DST surprises in scheduled tasks
    timezone="UTC",
    enable_utc=True,

    # Result expiration -- auto-delete results after 1 hour (seconds)
    result_expires=3600,

    # Worker settings
    worker_prefetch_multiplier=1,  # fetch one task at a time (fairer distribution)
    worker_max_tasks_per_child=100,  # restart worker child after 100 tasks (leak prevention)

    # Late acknowledgment -- acknowledge AFTER task completes, not when received.
    # This means if a worker crashes mid-task, the broker re-delivers the message.
    # CRITICAL: tasks with acks_late MUST be idempotent!
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Task tracking -- store STARTED state in backend (useful for monitoring)
    task_track_started=True,

    # =========================================================================
    # Task routing: directs specific tasks to specific queues.
    #
    # Workers subscribe to queues with: celery worker -Q default,high_priority
    # This lets you run dedicated workers for different workload types.
    # =========================================================================
    task_routes={
        "src.tasks_routing.high_priority_task": {"queue": "high_priority"},
        "src.tasks_routing.low_priority_task": {"queue": "low_priority"},
        # All other tasks go to the "default" queue (Celery's built-in default)
    },

    # Default queue for tasks without explicit routing
    task_default_queue="default",

    # =========================================================================
    # Beat schedule: periodic tasks dispatched by the Celery Beat process.
    #
    # Beat reads this config and publishes task messages at the specified
    # intervals. Workers then pick them up like any other task.
    # =========================================================================
    beat_schedule={
        "periodic-cleanup-every-30s": {
            "task": "src.tasks_beat.periodic_cleanup",
            "schedule": 30.0,  # every 30 seconds
            "args": (),
        },
        "daily-report-crontab": {
            "task": "src.tasks_beat.generate_daily_report",
            # For demo purposes, run every minute instead of once daily
            "schedule": crontab(minute="*/1"),
            "args": (),
        },
    },
)

# ---------------------------------------------------------------------------
# Auto-discover tasks in src.tasks_* modules
# ---------------------------------------------------------------------------
app.autodiscover_tasks(["src"], related_name=None, force=True)

# Explicitly register task modules so Celery finds all @app.task decorators
import src.tasks_basic  # noqa: F401, E402
import src.tasks_canvas  # noqa: F401, E402
import src.tasks_retry  # noqa: F401, E402
import src.tasks_routing  # noqa: F401, E402
import src.tasks_beat  # noqa: F401, E402
