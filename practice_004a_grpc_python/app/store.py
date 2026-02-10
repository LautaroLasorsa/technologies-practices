"""In-memory task store with event broadcasting for the WatchTasks stream.

This module is fully implemented — no TODOs here. It provides:
- Thread-safe dict-based storage for Task objects
- An event queue system so WatchTasks can pick up changes in real time

The store is intentionally simple: a dict + a list of subscriber queues.
In production you'd back this with a database + message broker, but the
gRPC patterns (servicer, streaming, interceptors) are identical.
"""

from __future__ import annotations

import queue
import threading
import uuid
from datetime import datetime, timezone

from app.generated import task_manager_pb2 as pb2


class TaskStore:
    """Thread-safe in-memory task storage with event fan-out."""

    def __init__(self) -> None:
        self._tasks: dict[str, pb2.Task] = {}
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue[pb2.TaskEvent]] = []
        self._subscribers_lock = threading.Lock()

    # ── Helpers ────────────────────────────────────

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex[:12]

    def _broadcast(self, event: pb2.TaskEvent) -> None:
        """Send an event to all active WatchTasks subscribers."""
        with self._subscribers_lock:
            for q in self._subscribers:
                q.put(event)

    # ── CRUD ──────────────────────────────────────

    def create(self, title: str, description: str) -> pb2.Task:
        now = self._now_iso()
        task = pb2.Task(
            id=self._new_id(),
            title=title,
            description=description,
            status=pb2.TASK_STATUS_TODO,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._tasks[task.id] = task

        self._broadcast(pb2.TaskEvent(
            event_type=pb2.TASK_EVENT_TYPE_CREATED,
            task=task,
        ))
        return task

    def get(self, task_id: str) -> pb2.Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def update(
        self,
        task_id: str,
        title: str,
        description: str,
        status: pb2.TaskStatus.ValueType,
    ) -> pb2.Task | None:
        with self._lock:
            existing = self._tasks.get(task_id)
            if existing is None:
                return None

            # Build updated task — only overwrite fields that were provided
            updated = pb2.Task(
                id=existing.id,
                title=title if title else existing.title,
                description=description if description else existing.description,
                status=status if status != pb2.TASK_STATUS_UNSPECIFIED else existing.status,
                created_at=existing.created_at,
                updated_at=self._now_iso(),
            )
            self._tasks[task_id] = updated

        self._broadcast(pb2.TaskEvent(
            event_type=pb2.TASK_EVENT_TYPE_UPDATED,
            task=updated,
        ))
        return updated

    def delete(self, task_id: str) -> pb2.Task | None:
        with self._lock:
            task = self._tasks.pop(task_id, None)

        if task is not None:
            self._broadcast(pb2.TaskEvent(
                event_type=pb2.TASK_EVENT_TYPE_DELETED,
                task=task,
            ))
        return task

    # ── Streaming ─────────────────────────────────

    def subscribe(self) -> queue.Queue[pb2.TaskEvent]:
        """Register a new subscriber queue. Returns the queue."""
        q: queue.Queue[pb2.TaskEvent] = queue.Queue()
        with self._subscribers_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[pb2.TaskEvent]) -> None:
        """Remove a subscriber queue (called when client disconnects)."""
        with self._subscribers_lock:
            self._subscribers.remove(q)
