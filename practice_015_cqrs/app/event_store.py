"""SQLite-backed append-only event store.

The event store is the write-side persistence layer. It stores every domain
event as an immutable row. Events are never updated or deleted --- only appended.

Key properties:
- Append-only: INSERT only, no UPDATE/DELETE
- Ordered: events for an aggregate are versioned sequentially
- Optimistic concurrency: expected_version prevents conflicting writes

This module is fully implemented --- it's infrastructure plumbing, not the
educational core.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from events import DomainEvent


class EventStore:
    """Append-only event store backed by SQLite."""

    def __init__(self, db_path: str = "event_store.db") -> None:
        self._db_path = db_path
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the events table if it doesn't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    aggregate_id TEXT    NOT NULL,
                    version      INTEGER NOT NULL,
                    event_type   TEXT    NOT NULL,
                    event_id     TEXT    NOT NULL UNIQUE,
                    data         TEXT    NOT NULL,
                    occurred_at  TEXT    NOT NULL,
                    UNIQUE(aggregate_id, version)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_aggregate
                ON events(aggregate_id, version)
            """)

    def _connect(self) -> sqlite3.Connection:
        """Create a new SQLite connection."""
        return sqlite3.connect(self._db_path)

    def append(
        self,
        events: list[DomainEvent],
        expected_version: int,
    ) -> None:
        """Append events to the store with optimistic concurrency control.

        Args:
            events: list of domain events to persist.
            expected_version: the aggregate version before these events.
                              If the actual latest version differs, raises
                              ConcurrencyError (another write happened first).

        Raises:
            ConcurrencyError: if the expected version doesn't match.
        """
        if not events:
            return

        aggregate_id = events[0].aggregate_id
        with self._connect() as conn:
            current = self._latest_version(conn, aggregate_id)
            if current != expected_version:
                raise ConcurrencyError(
                    f"Expected version {expected_version} for aggregate "
                    f"{aggregate_id}, but found {current}"
                )

            for i, event in enumerate(events):
                version = expected_version + i + 1
                event.version = version
                conn.execute(
                    """
                    INSERT INTO events
                        (aggregate_id, version, event_type, event_id, data, occurred_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.aggregate_id,
                        version,
                        event.event_type,
                        event.event_id,
                        event.to_json(),
                        event.occurred_at,
                    ),
                )

    def load_events(self, aggregate_id: str) -> list[DomainEvent]:
        """Load all events for an aggregate, ordered by version."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT data FROM events WHERE aggregate_id = ? ORDER BY version",
                (aggregate_id,),
            ).fetchall()
        return [DomainEvent.from_json(row[0]) for row in rows]

    def load_all_events(self, after_id: int = 0) -> list[tuple[int, DomainEvent]]:
        """Load all events across all aggregates (for rebuilding projections).

        Args:
            after_id: only return events with store id > after_id (for catch-up).

        Returns:
            List of (store_id, event) tuples ordered by store id.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, data FROM events WHERE id > ? ORDER BY id",
                (after_id,),
            ).fetchall()
        return [(row[0], DomainEvent.from_json(row[1])) for row in rows]

    def _latest_version(self, conn: sqlite3.Connection, aggregate_id: str) -> int:
        """Get the latest version for an aggregate (0 if none)."""
        row = conn.execute(
            "SELECT MAX(version) FROM events WHERE aggregate_id = ?",
            (aggregate_id,),
        ).fetchone()
        return row[0] or 0


class ConcurrencyError(Exception):
    """Raised when optimistic concurrency check fails."""
