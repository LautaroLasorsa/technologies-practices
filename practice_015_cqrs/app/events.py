"""Domain events for the Bank Account system.

Each event captures a business fact that happened. Events are immutable,
named in past tense, and carry all data needed to reconstruct state.

These are the "atoms" of Event Sourcing --- every state change is one of these.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------

class DomainEvent(BaseModel):
    """Base class for all domain events.

    Every event has:
    - event_id: unique identifier (for deduplication)
    - aggregate_id: which aggregate this event belongs to
    - event_type: discriminator for deserialization
    - occurred_at: when the event happened (UTC ISO-8601)
    - version: sequential version within the aggregate (for ordering)
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    aggregate_id: str
    event_type: str
    occurred_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    version: int = 0

    def to_json(self) -> str:
        """Serialize to JSON string for storage and publishing."""
        return self.model_dump_json()

    @staticmethod
    def from_json(raw: str) -> DomainEvent:
        """Deserialize a JSON string into the correct event subclass."""
        data = json.loads(raw)
        event_type = data.get("event_type")
        cls = EVENT_TYPE_MAP.get(event_type)
        if cls is None:
            raise ValueError(f"Unknown event type: {event_type}")
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Concrete events
# ---------------------------------------------------------------------------

class AccountOpened(DomainEvent):
    """A new bank account was opened."""

    event_type: Literal["AccountOpened"] = "AccountOpened"
    owner_name: str
    initial_balance: float = 0.0


class MoneyDeposited(DomainEvent):
    """Money was deposited into an account."""

    event_type: Literal["MoneyDeposited"] = "MoneyDeposited"
    amount: float
    description: str = ""


class MoneyWithdrawn(DomainEvent):
    """Money was withdrawn from an account."""

    event_type: Literal["MoneyWithdrawn"] = "MoneyWithdrawn"
    amount: float
    description: str = ""


# ---------------------------------------------------------------------------
# Registry for deserialization
# ---------------------------------------------------------------------------

EVENT_TYPE_MAP: dict[str, type[DomainEvent]] = {
    "AccountOpened": AccountOpened,
    "MoneyDeposited": MoneyDeposited,
    "MoneyWithdrawn": MoneyWithdrawn,
}
