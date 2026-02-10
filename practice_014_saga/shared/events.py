"""Event and command schemas shared across all saga participants.

Key concept: Commands vs Events
- **Command**: "Please do X" (imperative, directed at a specific service)
- **Event**: "X happened" (past tense, broadcast to anyone interested)

The orchestrator sends COMMANDS to services via `saga.commands`.
Services reply with EVENTS via `saga.events`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum


# =============================================================================
# Message types (commands and events flowing through the saga)
# =============================================================================

class MessageType(str, Enum):
    """All message types exchanged between orchestrator and services."""

    # Commands: Orchestrator -> Services
    RESERVE_INVENTORY = "reserve_inventory"
    RELEASE_INVENTORY = "release_inventory"  # compensating
    PROCESS_PAYMENT = "process_payment"
    REFUND_PAYMENT = "refund_payment"  # compensating

    # Events: Services -> Orchestrator
    INVENTORY_RESERVED = "inventory_reserved"
    INVENTORY_RESERVE_FAILED = "inventory_reserve_failed"
    INVENTORY_RELEASED = "inventory_released"
    PAYMENT_PROCESSED = "payment_processed"
    PAYMENT_FAILED = "payment_failed"
    PAYMENT_REFUNDED = "payment_refunded"

    # Saga lifecycle
    SAGA_START = "saga_start"
    SAGA_COMPLETED = "saga_completed"
    SAGA_FAILED = "saga_failed"


# =============================================================================
# Saga state machine
# =============================================================================

class SagaState(str, Enum):
    """
    Lifecycle states of a saga instance.

    TODO(human): Define ALL states the saga passes through.

    Happy path:  STARTED -> ??? -> ??? -> COMPLETED
    Failure path: ... -> COMPENSATING_??? -> ... -> FAILED

    Hint: Think about what the orchestrator is "waiting for" at each step.
    States should reflect both forward progress and backward compensation.

    Expected states (7 total):
    - STARTED: saga just created
    - (2 forward states: one for inventory, one for payment)
    - COMPLETED: all steps succeeded
    - (2 compensation states: one for payment refund, one for inventory release)
    - FAILED: compensation complete, saga is done
    """
    pass  # TODO(human): Replace with enum members


# =============================================================================
# Saga instance tracker
# =============================================================================

@dataclass
class OrderSaga:
    """
    Tracks the state of a single saga instance.

    TODO(human): Add the fields this dataclass needs to track a saga.

    Think about:
    - How do we identify this saga? (saga_id)
    - What order data does it carry? (order details)
    - What is its current state? (SagaState)
    - When was it created/updated? (timestamps)
    - What history of state transitions has it gone through? (audit trail)

    Hint: Use field(default_factory=...) for mutable defaults.
    """
    pass  # TODO(human): Replace with dataclass fields


# =============================================================================
# Message envelope (serialization helper -- fully implemented)
# =============================================================================

@dataclass
class SagaMessage:
    """Wire format for all messages on saga.commands and saga.events topics."""

    message_type: str
    saga_id: str
    payload: dict
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def serialize(self) -> bytes:
        """Serialize to JSON bytes for Kafka."""
        return json.dumps(asdict(self)).encode("utf-8")

    @classmethod
    def deserialize(cls, raw: bytes) -> SagaMessage:
        """Deserialize from JSON bytes."""
        data = json.loads(raw.decode("utf-8"))
        return cls(**data)

    @classmethod
    def create(
        cls,
        message_type: MessageType,
        saga_id: str,
        payload: dict | None = None,
    ) -> SagaMessage:
        """Factory for creating a typed message."""
        return cls(
            message_type=message_type.value,
            saga_id=saga_id,
            payload=payload or {},
        )


# =============================================================================
# Order data (carried inside SagaMessage.payload)
# =============================================================================

@dataclass
class OrderData:
    """Order details embedded in saga messages."""

    order_id: str
    customer_id: str
    item: str
    quantity: int
    price: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> OrderData:
        return cls(**data)
