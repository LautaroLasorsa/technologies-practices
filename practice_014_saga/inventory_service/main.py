"""
Inventory Service -- Reserves and releases inventory.

Listens on saga.commands for:
- RESERVE_INVENTORY: Reserve stock for an order
- RELEASE_INVENTORY: Release reserved stock (compensating transaction)

Replies on saga.events with:
- INVENTORY_RESERVED: Stock successfully reserved
- INVENTORY_RESERVE_FAILED: Not enough stock
- INVENTORY_RELEASED: Stock released (compensation done)
"""

from __future__ import annotations

import asyncio
import logging

from shared.config import (
    GROUP_INVENTORY_SERVICE,
    TOPIC_SAGA_COMMANDS,
    TOPIC_SAGA_EVENTS,
)
from shared.events import MessageType, SagaMessage
from shared.kafka_utils import (
    consume_loop,
    create_producer,
    publish,
    wait_for_redpanda,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [INVENTORY-SVC] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# In-memory inventory store (simulates a database)
# =============================================================================

inventory: dict[str, int] = {
    "laptop": 10,
    "keyboard": 50,
    "mouse": 100,
    "monitor": 5,
}

# Track reservations by saga_id for idempotent compensation
reservations: dict[str, dict] = {}


# =============================================================================
# Command handlers
# =============================================================================

def reserve_inventory(saga_id: str, payload: dict) -> SagaMessage:
    """
    Reserve inventory for an order.

    TODO(human): Implement the reservation logic.

    Steps:
    1. Extract `item` and `quantity` from payload
    2. Check if the item exists in `inventory` and has enough stock
    3. If sufficient stock:
       - Deduct the quantity from inventory[item]
       - Record the reservation in `reservations[saga_id]`
         (store item and quantity so we can undo it later)
       - Return an INVENTORY_RESERVED event
    4. If insufficient stock:
       - Return an INVENTORY_RESERVE_FAILED event
       - Include a reason in the payload (e.g., {"reason": "Not enough stock"})

    Idempotency hint: If saga_id is already in reservations, this is a retry.
    Return INVENTORY_RESERVED without deducting again.

    Args:
        saga_id: Unique identifier for this saga instance
        payload: Dict containing order data (item, quantity, etc.)

    Returns:
        SagaMessage with the appropriate event type
    """
    # TODO(human): Implement inventory reservation
    raise NotImplementedError("Implement reserve_inventory()")


def release_inventory(saga_id: str, payload: dict) -> SagaMessage:
    """
    Release previously reserved inventory (compensating transaction).

    TODO(human): Implement the release logic.

    Steps:
    1. Look up the reservation in `reservations[saga_id]`
    2. If found:
       - Add the reserved quantity back to inventory[item]
       - Remove the reservation from `reservations`
       - Return an INVENTORY_RELEASED event
    3. If not found (already released or never reserved):
       - Return INVENTORY_RELEASED anyway (idempotent -- safe to call twice)
       - Log a warning

    Why idempotent? If the orchestrator retries a compensation command
    (e.g., due to a network hiccup), we must not double-release.

    Args:
        saga_id: Unique identifier for this saga instance
        payload: Dict containing order data

    Returns:
        SagaMessage with INVENTORY_RELEASED event type
    """
    # TODO(human): Implement inventory release (compensating)
    raise NotImplementedError("Implement release_inventory()")


# =============================================================================
# Message dispatcher (fully implemented)
# =============================================================================

COMMAND_HANDLERS = {
    MessageType.RESERVE_INVENTORY.value: reserve_inventory,
    MessageType.RELEASE_INVENTORY.value: release_inventory,
}


async def handle_command(message: SagaMessage) -> list[SagaMessage] | None:
    """Dispatch a command to the appropriate handler."""
    handler = COMMAND_HANDLERS.get(message.message_type)
    if handler is None:
        return None  # Not our command, ignore

    reply = handler(message.saga_id, message.payload)
    return [reply]


# =============================================================================
# Entrypoint
# =============================================================================

async def main() -> None:
    await wait_for_redpanda()

    producer = await create_producer()

    logger.info("Inventory service starting consumer loop...")
    await consume_loop(
        topic=TOPIC_SAGA_COMMANDS,
        group_id=GROUP_INVENTORY_SERVICE,
        handler=handle_command,
        reply_topic=TOPIC_SAGA_EVENTS,
        producer=producer,
    )


if __name__ == "__main__":
    asyncio.run(main())
