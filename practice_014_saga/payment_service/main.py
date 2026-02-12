"""
Payment Service -- Processes and refunds payments.

Listens on saga.commands for:
- PROCESS_PAYMENT: Charge the customer
- REFUND_PAYMENT: Refund a previously processed payment (compensating transaction)

Replies on saga.events with:
- PAYMENT_PROCESSED: Payment successful
- PAYMENT_FAILED: Payment declined (e.g., total > $500)
- PAYMENT_REFUNDED: Refund completed (compensation done)

Failure simulation: Orders with total price > $500 are automatically declined.
This triggers the compensating transaction flow.
"""

from __future__ import annotations

import asyncio
import logging

from shared.config import (
    GROUP_PAYMENT_SERVICE,
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
    format="%(asctime)s [PAYMENT-SVC] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# In-memory payment store (simulates a database)
# =============================================================================

payments: dict[str, dict] = {}

# Failure threshold for simulating declined payments
MAX_ALLOWED_TOTAL = 500.0


# =============================================================================
# Command handlers
# =============================================================================

def process_payment(saga_id: str, payload: dict) -> SagaMessage:
    """
    Process a payment for an order.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches failure detection and error propagation in sagas.
    # The payment service simulates a declined transaction (price > $500), which
    # triggers the orchestrator's compensation flow. Production payment gateways
    # (Stripe, PayPal) return rich error codes; sagas must map these to appropriate
    # compensations or retries.
    # ──────────────────────────────────────────────────────────────────────

    TODO(human): Implement the payment logic.

    Steps:
    1. Calculate total = price * quantity from payload
    2. If total > MAX_ALLOWED_TOTAL:
       - Return a PAYMENT_FAILED event
       - Include {"reason": "Payment declined: total $X exceeds limit"} in payload
       - Log the decline
    3. If total <= MAX_ALLOWED_TOTAL:
       - Record the payment in `payments[saga_id]` (store total, customer_id, etc.)
       - Return a PAYMENT_PROCESSED event
       - Log the success

    Idempotency hint: If saga_id is already in payments, this is a retry.
    Return PAYMENT_PROCESSED without charging again.

    Args:
        saga_id: Unique identifier for this saga instance
        payload: Dict containing order data (price, quantity, customer_id, etc.)

    Returns:
        SagaMessage with the appropriate event type
    """
    # TODO(human): Implement payment processing
    raise NotImplementedError("Implement process_payment()")


def refund_payment(saga_id: str, payload: dict) -> SagaMessage:
    """
    Refund a previously processed payment (compensating transaction).

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches payment refund as a compensating transaction. In
    # production, refunds often interact with external systems (payment gateways)
    # that have their own idempotency semantics. SAGAs must handle cases where
    # the gateway confirms a refund but the confirmation message is lost, requiring
    # retry-safe compensations.
    # ──────────────────────────────────────────────────────────────────────

    TODO(human): Implement the refund logic.

    Steps:
    1. Look up the payment in `payments[saga_id]`
    2. If found:
       - Remove the payment from `payments`
       - Return a PAYMENT_REFUNDED event
       - Log the refund
    3. If not found (already refunded or never processed):
       - Return PAYMENT_REFUNDED anyway (idempotent)
       - Log a warning

    Args:
        saga_id: Unique identifier for this saga instance
        payload: Dict containing order data

    Returns:
        SagaMessage with PAYMENT_REFUNDED event type
    """
    # TODO(human): Implement payment refund (compensating)
    raise NotImplementedError("Implement refund_payment()")


# =============================================================================
# Message dispatcher (fully implemented)
# =============================================================================

COMMAND_HANDLERS = {
    MessageType.PROCESS_PAYMENT.value: process_payment,
    MessageType.REFUND_PAYMENT.value: refund_payment,
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

    logger.info("Payment service starting consumer loop...")
    await consume_loop(
        topic=TOPIC_SAGA_COMMANDS,
        group_id=GROUP_PAYMENT_SERVICE,
        handler=handle_command,
        reply_topic=TOPIC_SAGA_EVENTS,
        producer=producer,
    )


if __name__ == "__main__":
    asyncio.run(main())
