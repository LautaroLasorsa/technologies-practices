"""
Orchestrator entrypoint.

Consumes events from saga.events, processes them through the state machine,
and publishes commands to saga.commands (or status to order.status).
"""

from __future__ import annotations

import asyncio
import logging

from shared.config import (
    GROUP_ORCHESTRATOR,
    TOPIC_ORDER_STATUS,
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
from orchestrator.saga_orchestrator import SagaOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ORCHESTRATOR] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Status events go to order.status; commands go to saga.commands
STATUS_EVENTS = {MessageType.SAGA_COMPLETED.value, MessageType.SAGA_FAILED.value}


async def main() -> None:
    await wait_for_redpanda()

    orchestrator = SagaOrchestrator()
    producer = await create_producer()

    async def handle_message(message: SagaMessage) -> list[SagaMessage] | None:
        """
        Route incoming events through the orchestrator and publish replies.

        TODO(human): Implement the routing logic.

        Steps:
        1. Call orchestrator.handle_event(message) to get reply messages
        2. For each reply, decide the target topic:
           - If reply.message_type is in STATUS_EVENTS -> publish to TOPIC_ORDER_STATUS
           - Otherwise -> publish to TOPIC_SAGA_COMMANDS
        3. Publish each reply using the `publish()` helper

        Hint: This is the "glue" between the state machine and Kafka.
        The orchestrator doesn't know about topics -- this function does.
        """
        # TODO(human): Implement message routing
        raise NotImplementedError("Implement handle_message()")

    logger.info("Saga Orchestrator starting consumer loop...")
    await consume_loop(
        topic=TOPIC_SAGA_EVENTS,
        group_id=GROUP_ORCHESTRATOR,
        handler=handle_message,
    )


if __name__ == "__main__":
    asyncio.run(main())
