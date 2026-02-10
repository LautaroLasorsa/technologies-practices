"""Shared Kafka producer/consumer helpers using aiokafka.

All boilerplate for connecting to Redpanda, producing messages, and consuming
messages is implemented here. Services only need to provide their handler logic.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from shared.config import KAFKA_BOOTSTRAP
from shared.events import SagaMessage

logger = logging.getLogger(__name__)

# Type alias for message handlers
MessageHandler = Callable[[SagaMessage], Awaitable[list[SagaMessage] | None]]


async def create_producer() -> AIOKafkaProducer:
    """Create and start a Kafka producer connected to Redpanda."""
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP)
    await producer.start()
    logger.info("Kafka producer started (bootstrap=%s)", KAFKA_BOOTSTRAP)
    return producer


async def publish(
    producer: AIOKafkaProducer,
    topic: str,
    message: SagaMessage,
) -> None:
    """Publish a SagaMessage to a topic, keyed by saga_id."""
    await producer.send_and_wait(
        topic=topic,
        key=message.saga_id.encode("utf-8"),
        value=message.serialize(),
    )
    logger.info(
        "Published %s to %s (saga=%s)",
        message.message_type,
        topic,
        message.saga_id,
    )


async def consume_loop(
    topic: str,
    group_id: str,
    handler: MessageHandler,
    reply_topic: str | None = None,
    producer: AIOKafkaProducer | None = None,
) -> None:
    """
    Consume messages from a topic and dispatch to a handler.

    If the handler returns a list of SagaMessages, they are published to
    `reply_topic` (requires `producer` to be set).

    This function runs forever -- it is the main loop for each service.
    """
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )
    await consumer.start()
    logger.info(
        "Consumer started: topic=%s group=%s (bootstrap=%s)",
        topic,
        group_id,
        KAFKA_BOOTSTRAP,
    )

    try:
        async for record in consumer:
            try:
                message = SagaMessage.deserialize(record.value)
                logger.info(
                    "Received %s (saga=%s)",
                    message.message_type,
                    message.saga_id,
                )

                replies = await handler(message)

                if replies and reply_topic and producer:
                    for reply in replies:
                        await publish(producer, reply_topic, reply)

            except Exception:
                logger.exception("Error processing message from %s", topic)
    finally:
        await consumer.stop()


async def wait_for_redpanda(max_retries: int = 30, delay: float = 2.0) -> None:
    """Block until Redpanda is reachable. Used at service startup."""
    for attempt in range(1, max_retries + 1):
        try:
            consumer = AIOKafkaConsumer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
            )
            await consumer.start()
            await consumer.stop()
            logger.info("Redpanda is ready (attempt %d)", attempt)
            return
        except Exception:
            logger.warning(
                "Redpanda not ready (attempt %d/%d), retrying in %.0fs...",
                attempt,
                max_retries,
                delay,
            )
            await asyncio.sleep(delay)

    raise RuntimeError(f"Redpanda not reachable after {max_retries} attempts")
