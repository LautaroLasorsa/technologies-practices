"""Kafka/Redpanda event consumer.

Consumes domain events from the `bank-events` topic and feeds them
to the projection store to update read models.

This is the bridge between the event bus (Redpanda) and the query side.
It runs as a long-lived background task inside the query service.

This module is fully implemented --- it's infrastructure plumbing.
"""

from __future__ import annotations

import asyncio
import logging

from aiokafka import AIOKafkaConsumer

from events import DomainEvent
from projections import ProjectionStore

logger = logging.getLogger(__name__)

TOPIC = "bank-events"
BOOTSTRAP_SERVERS = "localhost:19092"
GROUP_ID = "query-service-projections"


class EventConsumer:
    """Consumes events from Redpanda and updates projections."""

    def __init__(self, projection_store: ProjectionStore) -> None:
        self._store = projection_store
        self._consumer: AIOKafkaConsumer | None = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the Kafka consumer and begin processing events."""
        self._consumer = AIOKafkaConsumer(
            TOPIC,
            bootstrap_servers=BOOTSTRAP_SERVERS,
            group_id=GROUP_ID,
            auto_offset_reset="earliest",
            value_deserializer=lambda v: v.decode("utf-8"),
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
        )
        await self._consumer.start()
        logger.info(
            "EventConsumer started (topic=%s, group=%s)", TOPIC, GROUP_ID
        )
        self._task = asyncio.create_task(self._consume_loop())

    async def stop(self) -> None:
        """Stop the consumer gracefully."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._consumer:
            await self._consumer.stop()
            logger.info("EventConsumer stopped")

    async def _consume_loop(self) -> None:
        """Main consume loop --- reads messages and projects events."""
        assert self._consumer is not None
        try:
            async for message in self._consumer:
                try:
                    event = DomainEvent.from_json(message.value)
                    logger.info(
                        "Consumed %s for %s (partition=%d, offset=%d)",
                        event.event_type,
                        event.aggregate_id,
                        message.partition,
                        message.offset,
                    )
                    self._store.project_event(event)
                except Exception:
                    logger.exception(
                        "Error projecting event at offset %d", message.offset
                    )
        except asyncio.CancelledError:
            logger.info("Consumer loop cancelled")
            raise
