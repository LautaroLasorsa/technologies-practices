"""Kafka/Redpanda event publisher.

Publishes domain events to Redpanda so the query side can consume them
and update read-model projections.

Events are published to the `bank-events` topic, keyed by aggregate_id
(so all events for one account land on the same partition, preserving order).

This module is fully implemented --- it's infrastructure plumbing.
"""

from __future__ import annotations

import logging

from aiokafka import AIOKafkaProducer

from events import DomainEvent

logger = logging.getLogger(__name__)

TOPIC = "bank-events"
BOOTSTRAP_SERVERS = "localhost:19092"


class EventPublisher:
    """Publishes domain events to Redpanda."""

    def __init__(self) -> None:
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        """Start the Kafka producer."""
        self._producer = AIOKafkaProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            key_serializer=lambda k: k.encode("utf-8"),
            value_serializer=lambda v: v.encode("utf-8"),
        )
        await self._producer.start()
        logger.info("EventPublisher started (bootstrap=%s)", BOOTSTRAP_SERVERS)

    async def stop(self) -> None:
        """Stop the Kafka producer."""
        if self._producer:
            await self._producer.stop()
            logger.info("EventPublisher stopped")

    async def publish(self, events: list[DomainEvent]) -> None:
        """Publish a batch of domain events to the bank-events topic.

        Each event is keyed by aggregate_id so partition ordering is preserved
        per account.
        """
        if not self._producer:
            raise RuntimeError("EventPublisher not started")

        for event in events:
            await self._producer.send_and_wait(
                topic=TOPIC,
                key=event.aggregate_id,
                value=event.to_json(),
            )
            logger.info(
                "Published %s for aggregate %s (v%d)",
                event.event_type,
                event.aggregate_id,
                event.version,
            )
