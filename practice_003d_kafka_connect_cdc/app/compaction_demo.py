"""Log compaction demonstration.

Kafka log compaction is a key-based retention mechanism: instead of
deleting old messages by time, it keeps only the LATEST value for
each key. This is perfect for CDC changelog topics where you only
need the current state of each row.

Run standalone:
    uv run python compaction_demo.py
"""

import json
import time

from confluent_kafka import Consumer, KafkaError, Producer
from confluent_kafka.admin import AdminClient, NewTopic, ConfigResource, ResourceType

import config


# ── Topic management ─────────────────────────────────────────────────


def create_compacted_topic(admin_client: AdminClient, topic_name: str) -> None:
    """Create a topic with log compaction enabled.

    # ── TODO(human) ──────────────────────────────────────────────────
    # Log compaction keeps the LATEST value per key and discards older
    # values for the same key. This is different from time-based
    # retention (default), which deletes ALL messages older than
    # retention.ms regardless of key.
    #
    # Steps:
    #   1. Create a NewTopic with:
    #      - topic: topic_name
    #      - num_partitions: 1 (single partition simplifies the demo)
    #      - replication_factor: 1
    #      - config: {
    #          "cleanup.policy": "compact",
    #              Enables log compaction instead of time-based deletion.
    #              Can also be "delete" (default) or "compact,delete" (both).
    #
    #          "segment.ms": "10000",
    #              Force log segment rollover every 10 seconds. By default
    #              this is 7 days! Compaction only runs on CLOSED segments
    #              (not the active one), so we need fast rollover for the
    #              demo. In production, keep the default or use segment.bytes.
    #
    #          "min.cleanable.dirty.ratio": "0.01",
    #              Ratio of dirty (uncompacted) to total log size that
    #              triggers compaction. Default is 0.5 (50%). Setting it
    #              very low (1%) makes compaction run aggressively.
    #              In production, higher values reduce compaction overhead.
    #
    #          "delete.retention.ms": "1000",
    #              How long tombstones (key + null value) are retained
    #              after compaction. Default is 24 hours. Short value
    #              for demo purposes.
    #        }
    #
    #   2. Call admin_client.create_topics([topic])
    #   3. Wait for the future to resolve (future.result())
    #   4. Handle TOPIC_ALREADY_EXISTS gracefully (print message, continue)
    #
    # Key insight: compaction NEVER runs on the active (open) segment.
    # That's why segment.ms=10000 matters -- without it, you'd need to
    # wait 7 days for compaction to kick in on the demo data.
    # ─────────────────────────────────────────────────────────────────
    """
    raise NotImplementedError("TODO(human)")


def demonstrate_compaction(topic_name: str) -> None:
    """Produce multiple values per key, then consume to show compaction.

    # ── TODO(human) ──────────────────────────────────────────────────
    # This function demonstrates log compaction by:
    #   1. Producing MULTIPLE messages for the SAME keys
    #   2. Waiting for compaction to run
    #   3. Consuming to show only the latest values survive
    #
    # Steps:
    #   1. Create a Producer with:
    #      {"bootstrap.servers": config.BOOTSTRAP_SERVERS}
    #
    #   2. Produce messages simulating state changes for 3 "users":
    #      Key: "user-1"  -> Values: "status=active", "status=inactive", "status=active"
    #      Key: "user-2"  -> Values: "status=active", "status=banned"
    #      Key: "user-3"  -> Values: "status=active"
    #      Key: "user-3"  -> Value: None (tombstone! signals deletion)
    #
    #      For each message:
    #        producer.produce(
    #            topic=topic_name,
    #            key=key.encode("utf-8"),
    #            value=value.encode("utf-8") if value else None,
    #        )
    #      Call producer.flush() after all messages.
    #
    #   3. Print what was produced (all 7 messages)
    #
    #   4. Print a message explaining we need to wait for compaction.
    #      Sleep for 30 seconds (segment.ms=10000 means the segment
    #      closes after 10s, then compaction runs on the next cycle).
    #      Print countdown or dots so user knows it's working.
    #
    #   5. Consume ALL messages from the topic (from beginning):
    #      Create a Consumer with:
    #        {
    #          "bootstrap.servers": config.BOOTSTRAP_SERVERS,
    #          "group.id": f"compaction-demo-{int(time.time())}",
    #          "auto.offset.reset": "earliest",
    #          "enable.auto.commit": False,
    #        }
    #      Use a unique group.id each run so we always read from start.
    #
    #   6. Poll until no messages for 5 seconds, collecting results.
    #      For each message, record key and value.
    #
    #   7. Print the consumed messages and explain:
    #      - "user-1" should have only the LAST value ("status=active")
    #      - "user-2" should have only "status=banned"
    #      - "user-3" may be absent (tombstone removed it) or present
    #        with null value (tombstone not yet cleaned)
    #      Note: compaction timing is non-deterministic. If all 7
    #      messages appear, compaction hasn't run yet on that segment.
    #      The user should run the script again after a minute.
    #
    #   8. Close the consumer.
    #
    # Why this matters for CDC:
    #   Debezium CDC topics use log compaction by default. For a table
    #   with 1M rows and frequent updates, compaction ensures the topic
    #   contains at most 1M messages (latest state per row) instead of
    #   growing unboundedly with every UPDATE.
    # ─────────────────────────────────────────────────────────────────
    """
    raise NotImplementedError("TODO(human)")


# ── Helpers (fully implemented) ──────────────────────────────────────


def produce_keyed_messages(
    producer: Producer,
    topic: str,
    messages: list[tuple[str, str | None]],
) -> None:
    """Produce a list of (key, value) messages to a topic."""
    for key, value in messages:
        producer.produce(
            topic=topic,
            key=key.encode("utf-8"),
            value=value.encode("utf-8") if value is not None else None,
        )
    producer.flush()


def consume_all_messages(
    bootstrap_servers: str,
    topic: str,
    timeout_no_message: float = 5.0,
) -> list[tuple[str, str | None]]:
    """Consume all available messages from a topic, return (key, value) pairs."""
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": f"compaction-reader-{int(time.time())}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])

    results: list[tuple[str, str | None]] = []
    last_msg_time = time.time()

    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            if time.time() - last_msg_time > timeout_no_message:
                break
            time.sleep(0.2)
            continue

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            print(f"  Consumer error: {msg.error()}")
            break

        last_msg_time = time.time()
        key = msg.key().decode("utf-8") if msg.key() else "(null)"
        value = msg.value().decode("utf-8") if msg.value() else None
        results.append((key, value))

    consumer.close()
    return results


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    """Run the log compaction demonstration."""
    print("=== Log Compaction Demo ===\n")

    admin = AdminClient({"bootstrap.servers": config.BOOTSTRAP_SERVERS})
    create_compacted_topic(admin, config.COMPACTION_TOPIC)
    demonstrate_compaction(config.COMPACTION_TOPIC)


if __name__ == "__main__":
    main()
