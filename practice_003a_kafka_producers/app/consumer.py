"""Kafka consumer for the Event Log System.

Demonstrates:
  - Subscribe to a topic and poll for messages
  - Manual offset commit (enable.auto.commit=False)
  - Graceful shutdown with consumer.close()
  - Deserializing JSON message values

Run after producing messages:
    uv run python consumer.py
"""

import json
import signal
import sys

from confluent_kafka import Consumer, KafkaError, KafkaException

import config


# ── Graceful shutdown flag ───────────────────────────────────────────

_running = True


def _signal_handler(signum, frame) -> None:
    """Set the shutdown flag on Ctrl+C."""
    global _running
    print("\nShutdown requested...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)


# ── TODO(human): Implement this function ─────────────────────────────


def consume_events(
    consumer: Consumer,
    topic: str,
    max_messages: int = 50,
) -> list[dict]:
    """Subscribe to a topic and consume messages until idle or max reached.

    TODO(human): Implement this function.

    Steps:
      1. Subscribe to the topic:
             consumer.subscribe([topic])
         This tells the consumer group coordinator which topics this
         consumer wants. Partition assignment happens automatically.

      2. Create an empty list to collect processed events.

      3. Enter a while loop that runs while `_running` is True and
         the number of processed events is below `max_messages`:

         a. Call consumer.poll(timeout=1.0).
            poll() returns a single Message object, or None if no
            message arrived within the timeout.

         b. If the result is None, print "No messages, waiting..." and continue.
            (After 3 consecutive Nones, you may want to break out of the loop.)

         c. If the message has an error, check:
            - If msg.error().code() == KafkaError._PARTITION_EOF, this is
              informational (you've reached the end of a partition). Print
              a message and continue.
            - Otherwise, raise KafkaException(msg.error()).

         d. Decode the message:
            - key:   msg.key().decode("utf-8") if msg.key() else None
            - value: json.loads(msg.value().decode("utf-8"))

         e. Print the consumed event info:
            topic, partition, offset, key, and the event_type from the value.

         f. Append the decoded value dict to your results list.

         g. Commit the offset for this message:
                consumer.commit(message=msg, asynchronous=False)
            Synchronous commit blocks until the broker confirms. This ensures
            at-least-once delivery: if the consumer crashes after processing
            but before committing, the message will be redelivered.

            Alternative: consumer.commit(asynchronous=True) for fire-and-forget
            commits (faster, but small risk of reprocessing on crash).

      4. Return the list of processed events.

    Key concept (auto.offset.reset):
      When a consumer group reads a topic for the FIRST time (no committed
      offsets exist), `auto.offset.reset` decides where to start:
        - "earliest": Read from the beginning of each partition (all history)
        - "latest":   Read only NEW messages produced after subscription
      For this practice we use "earliest" so you see all previously produced messages.

    Key concept (enable.auto.commit):
      When True (default), the consumer commits offsets periodically in the
      background (every auto.commit.interval.ms). This is convenient but
      can cause message loss: if processing fails after auto-commit, the
      message won't be redelivered. We set it to False for manual control.

    Docs: https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#confluent_kafka.Consumer.poll
    """
    raise NotImplementedError("TODO(human): implement consume_events")


# ── Orchestration (boilerplate) ──────────────────────────────────────


def create_consumer(group_id: str) -> Consumer:
    """Create a Kafka consumer with manual commit enabled.

    Key configuration:
      - bootstrap.servers: Broker address for initial connection.
      - group.id: Consumer group this consumer belongs to. All consumers
        in the same group share partition assignments.
      - auto.offset.reset: Where to start if no committed offset exists.
      - enable.auto.commit: False = you control when offsets are committed.
      - client.id: Identifies this consumer in broker logs.
    """
    return Consumer({
        "bootstrap.servers": config.BOOTSTRAP_SERVERS,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
        "client.id": "practice-003a-consumer",
    })


def main() -> None:
    consumer = create_consumer(config.EVENTS_GROUP)

    try:
        print(f"=== Consuming from '{config.EVENTS_TOPIC}' (group: {config.EVENTS_GROUP}) ===\n")
        events = consume_events(consumer, config.EVENTS_TOPIC)
        print(f"\nConsumed {len(events)} events total.")
    finally:
        # Always close the consumer to leave the group cleanly.
        # This triggers a rebalance so other group members pick up
        # the partitions this consumer was assigned.
        consumer.close()
        print("Consumer closed.")


if __name__ == "__main__":
    main()
