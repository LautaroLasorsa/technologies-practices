"""Kafka producer for the Event Log System.

Demonstrates:
  - Producing messages with keys and values (JSON serialized)
  - Delivery callbacks for async acknowledgment
  - Batching and flushing
  - The role of `acks` configuration

Run after admin.py has created the topics:
    uv run python producer.py
"""

import json
import time

from confluent_kafka import Producer, KafkaError

import config


# ── Sample data ──────────────────────────────────────────────────────

SAMPLE_EVENTS = [
    {"event_type": "user.signup",    "user_id": "user-001", "email": "alice@example.com"},
    {"event_type": "order.created",  "user_id": "user-002", "order_id": "ORD-100", "total": 59.99},
    {"event_type": "user.login",     "user_id": "user-001", "ip": "192.168.1.10"},
    {"event_type": "order.shipped",  "user_id": "user-002", "order_id": "ORD-100", "carrier": "FedEx"},
    {"event_type": "user.signup",    "user_id": "user-003", "email": "bob@example.com"},
    {"event_type": "order.created",  "user_id": "user-001", "order_id": "ORD-101", "total": 124.50},
    {"event_type": "user.login",     "user_id": "user-003", "ip": "10.0.0.5"},
    {"event_type": "order.created",  "user_id": "user-003", "order_id": "ORD-102", "total": 29.99},
    {"event_type": "order.shipped",  "user_id": "user-001", "order_id": "ORD-101", "carrier": "UPS"},
    {"event_type": "user.login",     "user_id": "user-002", "ip": "172.16.0.1"},
]


# ── Delivery callback (boilerplate) ─────────────────────────────────


def delivery_report(err: KafkaError | None, msg) -> None:
    """Called once per message to indicate delivery result.

    This callback is invoked by `producer.poll()` or `producer.flush()`.
    It runs in the producer's main thread, not a background thread.

    Args:
        err: None on success, a KafkaError on failure.
        msg: The Message object (has .topic(), .partition(), .offset(), .key(), .value()).
    """
    if err is not None:
        print(f"  FAILED: {err}")
    else:
        print(
            f"  Delivered: topic={msg.topic()} "
            f"partition={msg.partition()} "
            f"offset={msg.offset()} "
            f"key={msg.key()}"
        )


# ── TODO(human): Implement these functions ───────────────────────────


def produce_event(
    producer: Producer,
    topic: str,
    key: str,
    value: dict,
) -> None:
    """Produce a single event message to Kafka.

    TODO(human): Implement this function.

    Steps:
      1. Serialize `value` to JSON bytes:
             json.dumps(value).encode("utf-8")
      2. Encode `key` to UTF-8 bytes:
             key.encode("utf-8")
      3. Call producer.produce() with these arguments:
             producer.produce(
                 topic=topic,
                 key=<encoded key>,
                 value=<encoded value>,
                 callback=delivery_report,
             )
         The `key` determines which partition the message lands in.
         Messages with the same key always go to the same partition
         (consistent hashing), which guarantees ordering per key.
      4. Call producer.poll(0) to trigger delivery callbacks for any
         previously-produced messages that have been acknowledged.
         poll(0) is non-blocking -- it processes callbacks without waiting.

    Why poll(0)?
      produce() is asynchronous -- it queues the message in an internal
      buffer. The delivery_report callback only fires when poll() is
      called. Calling poll(0) after each produce() ensures callbacks
      are processed promptly instead of batching up.

    Docs: https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#confluent_kafka.Producer.produce
    """
    raise NotImplementedError("TODO(human): implement produce_event")


def produce_events_batch(
    producer: Producer,
    topic: str,
    events: list[dict],
    key_field: str = "user_id",
) -> int:
    """Produce a batch of events and flush to ensure all are delivered.

    TODO(human): Implement this function.

    Steps:
      1. Initialize a counter for successfully queued messages.
      2. Iterate over each event dict in `events`:
         a. Extract the key from event[key_field] (e.g., event["user_id"]).
         b. Add a "timestamp" field to the event: event["timestamp"] = time.time()
         c. Call produce_event(producer, topic, key, event).
         d. Increment the counter.
      3. After the loop, call producer.flush(timeout=10).
         flush() blocks until ALL messages in the buffer have been
         delivered (or the timeout expires). It returns the number
         of messages still in the queue (0 means all delivered).
         This is critical before shutting down -- without flush(),
         buffered messages would be lost.
      4. Print the flush result (number of messages remaining).
      5. Return the count of events that were queued.

    Key concept (acks):
      The producer's "acks" setting controls when the broker considers
      a message "committed":
        - acks=0  : Fire and forget. Producer doesn't wait. Fastest, but messages can be lost.
        - acks=1  : Leader broker writes to its log before responding. Fast, small risk of loss.
        - acks=all: All in-sync replicas must write before responding. Safest, slowest.
      For single-broker dev, acks=1 and acks=all behave identically.

    Docs: https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#confluent_kafka.Producer.flush
    """
    raise NotImplementedError("TODO(human): implement produce_events_batch")


# ── Orchestration (boilerplate) ──────────────────────────────────────


def create_producer() -> Producer:
    """Create a Kafka producer with recommended settings.

    Key configuration:
      - bootstrap.servers: Broker address(es) for initial connection.
      - acks: "all" for maximum durability (all replicas must confirm).
      - client.id: Identifies this producer in broker logs (debugging).
      - linger.ms: How long to wait before sending a batch (0 = send immediately).
    """
    return Producer({
        "bootstrap.servers": config.BOOTSTRAP_SERVERS,
        "acks": "all",
        "client.id": "practice-003a-producer",
        "linger.ms": 5,
    })


def main() -> None:
    producer = create_producer()

    print(f"=== Producing {len(SAMPLE_EVENTS)} events to '{config.EVENTS_TOPIC}' ===\n")
    count = produce_events_batch(producer, config.EVENTS_TOPIC, SAMPLE_EVENTS)
    print(f"\nDone. Queued {count} events.")


if __name__ == "__main__":
    main()
