"""Idempotent and transactional producers: exactly-once semantics in Kafka.

Demonstrates:
  - Idempotent producer configuration (enable.idempotence=true)
  - How PID + sequence numbers prevent duplicates
  - Transactional producer with begin/commit/abort
  - Atomic multi-partition writes

Run after admin.py has created the topics:
    uv run python idempotent_producer.py idempotent
    uv run python idempotent_producer.py transactional
"""

import json
import sys
import time

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer

import config


# -- Helpers (boilerplate) ----------------------------------------------------


def delivery_report(err: KafkaError | None, msg) -> None:
    """Called once per message to indicate delivery result."""
    if err is not None:
        print(f"  FAILED: {err}")
    else:
        print(
            f"  Delivered: topic={msg.topic()} "
            f"partition={msg.partition()} "
            f"offset={msg.offset()} "
            f"key={msg.key()}"
        )


def consume_and_count(topic: str, group_id: str, timeout: float = 15.0) -> int:
    """Consume all available messages from a topic and return the count.

    Used to verify idempotence and transactional behavior by counting
    how many messages actually landed in the topic.
    """
    consumer = Consumer(
        {
            "bootstrap.servers": config.BOOTSTRAP_SERVERS,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe([topic])

    count = 0
    empty_polls = 0
    start = time.time()

    while time.time() - start < timeout:
        msg = consumer.poll(1.0)
        if msg is None:
            empty_polls += 1
            if empty_polls > 5:
                break
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            raise KafkaException(msg.error())
        empty_polls = 0
        count += 1
        value = msg.value()
        if value:
            print(f"    [{count}] partition={msg.partition()} offset={msg.offset()} "
                  f"value={value.decode('utf-8')[:80]}")

    consumer.close()
    return count


# -- TODO(human): Implement these functions -----------------------------------


def create_idempotent_producer() -> Producer:
    """Create a Kafka producer with idempotence enabled.

    TODO(human): Implement this function.

    Background -- how idempotent producers work:
      When you set enable.idempotence=true, the producer is assigned a unique
      **PID (Producer ID)** by the broker on initialization. Each message the
      producer sends includes the PID and a **sequence number** (per partition).
      The broker tracks the last sequence number per PID per partition. If it
      receives a message with a sequence number it has already seen, it
      discards the duplicate and returns success (so the producer doesn't retry).

      Enabling idempotence automatically sets:
        - acks=all (all ISR replicas must acknowledge)
        - max.in.flight.requests.per.connection=5 (max concurrent requests)
        - retries=MAX_INT (retry indefinitely on transient errors)
      These settings together guarantee that messages are written exactly once
      to a single partition, even if the producer retries due to network errors.

    Steps:
      1. Create and return a Producer with these settings:
         - "bootstrap.servers": config.BOOTSTRAP_SERVERS
         - "enable.idempotence": True
         - "client.id": "practice-003e-idempotent"
         Note: You do NOT need to set acks, retries, or max.in.flight
         explicitly -- they are forced by enable.idempotence=True.

    Docs: https://docs.confluent.io/platform/current/clients/producer.html#idempotent-delivery

    Returns:
        A configured idempotent Producer instance.
    """
    raise NotImplementedError("TODO(human)")


def demonstrate_idempotence(producer: Producer, topic: str) -> None:
    """Produce messages and verify no duplicates land in the topic.

    TODO(human): Implement this function.

    Background -- what idempotence protects against:
      Without idempotence, a retry after a network timeout can cause the same
      message to be written twice (the broker received and wrote it, but the
      ack was lost, so the producer retries). With idempotence, the broker
      de-duplicates using PID + sequence number. Note: idempotence is per-
      partition, per-producer-session. It does NOT deduplicate across restarts
      or across multiple producers. For cross-session exactly-once, you need
      transactions (see create_transactional_producer below).

    Steps:
      1. Produce 10 messages to the topic. Use sequential keys like "key-0",
         "key-1", ... and JSON values like {"seq": 0, "data": "message-0"}.
         Use the delivery_report callback.
      2. Call producer.flush() to ensure all are delivered.
      3. Print how many messages were produced.
      4. Call consume_and_count(topic, "idempotent-verify-group") to count
         messages in the topic.
      5. Assert/verify that the count matches the number produced (10).
         Print the result: "Produced: 10, Consumed: 10 -- no duplicates".

    Args:
        producer: An idempotent producer (from create_idempotent_producer).
        topic: Topic to produce to.
    """
    raise NotImplementedError("TODO(human)")


def create_transactional_producer(transactional_id: str) -> Producer:
    """Create a Kafka producer with transactional support.

    TODO(human): Implement this function.

    Background -- transactional producers and exactly-once:
      A transactional producer wraps a batch of produce calls in an atomic
      transaction. Either ALL messages in the transaction are committed
      (visible to consumers with isolation.level=read_committed) or NONE
      are (if the transaction is aborted). This enables exactly-once
      semantics across multiple partitions and topics.

      The transactional.id is a unique string that identifies this producer
      across restarts. If a producer with the same transactional.id restarts,
      Kafka **fences** the old producer (any in-flight transactions from the
      old instance are aborted). This prevents zombie producers from causing
      duplicates.

    Steps:
      1. Create a Producer with these settings:
         - "bootstrap.servers": config.BOOTSTRAP_SERVERS
         - "transactional.id": transactional_id
         - "client.id": "practice-003e-transactional"
         Note: Setting transactional.id automatically enables idempotence.
      2. Call producer.init_transactions(timeout=10.0) to register this
         transactional producer with the transaction coordinator.
         This MUST be called before any begin_transaction/commit/abort.
      3. Return the producer.

    Docs: https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#confluent_kafka.Producer.init_transactions

    Returns:
        A configured transactional Producer instance, ready for begin_transaction().
    """
    raise NotImplementedError("TODO(human)")


def atomic_produce(
    producer: Producer, messages: list[tuple[str, str, str]]
) -> None:
    """Produce a batch of messages atomically within a single transaction.

    TODO(human): Implement this function.

    Background -- transaction lifecycle:
      A transaction has three phases:
        1. begin_transaction() -- marks the start of a new atomic batch
        2. produce() calls -- all messages are buffered and associated with
           the current transaction
        3. commit_transaction() -- atomically makes ALL messages visible to
           consumers (with isolation.level=read_committed)
      If anything fails between begin and commit, you MUST call
      abort_transaction() to roll back. Failing to abort leaves the
      transaction in a zombie state until it times out.

    Steps:
      1. Call producer.begin_transaction().
      2. Wrap the produce loop in a try/except:
         a. For each (topic, key, value) in messages:
            - Call producer.produce(topic=topic, key=key.encode("utf-8"),
              value=value.encode("utf-8"))
         b. Call producer.commit_transaction() to atomically commit all messages.
         c. Print "Transaction committed: {len(messages)} messages".
      3. In the except clause:
         a. Print the error.
         b. Call producer.abort_transaction() to roll back.
         c. Print "Transaction aborted".

    Args:
        producer: A transactional Producer (from create_transactional_producer).
        messages: List of (topic, key, value) tuples to produce atomically.
    """
    raise NotImplementedError("TODO(human)")


# -- Orchestration (boilerplate) ----------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python idempotent_producer.py <command>")
        print("Commands: idempotent, transactional")
        sys.exit(1)

    command = sys.argv[1]
    topic = config.TRANSACTION_TOPIC

    if command == "idempotent":
        print("=== Idempotent producer demo ===\n")
        producer = create_idempotent_producer()
        demonstrate_idempotence(producer, topic)

    elif command == "transactional":
        print("=== Transactional producer demo ===\n")
        producer = create_transactional_producer("practice-003e-txn-01")

        # Batch 1: should commit successfully
        batch_1 = [
            (topic, "order-1", json.dumps({"action": "create", "order_id": "ORD-001", "amount": 100})),
            (topic, "order-2", json.dumps({"action": "create", "order_id": "ORD-002", "amount": 250})),
            (topic, "order-1", json.dumps({"action": "confirm", "order_id": "ORD-001"})),
        ]
        print("--- Batch 1: atomic commit ---")
        atomic_produce(producer, batch_1)

        # Batch 2: another successful commit
        batch_2 = [
            (topic, "order-3", json.dumps({"action": "create", "order_id": "ORD-003", "amount": 75})),
            (topic, "order-3", json.dumps({"action": "ship", "order_id": "ORD-003"})),
        ]
        print("\n--- Batch 2: atomic commit ---")
        atomic_produce(producer, batch_2)

        # Count total messages in topic
        print("\n--- Verifying messages in topic ---")
        count = consume_and_count(topic, "txn-verify-group")
        print(f"\nTotal messages in topic: {count}")

    else:
        print(f"Unknown command: {command}")
        print("Commands: idempotent, transactional")
        sys.exit(1)


if __name__ == "__main__":
    main()
