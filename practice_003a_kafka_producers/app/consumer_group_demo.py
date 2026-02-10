"""Consumer group rebalancing demo.

Demonstrates:
  - Multiple consumers in the same group sharing partitions
  - Rebalance callbacks (on_assign / on_revoke)
  - How adding/removing consumers triggers partition redistribution

Run after producing messages to the orders topic:
    uv run python consumer_group_demo.py

This script uses threading to simulate multiple consumers in one process.
In production, each consumer would be a separate process or container.
"""

import json
import signal
import sys
import threading
import time

from confluent_kafka import Consumer, KafkaError, KafkaException, TopicPartition

import config


# ── Shared state ─────────────────────────────────────────────────────

_running = True


def _signal_handler(signum, frame) -> None:
    global _running
    print("\nShutdown requested...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)


# ── Rebalance callbacks (boilerplate) ────────────────────────────────


def make_rebalance_callbacks(worker_name: str):
    """Create on_assign and on_revoke callbacks for a named worker.

    These callbacks are invoked by the consumer group coordinator during
    a rebalance. A rebalance happens when:
      - A new consumer joins the group
      - An existing consumer leaves (close() or crash/timeout)
      - Topic partitions change (rare in practice)

    The protocol:
      1. on_revoke: All consumers release their current partitions
      2. Coordinator recalculates partition assignments
      3. on_assign: Each consumer receives its new partition set
    """

    def on_assign(consumer: Consumer, partitions: list[TopicPartition]) -> None:
        partition_ids = [p.partition for p in partitions]
        print(f"  [{worker_name}] ASSIGNED partitions: {partition_ids}")

    def on_revoke(consumer: Consumer, partitions: list[TopicPartition]) -> None:
        partition_ids = [p.partition for p in partitions]
        print(f"  [{worker_name}] REVOKED partitions: {partition_ids}")

    return on_assign, on_revoke


# ── TODO(human): Implement these functions ───────────────────────────


def run_consumer_worker(
    worker_name: str,
    group_id: str,
    topic: str,
    duration_seconds: float = 30.0,
) -> list[dict]:
    """Run a single consumer worker that processes messages for a given duration.

    TODO(human): Implement this function.

    This function creates a consumer, subscribes to the topic with rebalance
    callbacks, and polls for messages until the duration expires or shutdown
    is requested.

    Steps:
      1. Create a Consumer with this configuration:
             {
                 "bootstrap.servers": config.BOOTSTRAP_SERVERS,
                 "group.id": group_id,
                 "auto.offset.reset": "earliest",
                 "enable.auto.commit": False,
                 "client.id": worker_name,
                 "session.timeout.ms": 10000,
                 "heartbeat.interval.ms": 3000,
             }
         session.timeout.ms: If the broker doesn't receive a heartbeat from
         this consumer within 10s, it's considered dead and a rebalance starts.
         heartbeat.interval.ms: How often the consumer sends heartbeats.
         Rule of thumb: heartbeat = session_timeout / 3.

      2. Create rebalance callbacks using make_rebalance_callbacks(worker_name).

      3. Subscribe to the topic WITH the callbacks:
             consumer.subscribe([topic], on_assign=on_assign, on_revoke=on_revoke)

      4. Record the start time: start = time.time()

      5. Create an empty list for collected events.

      6. Enter a while loop that runs while:
         - _running is True
         - (time.time() - start) < duration_seconds

         a. Call consumer.poll(timeout=1.0).

         b. If None, continue (no message available).

         c. If message has an error:
            - _PARTITION_EOF: print end-of-partition and continue
            - Otherwise: print the error and break

         d. Decode key and value (same as in consumer.py).

         e. Print: [{worker_name}] partition=X offset=Y key=K event_type=...

         f. Append decoded value to results list.

         g. Commit synchronously: consumer.commit(message=msg, asynchronous=False)

      7. In a finally block, call consumer.close() and print that the worker stopped.
         close() is critical: it tells the group coordinator this consumer is
         leaving, which triggers an immediate rebalance. Without close(), the
         group must wait for session.timeout.ms before detecting the departure.

      8. Return the collected events list.

    Docs: https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#confluent_kafka.Consumer.subscribe
    """
    raise NotImplementedError("TODO(human): implement run_consumer_worker")


def demonstrate_rebalancing(topic: str, group_id: str) -> None:
    """Launch consumers at staggered intervals to observe rebalancing.

    TODO(human): Implement this function.

    This demonstrates how Kafka redistributes partitions when consumers
    join or leave a group. You'll see on_assign/on_revoke callbacks fire.

    Steps:
      1. Print a header explaining what will happen.

      2. Start Worker-1 in a thread (duration=30s):
             t1 = threading.Thread(
                 target=run_consumer_worker,
                 args=("Worker-1", group_id, topic, 30.0),
             )
             t1.start()
         Worker-1 initially gets ALL partitions (it's the only consumer).

      3. Sleep for 8 seconds. This gives Worker-1 time to join the group
         and start consuming before the next consumer arrives.

      4. Print that Worker-2 is joining, then start Worker-2 in a thread (duration=20s).
         When Worker-2 joins, the coordinator triggers a rebalance:
           - on_revoke fires on Worker-1 (gives up some partitions)
           - on_assign fires on both workers (each gets a subset)
         With 4 partitions and 2 consumers, expect ~2 partitions each.

      5. Sleep for 8 seconds.

      6. Print that Worker-3 is joining, then start Worker-3 in a thread (duration=10s).
         Another rebalance: 4 partitions / 3 consumers -> roughly 1-2 each.

      7. Join all threads (t1.join(), t2.join(), t3.join()).
         As workers finish (Worker-3 first, then 2, then 1), each close()
         triggers a rebalance where remaining consumers pick up the freed
         partitions.

      8. Print summary.

    What to observe:
      - Worker-1 starts with all 4 partitions
      - When Worker-2 joins: partitions are split ~2/2
      - When Worker-3 joins: partitions are split ~2/1/1 or ~1/1/2
      - When Worker-3 leaves: rebalance back to ~2/2
      - When Worker-2 leaves: Worker-1 gets all 4 again

    Docs: https://developer.confluent.io/courses/architecture/consumer-group-protocol/
    """
    raise NotImplementedError("TODO(human): implement demonstrate_rebalancing")


# ── Seed data producer (boilerplate) ─────────────────────────────────


def produce_seed_data(topic: str, num_messages: int = 40) -> None:
    """Produce test messages to the orders topic for the demo.

    Generates messages with different user keys so they spread across
    partitions. This is boilerplate -- the learning is in the consumer side.
    """
    from confluent_kafka import Producer

    producer = Producer({
        "bootstrap.servers": config.BOOTSTRAP_SERVERS,
        "acks": "all",
        "client.id": "group-demo-seeder",
    })

    users = ["user-001", "user-002", "user-003", "user-004", "user-005"]
    order_types = ["created", "paid", "shipped", "delivered"]

    for i in range(num_messages):
        user = users[i % len(users)]
        order_type = order_types[i % len(order_types)]
        event = {
            "event_type": f"order.{order_type}",
            "user_id": user,
            "order_id": f"ORD-{i:04d}",
            "timestamp": time.time(),
        }
        producer.produce(
            topic=topic,
            key=user.encode("utf-8"),
            value=json.dumps(event).encode("utf-8"),
        )
        producer.poll(0)

    remaining = producer.flush(timeout=10)
    print(f"Seeded {num_messages} messages to '{topic}' ({remaining} unsent).\n")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    print("=== Consumer Group Rebalancing Demo ===\n")
    print(f"Topic: {config.ORDERS_TOPIC} ({config.ORDERS_PARTITIONS} partitions)")
    print(f"Group: {config.ORDERS_GROUP}\n")

    # Seed the topic with messages first
    produce_seed_data(config.ORDERS_TOPIC, num_messages=40)

    # Run the rebalancing demonstration
    demonstrate_rebalancing(config.ORDERS_TOPIC, config.ORDERS_GROUP)


if __name__ == "__main__":
    main()
