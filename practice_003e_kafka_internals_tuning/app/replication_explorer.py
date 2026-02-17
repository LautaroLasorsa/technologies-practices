"""Explore Kafka replication internals: ISR, leader election, and min.insync.replicas.

Demonstrates:
  - Querying per-partition replica state (leader, replicas, ISR)
  - Monitoring ISR shrink/expand as brokers stop/start
  - The availability vs durability trade-off with min.insync.replicas + acks=all

Run after admin.py has created the topics:
    uv run python replication_explorer.py describe
    uv run python replication_explorer.py monitor
    uv run python replication_explorer.py min-isr

While monitoring, stop a broker in another terminal:
    docker compose stop kafka-internals-2
Then restart it:
    docker compose start kafka-internals-2
"""

import sys
import time

from confluent_kafka import KafkaError, KafkaException, Producer
from confluent_kafka.admin import AdminClient

import config


# -- Helpers (boilerplate) ----------------------------------------------------


def create_admin_client() -> AdminClient:
    """Create an AdminClient connected to the multi-broker cluster."""
    return AdminClient({"bootstrap.servers": config.BOOTSTRAP_SERVERS})


def print_replica_table(partition_info: list[dict]) -> None:
    """Print a formatted table of partition replica information.

    Args:
        partition_info: List of dicts with keys: partition, leader, replicas, isr.
    """
    header = f"{'Partition':>10} | {'Leader':>6} | {'Replicas':<20} | {'ISR':<20}"
    print(header)
    print("-" * len(header))
    for info in partition_info:
        replicas_str = ", ".join(str(r) for r in info["replicas"])
        isr_str = ", ".join(str(r) for r in info["isr"])
        print(
            f"{info['partition']:>10} | "
            f"{info['leader']:>6} | "
            f"{replicas_str:<20} | "
            f"{isr_str:<20}"
        )


# -- TODO(human): Implement these functions -----------------------------------


def describe_topic_replicas(admin_client: AdminClient, topic: str) -> list[dict]:
    """Query the cluster for per-partition replica state of a topic.

    TODO(human): Implement this function.

    Background -- what ISR, replicas, and leader mean:
      Every partition has a set of **replicas** (broker IDs that store a copy).
      One replica is the **leader** -- it handles all reads and writes for that
      partition. The other replicas are **followers** that fetch data from the
      leader. The **ISR (In-Sync Replica set)** is the subset of replicas that
      are "caught up" with the leader (within replica.lag.time.max.ms, default
      30 seconds). If a follower falls behind, it is removed from the ISR.

    Steps:
      1. Call admin_client.describe_topics([topic]) to get topic metadata.
         This returns a dict {topic_name: Future}. Call .result() on the future
         to get a TopicDescription object.
      2. Access the TopicDescription's .partitions attribute -- this is a list
         of TopicPartitionInfo objects, one per partition.
      3. For each TopicPartitionInfo, extract:
         - .partition (int): the partition ID
         - .leader (Node): the leader broker -- use .leader.id for the broker ID
         - .replicas (list[Node]): all replicas -- extract each .id
         - .isr (list[Node]): in-sync replicas -- extract each .id
      4. Build a list of dicts: [{"partition": int, "leader": int,
         "replicas": [int, ...], "isr": [int, ...]}, ...]
      5. Call print_replica_table() with the result to display it.
      6. Return the list of dicts.

    Docs: https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#confluent_kafka.admin.AdminClient.describe_topics

    Returns:
        List of dicts with per-partition replica information.
    """
    raise NotImplementedError("TODO(human)")


def monitor_isr_changes(
    admin_client: AdminClient, topic: str, interval: float = 2.0
) -> None:
    """Poll the cluster and log ISR changes as they happen.

    TODO(human): Implement this function.

    Background -- ISR dynamics during broker failure and recovery:
      When a broker stops, the controller detects the failure (via missed
      heartbeats) and removes that broker from the ISR of every partition it
      was part of. The ISR **shrinks**. When the broker restarts, it begins
      fetching from the leader to catch up. Once it is within
      replica.lag.time.max.ms of the leader's LEO (Log End Offset), the
      controller adds it back to the ISR -- the ISR **expands**.

      Key concepts:
        - **LEO (Log End Offset)**: The offset of the next message to be
          written. Each replica tracks its own LEO.
        - **HW (High Watermark)**: The offset up to which ALL ISR replicas
          have replicated. Consumers can only read up to the HW. When the
          ISR shrinks, the HW can advance faster (fewer replicas to wait for).
        - **Leader Epoch**: Incremented each time a new leader is elected.
          Followers use this to detect if they have divergent logs after a
          leader change (fencing stale leaders).

    Steps:
      1. Call describe_topic_replicas() to get the initial ISR state.
         Store it as "previous" state (e.g., a dict mapping partition -> ISR set).
      2. Enter an infinite loop (Ctrl+C to stop):
         a. Sleep for `interval` seconds.
         b. Call describe_topic_replicas() again to get current state.
         c. Compare current ISR sets with previous ISR sets per partition.
         d. If any partition's ISR changed, print a timestamped log line showing:
            - Which partition changed
            - Old ISR -> New ISR
            - Whether the ISR shrank (broker left) or expanded (broker rejoined)
         e. Update the previous state.
      3. Wrap in try/except KeyboardInterrupt for clean exit.

    Experiment:
      Run this in one terminal, then in another terminal:
        docker compose stop kafka-internals-2   # ISR shrinks (3 -> 2)
        docker compose start kafka-internals-2  # ISR expands (2 -> 3)

    Args:
        admin_client: AdminClient connected to the cluster.
        topic: Topic name to monitor.
        interval: Seconds between polls.
    """
    raise NotImplementedError("TODO(human)")


def demonstrate_min_isr_rejection(topic: str) -> None:
    """Show that acks=all + min.insync.replicas rejects writes when ISR is too small.

    TODO(human): Implement this function.

    Background -- the acks + min.insync.replicas interaction:
      When a producer uses acks=all, the leader waits for ALL ISR replicas
      to acknowledge the write before responding to the producer. The topic-level
      setting min.insync.replicas (default 1) sets the MINIMUM ISR size required
      for the leader to accept writes. If ISR < min.insync.replicas, the broker
      rejects the produce request with NotEnoughReplicasException.

      This is the core trade-off:
        - High min.insync.replicas (e.g., 2 of 3) = MORE durable (data survives
          N-1 broker failures) but LESS available (need >= min ISR brokers up)
        - Low min.insync.replicas (e.g., 1) = MORE available (only need 1 broker)
          but LESS durable (data can be lost if leader dies before replication)

    Steps:
      1. Create a Producer with acks=all and a short message.timeout.ms (e.g., 10000)
         so you don't wait forever on failure.
      2. Produce a test message and flush(). Verify it succeeds (all 3 brokers up).
      3. Print instructions asking the user to stop 2 brokers:
             docker compose stop kafka-internals-2 kafka-internals-3
      4. Wait for user input (input("Press Enter after stopping 2 brokers...")).
      5. Try to produce another message with flush().
         This should FAIL because ISR < min.insync.replicas (1 < 2).
      6. Catch the delivery error via a callback or check the flush return value.
         Print the error (should be NotEnoughReplicas or similar).
      7. Print instructions to restart the brokers:
             docker compose start kafka-internals-2 kafka-internals-3

    Args:
        topic: Topic configured with min.insync.replicas=2.
    """
    raise NotImplementedError("TODO(human)")


# -- Orchestration (boilerplate) ----------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python replication_explorer.py <command>")
        print("Commands: describe, monitor, min-isr")
        sys.exit(1)

    command = sys.argv[1]
    admin = create_admin_client()
    topic = config.REPLICATION_TOPIC

    if command == "describe":
        print(f"=== Replica state for '{topic}' ===\n")
        describe_topic_replicas(admin, topic)

    elif command == "monitor":
        print(f"=== Monitoring ISR changes for '{topic}' (Ctrl+C to stop) ===\n")
        monitor_isr_changes(admin, topic)

    elif command == "min-isr":
        print(f"=== min.insync.replicas rejection demo on '{topic}' ===\n")
        demonstrate_min_isr_rejection(topic)

    else:
        print(f"Unknown command: {command}")
        print("Commands: describe, monitor, min-isr")
        sys.exit(1)


if __name__ == "__main__":
    main()
