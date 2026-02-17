"""Consumer configuration experiments: assignment strategies, static membership, and tuning.

Demonstrates:
  - Benchmark different consumer configurations (fetch sizes, poll intervals)
  - Assignment strategies: Range, RoundRobin, CooperativeSticky
  - Static group membership to avoid unnecessary rebalances

Run after seeding the consumer-tuning topic with data:
    uv run python consumer_tuning.py benchmark
    uv run python consumer_tuning.py strategies
    uv run python consumer_tuning.py static-membership
"""

import json
import sys
import time
import uuid

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer

import config


# -- Helpers (boilerplate) ----------------------------------------------------


def seed_topic(topic: str, num_messages: int = 10000) -> None:
    """Produce messages to a topic for consumer benchmarks.

    This is a helper that quickly fills a topic with test data.
    """
    producer = Producer({
        "bootstrap.servers": config.BOOTSTRAP_SERVERS,
        "linger.ms": 50,
        "batch.size": 131072,
        "compression.type": "lz4",
    })

    for i in range(num_messages):
        msg = json.dumps({"seq": i, "ts": time.time(), "data": f"payload-{i}"})
        producer.produce(
            topic=topic,
            key=f"key-{i % 100}".encode("utf-8"),
            value=msg.encode("utf-8"),
        )
        if i % 1000 == 0:
            producer.poll(0)

    remaining = producer.flush(timeout=30)
    print(f"  Seeded {num_messages} messages to '{topic}' (remaining: {remaining})")


def _on_assign(consumer, partitions) -> None:
    """Rebalance callback: log partition assignments."""
    parts = [f"{p.topic}[{p.partition}]" for p in partitions]
    print(f"  Assigned: {', '.join(parts)}")


def _on_revoke(consumer, partitions) -> None:
    """Rebalance callback: log partition revocations."""
    parts = [f"{p.topic}[{p.partition}]" for p in partitions]
    print(f"  Revoked: {', '.join(parts)}")


# -- TODO(human): Implement these functions -----------------------------------


def benchmark_consumer_config(
    config_overrides: dict, topic: str, max_messages: int
) -> dict:
    """Benchmark a consumer with specific configuration overrides.

    TODO(human): Implement this function.

    Background -- key consumer tuning parameters:
      - fetch.min.bytes (default 1): Minimum data the broker should return
        per fetch request. Setting higher (e.g., 1024, 10240) makes the broker
        wait until it has enough data, reducing the number of fetch requests
        but increasing latency.
      - fetch.max.bytes (default 52428800 = 50 MB): Maximum data per fetch.
        Increase if messages are large.
      - max.poll.interval.ms (default 300000 = 5 min): Maximum time between
        poll() calls before the consumer is considered dead and removed from
        the group. Set lower for faster failure detection, but be careful not
        to set it shorter than your processing time.
      - max.partition.fetch.bytes (default 1048576 = 1 MB): Maximum bytes
        fetched per partition per request. Controls memory usage.
      - session.timeout.ms (default 45000): How long the broker waits for
        heartbeats before declaring the consumer dead.

    Steps:
      1. Build a base consumer config dict with:
         - "bootstrap.servers": config.BOOTSTRAP_SERVERS
         - "group.id": f"bench-{uuid.uuid4().hex[:8]}" (unique per run to avoid offset conflicts)
         - "auto.offset.reset": "earliest"
         - "enable.auto.commit": True
      2. Merge config_overrides into the base config (overrides take priority).
      3. Create a Consumer with the merged config.
      4. Subscribe to the topic.
      5. Record start time.
      6. Consume up to max_messages. Use poll(1.0). Track empty polls (break
         after 10 consecutive Nones to avoid hanging).
      7. Record end time. Close the consumer.
      8. Calculate elapsed and throughput (messages/sec).
      9. Return: {"config": config_overrides, "messages_consumed": int,
         "elapsed_s": float, "throughput_msg_s": float}

    Args:
        config_overrides: Dict of consumer config keys to override.
        topic: Topic to consume from.
        max_messages: Maximum messages to consume.

    Returns:
        Dict with benchmark results.
    """
    raise NotImplementedError("TODO(human)")


def compare_assignment_strategies(topic: str) -> None:
    """Create consumers with different assignment strategies and observe partition assignments.

    TODO(human): Implement this function.

    Background -- partition assignment strategies:
      When consumers join a group, the group coordinator (a broker) triggers
      a rebalance. The **assignment strategy** determines how partitions are
      distributed among consumers. confluent-kafka supports:

      1. **RangeAssignor** (default): Assigns partitions of each topic in
         contiguous ranges. Consumer 0 gets partitions 0-2, consumer 1 gets
         3-5, etc. Simple but can cause uneven distribution across topics.

      2. **RoundRobinAssignor**: Distributes partitions one-by-one in round-
         robin order. More even than Range when consuming multiple topics.

      3. **CooperativeStickyAssignor**: An "incremental cooperative" strategy.
         Unlike Range/RoundRobin which do EAGER rebalancing (revoke ALL
         partitions from ALL consumers, then reassign), CooperativeSticky
         only revokes the partitions that need to move. This means consumers
         that keep their partitions continue consuming during rebalance --
         no global stop-the-world pause.

      Eager vs Cooperative rebalancing:
        - Eager (Range, RoundRobin): ALL partitions revoked -> ALL reassigned.
          Simple but causes a brief consumption pause for the entire group.
        - Cooperative (CooperativeSticky): Only moved partitions are revoked.
          Remaining consumers keep consuming. Better for large groups.

    Steps:
      1. Use a shared group.id (e.g., "strategy-demo-group") so all consumers
         join the same group.
      2. For each strategy in ["range", "roundrobin", "cooperative-sticky"]:
         a. Create a Consumer with:
            - "bootstrap.servers": config.BOOTSTRAP_SERVERS
            - "group.id": f"strategy-demo-{strategy}" (separate group per strategy)
            - "partition.assignment.strategy": strategy
            - "auto.offset.reset": "earliest"
            - "client.id": f"consumer-{strategy}"
         b. Subscribe to the topic with on_assign=_on_assign callback.
         c. Poll a few times (e.g., 10 iterations with poll(2.0)) to trigger
            the rebalance and observe which partitions are assigned.
         d. Print the strategy name and the assigned partitions.
         e. Close the consumer.
      3. After all strategies, print a summary comparing the assignment patterns.

    Note: With a single consumer per group, all 6 partitions go to that one
    consumer regardless of strategy. The difference is visible when multiple
    consumers are in the same group. The main observation here is the
    REBALANCE PROTOCOL (eager vs cooperative) seen in the logs.

    Args:
        topic: Topic with 6 partitions (consumer-tuning).
    """
    raise NotImplementedError("TODO(human)")


def demonstrate_static_membership(topic: str) -> None:
    """Show that static group membership avoids rebalances on brief disconnects.

    TODO(human): Implement this function.

    Background -- static group membership (KIP-345):
      By default, when a consumer disconnects (even briefly), the group
      coordinator triggers a rebalance to redistribute its partitions. This
      causes a consumption pause for the entire group. With **static
      membership** (group.instance.id), the consumer gets a stable identity.
      When it disconnects, the coordinator waits for session.timeout.ms
      before triggering a rebalance. If the consumer reconnects within that
      window with the same group.instance.id, it gets its old partitions
      back with NO rebalance.

      This is critical for:
        - Rolling deployments (container restarts)
        - Transient network issues
        - Kubernetes pod rescheduling

    Steps:
      1. Create a Consumer with:
         - "bootstrap.servers": config.BOOTSTRAP_SERVERS
         - "group.id": "static-membership-demo"
         - "group.instance.id": "static-consumer-1"
         - "session.timeout.ms": 30000  (30 seconds grace period)
         - "auto.offset.reset": "earliest"
      2. Subscribe to the topic with on_assign and on_revoke callbacks.
      3. Poll a few messages to establish group membership and get partitions.
      4. Print the assigned partitions.
      5. Close the consumer (simulating a brief disconnect).
      6. Print "Consumer closed. Reconnecting within session.timeout.ms..."
      7. Wait 5 seconds (well within the 30s session timeout).
      8. Create a NEW Consumer with the SAME group.instance.id and group.id.
      9. Subscribe and poll to rejoin.
      10. Print the assigned partitions -- they should be the SAME as before,
          and the on_revoke callback should NOT have fired (no rebalance).
      11. Close the consumer.
      12. Print summary: "Static membership preserved partition assignment
          across reconnect -- no rebalance triggered."

    Args:
        topic: Topic to subscribe to.
    """
    raise NotImplementedError("TODO(human)")


# -- Orchestration (boilerplate) ----------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python consumer_tuning.py <command>")
        print("Commands: benchmark, strategies, static-membership, seed")
        sys.exit(1)

    command = sys.argv[1]
    topic = config.CONSUMER_TUNING_TOPIC

    if command == "seed":
        print(f"=== Seeding '{topic}' with test data ===\n")
        seed_topic(topic, num_messages=50000)

    elif command == "benchmark":
        print(f"=== Consumer Config Benchmark on '{topic}' ===\n")
        print("Seeding topic first...")
        seed_topic(topic, num_messages=50000)

        configs = [
            {"label": "defaults", "overrides": {}},
            {"label": "fetch.min.bytes=10KB", "overrides": {"fetch.min.bytes": 10240}},
            {"label": "fetch.min.bytes=100KB", "overrides": {"fetch.min.bytes": 102400}},
            {"label": "max.partition.fetch.bytes=256KB", "overrides": {"max.partition.fetch.bytes": 262144}},
            {"label": "max.partition.fetch.bytes=2MB", "overrides": {"max.partition.fetch.bytes": 2097152}},
        ]

        results = []
        for cfg in configs:
            print(f"\nBenchmarking: {cfg['label']}...")
            result = benchmark_consumer_config(cfg["overrides"], topic, max_messages=50000)
            result["label"] = cfg["label"]
            results.append(result)

        # Print comparison table
        print(f"\n{'Config':<40} | {'Messages':>10} | {'Elapsed (s)':>12} | {'Throughput (msg/s)':>18}")
        print("-" * 90)
        for r in results:
            print(
                f"{r['label']:<40} | "
                f"{r['messages_consumed']:>10} | "
                f"{r['elapsed_s']:>12.2f} | "
                f"{r['throughput_msg_s']:>18.1f}"
            )

    elif command == "strategies":
        print(f"=== Assignment Strategy Comparison on '{topic}' ===\n")
        compare_assignment_strategies(topic)

    elif command == "static-membership":
        print(f"=== Static Membership Demo on '{topic}' ===\n")
        demonstrate_static_membership(topic)

    else:
        print(f"Unknown command: {command}")
        print("Commands: benchmark, strategies, static-membership, seed")
        sys.exit(1)


if __name__ == "__main__":
    main()
