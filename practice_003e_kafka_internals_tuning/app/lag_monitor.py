"""Consumer lag monitoring: track how far behind consumers are.

Demonstrates:
  - Computing consumer lag per partition using AdminClient + Consumer watermarks
  - Continuous lag monitoring with tabular output
  - Understanding the relationship between committed offsets and log-end offsets

Run with a group ID to monitor:
    uv run python lag_monitor.py <group-id>

Example:
    uv run python lag_monitor.py consumer-tuning-group
"""

import sys
import time

from confluent_kafka import Consumer, KafkaError, TopicPartition
from confluent_kafka.admin import AdminClient

import config


# -- Helpers (boilerplate) ----------------------------------------------------


def create_admin_client() -> AdminClient:
    """Create an AdminClient connected to the multi-broker cluster."""
    return AdminClient({"bootstrap.servers": config.BOOTSTRAP_SERVERS})


def print_lag_table(lag_data: dict[str, int], group_id: str) -> None:
    """Print a formatted consumer lag table.

    Args:
        lag_data: Dict mapping "topic[partition]" to lag (int).
        group_id: Consumer group ID being monitored.
    """
    total_lag = sum(lag_data.values())
    timestamp = time.strftime("%H:%M:%S")

    print(f"\n[{timestamp}] Consumer group: {group_id}  |  Total lag: {total_lag}")
    print(f"{'Topic-Partition':<35} | {'Committed':>10} | {'Log-End':>10} | {'Lag':>10}")
    print("-" * 75)
    for tp_str, lag in sorted(lag_data.items()):
        # lag_data stores just lag; for full display we'd need offsets too
        print(f"{tp_str:<35} | {'--':>10} | {'--':>10} | {lag:>10}")


def print_full_lag_table(
    lag_details: list[dict], group_id: str
) -> None:
    """Print a detailed consumer lag table with committed and log-end offsets.

    Args:
        lag_details: List of dicts with keys: topic_partition, committed, log_end, lag.
        group_id: Consumer group ID being monitored.
    """
    total_lag = sum(d["lag"] for d in lag_details)
    timestamp = time.strftime("%H:%M:%S")

    print(f"\n[{timestamp}] Consumer group: {group_id}  |  Total lag: {total_lag}")
    print(f"{'Topic-Partition':<35} | {'Committed':>10} | {'Log-End':>10} | {'Lag':>10}")
    print("-" * 75)
    for d in sorted(lag_details, key=lambda x: x["topic_partition"]):
        committed_str = str(d["committed"]) if d["committed"] >= 0 else "none"
        print(
            f"{d['topic_partition']:<35} | "
            f"{committed_str:>10} | "
            f"{d['log_end']:>10} | "
            f"{d['lag']:>10}"
        )


# -- TODO(human): Implement these functions -----------------------------------


def get_consumer_lag(admin_client: AdminClient, group_id: str) -> list[dict]:
    """Compute consumer lag per partition for a consumer group.

    TODO(human): Implement this function.

    Background -- what consumer lag means:
      Consumer lag is the difference between the **log-end offset** (LEO) of
      a partition and the **committed offset** of the consumer group for that
      partition. It tells you how many messages the consumer has NOT yet
      processed.

      - **Log-End Offset (LEO)**: The offset of the NEXT message to be written
        to the partition. If the partition has offsets 0-99, LEO is 100.
      - **Committed Offset**: The last offset the consumer group has committed
        (acknowledged processing for). If committed=90, the consumer has
        processed messages 0-89.
      - **Lag = LEO - Committed**: Number of unprocessed messages. Lag=10
        means 10 messages are waiting to be consumed.

      Monitoring lag is essential in production:
        - Increasing lag = consumers can't keep up with producers
        - Sustained high lag = need more consumers or faster processing
        - Lag spikes after a deploy = possible regression in consumer code

    Steps:
      1. Use admin_client.list_consumer_group_offsets() to get committed offsets.
         This takes a list of [ConsumerGroupTopicPartitions(group_id)] and returns
         a dict {group_id: Future}. Call .result() to get a ConsumerGroupTopicPartitions
         object. Access .topic_partitions for a list of TopicPartition objects,
         each with .topic, .partition, and .offset (committed offset).
      2. Create a temporary Consumer (with a unique group.id so it doesn't
         interfere) to call get_watermark_offsets(TopicPartition) for each
         partition. This returns (low_watermark, high_watermark) where
         high_watermark == LEO.
      3. For each partition:
         - committed = the offset from step 1 (use -1 or 0 if no committed offset)
         - log_end = high_watermark from step 2
         - lag = log_end - max(committed, 0)
      4. Build a list of dicts: [{"topic_partition": "topic[partition]",
         "committed": int, "log_end": int, "lag": int}, ...]
      5. Return the list.

    Docs:
      - list_consumer_group_offsets: https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#confluent_kafka.admin.AdminClient.list_consumer_group_offsets
      - get_watermark_offsets: https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#confluent_kafka.Consumer.get_watermark_offsets

    Args:
        admin_client: AdminClient connected to the cluster.
        group_id: Consumer group ID to check lag for.

    Returns:
        List of dicts with per-partition lag information.
    """
    raise NotImplementedError("TODO(human)")


def monitor_lag_loop(group_id: str, interval: float = 3.0) -> None:
    """Continuously monitor and display consumer lag for a group.

    TODO(human): Implement this function.

    Steps:
      1. Create an AdminClient.
      2. Enter an infinite loop (Ctrl+C to stop):
         a. Call get_consumer_lag(admin_client, group_id).
         b. Call print_full_lag_table(lag_details, group_id) to display.
         c. Sleep for `interval` seconds.
      3. Wrap in try/except KeyboardInterrupt for clean exit.

    Usage:
      Run this in one terminal while running consumer experiments in another.
      You'll see lag increase as producers outpace consumers, and decrease
      as consumers catch up.

    Args:
        group_id: Consumer group ID to monitor.
        interval: Seconds between lag checks.
    """
    raise NotImplementedError("TODO(human)")


# -- Orchestration (boilerplate) ----------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python lag_monitor.py <group-id>")
        print(f"\nKnown groups:")
        print(f"  {config.REPLICATION_GROUP}")
        print(f"  {config.TRANSACTION_GROUP}")
        print(f"  {config.CONSUMER_TUNING_GROUP}")
        sys.exit(1)

    group_id = sys.argv[1]
    print(f"=== Monitoring lag for group '{group_id}' (Ctrl+C to stop) ===")
    monitor_lag_loop(group_id)


if __name__ == "__main__":
    main()
