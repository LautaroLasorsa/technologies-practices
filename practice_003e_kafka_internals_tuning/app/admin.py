"""Create all Kafka topics for practice 003e.

Run this once after starting the 3-broker cluster:
    uv run python admin.py

This is infrastructure setup (boilerplate) -- no TODO(human) here.
Topics are created with replication-factor=3 to exercise ISR,
leader election, and min.insync.replicas behavior across brokers.
"""

from confluent_kafka.admin import AdminClient, NewTopic

import config


# -- Helpers ------------------------------------------------------------------


def create_admin_client() -> AdminClient:
    """Create an AdminClient connected to the multi-broker cluster."""
    return AdminClient({"bootstrap.servers": config.BOOTSTRAP_SERVERS})


def create_topics(admin: AdminClient, topics: list[NewTopic]) -> None:
    """Create topics and report results.

    AdminClient.create_topics() returns a dict of {topic_name: Future}.
    Each future resolves when the broker responds. If the topic already
    exists, the future raises a KafkaException with error code
    TOPIC_ALREADY_EXISTS -- we catch that and print a friendly message.
    """
    futures = admin.create_topics(topics)

    for topic_name, future in futures.items():
        try:
            future.result()  # Block until topic is created
            print(f"  Created topic: {topic_name}")
        except Exception as exc:
            if "TOPIC_ALREADY_EXISTS" in str(exc):
                print(f"  Topic already exists: {topic_name}")
            else:
                print(f"  Failed to create topic {topic_name}: {exc}")


# -- Topic definitions --------------------------------------------------------


def define_practice_topics() -> list[NewTopic]:
    """Define all topics needed for this practice.

    Each topic is configured for the specific experiment it supports:
    - replication-demo: 3 partitions, RF=3, min.insync.replicas=2 for ISR experiments
    - compression-bench: 3 partitions, RF=3 for compression codec benchmarking
    - transaction-demo: 3 partitions, RF=3 for transactional producer experiments
    - consumer-tuning: 6 partitions, RF=3 for consumer config experiments
    - compaction-demo: 1 partition, RF=3, cleanup.policy=compact for log compaction
    """
    return [
        NewTopic(
            topic=config.REPLICATION_TOPIC,
            num_partitions=config.REPLICATION_PARTITIONS,
            replication_factor=3,
            config={
                "min.insync.replicas": "2",
            },
        ),
        NewTopic(
            topic=config.COMPRESSION_TOPIC,
            num_partitions=config.COMPRESSION_PARTITIONS,
            replication_factor=3,
        ),
        NewTopic(
            topic=config.TRANSACTION_TOPIC,
            num_partitions=config.TRANSACTION_PARTITIONS,
            replication_factor=3,
        ),
        NewTopic(
            topic=config.CONSUMER_TUNING_TOPIC,
            num_partitions=config.CONSUMER_TUNING_PARTITIONS,
            replication_factor=3,
        ),
        NewTopic(
            topic=config.COMPACTION_TOPIC,
            num_partitions=config.COMPACTION_PARTITIONS,
            replication_factor=3,
            config={
                "cleanup.policy": "compact",
                "segment.ms": "10000",
                "min.cleanable.dirty.ratio": "0.01",
            },
        ),
    ]


# -- Main ---------------------------------------------------------------------


def main() -> None:
    print("=== Creating Kafka topics for practice 003e ===\n")
    admin = create_admin_client()
    topics = define_practice_topics()
    create_topics(admin, topics)
    print("\nAll topics ready. You can now run the experiment scripts.")


if __name__ == "__main__":
    main()
