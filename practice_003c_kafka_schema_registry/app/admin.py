"""Create all Kafka topics for the Schema Registry practice.

Run this once after starting the Kafka broker:
    uv run python admin.py

This is infrastructure setup (boilerplate) -- no TODO(human) here.
"""

from confluent_kafka.admin import AdminClient, NewTopic

import config


# ── Helpers ──────────────────────────────────────────────────────────


def create_admin_client() -> AdminClient:
    """Create an AdminClient connected to the local broker."""
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


# ── Main ─────────────────────────────────────────────────────────────


def define_practice_topics() -> list[NewTopic]:
    """Define all topics needed for this practice."""
    return [
        NewTopic(
            topic=config.USERS_TOPIC,
            num_partitions=config.USERS_PARTITIONS,
            replication_factor=1,
        ),
        NewTopic(
            topic=config.SENSOR_READINGS_TOPIC,
            num_partitions=config.SENSOR_READINGS_PARTITIONS,
            replication_factor=1,
        ),
        NewTopic(
            topic=config.SENSOR_ALERTS_TOPIC,
            num_partitions=config.SENSOR_ALERTS_PARTITIONS,
            replication_factor=1,
        ),
    ]


def main() -> None:
    print("=== Creating Kafka topics for Schema Registry practice ===\n")
    admin = create_admin_client()
    topics = define_practice_topics()
    create_topics(admin, topics)
    print("\nAll topics ready. You can now run producer/consumer scripts.")


if __name__ == "__main__":
    main()
