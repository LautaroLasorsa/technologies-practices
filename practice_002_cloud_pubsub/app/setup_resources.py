"""Create all Pub/Sub topics and subscriptions for the Order Processing System.

Run this once after starting the emulator:
    uv run python setup_resources.py

This is infrastructure setup (boilerplate) -- no TODO(human) here.
"""

from google.cloud import pubsub_v1
from google.api_core.exceptions import AlreadyExists

import config


# ── Helpers ──────────────────────────────────────────────────────────


def create_topic(publisher: pubsub_v1.PublisherClient, topic_id: str) -> str:
    """Create a topic, ignoring AlreadyExists. Returns the topic path."""
    topic_path = publisher.topic_path(config.PROJECT_ID, topic_id)
    try:
        publisher.create_topic(request={"name": topic_path})
        print(f"  Created topic: {topic_id}")
    except AlreadyExists:
        print(f"  Topic already exists: {topic_id}")
    return topic_path


def create_subscription(
    subscriber: pubsub_v1.SubscriberClient,
    subscription_id: str,
    topic_path: str,
    *,
    enable_ordering: bool = False,
    dead_letter_topic_path: str | None = None,
    max_delivery_attempts: int = 0,
) -> str:
    """Create a subscription with optional ordering and dead-letter config."""
    sub_path = subscriber.subscription_path(config.PROJECT_ID, subscription_id)

    request: dict = {
        "name": sub_path,
        "topic": topic_path,
        "ack_deadline_seconds": 10,
        "enable_message_ordering": enable_ordering,
    }
    if dead_letter_topic_path and max_delivery_attempts > 0:
        request["dead_letter_policy"] = {
            "dead_letter_topic": dead_letter_topic_path,
            "max_delivery_attempts": max_delivery_attempts,
        }

    try:
        subscriber.create_subscription(request=request)
        print(f"  Created subscription: {subscription_id} -> {topic_path}")
    except AlreadyExists:
        print(f"  Subscription already exists: {subscription_id}")
    return sub_path


# ── Main ─────────────────────────────────────────────────────────────


def setup_topics(publisher: pubsub_v1.PublisherClient) -> dict[str, str]:
    """Create all topics and return a mapping of topic_id -> topic_path."""
    print("\n=== Creating topics ===")
    return {
        topic_id: create_topic(publisher, topic_id)
        for topic_id in [config.ORDERS_TOPIC, config.DEAD_LETTER_TOPIC]
    }


def setup_subscriptions(
    subscriber: pubsub_v1.SubscriberClient,
    topic_paths: dict[str, str],
) -> None:
    """Create all subscriptions against the given topic paths."""
    print("\n=== Creating subscriptions ===")
    orders_path = topic_paths[config.ORDERS_TOPIC]
    dl_path = topic_paths[config.DEAD_LETTER_TOPIC]

    # Basic pull subscription (Phase 2)
    create_subscription(subscriber, config.INVENTORY_SUB, orders_path)

    # Fan-out subscription (Phase 3)
    create_subscription(subscriber, config.NOTIFICATION_SUB, orders_path)

    # Ordering-enabled subscription (Phase 4)
    create_subscription(
        subscriber,
        config.ORDERED_SUB,
        orders_path,
        enable_ordering=True,
        dead_letter_topic_path=dl_path,
        max_delivery_attempts=5,
    )

    # Dead-letter subscription (Phase 4)
    create_subscription(subscriber, config.DEAD_LETTER_SUB, dl_path)


def main() -> None:
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()

    topic_paths = setup_topics(publisher)
    setup_subscriptions(subscriber, topic_paths)

    print("\nAll resources ready. You can now run publisher/subscriber scripts.")


if __name__ == "__main__":
    main()
