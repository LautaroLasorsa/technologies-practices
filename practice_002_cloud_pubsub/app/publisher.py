"""Publisher for the Order Processing System.

Demonstrates:
  - Publishing messages with attributes
  - Publishing with ordering keys (messages grouped by customer_id)

Run after setup_resources.py:
    uv run python publisher.py
"""

import json
import time

from google.cloud import pubsub_v1

import config

# ── Sample data ──────────────────────────────────────────────────────

SAMPLE_ORDERS = [
    {"order_id": "ORD-001", "customer_id": "CUST-A", "item": "Laptop", "quantity": 1},
    {"order_id": "ORD-002", "customer_id": "CUST-B", "item": "Mouse", "quantity": 3},
    {"order_id": "ORD-003", "customer_id": "CUST-A", "item": "Keyboard", "quantity": 1},
    {"order_id": "ORD-004", "customer_id": "CUST-C", "item": "Monitor", "quantity": 2},
    {
        "order_id": "ORD-005",
        "customer_id": "CUST-A",
        "item": "USB Cable",
        "quantity": 5,
    },
]


# ── TODO(human): Implement these two functions ──────────────────────


def publish_order(
    publisher: pubsub_v1.PublisherClient,
    topic_path: str,
    order_id: str,
    item: str,
    quantity: int,
) -> str:
    """Publish a single order message to the topic.

    TODO(human): Implement this function.

    Steps:
      1. Build a JSON payload dict with keys: order_id, item, quantity, timestamp.
         Use time.time() for the timestamp.
      2. Encode the payload as a UTF-8 bytes string (json.dumps(...).encode("utf-8")).
      3. Call publisher.publish(topic_path, data=..., **attributes) where attributes
         are keyword arguments: order_id=order_id, item=item.
         Attributes must be strings -- they're message metadata that subscribers
         can filter on without parsing the body.
      4. Call .result() on the returned future to block until the message is
         acknowledged by the server. This returns the message_id (a string).
      5. Return the message_id.

    Hint:
      future = publisher.publish(topic_path, data=encoded, order_id=order_id, item=item)
      message_id = future.result()

    Docs: https://cloud.google.com/pubsub/docs/publisher#publish_messages
    """
    data = json.dumps(
        {
            "order_id": order_id,
            "item": item,
            "quantity": quantity,
            "timestamp": time.time(),
        }
    ).encode("utf-8")

    future = publisher.publish(topic_path, data=data, order_id=order_id, item=item)
    return future.result()


def publish_orders_with_ordering(
    publisher: pubsub_v1.PublisherClient,
    topic_path: str,
    orders: list[dict],
) -> list[str | Exception]:
    """Publish multiple orders using ordering keys grouped by customer_id.

    TODO(human): Implement this function.

    Steps:
      1. Iterate over each order dict in `orders`.
      2. Build the same JSON payload as in publish_order (order_id, item, quantity, timestamp).
      3. Encode it as UTF-8 bytes.
      4. Call publisher.publish() with the additional keyword argument:
             ordering_key=order["customer_id"]
         This guarantees that messages with the same customer_id are delivered
         in the order they were published (within a single subscription that
         has enable_message_ordering=True).
      5. Call .result() on each future and collect the message_ids.
      6. Return the list of message_ids.

    IMPORTANT: For ordering to work, the PublisherClient must be created with:
        pubsub_v1.PublisherClient(
            publisher_options=pubsub_v1.types.PublisherOptions(
                enable_message_ordering=True,
            ),
        )
    This is already handled in main() below -- just use the publisher passed in.

    Docs: https://cloud.google.com/pubsub/docs/ordering
    """
    futures = []
    for order in orders:
        data = json.dumps(
            {
                "order_id": order["order_id"],
                "item": order["item"],
                "quantity": order["quantity"],
                "timestamp": time.time(),
            }
        ).encode("utf-8")

        futures.append(
            (
                publisher.publish(
                    topic_path,
                    data=data,
                    order_id=order["order_id"],
                    item=order["item"],
                    ordering_key=order["customer_id"],
                ),
                order["customer_id"],
            )
        )

    ids = []
    for future, ordering_key in futures:
        try:
            ids.append(future.result())
        except Exception as e:
            publisher.resume_publish(topic_path, ordering_key)
            ids.append(e)

    return ids


# ── Orchestration ────────────────────────────────────────────────────


def create_basic_publisher() -> pubsub_v1.PublisherClient:
    """Create a standard publisher (no ordering support)."""
    return pubsub_v1.PublisherClient()


def create_ordering_publisher() -> pubsub_v1.PublisherClient:
    """Create a publisher with message ordering enabled."""
    return pubsub_v1.PublisherClient(
        publisher_options=pubsub_v1.types.PublisherOptions(
            enable_message_ordering=True,
        ),
    )


def run_basic_publish(publisher: pubsub_v1.PublisherClient, topic_path: str) -> None:
    """Publish sample orders without ordering keys (Phase 2)."""
    print("\n=== Publishing orders (basic, no ordering) ===")
    for order in SAMPLE_ORDERS:
        msg_id = publish_order(
            publisher,
            topic_path,
            order_id=order["order_id"],
            item=order["item"],
            quantity=order["quantity"],
        )
        print(f"  Published {order['order_id']}: message_id={msg_id}")
    print(f"Published {len(SAMPLE_ORDERS)} orders.")


def run_ordered_publish(publisher: pubsub_v1.PublisherClient, topic_path: str) -> None:
    """Publish sample orders with ordering keys (Phase 4)."""
    print("\n=== Publishing orders (with ordering keys by customer_id) ===")
    msg_ids = publish_orders_with_ordering(publisher, topic_path, SAMPLE_ORDERS)
    for order, msg_id in zip(SAMPLE_ORDERS, msg_ids):
        print(
            f"  Published {order['order_id']} [key={order['customer_id']}]: message_id={msg_id}"
        )
    print(f"Published {len(msg_ids)} ordered orders.")


def main() -> None:
    topic_path = pubsub_v1.PublisherClient.topic_path(
        config.PROJECT_ID, config.ORDERS_TOPIC
    )

    # Phase 2: basic publish
    basic_pub = create_basic_publisher()
    run_basic_publish(basic_pub, topic_path)

    # Phase 4: ordered publish
    ordering_pub = create_ordering_publisher()
    run_ordered_publish(ordering_pub, topic_path)


if __name__ == "__main__":
    main()
