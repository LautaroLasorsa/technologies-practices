"""Fan-out test for the Order Processing System.

Demonstrates:
  - One topic with multiple subscriptions (fan-out pattern)
  - Each subscription independently receives ALL messages

Workflow:
  1. Publish N orders to the "orders" topic.
  2. Pull from "inventory-sub" -- should receive all N.
  3. Pull from "notification-sub" -- should also receive all N.
  4. Verify both got the full set.

Run after setup_resources.py:
    uv run python test_fanout.py
"""

import json
import time

from google.cloud import pubsub_v1

import config

# ── Helpers (boilerplate) ────────────────────────────────────────────

FANOUT_ORDERS = [
    {"order_id": "FAN-001", "item": "Tablet", "quantity": 1},
    {"order_id": "FAN-002", "item": "Headphones", "quantity": 2},
    {"order_id": "FAN-003", "item": "Charger", "quantity": 4},
]


def publish_fanout_orders(publisher: pubsub_v1.PublisherClient, topic_path: str) -> int:
    """Publish test orders and return the count."""
    print("\n=== Publishing fan-out test orders ===")
    for order in FANOUT_ORDERS:
        payload = json.dumps(
            {
                "order_id": order["order_id"],
                "item": order["item"],
                "quantity": order["quantity"],
                "timestamp": time.time(),
            }
        ).encode("utf-8")
        future = publisher.publish(
            topic_path,
            data=payload,
            order_id=order["order_id"],
            item=order["item"],
        )
        msg_id = future.result()
        print(f"  Published {order['order_id']}: message_id={msg_id}")
    return len(FANOUT_ORDERS)


def pull_all_messages(
    subscriber: pubsub_v1.SubscriberClient,
    subscription_path: str,
    max_messages: int = 10,
    max_rounds: int = 3,
) -> list[dict]:
    """Pull messages in rounds until none remain. Returns parsed orders."""
    all_orders: list[dict] = []
    for _ in range(max_rounds):
        response = subscriber.pull(
            request={"subscription": subscription_path, "max_messages": max_messages},
        )
        if not response.received_messages:
            break

        ack_ids = []
        for received in response.received_messages:
            data = json.loads(received.message.data.decode("utf-8"))
            all_orders.append(data)
            ack_ids.append(received.ack_id)

        subscriber.acknowledge(
            request={"subscription": subscription_path, "ack_ids": ack_ids},
        )
    return all_orders


# ── TODO(human): Implement this function ─────────────────────────────


def verify_fanout(
    subscriber: pubsub_v1.SubscriberClient,
    sub_paths: list[str],
    expected_count: int,
) -> bool:
    """Verify that every subscription received all published messages.

    TODO(human): Implement this function.

    Steps:
      1. For each subscription_path in sub_paths:
         a. Call pull_all_messages(subscriber, subscription_path) to get the
            list of order dicts received by that subscription.
         b. Print how many messages the subscription received.
         c. Check if len(received) == expected_count.

      2. After checking all subscriptions, determine if fan-out was successful:
         - ALL subscriptions must have received exactly `expected_count` messages.

      3. Print a summary:
         - If all passed: "Fan-out PASSED: all {len(sub_paths)} subscriptions
           received {expected_count} messages."
         - If any failed: "Fan-out FAILED: not all subscriptions received
           {expected_count} messages."

      4. Return True if all passed, False otherwise.

    Why fan-out works this way:
      In Pub/Sub, each *subscription* is an independent delivery channel.
      When a message is published to a topic, a copy is placed in EVERY
      subscription attached to that topic. This is different from competing
      consumers on the SAME subscription (where each message goes to only
      one consumer). This is the key distinction.

    Hint:
      results = []
      for sub_path in sub_paths:
          orders = pull_all_messages(subscriber, sub_path)
          results.append(len(orders) == expected_count)
      return all(results)
    """
    ok = True

    for subpath in sub_paths:
        pulled = pull_all_messages(subscriber, subpath)
        print(f"{subpath} pulled {len(pulled)} messages")
        ok = ok and (len(pulled) == expected_count)

    if ok:
        print(
            f"Fan-out PASSED: all {len(sub_paths)} subscriptions received {expected_count} messages."
        )
    else:
        print(
            f"Fan-out FAILED: not all subscriptions received {expected_count} messages."
        )

    return ok


# ── Orchestration ────────────────────────────────────────────────────


def main() -> None:
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()

    topic_path = publisher.topic_path(config.PROJECT_ID, config.ORDERS_TOPIC)

    expected = publish_fanout_orders(publisher, topic_path)

    # Give the emulator a moment to propagate messages to subscriptions
    print("\nWaiting 2 seconds for message propagation...")
    time.sleep(2)

    sub_paths = [
        subscriber.subscription_path(config.PROJECT_ID, config.INVENTORY_SUB),
        subscriber.subscription_path(config.PROJECT_ID, config.NOTIFICATION_SUB),
    ]

    print("\n=== Verifying fan-out ===")
    success = verify_fanout(subscriber, sub_paths, expected)

    if success:
        print("\nAll good! Both subscriptions received every message.")
    else:
        print("\nSomething went wrong. Check output above for details.")


if __name__ == "__main__":
    main()
