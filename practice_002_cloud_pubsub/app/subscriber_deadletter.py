"""Dead-letter subscriber for the Order Processing System.

Demonstrates:
  - Poison message handling (NACKing messages to trigger dead-letter routing)
  - Dead-letter topic monitoring

Workflow:
  1. Publisher sends orders to the "orders" topic.
  2. The "ordered-sub" subscription has a dead-letter policy (max_delivery_attempts=3).
  3. The poison_message_handler NACKs messages matching a condition.
  4. After 3 NACKs, Pub/Sub routes the message to "dead-letter-topic".
  5. A separate listener on "dead-letter-sub" logs the failed messages.

Run after setup_resources.py and publisher.py:
    uv run python subscriber_deadletter.py
"""

import json

from concurrent.futures import TimeoutError

from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.subscriber.message import Message

import config


# ── TODO(human): Implement this function ─────────────────────────────


def poison_message_handler(message: Message) -> None:
    """Callback that NACKs 'poison' messages to trigger dead-letter routing.

    TODO(human): Implement this function.

    Steps:
      1. Decode the message data: json.loads(message.data.decode("utf-8"))
      2. Read the "item" field from the decoded payload.
      3. Define a condition for "poison" messages. For this exercise, consider
         any order where item == "Monitor" as a poison message.
      4. If the message is poison:
         - Print a warning: "NACK poison message: {order_id} (item={item})"
         - Call message.nack() to reject it. Pub/Sub will redeliver it.
         - After max_delivery_attempts (3) NACKs, Pub/Sub moves the message
           to the dead-letter topic automatically.
      5. If the message is NOT poison:
         - Print: "ACK order: {order_id} (item={item})"
         - Call message.ack()

    Key concept (dead-letter routing):
      When a subscription has a dead_letter_policy with max_delivery_attempts=N,
      Pub/Sub tracks how many times a message has been delivered. After N
      deliveries (NACKs or ack_deadline expirations), the message is
      automatically published to the dead_letter_topic and removed from
      the original subscription. (Emulator minimum is 5.)

    Docs: https://cloud.google.com/pubsub/docs/dead-letter-topics
    """
    raise NotImplementedError("TODO(human): implement poison_message_handler")


# ── Dead-letter monitor ──────────────────────────────────────────────


def log_dead_letter(message: Message) -> None:
    """Log messages that arrived at the dead-letter topic (boilerplate)."""
    data = json.loads(message.data.decode("utf-8"))
    order_id = data.get("order_id", "unknown")
    item = data.get("item", "unknown")
    print(f"  [DEAD-LETTER] Received failed message: order_id={order_id}, item={item}")
    message.ack()


# ── Orchestration ────────────────────────────────────────────────────


def run_poison_subscriber(
    subscriber: pubsub_v1.SubscriberClient,
    timeout: float = 30.0,
) -> None:
    """Subscribe to ordered-sub and NACK poison messages."""
    sub_path = subscriber.subscription_path(config.PROJECT_ID, config.ORDERED_SUB)
    print(f"\n=== Listening on {config.ORDERED_SUB} (will NACK poison messages) ===")

    future = subscriber.subscribe(sub_path, callback=poison_message_handler)
    try:
        future.result(timeout=timeout)
    except TimeoutError:
        print("Timeout. Stopping poison subscriber.")
        future.cancel()
        future.result()


def run_deadletter_monitor(
    subscriber: pubsub_v1.SubscriberClient,
    timeout: float = 15.0,
) -> None:
    """Subscribe to dead-letter-sub and log any messages that arrived."""
    sub_path = subscriber.subscription_path(config.PROJECT_ID, config.DEAD_LETTER_SUB)
    print(f"\n=== Monitoring {config.DEAD_LETTER_SUB} for failed messages ===")

    future = subscriber.subscribe(sub_path, callback=log_dead_letter)
    try:
        future.result(timeout=timeout)
    except TimeoutError:
        print("Timeout. Stopping dead-letter monitor.")
        future.cancel()
        future.result()


def main() -> None:
    subscriber = pubsub_v1.SubscriberClient()

    # Step 1: Process messages from ordered-sub, NACKing poison ones.
    # After max_delivery_attempts NACKs, Pub/Sub routes them to dead-letter-topic.
    run_poison_subscriber(subscriber, timeout=30.0)

    # Step 2: Check what ended up in the dead-letter subscription.
    run_deadletter_monitor(subscriber, timeout=15.0)


if __name__ == "__main__":
    main()
