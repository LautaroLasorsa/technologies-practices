"""Synchronous pull subscriber for the Order Processing System.

Demonstrates:
  - Synchronous (non-streaming) pull of messages
  - Message acknowledgment (ACK)
  - Processing message data and attributes

Run after publishing messages:
    uv run python subscriber_pull.py
"""

import json

from google.cloud import pubsub_v1

import config


# ── TODO(human): Implement this function ─────────────────────────────


def pull_and_process(
    subscriber: pubsub_v1.SubscriberClient,
    subscription_path: str,
    max_messages: int,
) -> list[dict]:
    """Pull messages synchronously, process them, and ACK each one.

    TODO(human): Implement this function.

    Steps:
      1. Call subscriber.pull() with:
             request={"subscription": subscription_path, "max_messages": max_messages}
         This returns a PullResponse with a `received_messages` list.

      2. If no messages were received, print a message and return an empty list.

      3. For each received_message in the response:
         a. The actual Pub/Sub message is at received_message.message
         b. Decode the message data: json.loads(received_message.message.data.decode("utf-8"))
         c. Access attributes via received_message.message.attributes (a dict-like object)
         d. Collect the parsed order dict into a results list
         e. Print the order info (order_id, item, quantity)

      4. Collect all ack_ids from the received messages into a list.

      5. Call subscriber.acknowledge() with:
             request={"subscription": subscription_path, "ack_ids": ack_ids}
         This tells Pub/Sub these messages have been successfully processed.
         Without this, messages will be redelivered after the ack_deadline.

      6. Return the list of parsed order dicts.

    Hint:
      response = subscriber.pull(request={"subscription": subscription_path, "max_messages": max_messages})
      ack_ids = [msg.ack_id for msg in response.received_messages]
      subscriber.acknowledge(request={"subscription": subscription_path, "ack_ids": ack_ids})

    Key concept (at-least-once delivery):
      If you process a message but crash before calling acknowledge(), Pub/Sub
      will redeliver that message. Your processing logic should be idempotent
      (safe to run multiple times for the same message).

    Docs: https://cloud.google.com/pubsub/docs/pull#synchronous_pull
    """
    raise NotImplementedError("TODO(human): implement pull_and_process")


# ── Orchestration ────────────────────────────────────────────────────


def create_subscriber() -> pubsub_v1.SubscriberClient:
    """Create a Pub/Sub subscriber client."""
    return pubsub_v1.SubscriberClient()


def run_pull_loop(
    subscriber: pubsub_v1.SubscriberClient,
    subscription_path: str,
    max_messages: int = 10,
    rounds: int = 3,
) -> list[dict]:
    """Pull messages in multiple rounds until no more are available."""
    all_orders: list[dict] = []
    for i in range(rounds):
        print(f"\n--- Pull round {i + 1}/{rounds} ---")
        orders = pull_and_process(subscriber, subscription_path, max_messages)
        all_orders.extend(orders)
        if not orders:
            print("No more messages. Stopping.")
            break
    return all_orders


def main() -> None:
    subscriber = create_subscriber()
    sub_path = subscriber.subscription_path(config.PROJECT_ID, config.INVENTORY_SUB)

    print(f"Pulling from: {config.INVENTORY_SUB}")
    orders = run_pull_loop(subscriber, sub_path)
    print(f"\nTotal orders received: {len(orders)}")


if __name__ == "__main__":
    main()
