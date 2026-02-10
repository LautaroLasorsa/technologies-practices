"""Streaming pull subscriber for the Order Processing System.

Demonstrates:
  - Callback-based streaming pull (long-lived connection)
  - Graceful shutdown with signal handling
  - ACK/NACK semantics in callbacks

Run after publishing messages:
    uv run python subscriber_streaming.py
"""

import json
import signal
import sys

from concurrent.futures import TimeoutError

from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.subscriber.message import Message

import config


# ── TODO(human): Implement this function ─────────────────────────────


def process_message(message: Message) -> None:
    """Callback that processes a single Pub/Sub message.

    TODO(human): Implement this function.

    Steps:
      1. Decode the message data: json.loads(message.data.decode("utf-8"))
      2. Read message attributes via message.attributes (dict-like).
         For example: message.attributes.get("order_id", "unknown")
      3. Print the order details (order_id, item, quantity from the payload).
      4. Call message.ack() to acknowledge successful processing.

    ACK vs NACK:
      - message.ack()  -> tells Pub/Sub this message is done; won't be redelivered
      - message.nack() -> tells Pub/Sub to redeliver this message immediately
      Use nack() when processing fails and you want a retry.

    IMPORTANT: If you neither ack nor nack, the message will be redelivered
    after the subscription's ack_deadline_seconds (default 10s in our setup).

    Docs: https://cloud.google.com/pubsub/docs/pull#stream_pull
    """
    raise NotImplementedError("TODO(human): implement process_message")


# ── Orchestration ────────────────────────────────────────────────────


def start_streaming_pull(
    subscriber: pubsub_v1.SubscriberClient,
    subscription_path: str,
    callback,
    timeout: float = 30.0,
) -> None:
    """Start a streaming pull and block until timeout or interrupt."""
    streaming_future = subscriber.subscribe(subscription_path, callback=callback)
    print(f"Listening on {subscription_path} (timeout={timeout}s, Ctrl+C to stop)...")

    try:
        streaming_future.result(timeout=timeout)
    except TimeoutError:
        print("\nTimeout reached. Shutting down.")
        streaming_future.cancel()
        streaming_future.result()  # Block until shutdown completes
    except KeyboardInterrupt:
        print("\nInterrupted. Shutting down.")
        streaming_future.cancel()
        streaming_future.result()


def main() -> None:
    subscriber = pubsub_v1.SubscriberClient()
    sub_path = subscriber.subscription_path(config.PROJECT_ID, config.NOTIFICATION_SUB)

    print(f"Streaming pull from: {config.NOTIFICATION_SUB}")
    start_streaming_pull(subscriber, sub_path, callback=process_message)


if __name__ == "__main__":
    main()
