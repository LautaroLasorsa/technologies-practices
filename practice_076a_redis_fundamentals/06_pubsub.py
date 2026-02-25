# /// script
# requires-python = ">=3.12"
# dependencies = ["redis>=5.0"]
# ///
"""Redis Pub/Sub -- Publisher and Subscriber.

Demonstrates:
  - PUBLISH messages to a channel
  - SUBSCRIBE to a channel and process messages
  - Pattern subscriptions with PSUBSCRIBE
  - The fire-and-forget nature of Pub/Sub (at-most-once delivery)

Run after starting Redis:
    uv run 06_pubsub.py
"""

from __future__ import annotations

import threading
import time

import redis


def get_client() -> redis.Redis:
    return redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# -- TODO(human): Implement these functions --------------------------------


def publisher(r: redis.Redis, channel: str, messages: list[str], delay: float) -> None:
    """Publish messages to a Redis channel with a delay between each.

    TODO(human): Implement this function.

    This function demonstrates the PUBLISH command. In Redis Pub/Sub,
    publishing is fire-and-forget: the publisher doesn't know (or care)
    how many subscribers are listening. If no subscribers are on the
    channel, the message is simply discarded.

    Steps:

    1. Wait a short moment for the subscriber to be ready:
           time.sleep(0.5)

    2. Loop through each message in the `messages` list:
       a. Publish the message to the channel:
              num_receivers = r.publish(channel, message)
          PUBLISH returns the number of clients that received the message.
          If no subscribers are listening, it returns 0 (and the message
          is lost forever -- this is at-most-once delivery).
          Print: the message, channel, and number of receivers.

       b. Wait between messages:
              time.sleep(delay)

    3. After all messages, publish a special "QUIT" message so the
       subscriber knows to stop:
           r.publish(channel, "QUIT")

    Docs:
      - PUBLISH: https://redis.io/docs/latest/commands/publish/
    """
    raise NotImplementedError("TODO(human): implement publisher")


def subscriber(r: redis.Redis, channel: str) -> list[str]:
    """Subscribe to a channel and collect messages until "QUIT" is received.

    TODO(human): Implement this function.

    This function demonstrates the SUBSCRIBE workflow using redis-py's
    PubSub helper. A subscribed Redis connection enters a special mode
    where it can ONLY receive messages (no regular commands). The PubSub
    object handles this by creating a dedicated connection.

    Steps:

    1. Create a PubSub object:
           pubsub = r.pubsub()

    2. Subscribe to the channel:
           pubsub.subscribe(channel)
       This sends the SUBSCRIBE command. The first message you receive
       will be a confirmation message of type "subscribe" with data=1
       (number of channels subscribed to).

    3. Initialize an empty list to collect received messages.

    4. Enter a message loop using pubsub.listen():
           for message in pubsub.listen():
       listen() is a generator that yields messages as dicts:
         {
           "type": "subscribe" | "message" | "unsubscribe" | ...,
           "channel": "channel_name",
           "data": <message_data or subscription_count>,
           "pattern": None,
         }

       For each message:
       a. If message["type"] == "subscribe":
             Print a confirmation (e.g., "Subscribed to {channel}").
             Continue to the next message.

       b. If message["type"] == "message":
             The actual published data is in message["data"].
             - If the data is "QUIT", break out of the loop.
             - Otherwise, print the message and append to your results list.

    5. Unsubscribe and close the PubSub connection:
           pubsub.unsubscribe()
           pubsub.close()

    6. Return the list of collected messages.

    Docs:
      - redis-py PubSub: https://redis-py.readthedocs.io/en/stable/advanced_features.html#publish-subscribe
    """
    raise NotImplementedError("TODO(human): implement subscriber")


# -- Boilerplate: Pattern subscription demo --------------------------------


def pattern_subscriber_demo(r: redis.Redis) -> None:
    """Demonstrate pattern-based subscriptions (boilerplate, not a TODO).

    PSUBSCRIBE lets you subscribe to channels matching a glob pattern.
    For example, "news.*" matches "news.sports", "news.tech", "news.finance".
    This is useful for topic hierarchies.
    """
    pubsub = r.pubsub()
    pubsub.psubscribe("events.*")
    print("  Subscribed to pattern 'events.*'")

    # Give the subscription a moment to register
    time.sleep(0.3)

    # Publish to different channels matching the pattern
    channels_and_messages = [
        ("events.login", "user-42 logged in"),
        ("events.purchase", "order-99 placed"),
        ("events.error", "disk full on node-3"),
        ("other.channel", "this won't be received"),
    ]

    r2 = get_client()
    for ch, msg in channels_and_messages:
        receivers = r2.publish(ch, msg)
        print(f"  Published to '{ch}': receivers={receivers}")

    # Read the messages (with a short timeout to not block forever)
    time.sleep(0.5)
    received = 0
    while True:
        message = pubsub.get_message(timeout=1.0)
        if message is None:
            break
        if message["type"] in ("pmessage",):
            print(
                f"  Pattern '{message['pattern']}' matched channel "
                f"'{message['channel']}': {message['data']}"
            )
            received += 1

    print(f"  Received {received} messages via pattern subscription")
    pubsub.punsubscribe()
    pubsub.close()


# -- Orchestration (boilerplate) -------------------------------------------


def main() -> None:
    r = get_client()
    r.ping()
    print("Connected to Redis!")

    section("Exercise: Pub/Sub Publisher & Subscriber")

    messages_to_send = [
        "Hello from publisher!",
        "Redis Pub/Sub is fire-and-forget",
        "No message persistence",
        "At-most-once delivery",
    ]

    collected: list[str] = []

    def subscriber_thread() -> None:
        nonlocal collected
        collected = subscriber(get_client(), "notifications")

    # Start subscriber first so it's ready to receive
    sub_thread = threading.Thread(target=subscriber_thread, daemon=True)
    sub_thread.start()

    # Give subscriber time to connect and subscribe
    time.sleep(0.3)

    # Run publisher in main thread
    publisher(r, "notifications", messages_to_send, delay=0.5)

    # Wait for subscriber to finish
    sub_thread.join(timeout=10)

    print(f"\n  Subscriber collected {len(collected)} messages:")
    for msg in collected:
        print(f"    - {msg}")

    # Demonstrate what happens when publishing with no subscribers
    section("Demo: Publishing with No Subscribers")
    receivers = r.publish("empty_channel", "hello?")
    print(f"  Published to 'empty_channel': {receivers} receivers")
    print("  (Message is lost -- no one was listening)")

    section("Demo: Pattern Subscriptions (PSUBSCRIBE)")
    pattern_subscriber_demo(r)

    print("\n--- All pub/sub exercises completed ---")


if __name__ == "__main__":
    main()
