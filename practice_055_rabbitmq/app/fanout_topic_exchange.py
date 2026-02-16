"""Exercise 3: Fanout and Topic exchanges -- broadcast and pattern matching.

Demonstrates:
  - Fanout exchange: broadcast every message to all bound queues
  - Topic exchange: route based on wildcard patterns (* and #)
  - How the same message can be routed to multiple queues simultaneously
  - Comparing routing behavior across exchange types

Run after setup_infrastructure.py:
    uv run python fanout_topic_exchange.py
"""

import json
import time

import pika

import config

# -- Sample data ---------------------------------------------------------------

BROADCAST_EVENTS = [
    {"event": "system.startup", "service": "api-gateway", "version": "2.1.0"},
    {"event": "system.deploy", "service": "payment-service", "version": "3.0.1"},
    {"event": "system.alert", "service": "order-service", "level": "warning", "msg": "high latency"},
]

TOPIC_EVENTS = [
    # Routing key format: "<entity>.<action>.<region>"
    ("order.created.us", {"action": "created", "order_id": "ORD-100", "region": "us", "total": 150.00}),
    ("order.created.eu", {"action": "created", "order_id": "ORD-101", "region": "eu", "total": 89.99}),
    ("order.shipped.us", {"action": "shipped", "order_id": "ORD-100", "region": "us", "carrier": "FedEx"}),
    ("order.shipped.eu", {"action": "shipped", "order_id": "ORD-101", "region": "eu", "carrier": "DHL"}),
    ("order.cancelled.us", {"action": "cancelled", "order_id": "ORD-102", "region": "us", "reason": "customer request"}),
    ("item.created.us", {"action": "created", "item_id": "ITEM-500", "region": "us", "name": "Widget Pro"}),
    ("item.created.eu", {"action": "created", "item_id": "ITEM-501", "region": "eu", "name": "Gadget X"}),
]


# -- TODO(human): Implement these functions ------------------------------------


def publish_to_fanout(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    messages: list[dict],
) -> int:
    """Publish messages to the fanout exchange (broadcast to all bound queues).

    TODO(human): Implement this function.

    A fanout exchange delivers a COPY of every message to EVERY bound queue,
    completely ignoring the routing key. This is the simplest exchange type
    and implements the classic pub/sub (publish-subscribe) pattern.

    Steps:
      1. Initialize a counter for published messages.

      2. Iterate over each message dict in `messages`:
         a. Add a "timestamp" field: message["timestamp"] = time.time()

         b. Serialize to JSON bytes:
                body = json.dumps(message).encode("utf-8")

         c. Publish to the fanout exchange:
                channel.basic_publish(
                    exchange=config.FANOUT_EXCHANGE,
                    routing_key="",  # <-- ignored by fanout, but required by AMQP
                    body=body,
                    properties=pika.BasicProperties(
                        content_type="application/json",
                    ),
                )

            IMPORTANT: The routing_key parameter is REQUIRED by the AMQP protocol
            (basic_publish always takes a routing_key), but fanout exchanges
            completely ignore it. You can pass any string -- empty string is
            conventional. This is a common source of confusion for beginners.

         d. Print: the event name and that it was broadcast.
         e. Increment counter.

      3. Return the total count.

    Real-world use cases for fanout:
      - Broadcasting system-wide events (deploys, config changes, alerts)
      - Sending the same event to multiple services for different processing
        (one service logs it, another updates analytics, another backs it up)
      - Chat systems where all participants in a room see every message

    Docs: https://www.rabbitmq.com/tutorials/tutorial-three-python (fanout tutorial)
    """
    raise NotImplementedError("TODO(human): Publish messages to fanout exchange")


def publish_to_topic(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    events: list[tuple[str, dict]],
) -> int:
    """Publish messages to the topic exchange with multi-segment routing keys.

    TODO(human): Implement this function.

    A topic exchange routes messages based on wildcard matching between the
    message's routing key and the binding patterns on queues. Routing keys
    are dot-separated words (e.g., "order.created.us").

    The queues in this practice are bound with these patterns:
      - QUEUE_TOPIC_ALL_ORDERS:  "order.#"      -> all order events, any depth
      - QUEUE_TOPIC_US_ORDERS:   "order.*.us"    -> order events in US only
      - QUEUE_TOPIC_CREATED:     "*.created.*"   -> any "created" event

    Steps:
      1. Initialize a counter for published messages.

      2. Iterate over each (routing_key, message) tuple in `events`:
         a. Add "timestamp" and "routing_key" fields to the message:
                message["timestamp"] = time.time()
                message["routing_key"] = routing_key

         b. Serialize to JSON bytes.

         c. Publish to the topic exchange:
                channel.basic_publish(
                    exchange=config.TOPIC_EXCHANGE,
                    routing_key=routing_key,
                    body=body,
                    properties=pika.BasicProperties(
                        content_type="application/json",
                    ),
                )

            The routing_key here is CRITICAL -- it determines which queues
            receive this message. For example:
              - "order.created.us" matches "order.#", "order.*.us", AND "*.created.*"
                -> delivered to ALL THREE topic queues
              - "order.shipped.eu" matches only "order.#"
                -> delivered to QUEUE_TOPIC_ALL_ORDERS only
              - "item.created.eu" matches only "*.created.*"
                -> delivered to QUEUE_TOPIC_CREATED only

         d. Print: the routing key and which patterns it should match.
         e. Increment counter.

      3. Return the total count.

    Topic exchange wildcard rules:
      * (star)  = exactly ONE word    -> "order.*.us" matches "order.FOO.us" (one word between dots)
      # (hash)  = zero or MORE words  -> "order.#" matches "order", "order.x", "order.x.y.z"
      No wildcards = exact match (behaves like direct exchange)
      "#" alone = matches everything (behaves like fanout exchange)

    Docs: https://www.rabbitmq.com/tutorials/tutorial-five-python (topic tutorial)
    """
    raise NotImplementedError("TODO(human): Publish messages to topic exchange with routing keys")


def consume_and_display(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    queue_name: str,
    label: str,
    max_messages: int = 20,
) -> list[dict]:
    """Consume all available messages from a queue and display them.

    TODO(human): Implement this function.

    This function drains a queue using basic_get (pull mode) and displays
    each message with its routing key, showing which messages matched the
    queue's binding pattern.

    Steps:
      1. Create an empty list for collected messages.

      2. Print a header: f"  [{label}] Queue: {queue_name}"

      3. Loop up to max_messages times:
         a. Call channel.basic_get(queue=queue_name, auto_ack=True).
            We use auto_ack=True here (simpler) because this is a display
            exercise -- we don't need the reliability of manual acks.

            auto_ack=True means RabbitMQ removes the message from the queue
            immediately upon delivery, before we even process it. If our
            code crashes after receiving but before printing, the message is
            gone forever. This is acceptable for demo/display purposes but
            NEVER appropriate for production message processing.

         b. If method is None (queue empty), break.

         c. Decode the body from JSON.

         d. Print the message's routing key (from the message body if available,
            or from method.routing_key) and a brief summary.

         e. Append to results list.

      4. If no messages were received, print "  (empty -- no messages matched)"

      5. Return the collected messages.

    Docs: https://pika.readthedocs.io/en/stable/modules/channel.html#pika.channel.Channel.basic_get
    """
    raise NotImplementedError("TODO(human): Consume and display messages from a queue")


# -- Orchestration (boilerplate) -----------------------------------------------


def main() -> None:
    print("=== Exercise 3: Fanout & Topic Exchange Routing ===\n")

    connection = config.get_connection()
    channel = connection.channel()

    try:
        # ======== Part A: Fanout Exchange (Broadcast) ========
        print("=" * 60)
        print("PART A: Fanout Exchange -- Broadcasting to All Queues")
        print("=" * 60)
        print()

        print(f"Publishing {len(BROADCAST_EVENTS)} events to fanout exchange "
              f"'{config.FANOUT_EXCHANGE}':")
        fanout_count = publish_to_fanout(channel, BROADCAST_EVENTS)
        print(f"\nPublished {fanout_count} messages. Each should appear in ALL 3 fanout queues.\n")

        print("Consuming from fanout queues:\n")
        logger_msgs = consume_and_display(channel, config.QUEUE_FANOUT_LOGGER, "Logger")
        print()
        analytics_msgs = consume_and_display(channel, config.QUEUE_FANOUT_ANALYTICS, "Analytics")
        print()
        backup_msgs = consume_and_display(channel, config.QUEUE_FANOUT_BACKUP, "Backup")
        print()

        print(f"Fanout summary: Logger={len(logger_msgs)}, "
              f"Analytics={len(analytics_msgs)}, Backup={len(backup_msgs)}")
        print("All three queues received the SAME messages -- that's fanout.\n")

        # ======== Part B: Topic Exchange (Pattern Matching) ========
        print("=" * 60)
        print("PART B: Topic Exchange -- Wildcard Pattern Routing")
        print("=" * 60)
        print()

        print("Binding patterns:")
        print(f"  {config.QUEUE_TOPIC_ALL_ORDERS:30s} <- '{config.TOPIC_BIND_ALL_ORDERS}'")
        print(f"  {config.QUEUE_TOPIC_US_ORDERS:30s} <- '{config.TOPIC_BIND_US_ORDERS}'")
        print(f"  {config.QUEUE_TOPIC_CREATED:30s} <- '{config.TOPIC_BIND_CREATED}'")
        print()

        print(f"Publishing {len(TOPIC_EVENTS)} events to topic exchange "
              f"'{config.TOPIC_EXCHANGE}':")
        topic_count = publish_to_topic(channel, TOPIC_EVENTS)
        print(f"\nPublished {topic_count} messages.\n")

        print("Consuming from topic queues:\n")
        all_orders = consume_and_display(channel, config.QUEUE_TOPIC_ALL_ORDERS, "All Orders (order.#)")
        print()
        us_orders = consume_and_display(channel, config.QUEUE_TOPIC_US_ORDERS, "US Orders (order.*.us)")
        print()
        created = consume_and_display(channel, config.QUEUE_TOPIC_CREATED, "Created Events (*.created.*)")
        print()

        # ======== Analysis ========
        print("=" * 60)
        print("TOPIC EXCHANGE ANALYSIS")
        print("=" * 60)
        print()
        print(f"  'order.#' (all orders):   {len(all_orders)} messages")
        print(f"    -> Matches: order.created.us, order.created.eu, order.shipped.us,")
        print(f"       order.shipped.eu, order.cancelled.us (all 'order.*' events)")
        print()
        print(f"  'order.*.us' (US orders): {len(us_orders)} messages")
        print(f"    -> Matches: order.created.us, order.shipped.us, order.cancelled.us")
        print(f"       (only 'order' events with 'us' as third segment)")
        print()
        print(f"  '*.created.*' (created):  {len(created)} messages")
        print(f"    -> Matches: order.created.us, order.created.eu, item.created.us, item.created.eu")
        print(f"       (any entity with 'created' as second segment)")
        print()
        print("Notice: 'order.created.us' appears in ALL THREE queues!")
        print("A single message can match multiple binding patterns simultaneously.")

    finally:
        connection.close()
        print("\nConnection closed.")


if __name__ == "__main__":
    main()
