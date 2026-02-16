"""Exercise 2: Direct exchange -- route messages by exact routing key match.

Demonstrates:
  - Publishing messages to a direct exchange with specific routing keys
  - Consuming messages from queues bound with matching routing keys
  - Understanding how direct routing determines message destinations
  - The default exchange as a special case of direct routing

Run after setup_infrastructure.py:
    uv run python direct_exchange.py
"""

import json
import time

import pika

import config

# -- Sample data ---------------------------------------------------------------

PAYMENT_EVENTS = [
    {"type": "payment.processed", "order_id": "ORD-001", "amount": 99.99, "currency": "USD"},
    {"type": "payment.processed", "order_id": "ORD-002", "amount": 45.00, "currency": "EUR"},
    {"type": "payment.processed", "order_id": "ORD-003", "amount": 199.50, "currency": "USD"},
]

NOTIFICATION_EVENTS = [
    {"type": "notification.send", "user_id": "user-001", "channel": "email", "subject": "Order confirmed"},
    {"type": "notification.send", "user_id": "user-002", "channel": "sms", "subject": "Payment received"},
]

AUDIT_EVENTS = [
    {"type": "audit.log", "action": "user.login", "user_id": "user-001", "ip": "192.168.1.10"},
    {"type": "audit.log", "action": "order.created", "user_id": "user-002", "details": "ORD-002"},
    {"type": "audit.log", "action": "payment.failed", "user_id": "user-003", "reason": "insufficient funds"},
]


# -- TODO(human): Implement these functions ------------------------------------


def publish_with_routing_key(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    exchange: str,
    routing_key: str,
    messages: list[dict],
) -> int:
    """Publish messages to a direct exchange with a specific routing key.

    TODO(human): Implement this function.

    This function publishes a list of messages to the given exchange using
    the specified routing key. For a direct exchange, only queues bound with
    a binding key that EXACTLY matches this routing key will receive the messages.

    Steps:
      1. Initialize a counter for published messages.

      2. Iterate over each message dict in `messages`:
         a. Add a "timestamp" field: message["timestamp"] = time.time()

         b. Serialize the message to JSON bytes:
                body = json.dumps(message).encode("utf-8")

         c. Publish to the exchange using channel.basic_publish():
                channel.basic_publish(
                    exchange=exchange,
                    routing_key=routing_key,
                    body=body,
                    properties=pika.BasicProperties(
                        content_type="application/json",
                        delivery_mode=pika.DeliveryMode.Transient,
                    ),
                )

            Key parameters explained:
              - exchange: Name of the exchange to publish to. For the default
                exchange, use "" (empty string).
              - routing_key: The routing key that the direct exchange will
                match against queue binding keys.
              - body: The message payload as bytes.
              - properties: AMQP message properties (metadata).
                - content_type: Tells consumers how to deserialize the body.
                - delivery_mode: Transient (1) = in-memory only, lost on restart.
                                 Persistent (2) = written to disk, survives restart.
                  We use Transient here since these are demo messages. Exercise 4
                  covers persistent messages.

         d. Print confirmation: routing_key, message type, and a snippet.
         e. Increment the counter.

      3. Return the total count of published messages.

    Important concept -- what happens with unmatched routing keys:
      If no queue is bound with a matching key, the message is SILENTLY DROPPED
      by default. This is different from Kafka (where an unmatched topic causes
      an error). To detect unroutable messages, set mandatory=True in
      basic_publish and register a return callback -- but that's advanced usage.

    Docs: https://pika.readthedocs.io/en/stable/modules/channel.html#pika.channel.Channel.basic_publish
    """
    raise NotImplementedError("TODO(human): Publish messages with routing key to direct exchange")


def consume_from_queue(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    queue_name: str,
    max_messages: int = 10,
) -> list[dict]:
    """Consume messages from a specific queue using basic_get (polling).

    TODO(human): Implement this function.

    This function uses basic_get (synchronous pull) to retrieve messages
    one at a time from a queue. This is simpler than basic_consume (push)
    and better suited for scripts that need to drain a queue and exit.

    Steps:
      1. Create an empty list to collect consumed messages.

      2. Loop up to `max_messages` times:
         a. Call channel.basic_get(queue=queue_name, auto_ack=False).
            basic_get returns a tuple of (method, properties, body):
              - method: Contains delivery_tag, routing_key, exchange, etc.
                        If the queue is empty, method is None.
              - properties: AMQP message properties (content_type, headers, etc.)
              - body: The raw message bytes.

         b. If method is None, the queue is empty -- break out of the loop.

         c. Decode the message body:
                message = json.loads(body.decode("utf-8"))

         d. Print the consumed message info:
            - Queue name
            - Routing key (from method.routing_key)
            - A summary of the message content

         e. Acknowledge the message:
                channel.basic_ack(delivery_tag=method.delivery_tag)

            The delivery_tag is a unique integer identifying this message
            delivery on this channel. You MUST use the correct delivery_tag
            when acking -- acking with a wrong tag is a protocol error.

            basic_ack tells RabbitMQ: "I have successfully processed this
            message, you can remove it from the queue." Without this ack,
            the message stays in the "unacked" state and will be redelivered
            when the consumer disconnects.

         f. Append the decoded message dict to the results list.

      3. Return the collected messages list.

    basic_get vs basic_consume:
      - basic_get: Pull model. You ask for one message at a time. Simple,
        but inefficient for high-throughput scenarios (one network round-trip
        per message).
      - basic_consume: Push model. RabbitMQ delivers messages to your callback
        as they arrive. More efficient, but requires handling the event loop.
      We use basic_get here for simplicity. Exercise 4 uses basic_consume.

    Docs: https://pika.readthedocs.io/en/stable/modules/channel.html#pika.channel.Channel.basic_get
    """
    raise NotImplementedError("TODO(human): Consume messages from queue using basic_get")


# -- Orchestration (boilerplate) -----------------------------------------------


def main() -> None:
    print("=== Exercise 2: Direct Exchange Routing ===\n")

    connection = config.get_connection()
    channel = connection.channel()

    try:
        # --- Publish Phase ---
        print("--- Publishing Messages ---\n")

        print(f"Publishing {len(PAYMENT_EVENTS)} payment events "
              f"(routing_key='{config.ROUTING_KEY_PAYMENT}'):")
        payment_count = publish_with_routing_key(
            channel, config.DIRECT_EXCHANGE, config.ROUTING_KEY_PAYMENT, PAYMENT_EVENTS,
        )
        print()

        print(f"Publishing {len(NOTIFICATION_EVENTS)} notification events "
              f"(routing_key='{config.ROUTING_KEY_NOTIFICATION}'):")
        notif_count = publish_with_routing_key(
            channel, config.DIRECT_EXCHANGE, config.ROUTING_KEY_NOTIFICATION, NOTIFICATION_EVENTS,
        )
        print()

        print(f"Publishing {len(AUDIT_EVENTS)} audit events "
              f"(routing_key='{config.ROUTING_KEY_AUDIT}'):")
        audit_count = publish_with_routing_key(
            channel, config.DIRECT_EXCHANGE, config.ROUTING_KEY_AUDIT, AUDIT_EVENTS,
        )
        print()

        total = payment_count + notif_count + audit_count
        print(f"Published {total} messages total.\n")

        # --- Consume Phase ---
        print("--- Consuming Messages ---\n")

        print(f"Consuming from '{config.QUEUE_PAYMENTS}':")
        payments = consume_from_queue(channel, config.QUEUE_PAYMENTS)
        print(f"  -> Got {len(payments)} messages\n")

        print(f"Consuming from '{config.QUEUE_NOTIFICATIONS}':")
        notifications = consume_from_queue(channel, config.QUEUE_NOTIFICATIONS)
        print(f"  -> Got {len(notifications)} messages\n")

        print(f"Consuming from '{config.QUEUE_AUDIT}':")
        audit_msgs = consume_from_queue(channel, config.QUEUE_AUDIT)
        print(f"  -> Got {len(audit_msgs)} messages\n")

        # --- Summary ---
        print("=== Direct Exchange Summary ===")
        print(f"  Payments queue:      {len(payments)} messages (expected {len(PAYMENT_EVENTS)})")
        print(f"  Notifications queue: {len(notifications)} messages (expected {len(NOTIFICATION_EVENTS)})")
        print(f"  Audit queue:         {len(audit_msgs)} messages (expected {len(AUDIT_EVENTS)})")
        print("\nEach queue received ONLY the messages with its matching routing key.")
        print("This is the power of direct exchange routing.")

    finally:
        connection.close()
        print("\nConnection closed.")


if __name__ == "__main__":
    main()
