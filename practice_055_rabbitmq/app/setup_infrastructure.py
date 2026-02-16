"""Exercise 1: Declare RabbitMQ exchanges, queues, and bindings.

Demonstrates:
  - Declaring exchanges of all four types (direct, fanout, topic, headers)
  - Declaring queues with various properties (durable, auto-delete)
  - Creating bindings between exchanges and queues with routing/binding keys
  - Understanding the AMQP resource model

This is the "infrastructure as code" for our messaging system. In production,
these declarations are typically run at application startup -- they are
idempotent (safe to run multiple times) as long as the properties match.

Run:
    uv run python setup_infrastructure.py
"""

import pika

import config


# -- TODO(human): Implement these functions ------------------------------------


def declare_exchanges(channel: pika.adapters.blocking_connection.BlockingChannel) -> None:
    """Declare all four exchange types used in this practice.

    TODO(human): Implement this function.

    You need to declare four exchanges using channel.exchange_declare().
    Each exchange has a different type that determines its routing behavior.

    Steps:
      1. Declare a DIRECT exchange named config.DIRECT_EXCHANGE:
             channel.exchange_declare(
                 exchange=config.DIRECT_EXCHANGE,
                 exchange_type="direct",
                 durable=True,
             )
         A direct exchange routes messages to queues whose binding key
         exactly matches the message's routing key. This is the most
         common exchange type -- use it when you know exactly which
         queue(s) should receive each message.

         durable=True means the exchange definition survives broker restarts.
         This does NOT affect message persistence -- that's a separate setting.

      2. Declare a FANOUT exchange named config.FANOUT_EXCHANGE:
             exchange_type="fanout", durable=True
         A fanout exchange broadcasts every message to ALL bound queues,
         completely ignoring the routing key. Think of it as a radio tower:
         every listener (queue) tuned in (bound) receives every broadcast.
         Use it for notifications, logging, or any "publish to all" pattern.

      3. Declare a TOPIC exchange named config.TOPIC_EXCHANGE:
             exchange_type="topic", durable=True
         A topic exchange routes based on wildcard pattern matching.
         Routing keys are dot-separated words (e.g., "order.created.us").
         Binding patterns use * (exactly one word) and # (zero or more words).
         This is the most flexible exchange type -- it can behave as both
         direct (no wildcards) and fanout (binding key "#").

      4. Declare a HEADERS exchange named config.HEADERS_EXCHANGE:
             exchange_type="headers", durable=True
         A headers exchange routes based on message header attributes instead
         of routing keys. When binding, you specify which headers must match.
         Less common than the other types, but useful when routing logic
         depends on multiple attributes that don't fit in a single string.

    After each declaration, print a confirmation message.

    Docs: https://pika.readthedocs.io/en/stable/modules/channel.html#pika.channel.Channel.exchange_declare
    """
    raise NotImplementedError("TODO(human): Declare all four exchange types")


def declare_queues(channel: pika.adapters.blocking_connection.BlockingChannel) -> None:
    """Declare all queues used across the practice exercises.

    TODO(human): Implement this function.

    You need to declare multiple queues using channel.queue_declare().
    Each queue has properties that control its behavior and lifetime.

    Steps:
      1. Declare the DIRECT exchange queues (durable=True for all three):
             channel.queue_declare(queue=config.QUEUE_PAYMENTS, durable=True)
             channel.queue_declare(queue=config.QUEUE_NOTIFICATIONS, durable=True)
             channel.queue_declare(queue=config.QUEUE_AUDIT, durable=True)

         durable=True means the queue definition (its name and properties)
         survives a broker restart. NOTE: this does NOT make messages persistent --
         for that, you must also publish with delivery_mode=2. A durable queue
         with non-persistent messages will exist after restart but be empty.

      2. Declare the FANOUT exchange queues (durable=True):
             config.QUEUE_FANOUT_LOGGER
             config.QUEUE_FANOUT_ANALYTICS
             config.QUEUE_FANOUT_BACKUP

         These three queues demonstrate the fanout pattern: all three will
         be bound to the same fanout exchange, and each will receive a copy
         of every message published to that exchange.

      3. Declare the TOPIC exchange queues (durable=True):
             config.QUEUE_TOPIC_ALL_ORDERS
             config.QUEUE_TOPIC_US_ORDERS
             config.QUEUE_TOPIC_CREATED

         These queues will be bound to the topic exchange with different
         wildcard patterns. The pattern determines which messages each queue
         receives -- this is the power of topic exchanges.

      4. Declare the ACK exercise queue (durable=True):
             config.QUEUE_ACK_TASKS

         This queue will be used in Exercise 4 to practice manual
         acknowledgments and prefetch/QoS settings.

      5. Declare the DURABLE TASKS queue (durable=True):
             config.QUEUE_DURABLE_TASKS

         This queue will hold persistent messages and demonstrate
         publisher confirms.

    After all declarations, print a summary of the queues created.

    Key concept -- queue properties:
      - durable: Queue survives broker restart (metadata on disk)
      - exclusive: Only the declaring connection can use it; auto-deleted on disconnect
      - auto_delete: Queue is deleted when the last consumer unsubscribes
      - arguments: Extra args like TTL, max-length, dead-letter-exchange (used in Exercise 5)

    Note: The dead letter and TTL queues (Exercise 5) are declared in their
    own setup function in dead_letter.py because they require special arguments.

    Docs: https://pika.readthedocs.io/en/stable/modules/channel.html#pika.channel.Channel.queue_declare
    """
    raise NotImplementedError("TODO(human): Declare all queues for the practice exercises")


def create_bindings(channel: pika.adapters.blocking_connection.BlockingChannel) -> None:
    """Bind queues to exchanges with appropriate routing/binding keys.

    TODO(human): Implement this function.

    A binding is a rule that tells an exchange "send matching messages to this queue".
    Without bindings, exchanges have nowhere to route messages (they'd be dropped).
    The meaning of the binding key depends on the exchange type.

    Steps:
      1. Bind queues to the DIRECT exchange:
             channel.queue_bind(
                 queue=config.QUEUE_PAYMENTS,
                 exchange=config.DIRECT_EXCHANGE,
                 routing_key=config.ROUTING_KEY_PAYMENT,
             )
         Repeat for QUEUE_NOTIFICATIONS with ROUTING_KEY_NOTIFICATION,
         and QUEUE_AUDIT with ROUTING_KEY_AUDIT.

         For direct exchanges, the routing_key in the binding must EXACTLY
         match the routing_key used when publishing. If you publish with
         routing_key="payment.processed", only queues bound with that same
         key receive the message.

      2. Bind queues to the FANOUT exchange:
             channel.queue_bind(
                 queue=config.QUEUE_FANOUT_LOGGER,
                 exchange=config.FANOUT_EXCHANGE,
             )
         Repeat for QUEUE_FANOUT_ANALYTICS and QUEUE_FANOUT_BACKUP.

         For fanout exchanges, the routing_key is IGNORED -- you can omit it
         or pass any string. Every bound queue receives every message. This
         is why fanout is the simplest exchange type.

      3. Bind queues to the TOPIC exchange with wildcard patterns:
             channel.queue_bind(
                 queue=config.QUEUE_TOPIC_ALL_ORDERS,
                 exchange=config.TOPIC_EXCHANGE,
                 routing_key=config.TOPIC_BIND_ALL_ORDERS,  # "order.#"
             )
             channel.queue_bind(
                 queue=config.QUEUE_TOPIC_US_ORDERS,
                 exchange=config.TOPIC_EXCHANGE,
                 routing_key=config.TOPIC_BIND_US_ORDERS,   # "order.*.us"
             )
             channel.queue_bind(
                 queue=config.QUEUE_TOPIC_CREATED,
                 exchange=config.TOPIC_EXCHANGE,
                 routing_key=config.TOPIC_BIND_CREATED,     # "*.created.*"
             )

         Topic exchange binding patterns:
           - "order.#"     matches: "order.created", "order.created.us", "order.shipped.eu.priority"
           - "order.*.us"  matches: "order.created.us", "order.shipped.us"
                           NOT:     "order.created.eu", "order.us" (wrong segment count)
           - "*.created.*" matches: "order.created.us", "item.created.eu"
                           NOT:     "created.us" (missing first segment), "order.created" (missing third)

    After creating all bindings, print a summary showing which queues are
    bound to which exchanges.

    Docs: https://pika.readthedocs.io/en/stable/modules/channel.html#pika.channel.Channel.queue_bind
    """
    raise NotImplementedError("TODO(human): Create all exchange-to-queue bindings")


# -- Orchestration (boilerplate) -----------------------------------------------


def main() -> None:
    print("=== Exercise 1: Declaring RabbitMQ Infrastructure ===\n")

    connection = config.get_connection()
    channel = connection.channel()

    try:
        print("--- Declaring Exchanges ---")
        declare_exchanges(channel)
        print()

        print("--- Declaring Queues ---")
        declare_queues(channel)
        print()

        print("--- Creating Bindings ---")
        create_bindings(channel)
        print()

        print("=== Infrastructure setup complete! ===")
        print("Open http://localhost:15672 (guest/guest) to verify in the management UI.")
    finally:
        connection.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()
