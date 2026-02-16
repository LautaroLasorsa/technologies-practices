"""Exercise 5: Dead letter exchanges and TTL (time-to-live).

Demonstrates:
  - Dead letter exchange (DLX): catching rejected and expired messages
  - Per-queue TTL: messages expire after a configured time
  - Per-message TTL: individual message expiration
  - Rejected messages routing to DLX
  - The DLX + TTL pattern for delayed retry

Run after setup_infrastructure.py:
    uv run python dead_letter.py
"""

import json
import time

import pika

import config

# -- Sample data ---------------------------------------------------------------

WORK_MESSAGES = [
    {"task_id": "WRK-001", "action": "process_payment", "amount": 100.00},
    {"task_id": "WRK-002", "action": "CORRUPT_DATA", "amount": -1},
    {"task_id": "WRK-003", "action": "send_receipt", "amount": 50.00},
    {"task_id": "WRK-004", "action": "CORRUPT_DATA", "amount": 0},
    {"task_id": "WRK-005", "action": "update_inventory", "amount": 75.50},
]

TTL_MESSAGES = [
    {"task_id": "TTL-001", "action": "time_sensitive_alert", "expires_in": "5s"},
    {"task_id": "TTL-002", "action": "flash_sale_notification", "expires_in": "5s"},
    {"task_id": "TTL-003", "action": "urgent_update", "expires_in": "5s"},
]


# -- TODO(human): Implement these functions ------------------------------------


def setup_dead_letter_infrastructure(
    channel: pika.adapters.blocking_connection.BlockingChannel,
) -> None:
    """Declare the dead letter exchange, work queues, and dead letter queue.

    TODO(human): Implement this function.

    Dead letter exchanges handle the "failure path" of messaging. When a message
    cannot be processed (rejected, expired, or queue overflow), instead of being
    lost, it's routed to a DLX for inspection, alerting, or retry.

    Steps:
      1. Declare the dead letter exchange:
             channel.exchange_declare(
                 exchange=config.DEAD_LETTER_EXCHANGE,
                 exchange_type="fanout",
                 durable=True,
             )

         We use a FANOUT exchange as the DLX so that ALL dead-lettered messages
         (from any queue) end up in one place. In production, you might use a
         direct or topic DLX to route different failure types differently.

      2. Declare the dead letter QUEUE (where dead-lettered messages land):
             channel.queue_declare(
                 queue=config.QUEUE_DEAD_LETTER,
                 durable=True,
             )

         This is a normal queue -- it just happens to receive messages from
         the DLX. You can consume from it like any other queue, which is
         useful for monitoring, alerting, or manual retry.

      3. Bind the dead letter queue to the DLX:
             channel.queue_bind(
                 queue=config.QUEUE_DEAD_LETTER,
                 exchange=config.DEAD_LETTER_EXCHANGE,
             )

      4. Declare the WORK QUEUE with DLX arguments:
             channel.queue_declare(
                 queue=config.QUEUE_WORK,
                 durable=True,
                 arguments={
                     "x-dead-letter-exchange": config.DEAD_LETTER_EXCHANGE,
                 },
             )

         The "x-dead-letter-exchange" argument tells RabbitMQ: "When messages
         in this queue are dead-lettered (rejected with requeue=False, expired,
         or dropped due to queue overflow), route them to this exchange."

         Optional additional arguments:
           - "x-dead-letter-routing-key": Override the original routing key
             when publishing to the DLX. Useful for routing dead letters from
             multiple queues to specific dead letter queues.
           - "x-max-length": Maximum number of messages in the queue. When
             exceeded, oldest messages are dead-lettered.

      5. Declare the TTL WORK QUEUE with both DLX and per-queue TTL:
             channel.queue_declare(
                 queue=config.QUEUE_TTL_WORK,
                 durable=True,
                 arguments={
                     "x-dead-letter-exchange": config.DEAD_LETTER_EXCHANGE,
                     "x-message-ttl": 5000,  # 5 seconds in milliseconds
                 },
             )

         "x-message-ttl" sets a per-queue TTL: ANY message published to this
         queue will expire after 5000ms (5 seconds) if not consumed.

         When a message expires AND a DLX is configured, the message is
         dead-lettered (routed to the DLX). Without a DLX, expired messages
         are simply discarded.

         Per-queue TTL vs Per-message TTL:
           - Per-queue (x-message-ttl): ALL messages get the same TTL. Simpler.
             Expired messages are removed from the HEAD of the queue in order.
           - Per-message (expiration property): Each message can have a different
             TTL. BUT: expired messages are only removed when they reach the
             head of the queue (not immediately). This can cause unexpected
             delays if a long-TTL message is ahead of a short-TTL one.

      6. Print a summary of what was declared.

    The DLX pattern is essential for production systems:
      - Monitor failed messages for debugging
      - Implement retry logic (DLX -> delay queue -> re-route to original queue)
      - Alerting when too many messages are dead-lettered
      - Compliance (never lose a message, even if processing fails)

    Docs:
      - DLX: https://www.rabbitmq.com/docs/dlx
      - TTL: https://www.rabbitmq.com/docs/ttl
    """
    raise NotImplementedError("TODO(human): Set up dead letter exchange, queues, and TTL configuration")


def publish_and_expire(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    messages: list[dict],
) -> int:
    """Publish messages to the TTL queue and wait for them to expire into the DLX.

    TODO(human): Implement this function.

    This function publishes messages to a queue that has a 5-second TTL.
    If the messages are not consumed within 5 seconds, they expire and are
    routed to the dead letter exchange.

    Steps:
      1. Initialize a counter.

      2. Iterate over each message in `messages`:
         a. Add a timestamp.
         b. Serialize to JSON bytes.
         c. Publish to the TTL work queue via the default exchange:
                channel.basic_publish(
                    exchange="",
                    routing_key=config.QUEUE_TTL_WORK,
                    body=body,
                    properties=pika.BasicProperties(
                        content_type="application/json",
                        delivery_mode=pika.DeliveryMode.Persistent,
                    ),
                )

            We publish to the DEFAULT EXCHANGE ("") using the queue name as
            the routing key. The default exchange is a special direct exchange
            that automatically has a binding for every queue with the queue's
            name as the binding key. This is a shortcut for "publish directly
            to this queue."

            Note: We do NOT set the "expiration" property on the message itself
            because the queue already has x-message-ttl=5000. Both methods work:
              - Per-queue TTL: Set on queue declaration (applies to all messages)
              - Per-message TTL: Set expiration="5000" in BasicProperties
            When both are set, the LOWER value wins.

         d. Print the task_id and that it was published to the TTL queue.
         e. Increment counter.

      3. Print how many messages were published and that we're waiting for them
         to expire.

      4. Wait for the TTL to expire:
             print("  Waiting 7 seconds for messages to expire...")
             time.sleep(7)

         We wait 7 seconds (longer than the 5s TTL) to ensure all messages
         have expired and been dead-lettered.

      5. Return the count.

    What happens during the wait:
      After 5 seconds, RabbitMQ checks the head of the TTL queue. Messages that
      have exceeded their TTL are removed and published to the dead letter exchange
      (config.DEAD_LETTER_EXCHANGE). From there, the fanout DLX delivers them to
      config.QUEUE_DEAD_LETTER. The messages gain an "x-death" header containing
      metadata about why and where they were dead-lettered.

    Docs: https://www.rabbitmq.com/docs/ttl
    """
    raise NotImplementedError("TODO(human): Publish messages to TTL queue and wait for expiration")


def reject_to_dead_letter(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    queue_name: str,
    max_messages: int = 20,
) -> tuple[list[dict], list[dict]]:
    """Consume messages, rejecting invalid ones to the dead letter exchange.

    TODO(human): Implement this function.

    This function consumes from a queue with a DLX configured. Valid messages
    are acked normally. Invalid messages (action == "CORRUPT_DATA") are
    rejected with requeue=False, which sends them to the dead letter exchange.

    Steps:
      1. Create two lists: `processed` and `rejected`.

      2. Loop up to max_messages times:
         a. basic_get with auto_ack=False.
         b. If method is None, break.
         c. Decode message body from JSON.

         d. IF the message's action is "CORRUPT_DATA":
              - Print: "[REJECT] {task_id} -- corrupt data, sending to DLX"
              - REJECT the message:
                    channel.basic_reject(
                        delivery_tag=method.delivery_tag,
                        requeue=False,
                    )

                basic_reject with requeue=False does two things:
                  1. Removes the message from the CURRENT queue
                  2. Publishes it to the queue's dead letter exchange (if configured)

                If we used requeue=True instead, the message would go BACK to the
                same queue (not to the DLX), potentially causing an infinite loop.

                The rejected message arrives in the dead letter queue with an
                "x-death" header added by RabbitMQ containing:
                  - queue: the queue it was dead-lettered from
                  - reason: "rejected"
                  - exchange: the original exchange
                  - routing-keys: the original routing keys
                  - count: how many times it was dead-lettered
                  - time: when it was dead-lettered

              - Append to `rejected` list.

            ELSE (valid message):
              - Print: "[ACK] {task_id} -- processed successfully"
              - Acknowledge: channel.basic_ack(delivery_tag=method.delivery_tag)
              - Append to `processed` list.

      3. Return (processed, rejected).

    basic_reject vs basic_nack:
      - basic_reject: AMQP 0-9-1 standard. Rejects ONE message at a time.
      - basic_nack: RabbitMQ extension. Can reject MULTIPLE messages at once
        using the `multiple=True` parameter (rejects all unacked messages up
        to and including the given delivery_tag).
      Both support the requeue parameter and both trigger DLX routing when
      requeue=False.

    Docs:
      - basic_reject: https://pika.readthedocs.io/en/stable/modules/channel.html#pika.channel.Channel.basic_reject
      - DLX docs: https://www.rabbitmq.com/docs/dlx
    """
    raise NotImplementedError("TODO(human): Consume and reject invalid messages to dead letter exchange")


# -- Helper (boilerplate) -----------------------------------------------------


def drain_queue(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    queue_name: str,
    label: str,
) -> list[dict]:
    """Drain all messages from a queue using auto_ack (for display purposes)."""
    messages = []
    while True:
        method, properties, body = channel.basic_get(queue=queue_name, auto_ack=True)
        if method is None:
            break
        msg = json.loads(body.decode("utf-8"))
        messages.append(msg)
        task_id = msg.get("task_id", "unknown")
        print(f"  [{label}] {task_id}: {msg.get('action', 'n/a')}")
    return messages


# -- Orchestration (boilerplate) -----------------------------------------------


def main() -> None:
    print("=== Exercise 5: Dead Letter Exchanges and TTL ===\n")

    connection = config.get_connection()
    channel = connection.channel()

    try:
        # ======== Setup DLX Infrastructure ========
        print("--- Setting up Dead Letter Infrastructure ---\n")
        setup_dead_letter_infrastructure(channel)
        print()

        # ======== Part A: Rejection -> DLX ========
        print("=" * 60)
        print("PART A: Rejected Messages -> Dead Letter Exchange")
        print("=" * 60)
        print()

        print("Seeding work queue with valid and invalid messages...")
        for msg in WORK_MESSAGES:
            msg["timestamp"] = time.time()
            body = json.dumps(msg).encode("utf-8")
            channel.basic_publish(
                exchange="",
                routing_key=config.QUEUE_WORK,
                body=body,
                properties=pika.BasicProperties(
                    content_type="application/json",
                    delivery_mode=pika.DeliveryMode.Persistent,
                ),
            )
        print(f"  Published {len(WORK_MESSAGES)} messages to '{config.QUEUE_WORK}'\n")

        print("Consuming from work queue (rejecting CORRUPT_DATA messages):")
        processed, rejected = reject_to_dead_letter(channel, config.QUEUE_WORK)
        print(f"\n  Processed: {len(processed)}, Rejected: {len(rejected)}")
        print()

        print("Checking dead letter queue for rejected messages:")
        dead_letters = drain_queue(channel, config.QUEUE_DEAD_LETTER, "DLX")
        print(f"\n  Found {len(dead_letters)} messages in dead letter queue")
        if dead_letters:
            print("  These messages were rejected with requeue=False and routed to the DLX.")
        print()

        # ======== Part B: TTL Expiration -> DLX ========
        print("=" * 60)
        print("PART B: TTL Expiration -> Dead Letter Exchange")
        print("=" * 60)
        print()

        print(f"Publishing messages to TTL queue (x-message-ttl=5000ms):")
        expired_count = publish_and_expire(channel, TTL_MESSAGES)
        print()

        print("Checking dead letter queue for expired messages:")
        expired_dead_letters = drain_queue(channel, config.QUEUE_DEAD_LETTER, "DLX-Expired")
        print(f"\n  Found {len(expired_dead_letters)} expired messages in dead letter queue")
        if expired_dead_letters:
            print("  These messages exceeded the 5-second TTL and were dead-lettered.")
        print()

        # ======== Summary ========
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print()
        print("  Messages enter the Dead Letter Exchange when:")
        print("    1. Rejected/nack'd with requeue=False")
        print("    2. TTL expires (per-queue or per-message)")
        print("    3. Queue length limit exceeded (x-max-length)")
        print()
        print("  DLX use cases in production:")
        print("    - Error monitoring and alerting")
        print("    - Delayed retry (DLX -> TTL queue -> back to original)")
        print("    - Audit trail of failed processing")
        print("    - Poison message isolation")

    finally:
        connection.close()
        print("\nConnection closed.")


if __name__ == "__main__":
    main()
