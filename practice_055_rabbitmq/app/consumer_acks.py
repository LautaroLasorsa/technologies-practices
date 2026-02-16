"""Exercise 4: Consumer acknowledgments, message durability, and QoS.

Demonstrates:
  - Manual ack vs auto ack: controlling when messages are removed from the queue
  - basic_nack and basic_reject: negative acknowledgments with requeue option
  - Prefetch count (QoS): limiting unacknowledged messages per consumer
  - Message durability: persistent messages and publisher confirms
  - Fair dispatch: distributing work evenly across consumers

Run after setup_infrastructure.py:
    uv run python consumer_acks.py
"""

import json
import time

import pika

import config

# -- Sample data ---------------------------------------------------------------

TASK_MESSAGES = [
    {"task_id": "TASK-001", "action": "process_order", "priority": "high", "payload": "order data..."},
    {"task_id": "TASK-002", "action": "send_email", "priority": "low", "payload": "email content..."},
    {"task_id": "TASK-003", "action": "generate_report", "priority": "medium", "payload": "report params..."},
    {"task_id": "TASK-004", "action": "INVALID_ACTION", "priority": "high", "payload": "bad data"},
    {"task_id": "TASK-005", "action": "resize_image", "priority": "low", "payload": "image bytes..."},
    {"task_id": "TASK-006", "action": "process_order", "priority": "high", "payload": "order data..."},
    {"task_id": "TASK-007", "action": "INVALID_ACTION", "priority": "low", "payload": "bad data again"},
    {"task_id": "TASK-008", "action": "send_email", "priority": "medium", "payload": "newsletter..."},
]

VALID_ACTIONS = {"process_order", "send_email", "generate_report", "resize_image"}


# -- Seed helper (boilerplate) ------------------------------------------------


def seed_tasks(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    queue: str,
    tasks: list[dict],
) -> None:
    """Publish task messages to a queue for the consumer exercises."""
    for task in tasks:
        task["timestamp"] = time.time()
        body = json.dumps(task).encode("utf-8")
        channel.basic_publish(
            exchange="",  # default exchange: routes by queue name
            routing_key=queue,
            body=body,
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=pika.DeliveryMode.Transient,
            ),
        )
    print(f"  Seeded {len(tasks)} tasks to '{queue}'")


# -- TODO(human): Implement these functions ------------------------------------


def consume_with_manual_ack(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    queue_name: str,
    max_messages: int = 20,
) -> tuple[list[dict], list[dict]]:
    """Consume messages with manual acknowledgment, acking or nacking based on content.

    TODO(human): Implement this function.

    This function demonstrates the heart of RabbitMQ reliability: manual
    acknowledgments. You'll process each message and decide whether to
    ack (success) or nack (failure) based on whether the task action is valid.

    Steps:
      1. Create two lists: `acked` and `nacked` to track results.

      2. Loop up to `max_messages` times:
         a. Call channel.basic_get(queue=queue_name, auto_ack=False).
            auto_ack=False is CRITICAL here -- it means RabbitMQ will NOT
            remove the message upon delivery. The message stays in "unacked"
            state until we explicitly ack or nack it.

         b. If method is None (queue empty), break.

         c. Decode the message body from JSON.

         d. Check if the task's "action" is in VALID_ACTIONS:

            IF VALID (action in VALID_ACTIONS):
              - Print: "[ACK] Task {task_id}: {action} -- processing..."
              - Acknowledge the message:
                    channel.basic_ack(delivery_tag=method.delivery_tag)

                basic_ack tells RabbitMQ: "This message has been successfully
                processed. Remove it from the queue permanently."

                The delivery_tag is a channel-specific sequential integer that
                identifies this delivery. It's NOT the message ID -- it's the
                delivery attempt number on THIS channel.

              - Append to `acked` list.

            IF INVALID (action not in VALID_ACTIONS):
              - Print: "[NACK] Task {task_id}: {action} -- invalid, rejecting!"
              - Negatively acknowledge the message:
                    channel.basic_nack(
                        delivery_tag=method.delivery_tag,
                        requeue=False,
                    )

                basic_nack tells RabbitMQ: "I cannot process this message."
                The requeue parameter is crucial:
                  - requeue=True:  Put the message BACK in the queue for another
                                   consumer (or the same one) to try again.
                                   WARNING: This can create infinite retry loops
                                   if the message is fundamentally unprocessable!
                  - requeue=False: DISCARD the message (or send to Dead Letter
                                   Exchange if one is configured on the queue).
                                   This is what we want for invalid messages.

                Alternative: channel.basic_reject(delivery_tag=..., requeue=False)
                basic_reject does the same thing but for a SINGLE message only.
                basic_nack is a RabbitMQ extension that also supports batch
                rejection via the `multiple` parameter.

              - Append to `nacked` list.

      3. Return the tuple (acked, nacked).

    Why manual ack matters:
      With auto_ack=True, if your code crashes mid-processing, the message is
      GONE -- RabbitMQ already removed it. With manual ack, unacked messages
      are redelivered when the consumer disconnects, providing at-least-once
      delivery semantics. The trade-off: you might process a message twice if
      you crash after processing but before acking.

    Docs:
      - basic_ack: https://pika.readthedocs.io/en/stable/modules/channel.html#pika.channel.Channel.basic_ack
      - basic_nack: https://pika.readthedocs.io/en/stable/modules/channel.html#pika.channel.Channel.basic_nack
    """
    raise NotImplementedError("TODO(human): Consume with manual ack/nack based on message validity")


def consume_with_prefetch(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    queue_name: str,
    prefetch_count: int = 1,
    max_messages: int = 20,
) -> list[dict]:
    """Consume messages with prefetch/QoS limiting.

    TODO(human): Implement this function.

    Prefetch count (Quality of Service) controls how many messages RabbitMQ
    delivers to a consumer before requiring an acknowledgment. This is
    essential for fair dispatch and preventing consumer overload.

    Steps:
      1. Set the prefetch count using basic_qos:
             channel.basic_qos(prefetch_count=prefetch_count)

         This tells RabbitMQ: "Don't send me more than N unacknowledged
         messages at a time." The broker tracks the count per-consumer
         and stops delivering once the limit is reached.

         Why this matters:
           Without QoS, RabbitMQ uses round-robin dispatch but sends messages
           as fast as possible. If Consumer A is slow and Consumer B is fast,
           A builds up a backlog while B sits idle. With prefetch_count=1,
           RabbitMQ only sends the next message after the current one is acked,
           naturally directing messages to available consumers (fair dispatch).

         Trade-offs:
           - prefetch_count=1:  Maximum fairness, but low throughput (one network
                                round-trip per message). Good for slow, expensive tasks.
           - prefetch_count=10-50: Good balance of throughput and fairness. Messages
                                   are pre-fetched into a local buffer.
           - No QoS (unlimited): Maximum throughput, but unfair -- slow consumers
                                 build up huge backlogs.

      2. Create an empty list for processed messages.

      3. Loop up to max_messages times:
         a. basic_get with auto_ack=False.
         b. If method is None, break.
         c. Decode the message body.
         d. Print: f"  [Prefetch={prefetch_count}] Processing {task_id}..."
         e. Simulate work: time.sleep(0.1)
            In a real system, this would be actual processing. The sleep
            demonstrates that with prefetch_count=1, we only hold one
            message at a time during processing.
         f. Ack the message: channel.basic_ack(delivery_tag=method.delivery_tag)
         g. Append to results.

      4. Return the processed messages list.

    Note on basic_qos scope:
      basic_qos(prefetch_count=N) applies to the CHANNEL by default.
      If you have multiple consumers on the same channel, the limit is shared.
      In pika's BlockingConnection, each basic_consume gets its own consumer
      but shares the channel's QoS setting.

    Docs: https://pika.readthedocs.io/en/stable/modules/channel.html#pika.channel.Channel.basic_qos
    """
    raise NotImplementedError("TODO(human): Consume with prefetch/QoS limiting")


def publish_durable_messages(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    queue_name: str,
    messages: list[dict],
) -> int:
    """Publish persistent messages with publisher confirms.

    TODO(human): Implement this function.

    This function demonstrates the full reliability stack:
    1. Publisher confirms: broker acknowledges receipt of each message
    2. Persistent messages: written to disk, survive broker restart
    3. Durable queue: queue definition survives broker restart

    Steps:
      1. Enable publisher confirms on the channel:
             channel.confirm_delivery()

         After this call, every basic_publish will BLOCK until the broker
         confirms (acks) that it received the message. If the message cannot
         be routed (no matching queue), pika raises pika.exceptions.UnroutableError.

         Without publisher confirms, basic_publish is fire-and-forget -- the
         message goes into a local buffer and you have NO GUARANTEE it reached
         the broker. Publisher confirms close this reliability gap.

      2. Initialize a counter for confirmed messages.

      3. Iterate over each message dict in `messages`:
         a. Add a timestamp.
         b. Serialize to JSON bytes.
         c. Publish with persistent delivery mode:
                channel.basic_publish(
                    exchange="",
                    routing_key=queue_name,
                    body=body,
                    properties=pika.BasicProperties(
                        content_type="application/json",
                        delivery_mode=pika.DeliveryMode.Persistent,
                    ),
                )

            delivery_mode=Persistent (value 2) tells RabbitMQ to write the
            message to disk. Combined with a durable queue, this ensures
            the message survives a broker restart.

            The THREE conditions for full message durability:
              1. Exchange is durable    (declared with durable=True)
              2. Queue is durable       (declared with durable=True)
              3. Message is persistent   (delivery_mode=2 / Persistent)

            If ANY of these is missing, messages can be lost on restart.

            Note: Even with all three, there's a tiny window where the broker
            has received the message but hasn't flushed to disk yet. For
            absolute guarantees, use transactions (slow) or publisher confirms
            (much faster, what we're using here).

         d. Print confirmation with the task_id.
         e. Increment counter.

      4. Return the confirmed count.

    Docs:
      - confirm_delivery: https://pika.readthedocs.io/en/stable/modules/channel.html#pika.channel.Channel.confirm_delivery
      - DeliveryMode: https://pika.readthedocs.io/en/stable/modules/spec.html
    """
    raise NotImplementedError("TODO(human): Publish persistent messages with publisher confirms")


# -- Orchestration (boilerplate) -----------------------------------------------


def main() -> None:
    print("=== Exercise 4: Consumer Acknowledgments, Durability & QoS ===\n")

    connection = config.get_connection()
    channel = connection.channel()

    try:
        # ======== Part A: Manual Ack/Nack ========
        print("=" * 60)
        print("PART A: Manual Acknowledgments (ACK/NACK)")
        print("=" * 60)
        print()
        print("Seeding task queue...")
        seed_tasks(channel, config.QUEUE_ACK_TASKS, TASK_MESSAGES)
        print()

        print(f"Consuming from '{config.QUEUE_ACK_TASKS}' with manual ack:")
        print(f"  Valid actions: {VALID_ACTIONS}")
        print(f"  Invalid tasks will be NACK'd (rejected, not requeued)\n")
        acked, nacked = consume_with_manual_ack(channel, config.QUEUE_ACK_TASKS)
        print(f"\n  Results: {len(acked)} acked, {len(nacked)} nacked")
        print(f"  Nacked tasks: {[t['task_id'] for t in nacked]}")
        print()

        # ======== Part B: Prefetch/QoS ========
        print("=" * 60)
        print("PART B: Prefetch Count (QoS)")
        print("=" * 60)
        print()

        # Re-seed for prefetch demo
        prefetch_tasks = [
            {"task_id": f"PF-{i:03d}", "action": "process_order", "priority": "medium", "payload": f"data-{i}"}
            for i in range(6)
        ]
        print("Seeding tasks for prefetch demo...")
        seed_tasks(channel, config.QUEUE_ACK_TASKS, prefetch_tasks)
        print()

        print(f"Consuming with prefetch_count=1 (fair dispatch):")
        processed = consume_with_prefetch(channel, config.QUEUE_ACK_TASKS, prefetch_count=1)
        print(f"\n  Processed {len(processed)} tasks with prefetch=1")
        print("  Each message was fully processed (acked) before the next was delivered.")
        print()

        # ======== Part C: Publisher Confirms & Durability ========
        print("=" * 60)
        print("PART C: Publisher Confirms & Message Durability")
        print("=" * 60)
        print()

        durable_tasks = [
            {"task_id": "DUR-001", "action": "critical_update", "payload": "important data 1"},
            {"task_id": "DUR-002", "action": "critical_update", "payload": "important data 2"},
            {"task_id": "DUR-003", "action": "critical_update", "payload": "important data 3"},
        ]

        print("Publishing persistent messages with publisher confirms:")
        confirmed = publish_durable_messages(channel, config.QUEUE_DURABLE_TASKS, durable_tasks)
        print(f"\n  {confirmed} messages confirmed by broker (written to disk)")
        print("  These messages would survive a broker restart.")
        print()

        # ======== Summary ========
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print()
        print("  Manual ack:     Gives you control over when messages are removed")
        print("  basic_nack:     Reject invalid messages (with or without requeue)")
        print("  Prefetch (QoS): Limits unacked messages, enables fair dispatch")
        print("  Persistent:     delivery_mode=2 writes messages to disk")
        print("  Confirms:       Broker acks receipt, closing the reliability gap")

    finally:
        connection.close()
        print("\nConnection closed.")


if __name__ == "__main__":
    main()
