"""Exercise 4: MQTT 5.0 Features -- Shared Subscriptions, Message Expiry, Request/Reply.

Demonstrates:
  - Shared subscriptions: $share/<group>/<topic> for load balancing
  - Message expiry interval: TTL for messages
  - Request/reply pattern: ResponseTopic + CorrelationData

Run after starting Mosquitto:
    uv run python mqtt5_features.py
"""

import json
import time
import uuid
from dataclasses import dataclass, field

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion, MQTTProtocolVersion
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.properties import Properties

import config

# ── Constants ────────────────────────────────────────────────────────

SHARED_TOPIC = config.SHARED_SUB_TOPIC  # "tasks/processing"
SHARED_GROUP = "workers"
NUM_TASKS = 12
NUM_WORKERS = 3

EXPIRY_TOPIC = "notifications/alerts"
EXPIRY_SECONDS = 3  # Short TTL for demonstration

REQUEST_TOPIC = config.REQUEST_TOPIC  # "services/calculator/request"
RESPONSE_BASE = config.RESPONSE_TOPIC_PREFIX  # "services/calculator/response"


# ── Data tracking ───────────────────────────────────────────────────


@dataclass
class WorkerStats:
    """Track messages received by a shared subscription worker."""

    worker_id: str
    messages_received: int = 0
    task_ids: list[str] = field(default_factory=list)


@dataclass
class ReplyTracker:
    """Track request/reply correlation."""

    correlation_id: str
    request_sent_at: float = 0.0
    reply_received_at: float = 0.0
    reply_payload: str = ""


# ── Callbacks (boilerplate) ──────────────────────────────────────────


def on_connect(
    client: mqtt.Client,
    userdata: object,
    flags: mqtt.ConnectFlags,
    reason_code: mqtt.ReasonCode,
    properties: mqtt.Properties | None,
) -> None:
    """Log connection."""
    if reason_code == 0:
        print(f"[CONNECTED] {client._client_id.decode()}")
    else:
        print(f"[ERROR] {reason_code}")


def make_worker_on_message(
    stats: WorkerStats,
) -> mqtt.CallbackOnMessage:
    """Create a per-worker on_message callback that tracks received tasks.

    Each worker gets its own callback closure so we can track which
    worker received which messages independently.
    """

    def on_message(
        client: mqtt.Client,
        userdata: object,
        msg: mqtt.MQTTMessage,
    ) -> None:
        payload = json.loads(msg.payload.decode("utf-8"))
        stats.messages_received += 1
        task_id = payload.get("task_id", "unknown")
        stats.task_ids.append(task_id)
        print(
            f"  [{stats.worker_id}] received task={task_id}  "
            f"(total: {stats.messages_received})"
        )

    return on_message


# ── TODO(human): Implement these functions ───────────────────────────


def demonstrate_shared_subscriptions(
    pub_client: mqtt.Client,
) -> dict[str, WorkerStats]:
    """Demonstrate MQTT 5.0 shared subscriptions for load balancing.

    TODO(human): Implement this function.

    Shared subscriptions allow multiple clients to share the processing
    of messages from a single topic, similar to Kafka consumer groups.
    The broker distributes messages round-robin across group members.

    Syntax: $share/<ShareName>/<TopicFilter>
    Example: $share/workers/tasks/processing

    Steps:
      1. Create NUM_WORKERS (3) worker clients, each with its own stats:
             all_stats = {}
             worker_clients = []
             for i in range(NUM_WORKERS):
                 worker_id = f"worker-{i}"
                 stats = WorkerStats(worker_id=worker_id)
                 all_stats[worker_id] = stats

                 worker = mqtt.Client(
                     callback_api_version=CallbackAPIVersion.VERSION2,
                     client_id=worker_id,
                     protocol=MQTTProtocolVersion.MQTTv5,
                 )
                 worker.on_connect = on_connect
                 worker.on_message = make_worker_on_message(stats)
                 worker.connect(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE)
                 worker.loop_start()
                 worker_clients.append(worker)

      2. Subscribe each worker to the SHARED subscription:
             shared_filter = f"$share/{SHARED_GROUP}/{SHARED_TOPIC}"
             for worker in worker_clients:
                 worker.subscribe(shared_filter, qos=1)

         The "$share/workers/" prefix tells the broker: "distribute
         messages published to 'tasks/processing' among all clients
         subscribed to this shared group." Each message goes to
         exactly ONE worker in the group (round-robin by default in
         Mosquitto).

         This is fundamentally different from a normal subscription,
         where EVERY subscriber gets EVERY message. Shared subscriptions
         enable horizontal scaling of message processing.

         Wait for subscriptions to establish:
             time.sleep(1)

      3. Publish NUM_TASKS (12) task messages to the SHARED_TOPIC:
         For i in range(NUM_TASKS):
             payload = json.dumps({
                 "task_id": f"task-{i:03d}",
                 "operation": "process",
                 "data": f"payload_{i}",
             })
             pub_client.publish(SHARED_TOPIC, payload, qos=1)
             time.sleep(0.1)  # Small delay to allow distribution

         IMPORTANT: Publish to "tasks/processing" (the actual topic),
         NOT to "$share/workers/tasks/processing". The $share prefix
         is ONLY for subscribing. Publishers use the normal topic.

      4. Wait for all messages to be processed:
             time.sleep(2)

      5. Print distribution results:
             print("\\n  Task distribution across workers:")
             for worker_id, stats in all_stats.items():
                 print(f"    {worker_id}: {stats.messages_received} tasks "
                       f"({stats.task_ids})")
             total = sum(s.messages_received for s in all_stats.values())
             print(f"  Total processed: {total}/{NUM_TASKS}")

         You should see roughly 4 tasks per worker (12 tasks / 3 workers).
         The exact distribution depends on the broker's scheduling.

      6. Cleanup all worker clients:
             for worker in worker_clients:
                 worker.loop_stop()
                 worker.disconnect()

      7. Return all_stats

    Comparison with Kafka consumer groups:
      - Kafka: partitions are assigned to consumers. A consumer reads
        ALL messages from its assigned partitions. Rebalancing on join/leave.
      - MQTT shared subs: the broker distributes individual messages
        round-robin. No partition concept. No rebalancing delay.
      - Kafka provides ordering per partition; MQTT shared subs do NOT
        guarantee ordering across workers.
      - Kafka consumers can replay (seek to offset); MQTT has no replay.

    Docs: https://www.hivemq.com/blog/mqtt5-essentials-part7-shared-subscriptions/
    """
    raise NotImplementedError(
        "TODO(human): Implement shared subscriptions with worker clients"
    )


def demonstrate_message_expiry(
    pub_client: mqtt.Client,
) -> None:
    """Demonstrate MQTT 5.0 message expiry interval.

    TODO(human): Implement this function.

    Message expiry allows publishers to set a TTL (Time-To-Live) on
    messages. If the message hasn't been delivered to a subscriber
    before the expiry interval elapses, the broker discards it.

    Steps:
      1. Create MQTT 5.0 publish properties with MessageExpiryInterval:
             publish_props = Properties(PacketTypes.PUBLISH)
             publish_props.MessageExpiryInterval = EXPIRY_SECONDS

         The Properties class is imported from paho.mqtt.properties.
         PacketTypes.PUBLISH tells the Properties object which packet
         type these properties apply to (PUBLISH, CONNECT, SUBSCRIBE, etc.).

         MessageExpiryInterval is in seconds. The broker decrements it
         while the message is queued. If it reaches 0, the broker discards
         the message instead of delivering it.

      2. Publish a message with expiry to EXPIRY_TOPIC:
             payload_expire = json.dumps({
                 "alert": "temperature_high",
                 "expires_in": EXPIRY_SECONDS,
                 "timestamp": time.time(),
             })
             pub_client.publish(
                 EXPIRY_TOPIC,
                 payload_expire,
                 qos=1,
                 retain=True,
                 properties=publish_props,
             )
             print(f"  Published with {EXPIRY_SECONDS}s expiry (retained)")

         We use retain=True so the message is stored by the broker.
         The expiry interval applies to retained messages too -- the
         broker will remove the retained message once it expires.

      3. Immediately subscribe and verify the message is received:
             sub_client = mqtt.Client(
                 callback_api_version=CallbackAPIVersion.VERSION2,
                 client_id="expiry-sub-immediate",
                 protocol=MQTTProtocolVersion.MQTTv5,
             )

             received_messages = []
             def track_msg(client, userdata, msg):
                 received_messages.append(msg)
                 payload = json.loads(msg.payload.decode("utf-8"))
                 props = msg.properties
                 # In MQTT 5.0, the broker decrements MessageExpiryInterval
                 # and includes the remaining TTL in delivered messages
                 remaining = getattr(props, 'MessageExpiryInterval', None)
                 print(f"  [RECV] {msg.topic}: {payload.get('alert')} "
                       f"(remaining TTL: {remaining}s)")

             sub_client.on_message = track_msg
             sub_client.connect(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE)
             sub_client.loop_start()
             sub_client.subscribe(EXPIRY_TOPIC, qos=1)
             time.sleep(1)

             print(f"  Immediate subscriber received: {len(received_messages)} message(s)")
             sub_client.loop_stop()
             sub_client.disconnect()

      4. Wait for the message to expire:
             print(f"\\n  Waiting {EXPIRY_SECONDS + 1}s for message to expire...")
             time.sleep(EXPIRY_SECONDS + 1)

      5. Subscribe AFTER expiry and verify the message is NOT received:
             late_sub = mqtt.Client(
                 callback_api_version=CallbackAPIVersion.VERSION2,
                 client_id="expiry-sub-late",
                 protocol=MQTTProtocolVersion.MQTTv5,
             )

             late_messages = []
             def track_late(client, userdata, msg):
                 late_messages.append(msg)
                 print(f"  [RECV LATE] {msg.topic}: {msg.payload.decode()}")

             late_sub.on_message = track_late
             late_sub.connect(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE)
             late_sub.loop_start()
             late_sub.subscribe(EXPIRY_TOPIC, qos=1)
             time.sleep(1.5)

             print(f"  Late subscriber received: {len(late_messages)} message(s)")
             if len(late_messages) == 0:
                 print("  SUCCESS: Expired message was NOT delivered to late subscriber")
             else:
                 print("  NOTE: Message was still delivered (may not have expired yet)")

             late_sub.loop_stop()
             late_sub.disconnect()

    Key concept -- message expiry vs retained messages:
      Without expiry, retained messages persist FOREVER (or until replaced).
      With MessageExpiryInterval, the broker automatically cleans up
      stale retained messages. This is important for:
      - Alerts that are only relevant for a short time
      - Sensor readings that become outdated
      - Notifications that should not be delivered hours later

    Comparison:
      - Kafka: retention is time-based (topic-level), not per-message.
        All messages in a partition expire at the same time.
      - RabbitMQ: per-message TTL via x-message-ttl header.
      - Cloud Pub/Sub: 7-day max retention, no per-message TTL.
      - MQTT 5.0: per-message expiry, broker decrements TTL over time.

    Docs: https://www.hivemq.com/blog/mqtt5-essentials-part4-session-and-message-expiry/
    """
    raise NotImplementedError(
        "TODO(human): Demonstrate message expiry with early and late subscribers"
    )


def demonstrate_request_reply(
    pub_client: mqtt.Client,
) -> list[ReplyTracker]:
    """Demonstrate MQTT 5.0 request/reply pattern with ResponseTopic.

    TODO(human): Implement this function.

    MQTT 5.0 introduces ResponseTopic and CorrelationData properties
    to enable request/reply (RPC-style) patterns. The requester tells
    the responder WHERE to send the reply and includes a correlation
    ID to match replies to requests.

    Steps:
      1. Create a "service" client that listens for requests and sends replies:
             service = mqtt.Client(
                 callback_api_version=CallbackAPIVersion.VERSION2,
                 client_id="calculator-service",
                 protocol=MQTTProtocolVersion.MQTTv5,
             )

      2. Define the service's on_message handler:
             def service_handler(client, userdata, msg):
                 # Extract MQTT 5.0 properties from the received message
                 props = msg.properties

                 # Get the response topic (where to send the reply)
                 response_topic = props.ResponseTopic
                 # Get the correlation data (to match request <-> reply)
                 correlation_data = props.CorrelationData

                 # Process the request
                 request = json.loads(msg.payload.decode("utf-8"))
                 a, b, op = request["a"], request["b"], request["op"]
                 if op == "add":
                     result = a + b
                 elif op == "multiply":
                     result = a * b
                 else:
                     result = None

                 # Build reply properties with the same correlation data
                 reply_props = Properties(PacketTypes.PUBLISH)
                 reply_props.CorrelationData = correlation_data

                 # Publish reply to the response topic
                 reply_payload = json.dumps({"result": result, "op": op})
                 client.publish(response_topic, reply_payload, qos=1, properties=reply_props)
                 print(f"  [SERVICE] {op}({a}, {b}) = {result}, "
                       f"replied to {response_topic}")

             service.on_connect = on_connect
             service.on_message = service_handler
             service.connect(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE)
             service.loop_start()
             service.subscribe(REQUEST_TOPIC, qos=1)

      3. Create a "requester" client that sends requests and receives replies:
             trackers: list[ReplyTracker] = []

             # Generate a unique response topic for this requester
             requester_id = f"requester-{uuid.uuid4().hex[:8]}"
             my_response_topic = f"{RESPONSE_BASE}/{requester_id}"

             requester = mqtt.Client(
                 callback_api_version=CallbackAPIVersion.VERSION2,
                 client_id=requester_id,
                 protocol=MQTTProtocolVersion.MQTTv5,
             )

      4. Define the requester's on_message handler to receive replies:
             def reply_handler(client, userdata, msg):
                 props = msg.properties
                 correlation_data = props.CorrelationData
                 # Find the matching tracker by correlation ID
                 corr_id = correlation_data.decode("utf-8") if correlation_data else ""
                 for tracker in trackers:
                     if tracker.correlation_id == corr_id:
                         tracker.reply_received_at = time.perf_counter()
                         tracker.reply_payload = msg.payload.decode("utf-8")
                         break
                 payload = json.loads(msg.payload.decode("utf-8"))
                 print(f"  [REPLY] correlation={corr_id}  result={payload.get('result')}")

             requester.on_connect = on_connect
             requester.on_message = reply_handler
             requester.connect(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE)
             requester.loop_start()

         Subscribe to our unique response topic:
             requester.subscribe(my_response_topic, qos=1)
             time.sleep(1)

      5. Send several requests with ResponseTopic and CorrelationData:
         requests = [
             {"a": 10, "b": 5, "op": "add"},
             {"a": 7, "b": 3, "op": "multiply"},
             {"a": 100, "b": 42, "op": "add"},
         ]

         For each request:
           a. Generate a unique correlation ID:
                  corr_id = uuid.uuid4().hex[:8]
                  tracker = ReplyTracker(correlation_id=corr_id)
                  trackers.append(tracker)

           b. Build publish properties with ResponseTopic and CorrelationData:
                  req_props = Properties(PacketTypes.PUBLISH)
                  req_props.ResponseTopic = my_response_topic
                  req_props.CorrelationData = corr_id.encode("utf-8")

              - ResponseTopic: tells the service where to send the reply.
                Each requester uses a unique response topic so replies
                don't get mixed up between requesters.
              - CorrelationData: opaque bytes that the service echoes back
                in the reply. Allows the requester to match replies to
                the original requests (critical when multiple requests
                are in flight).

           c. Publish the request:
                  tracker.request_sent_at = time.perf_counter()
                  requester.publish(
                      REQUEST_TOPIC,
                      json.dumps(request),
                      qos=1,
                      properties=req_props,
                  )
                  print(f"  [REQUEST] {request['op']}({request['a']}, {request['b']}) "
                        f"corr={corr_id}")
                  time.sleep(0.5)

      6. Wait for all replies:
             time.sleep(2)

      7. Print round-trip summary:
             print("\\n  Request/Reply Summary:")
             for tracker in trackers:
                 if tracker.reply_received_at > 0:
                     rtt = (tracker.reply_received_at - tracker.request_sent_at) * 1000
                     print(f"    corr={tracker.correlation_id}: "
                           f"RTT={rtt:.1f}ms  reply={tracker.reply_payload}")
                 else:
                     print(f"    corr={tracker.correlation_id}: NO REPLY")

      8. Cleanup:
             service.loop_stop()
             service.disconnect()
             requester.loop_stop()
             requester.disconnect()

      9. Return trackers

    Key concept -- request/reply in MQTT vs other protocols:
      - MQTT 5.0: Uses ResponseTopic + CorrelationData properties.
        Lightweight, built into the protocol. No dedicated reply queue.
      - RabbitMQ: Uses reply-to header + correlation-id. Similar pattern
        but requires an explicit reply queue.
      - Kafka: No built-in request/reply. Must be implemented manually
        with a response topic + correlation.
      - HTTP/gRPC: Native request/reply (synchronous). MQTT makes
        this asynchronous and decoupled via the broker.

    Docs:
      - Properties: https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html
      - Response Topic: https://www.hivemq.com/blog/mqtt5-essentials-part9-request-response-pattern/
    """
    raise NotImplementedError(
        "TODO(human): Implement request/reply with ResponseTopic and CorrelationData"
    )


# ── Orchestration (boilerplate) ──────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("EXERCISE 4: MQTT 5.0 Features")
    print("=" * 60)

    # Publisher client for shared subs and expiry demos
    pub_client = mqtt.Client(
        callback_api_version=CallbackAPIVersion.VERSION2,
        client_id="mqtt5-publisher",
        protocol=MQTTProtocolVersion.MQTTv5,
    )
    pub_client.on_connect = on_connect
    pub_client.connect(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE)
    pub_client.loop_start()
    time.sleep(0.5)

    # Part 1: Shared Subscriptions
    print("\n--- Part 1: Shared Subscriptions ($share/workers/tasks/processing) ---")
    worker_stats = demonstrate_shared_subscriptions(pub_client)
    print("\n  Distribution summary:")
    for wid, stats in worker_stats.items():
        print(f"    {wid}: {stats.messages_received} tasks")

    # Part 2: Message Expiry
    print("\n--- Part 2: Message Expiry Interval ---")
    demonstrate_message_expiry(pub_client)

    # Part 3: Request/Reply
    print("\n--- Part 3: Request/Reply (ResponseTopic + CorrelationData) ---")
    trackers = demonstrate_request_reply(pub_client)
    replies_received = sum(1 for t in trackers if t.reply_received_at > 0)
    print(f"\n  Replies received: {replies_received}/{len(trackers)}")

    # Cleanup
    pub_client.loop_stop()
    pub_client.disconnect()

    print("\nDone. All clients disconnected.")


if __name__ == "__main__":
    main()
