# Practice 057 -- MQTT: Lightweight Messaging & Protocol Comparison

## Technologies

- **MQTT 5.0** -- Lightweight publish/subscribe messaging protocol (OASIS standard)
- **Eclipse Mosquitto 2** -- Open-source MQTT broker supporting v5.0, v3.1.1, v3.1
- **paho-mqtt 2.x** -- Eclipse Paho Python MQTT client (CallbackAPIVersion.VERSION2)
- **Docker / Docker Compose** -- Local Mosquitto broker

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Theoretical Context

### What MQTT Is

MQTT (Message Queuing Telemetry Transport) is a lightweight publish/subscribe messaging protocol designed for constrained devices and low-bandwidth, high-latency, or unreliable networks. Originally created by Andy Stanford-Clark (IBM) and Arlen Nipper in 1999 for monitoring oil pipelines over satellite links, it became an OASIS standard in 2014 (v3.1.1) and was updated to v5.0 in 2019.

**The problem it solves**: Traditional request/response protocols like HTTP are too heavy for IoT scenarios -- hundreds of bytes of headers per message, no persistent connections, no push-based delivery. MQTT provides a binary protocol with a **2-byte minimum fixed header**, persistent TCP connections, and broker-mediated pub/sub -- making it ideal for scenarios where thousands of constrained devices need to communicate efficiently.

### Design Philosophy

MQTT's design is driven by three constraints:

1. **Minimal overhead**: The fixed header is just 2 bytes (1 byte control + 1 byte remaining length). Compare: HTTP headers are typically 200--800 bytes, AMQP frame headers are 8 bytes. A complete MQTT PUBLISH with a short topic and small payload can be under 20 bytes total.
2. **Persistent connections**: Clients maintain a single long-lived TCP connection to the broker. No connection setup overhead per message (unlike HTTP/1.1 without keep-alive). The broker tracks connection state and can detect client failures.
3. **Push-based delivery**: The broker pushes messages to subscribers immediately -- no polling required. This reduces latency and battery consumption on constrained devices.

### How MQTT Works Internally

```
  Publisher                 Broker                  Subscriber
     |                       |                         |
     |-- CONNECT ----------->|                         |
     |<- CONNACK ------------|                         |
     |                       |<-- CONNECT -------------|
     |                       |--- CONNACK ------------>|
     |                       |<-- SUBSCRIBE -----------|
     |                       |--- SUBACK ------------->|
     |-- PUBLISH ----------->|                         |
     |                       |--- PUBLISH ------------>|
     |                       |                         |
```

**Broker**: The central hub. All messages flow through the broker -- publishers and subscribers never communicate directly. The broker handles authentication, authorization, message routing, QoS enforcement, session state, and retained messages. Popular brokers: Mosquitto (lightweight, single-node), EMQX (clustered, high-throughput), HiveMQ (enterprise).

**Topics**: Hierarchical strings using `/` as a level separator. Topics are created implicitly on first publish -- no pre-creation required (unlike Kafka topics or RabbitMQ queues).

Examples:
- `sensors/temperature/room1`
- `home/livingroom/light/status`
- `factory/line3/machine7/vibration`

**Subscriptions**: Clients subscribe to topic filters, which can include wildcards:

| Wildcard | Symbol | Matches | Example |
|----------|--------|---------|---------|
| Single-level | `+` | Exactly one level | `sensors/+/room1` matches `sensors/temperature/room1` and `sensors/humidity/room1` |
| Multi-level | `#` | Zero or more levels (must be last) | `sensors/#` matches `sensors`, `sensors/temperature`, `sensors/temperature/room1` |

**Important**: Wildcards are for subscribing only -- you cannot publish to a wildcard topic.

### QoS Levels (Critical)

MQTT defines three Quality of Service levels that control delivery guarantees. QoS is set **per message** by the publisher and can be **downgraded** by the broker based on the subscriber's requested QoS (the effective QoS is `min(publisher_qos, subscriber_qos)`).

#### QoS 0 -- At Most Once (Fire and Forget)

```
  Sender                    Receiver
    |--- PUBLISH (QoS 0) ---->|
    |         (done)          |
```

- **One packet**: PUBLISH only, no acknowledgment
- **Guarantee**: Message may be lost. No retry.
- **Use case**: High-frequency sensor data where occasional loss is acceptable (e.g., temperature readings every second)
- **Trade-off**: Lowest overhead, lowest latency, no delivery guarantee

#### QoS 1 -- At Least Once (Acknowledged Delivery)

```
  Sender                    Receiver
    |--- PUBLISH (QoS 1) ---->|
    |       (store msg)       | (process msg)
    |<-- PUBACK --------------|
    |     (delete msg)        |
```

- **Two packets**: PUBLISH + PUBACK
- **Guarantee**: Message delivered at least once. If PUBACK is lost, sender retransmits with DUP flag -- receiver may get duplicates.
- **Use case**: Important events where duplicates can be handled (e.g., device status updates with idempotent processing)
- **Trade-off**: Reliable delivery, possible duplicates

#### QoS 2 -- Exactly Once (Four-Step Handshake)

```
  Sender                    Receiver
    |--- PUBLISH (QoS 2) ---->|
    |       (store msg)       | (store packet ID)
    |<-- PUBREC --------------|
    |       (store PUBREC)    |
    |--- PUBREL ------------->|
    |                         | (process msg, delete packet ID)
    |<-- PUBCOMP -------------|
    |     (delete msg)        |
```

- **Four packets**: PUBLISH -> PUBREC -> PUBREL -> PUBCOMP
- **Guarantee**: Message delivered exactly once. The PUBREL packet acts as a boundary -- any PUBLISH received before it is a duplicate; any after is new.
- **Use case**: Critical commands where duplicates cause harm (e.g., billing events, actuator commands, financial transactions)
- **Trade-off**: Highest overhead (4x packets), highest latency, strongest guarantee

**Why the 4-step handshake?** The two-phase approach (PUBREC acknowledges receipt, PUBREL/PUBCOMP release the packet ID) ensures that even if any single packet is lost and retransmitted, the receiver can always distinguish duplicates from new messages. This is something QoS 1 cannot guarantee.

### Retained Messages

When a message is published with the **retained flag** set to `true`, the broker stores the **last** retained message for that topic. When a new client subscribes to that topic, the broker immediately delivers the retained message -- the subscriber doesn't have to wait for the next publish.

Key behaviors:
- Only **one** retained message is stored per topic (the latest one replaces the previous)
- To **clear** a retained message, publish an empty payload with retained=True to the same topic
- Retained messages are independent of sessions -- they persist even if no clients are connected
- The broker delivers retained messages with the retained flag set, so subscribers can distinguish them from live messages

**Use case**: Device status/presence. A sensor publishes its current state as retained. Any new monitoring dashboard that subscribes immediately gets the current state without waiting for the next reading.

### Last Will and Testament (LWT)

When a client connects, it can register a **will message** with the broker. If the client disconnects **ungracefully** (TCP connection drops, keepalive timeout expires, protocol error) -- but NOT on a clean `DISCONNECT` -- the broker publishes the will message on the client's behalf.

The will message has all the properties of a normal message: topic, payload, QoS, and retained flag. MQTT 5.0 adds a **Will Delay Interval** -- the broker waits a specified number of seconds before publishing the will, giving the client a chance to reconnect.

**Use case**: Device presence. Client connects with LWT `devices/sensor42/status` = "offline" (retained). On connect, it publishes "online" (retained) to the same topic. If it crashes, the broker publishes "offline" automatically.

### Clean Session vs Persistent Session

**Clean Session (MQTT 3.1.1) / Clean Start (MQTT 5.0)**:

| Setting | Behavior |
|---------|----------|
| `clean_session=True` (3.1.1) / `clean_start=True` (5.0) | Broker discards any previous session state. Subscriptions must be re-established. Queued QoS 1/2 messages are lost. |
| `clean_session=False` / `clean_start=False` | Broker resumes previous session. Subscriptions persist. QoS 1/2 messages queued while offline are delivered on reconnect. |

**Session state includes**: client subscriptions, QoS 1/2 messages pending delivery, QoS 2 messages pending acknowledgment.

MQTT 5.0 adds **Session Expiry Interval**: how long the broker keeps session state after disconnect. A value of 0 means "delete immediately" (equivalent to clean start). A value of `0xFFFFFFFF` means "never expire".

### MQTT 5.0 Features

MQTT 5.0 (2019) is the most significant protocol update, adding features for production-grade deployments:

| Feature | Description |
|---------|-------------|
| **Shared Subscriptions** | Multiple clients subscribe to `$share/<group>/<topic>`. The broker distributes messages round-robin across group members -- enabling horizontal scaling of message processing (similar to Kafka consumer groups). |
| **Message Expiry Interval** | Publisher sets TTL in seconds. Broker discards the message if it hasn't been delivered before expiry. Prevents stale data delivery to reconnecting clients. |
| **Topic Aliases** | Client and broker negotiate short numeric IDs (1-65535) to replace long topic strings, reducing per-message bandwidth. |
| **User Properties** | Arbitrary key-value string pairs attached to most packet types. Enables custom metadata (content-type, correlation IDs, routing hints) without encoding in the payload. |
| **Response Topic + Correlation Data** | Request/reply pattern: publisher sets a response topic and correlation data in the PUBLISH. The responder publishes the reply to that topic with the same correlation data. This enables RPC-style patterns over MQTT. |
| **Reason Codes** | Every ACK packet (CONNACK, PUBACK, SUBACK, etc.) includes a reason code explaining success or failure. Replaces the limited return codes of v3.1.1. |
| **Flow Control** | Client and broker negotiate `Receive Maximum` -- the max number of unacknowledged QoS 1/2 messages in flight. Prevents fast publishers from overwhelming slow subscribers. |
| **Server Disconnect** | Broker can send DISCONNECT with a reason code before closing the connection (e.g., "server shutting down", "quota exceeded"). In v3.1.1, the broker just dropped the TCP connection. |
| **Auth Packet** | Enhanced authentication with challenge/response exchanges (e.g., SCRAM, OAuth 2.0). |

### Structured Protocol Comparison

This comparison covers MQTT alongside Kafka (practice 003), RabbitMQ (practice 055), and Google Cloud Pub/Sub (practice 002) -- the four messaging systems explored in this repository.

| Dimension | MQTT | Kafka | RabbitMQ | Cloud Pub/Sub |
|-----------|------|-------|----------|---------------|
| **Protocol type** | Binary pub/sub (OASIS standard) | Binary log-based streaming (custom protocol) | AMQP 0-9-1 (OASIS), also STOMP, MQTT adapters | HTTP/gRPC (Google proprietary) |
| **Protocol overhead** | 2-byte minimum header. ~20 bytes for a small PUBLISH. | 24+ byte record header + batch framing. Optimized for throughput, not per-msg size. | 8-byte AMQP frame header + method/content frames. ~60+ bytes per message. | HTTP/gRPC headers. ~200+ bytes per request. |
| **Throughput** | 10K--100K+ msg/sec per broker (depends on QoS). Designed for many small messages. | Millions of msg/sec per broker. Designed for high-throughput event streaming. | 10K--50K msg/sec per node. Designed for flexible routing. | Scales elastically (managed). ~100K+ msg/sec per subscription. |
| **Latency** | Sub-millisecond typical (0.02 ms measured in benchmarks). Push-based, no polling. | 5--100+ ms typical (batching + disk writes). Pull-based consumer model. | Sub-millisecond to low-ms at low throughput. Increases under load. | 10--100 ms typical (network + HTTP overhead). Managed service latency. |
| **Delivery guarantees** | QoS 0 (at-most-once), QoS 1 (at-least-once), QoS 2 (exactly-once). Per-message choice. | At-least-once (default), exactly-once (with transactions). Per-producer/consumer config. | At-most-once or at-least-once (with acks + persistent queues). No exactly-once. | At-least-once (default). No exactly-once. |
| **Message persistence / replay** | Retained: only last message per topic. No replay / no log. Messages are ephemeral after delivery. | Full commit log with configurable retention (time/size). Consumers replay by seeking offsets. | Queues store unacked messages. No replay after ack. Optional dead-letter queues. | Messages retained until acked (7-day default). No offset-based replay. |
| **Ordering** | Per-topic ordering (single broker). No partitioning built in. | Per-partition ordering. No cross-partition ordering. | Per-queue FIFO (with caveats under competing consumers). | Per-message ordering key (optional). No global ordering by default. |
| **Consumer groups / load balancing** | Shared subscriptions (MQTT 5.0): `$share/<group>/<topic>`, round-robin distribution. | Consumer groups with partition assignment. Automatic rebalancing. | Competing consumers on the same queue. No built-in group coordination. | Multiple subscribers to the same subscription. Automatic load balancing. |
| **Connection model** | Long-lived TCP connection per client. Keepalive heartbeats. Broker detects disconnects. | Long-lived TCP connection. Consumer poll loop. | Long-lived TCP (AMQP channel multiplexing). Heartbeats. | HTTP/gRPC per-request or streaming pull. |
| **Topic/queue model** | Hierarchical topics with wildcards (`+`, `#`). No pre-creation needed. | Named topics with partitions. Must be pre-created (or auto-create). | Exchanges + queues + bindings. Flexible routing (direct, fanout, topic, headers). | Named topics + subscriptions. Must be pre-created. |
| **Typical deployment** | Edge/IoT: thousands to millions of lightweight devices. Single broker or small cluster. | Data center: event streaming backbone. Multi-broker clusters. | Enterprise: microservice integration, task queues. Single node or cluster. | Cloud-native: fully managed, auto-scaling. No infrastructure to manage. |
| **Deployment complexity** | Low. Single binary, minimal config. | High. Multi-broker cluster, ZooKeeper/KRaft, topic management. | Medium. Erlang runtime, cluster setup, exchange/queue topology. | None (managed). Just API calls. |
| **Best fit** | IoT telemetry, device control, edge computing, mobile push, real-time notifications | Event sourcing, log aggregation, stream processing, data pipelines, CDC | Task queues, RPC, complex routing, microservice integration | Cloud-native pub/sub, serverless triggers, cross-service events |

**Key insight**: These are not interchangeable -- each occupies a distinct niche. MQTT excels where protocol overhead and client footprint matter (IoT, edge). Kafka excels where throughput and replay matter (data pipelines). RabbitMQ excels where routing flexibility matters (enterprise). Pub/Sub excels where operational simplicity matters (cloud-native).

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Broker** | Central server that receives all messages from publishers and routes them to matching subscribers. |
| **Topic** | Hierarchical string (e.g., `sensors/temp/room1`) that messages are published to. Created implicitly. |
| **Wildcard** | `+` (single level) or `#` (multi-level) in subscription filters. Subscribe-only. |
| **QoS** | Quality of Service level (0, 1, or 2) controlling delivery guarantees per message. |
| **Retained message** | Last message on a topic stored by broker, delivered immediately to new subscribers. |
| **Last Will (LWT)** | Message published by the broker when a client disconnects ungracefully. |
| **Clean Start** | MQTT 5.0 flag controlling whether to resume a previous session or start fresh. |
| **Session Expiry** | MQTT 5.0: how long the broker keeps session state after disconnect. |
| **Shared Subscription** | MQTT 5.0: `$share/<group>/<topic>` enables load balancing across subscribers. |
| **Message Expiry** | MQTT 5.0: TTL in seconds. Broker discards undelivered messages after expiry. |
| **User Properties** | MQTT 5.0: arbitrary key-value metadata attached to PUBLISH and other packets. |
| **Response Topic** | MQTT 5.0: enables request/reply pattern. Publisher specifies where to send the response. |
| **Keepalive** | Client sends PINGREQ if idle for the keepalive interval; broker sends PINGRESP. Detects half-open connections. |

## Description

Build an **IoT Telemetry System** that demonstrates MQTT's unique features through progressively complex exercises: basic pub/sub with topic hierarchies and wildcards, QoS levels and their protocol differences, retained messages and Last Will for device presence tracking, MQTT 5.0 advanced features (shared subscriptions, message expiry, request/reply), and a structured comparison against Kafka, RabbitMQ, and Cloud Pub/Sub.

### What you'll learn

1. **MQTT fundamentals** -- broker, topics (hierarchical with `/`), subscriptions, wildcards (`+` and `#`)
2. **QoS levels** -- QoS 0 (fire-and-forget), QoS 1 (acknowledged), QoS 2 (exactly-once 4-step handshake)
3. **Retained messages** -- broker stores last message per topic, delivers to new subscribers immediately
4. **Last Will and Testament** -- broker publishes on ungraceful disconnect; device presence pattern
5. **MQTT 5.0 features** -- shared subscriptions, message expiry, user properties, request/reply with response topics
6. **Protocol comparison** -- structured analysis of MQTT vs Kafka vs RabbitMQ vs Cloud Pub/Sub across throughput, latency, reliability, and use-case fit

## Instructions

### Exercise 1: Setup & Basic Pub/Sub (~20 min)

**Context**: MQTT topics are hierarchical strings separated by `/`, and wildcard subscriptions (`+` for single-level, `#` for multi-level) let a single subscriber receive messages from many topics. Understanding this topic model is the foundation of MQTT -- unlike Kafka (flat topics with partitions) or RabbitMQ (exchanges + bindings + queues), MQTT's routing is built into the topic hierarchy itself.

1. Start Mosquitto with `docker compose up -d` from the practice root
2. Verify the broker is running: `docker compose logs mosquitto`
3. Install Python dependencies: `cd app && uv sync`
4. Open `app/basic_pubsub.py` and implement the TODO(human) functions:
   - `create_client()` -- create an MQTT 5.0 client with paho-mqtt v2 API
   - `publish_sensor_data()` -- publish to hierarchical topics like `sensors/temperature/room1`
   - `subscribe_with_wildcards()` -- subscribe with `+` and `#` wildcards, observe which messages match
5. Run: `uv run python basic_pubsub.py`
6. Key question: How does MQTT's implicit topic creation compare to Kafka's explicit topic/partition management? What are the trade-offs?

### Exercise 2: QoS Levels (~25 min)

**Context**: QoS is MQTT's most distinctive feature -- no other major messaging system offers per-message delivery guarantees at three levels. Understanding the protocol flow for each QoS level (especially the 4-step handshake for QoS 2) is essential for choosing the right trade-off between reliability and performance in real systems.

1. Open `app/qos_levels.py` and implement the TODO(human) functions:
   - `demonstrate_qos0()` -- fire-and-forget publishing. Observe no acknowledgment.
   - `demonstrate_qos1()` -- at-least-once. Use on_publish callback to observe PUBACK.
   - `demonstrate_qos2()` -- exactly-once. Observe the 4-packet flow.
   - `compare_qos_levels()` -- publish the same message at all 3 QoS levels and compare timing/behavior
2. Run: `uv run python qos_levels.py`
3. Key question: Why would you ever use QoS 0 if messages can be lost? (Hint: think about 10,000 sensors publishing every second.)

### Exercise 3: Retained Messages & Last Will (~25 min)

**Context**: Retained messages and LWT together solve the "device presence" problem -- knowing whether a device is online or offline without polling. This is a pattern unique to MQTT and is one of the main reasons it dominates IoT. No equivalent exists in Kafka (no retained messages) or standard RabbitMQ (no LWT).

1. Open `app/retained_lastwill.py` and implement the TODO(human) functions:
   - `publish_retained_status()` -- publish device status as retained message; observe that new subscribers get it immediately
   - `setup_last_will()` -- configure LWT so the broker publishes "offline" on ungraceful disconnect
   - `demonstrate_device_presence()` -- combine retained + LWT for a complete device presence system
2. Run: `uv run python retained_lastwill.py`
3. Key question: What happens if you publish a retained message with an empty payload? Why is this useful?

### Exercise 4: MQTT 5.0 Features (~25 min)

**Context**: MQTT 5.0 bridges the gap between "simple IoT protocol" and "production messaging system." Shared subscriptions enable horizontal scaling (like Kafka consumer groups), message expiry prevents stale data delivery, and response topics enable RPC-style request/reply -- all within MQTT's lightweight protocol.

1. Open `app/mqtt5_features.py` and implement the TODO(human) functions:
   - `demonstrate_shared_subscriptions()` -- subscribe multiple clients to `$share/workers/<topic>`, observe round-robin distribution
   - `demonstrate_message_expiry()` -- publish with MessageExpiryInterval, verify expired messages are not delivered
   - `demonstrate_request_reply()` -- use ResponseTopic and CorrelationData for request/reply pattern
2. Run: `uv run python mqtt5_features.py`
3. Key question: How do MQTT shared subscriptions compare to Kafka consumer groups? What does Kafka provide that shared subs don't? (Hint: think about ordering and rebalancing.)

### Exercise 5: Protocol Comparison (~25 min)

**Context**: This exercise ties together everything learned across messaging practices (002 Cloud Pub/Sub, 003 Kafka, 055 RabbitMQ, 057 MQTT). The goal is to build a structured, code-driven comparison that analyzes each protocol across key dimensions, producing a formatted report.

1. Open `app/protocol_comparison.py` and implement the TODO(human) functions:
   - `analyze_overhead()` -- compare protocol overhead (header sizes, connection setup)
   - `analyze_delivery_guarantees()` -- compare QoS/ack models across protocols
   - `analyze_scalability_patterns()` -- compare consumer groups, shared subs, partitions
   - `analyze_use_case_fit()` -- match protocols to scenarios (IoT, streaming, microservices, cloud)
   - `generate_comparison_report()` -- orchestrate all analyses and produce a formatted report
2. Run: `uv run python protocol_comparison.py`
3. Key question: If you were designing an IoT fleet management system with 100K devices sending telemetry and a cloud backend doing analytics, which combination of protocols would you use and why?

### Reflection (~5 min)

1. Compare the messaging systems you've now practiced: Pub/Sub (002), Kafka (003), and MQTT (057)
2. When would you use MQTT as a bridge to Kafka rather than using Kafka directly?
3. Discuss: Why hasn't a single "universal messaging protocol" emerged? What fundamental trade-offs prevent it?

## Motivation

- **Completes the messaging trifecta**: Kafka for event streaming (003), Cloud Pub/Sub for managed messaging (002), and now MQTT for IoT/lightweight messaging -- covering the three dominant paradigms
- **IoT & edge computing**: MQTT is the de-facto standard for IoT (used by AWS IoT Core, Azure IoT Hub, Google Cloud IoT). Edge computing and device management roles consistently require MQTT knowledge
- **Protocol design understanding**: Studying MQTT's QoS levels deepens understanding of delivery guarantees, a concept that applies to all distributed systems
- **Increasingly relevant at scale**: As IoT, connected vehicles, and industrial automation grow, MQTT usage is expanding beyond traditional IoT into real-time notifications, mobile push, and microservice communication
- **Structured comparison skill**: Evaluating protocols across dimensions (overhead, latency, guarantees, use-case fit) is a critical architectural skill for system design interviews and production decisions

## References

- [MQTT 5.0 Specification (OASIS)](https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html)
- [Eclipse Mosquitto Documentation](https://mosquitto.org/man/mosquitto-conf-5.html)
- [Eclipse Paho MQTT Python Client (GitHub)](https://github.com/eclipse-paho/paho.mqtt.python)
- [Paho MQTT Python API Docs](https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html)
- [Paho MQTT v1 to v2 Migration Guide](https://eclipse.dev/paho/files/paho.mqtt.python/html/migrations.html)
- [MQTT QoS Levels Explained (EMQ)](https://www.emqx.com/en/blog/introduction-to-mqtt-qos)
- [MQTT 5.0 New Features (EMQ)](https://www.emqx.com/en/blog/introduction-to-mqtt-5)
- [MQTT Shared Subscriptions (HiveMQ)](https://www.hivemq.com/blog/mqtt5-essentials-part7-shared-subscriptions/)
- [Mosquitto Docker Configuration (Cedalo)](https://cedalo.com/blog/mosquitto-docker-configuration-ultimate-guide/)
- [MQTT vs HTTP for IoT (EMQ)](https://www.emqx.com/en/blog/mqtt-vs-http)

## Commands

All commands are run from `practice_057_mqtt/`.

### Phase 1: Docker & Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Mosquitto broker (detached) |
| `docker compose logs mosquitto` | View Mosquitto broker logs |
| `docker compose logs -f mosquitto` | Follow Mosquitto logs in real time |
| `docker compose ps` | Check container status and health |
| `docker compose down` | Stop and remove the Mosquitto container |
| `docker compose down -v` | Stop, remove container, and delete volumes (full reset) |

### Phase 2: Python Setup

| Command | Description |
|---------|-------------|
| `cd app && uv sync` | Install Python dependencies from pyproject.toml |

### Phase 3: Exercise 1 -- Basic Pub/Sub

| Command | Description |
|---------|-------------|
| `uv run python basic_pubsub.py` | Run basic publish/subscribe with topic hierarchies and wildcards |

### Phase 4: Exercise 2 -- QoS Levels

| Command | Description |
|---------|-------------|
| `uv run python qos_levels.py` | Demonstrate QoS 0, 1, 2 and compare delivery behavior |

### Phase 5: Exercise 3 -- Retained Messages & Last Will

| Command | Description |
|---------|-------------|
| `uv run python retained_lastwill.py` | Demonstrate retained messages and Last Will (device presence) |

### Phase 6: Exercise 4 -- MQTT 5.0 Features

| Command | Description |
|---------|-------------|
| `uv run python mqtt5_features.py` | Demonstrate shared subscriptions, message expiry, and request/reply |

### Phase 7: Exercise 5 -- Protocol Comparison

| Command | Description |
|---------|-------------|
| `uv run python protocol_comparison.py` | Generate structured comparison report (MQTT vs Kafka vs RabbitMQ vs Pub/Sub) |

**Note:** Phase 3--7 Python commands must be run from the `app/` subdirectory (where `pyproject.toml` lives).

## State

`not-started`
