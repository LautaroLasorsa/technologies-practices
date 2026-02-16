# Practice 055 -- RabbitMQ: Queues, Exchanges & AMQP

## Technologies

- **RabbitMQ 4.x** -- Traditional message broker implementing AMQP 0-9-1
- **pika 1.3+** -- Pure Python AMQP 0-9-1 client library for RabbitMQ
- **Docker / Docker Compose** -- Local RabbitMQ with management UI

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Theoretical Context

RabbitMQ is an open-source message broker originally built by Rabbit Technologies (now part of Broadcom/VMware) that implements the **Advanced Message Queuing Protocol (AMQP 0-9-1)**. It acts as an intermediary for messaging -- applications connect to it to publish and consume messages, enabling decoupled, asynchronous communication between services. RabbitMQ is one of the most widely deployed message brokers, used by companies like Bloomberg, Goldman Sachs, and Instagram.

### The Problem RabbitMQ Solves

In a distributed system, services need to communicate without being tightly coupled. Direct service-to-service calls create dependencies: if the receiving service is down, the sender blocks or fails. A message broker sits between producers and consumers, accepting messages from producers and delivering them to consumers. This provides:

- **Temporal decoupling**: Producers and consumers don't need to be running simultaneously
- **Load leveling**: The broker buffers messages during traffic spikes, letting consumers process at their own pace
- **Flexible routing**: Messages can be routed to one consumer, many consumers, or specific consumers based on rules
- **Reliability**: Messages are persisted and acknowledged, preventing data loss

### AMQP 0-9-1 Protocol

AMQP (Advanced Message Queuing Protocol) is a wire-level protocol that defines how messages are structured, routed, queued, and acknowledged. RabbitMQ's core model implements AMQP 0-9-1 with extensions. The key insight of AMQP is that **producers never send messages directly to queues**. Instead, producers send messages to **exchanges**, which route them to queues based on **bindings** and **routing keys**.

The message flow is: `Producer --> Exchange --> [Binding Rules] --> Queue --> Consumer`

### Core Architecture

#### Connections and Channels

A **connection** is a TCP connection between an application and the RabbitMQ broker. Creating TCP connections is expensive, so AMQP multiplexes multiple **channels** over a single connection. A channel is a lightweight virtual connection inside a real TCP connection. Every AMQP operation (publish, consume, declare) happens on a channel. Rule of thumb: one channel per thread, one connection per application.

#### Virtual Hosts (vhosts)

A **virtual host** is a logical grouping of resources (exchanges, queues, bindings, users, permissions). It provides multi-tenancy within a single broker instance -- different applications or environments can share one RabbitMQ server with full isolation. The default vhost is `/`.

#### Exchanges

An exchange receives messages from producers and routes them to zero or more queues based on the exchange type and binding rules. When a producer publishes a message, it specifies:
- The **exchange name** (which exchange to send to)
- A **routing key** (a string used by the exchange's routing algorithm)

Exchanges are defined by these attributes:
- **Name**: Identifier used by producers
- **Type**: Routing algorithm (direct, fanout, topic, headers)
- **Durability**: Survives broker restart if durable
- **Auto-delete**: Deleted when the last queue unbinds

#### Queues

A **queue** is a buffer that stores messages. Queues have these key properties:
- **Name**: Identifier used by consumers (can be server-generated)
- **Durable**: Queue definition survives broker restart (messages survive only if also marked persistent)
- **Exclusive**: Used by only one connection and deleted when that connection closes
- **Auto-delete**: Deleted when the last consumer unsubscribes

Messages in a queue are delivered to consumers in **FIFO order**. Unlike Kafka's log, once a message is acknowledged by a consumer, it is **removed from the queue** -- there is no replay.

#### Bindings

A **binding** is a rule that links an exchange to a queue. It can include a **binding key** (sometimes called routing key) that the exchange uses to filter which messages go to which queue. The semantics of the binding key depend on the exchange type.

### Exchange Types

#### 1. Direct Exchange

Routes messages to queues whose **binding key exactly matches** the message's routing key. This is the simplest and most common exchange type.

Example: A queue bound with key `"payment.processed"` only receives messages published with routing key `"payment.processed"`.

The **default exchange** (empty string `""`) is a special direct exchange pre-declared by RabbitMQ. Every queue is automatically bound to it with a binding key equal to the queue's name. This lets you publish directly to a queue by name without explicitly creating bindings.

#### 2. Fanout Exchange

Routes messages to **all bound queues**, ignoring the routing key entirely. This is broadcast/pub-sub: every consumer with a queue bound to the fanout exchange gets a copy of every message.

Use cases: broadcasting events to all services, distributing log messages to multiple outputs.

#### 3. Topic Exchange

Routes messages based on **wildcard pattern matching** between the routing key and binding patterns. Routing keys are dot-separated words (e.g., `"order.created.us"`). Binding patterns can use:
- `*` (star) -- matches **exactly one word**: `"order.*.us"` matches `"order.created.us"` but NOT `"order.created.shipped.us"`
- `#` (hash) -- matches **zero or more words**: `"order.#"` matches `"order.created"`, `"order.created.us"`, and just `"order"`

Topic exchanges are the most flexible -- a binding key of `"#"` makes it behave like fanout, and a binding key without wildcards makes it behave like direct.

#### 4. Headers Exchange

Routes based on **message header attributes** instead of routing keys. When binding a queue, you specify a set of header key-value pairs and an `x-match` argument:
- `x-match: all` -- **all** specified headers must match (AND logic)
- `x-match: any` -- **at least one** header must match (OR logic)

Headers exchanges are less common but useful when routing logic is too complex for string-based routing keys.

### Acknowledgments and Reliability

#### Consumer Acknowledgments

After delivering a message, RabbitMQ needs to know when to remove it from the queue. AMQP provides three acknowledgment methods:

| Method | Effect |
|--------|--------|
| `basic_ack` | Positive acknowledgment -- message processed successfully, remove from queue |
| `basic_nack` | Negative acknowledgment -- processing failed, optionally requeue (RabbitMQ extension, supports multiple messages) |
| `basic_reject` | Negative acknowledgment -- processing failed, optionally requeue (AMQP standard, single message only) |

**Auto-ack mode** (`auto_ack=True`): RabbitMQ removes the message immediately after delivery. Fast, but if the consumer crashes during processing, the message is lost. Suitable only for non-critical messages.

**Manual ack mode** (`auto_ack=False`): Consumer explicitly acknowledges after processing. If the consumer disconnects without acking, RabbitMQ requeues the message. This provides **at-least-once delivery** (messages may be delivered more than once if ack is lost).

#### Prefetch Count (QoS)

`basic_qos(prefetch_count=N)` limits the number of unacknowledged messages a consumer can hold. Without a prefetch limit, RabbitMQ pushes all available messages to the consumer as fast as possible, which can overwhelm it. With `prefetch_count=1`, RabbitMQ waits for an ack before sending the next message -- this enables **fair dispatch** among multiple consumers.

#### Message Durability

For messages to survive a broker restart, **three things** must be true:
1. The **exchange** must be declared as durable
2. The **queue** must be declared as durable
3. The **message** must be published with `delivery_mode=2` (persistent)

Even persistent messages are not 100% guaranteed -- there's a short window between receiving and writing to disk. For stronger guarantees, use **publisher confirms**.

#### Publisher Confirms

An extension to AMQP where the broker confirms (acks back to the producer) that it has received and persisted a message. In pika, you enable this with `channel.confirm_delivery()`, and then `basic_publish` raises `UnroutableError` if the message cannot be routed.

### Dead Letter Exchanges (DLX) and TTL

A **dead letter exchange** is an exchange that receives messages that were "dead-lettered" from a queue. A message is dead-lettered when:
1. It is **rejected** or **nack'd** by a consumer (with `requeue=False`)
2. The message's **TTL expires** (time-to-live)
3. The **queue length limit** is exceeded (oldest messages are dropped)

You configure a DLX on a queue with the `x-dead-letter-exchange` argument. Optionally, `x-dead-letter-routing-key` overrides the original routing key.

**TTL (Time-To-Live)** can be set:
- **Per-queue**: All messages in the queue expire after N milliseconds (`x-message-ttl` argument)
- **Per-message**: Set the `expiration` property on individual messages (milliseconds as string)

When both are set, the lower value wins.

DLX + TTL is a common pattern for **delayed retry**: failed messages go to a DLX queue with a TTL, then are re-routed back to the original queue after the delay expires.

### Comparison with Other Messaging Systems

| Feature | RabbitMQ | Apache Kafka | Google Pub/Sub | MQTT |
|---------|----------|-------------|----------------|------|
| **Model** | Queue-based broker (AMQP) | Distributed log (append-only) | Managed pub/sub | Lightweight pub/sub |
| **Message lifetime** | Removed after ack | Retained for configured period | Retained until ack (7 days max) | Fire-and-forget or QoS 1/2 |
| **Replay** | No (consumed = gone) | Yes (consumers track offsets) | Limited (seek to timestamp) | No |
| **Routing** | Flexible (4 exchange types) | Topic + partition key | Topic + filtering | Topic hierarchy |
| **Ordering** | Per-queue FIFO | Per-partition only | Per-subscription (ordering key) | Per-topic (QoS dependent) |
| **Throughput** | ~50K msg/s per node | Millions msg/s per cluster | Managed, auto-scales | Very high (lightweight protocol) |
| **Consumer model** | Push (broker pushes to consumer) | Pull (consumer polls broker) | Push (subscriber pulls/pushes) | Push |
| **Best for** | Task queues, RPC, complex routing | Event streaming, data pipelines | Cloud-native, serverless | IoT, constrained devices |
| **Protocol** | AMQP 0-9-1 (also STOMP, MQTT) | Custom binary protocol | gRPC / HTTP | MQTT 3.1.1 / 5.0 |

**Key distinction**: RabbitMQ is a **smart broker / dumb consumer** model -- the broker tracks what has been delivered and acknowledged. Kafka is a **dumb broker / smart consumer** model -- consumers track their own position (offset) in the log. RabbitMQ excels at complex routing and task distribution; Kafka excels at event streaming and replay.

### Key Concepts Table

| Concept | Description |
|---------|-------------|
| **Broker** | The RabbitMQ server that receives, stores, and delivers messages |
| **Connection** | A TCP connection between an application and the broker |
| **Channel** | A lightweight virtual connection multiplexed over a single TCP connection |
| **Exchange** | Receives messages from producers and routes them to queues via bindings |
| **Queue** | A named buffer that stores messages until consumers process them |
| **Binding** | A rule linking an exchange to a queue, with an optional binding key |
| **Routing Key** | A string attribute on a message used by exchanges to determine routing |
| **Virtual Host** | Logical grouping of resources for multi-tenancy isolation |
| **Consumer Tag** | A unique identifier for a consumer subscription on a channel |
| **Prefetch Count** | Max number of unacknowledged messages a consumer can hold (QoS) |
| **Dead Letter Exchange** | An exchange that receives rejected, expired, or overflow messages |
| **TTL** | Time-to-live -- how long a message or queue can exist before expiring |
| **Publisher Confirm** | Broker acknowledgment back to the producer that a message was persisted |
| **Durable** | Exchange/queue survives broker restart; persistent message is written to disk |

## Description

Build a **message routing system** that demonstrates all four AMQP exchange types (direct, fanout, topic, headers), consumer acknowledgment patterns, QoS/prefetch control, dead letter exchanges, and TTL-based expiration -- all running against a local RabbitMQ broker with Docker.

### What you'll learn

1. **AMQP 0-9-1 model** -- connections, channels, exchanges, queues, bindings
2. **Exchange types** -- direct (exact match), fanout (broadcast), topic (wildcard patterns), headers (attribute-based)
3. **Infrastructure as code** -- declaring exchanges, queues, and bindings programmatically
4. **Consumer acknowledgments** -- manual ack, nack, reject, requeue behavior
5. **QoS & prefetch** -- controlling message delivery rate to consumers
6. **Message durability** -- persistent messages, durable queues, publisher confirms
7. **Dead letter exchanges** -- handling rejected/expired messages with DLX and TTL

## Instructions

### Exercise 1: Setup & Infrastructure (~15 min)

Start RabbitMQ and declare the messaging infrastructure (exchanges, queues, bindings). This exercise teaches you how RabbitMQ resources are created programmatically -- in production, infrastructure is usually declared at application startup, not manually via the management UI.

1. Start RabbitMQ: `docker compose up -d` from the practice root
2. Verify health: `docker compose ps` (wait for "healthy" status)
3. Open management UI at `http://localhost:15672` (guest/guest) -- explore the Exchanges, Queues, and Bindings tabs
4. Run `cd app && uv sync` to install dependencies
5. Open `app/setup_infrastructure.py` and implement the TODO(human) functions:
   - `declare_exchanges()` -- create direct, fanout, topic, and headers exchanges
   - `declare_queues()` -- create queues with various properties (durable, exclusive, TTL)
   - `create_bindings()` -- bind queues to exchanges with routing/binding keys
6. Run: `uv run python setup_infrastructure.py`
7. Verify in the management UI that all exchanges, queues, and bindings appear
8. **Key question**: Why does AMQP separate exchanges from queues instead of having producers publish directly to queues?

### Exercise 2: Direct Exchange -- Route by Routing Key (~20 min)

The direct exchange is the workhorse of RabbitMQ -- it routes messages based on exact routing key match. This exercise teaches the fundamental publish/consume loop and how routing keys determine message destinations.

1. Open `app/direct_exchange.py` and implement the TODO(human) functions:
   - `publish_with_routing_key()` -- publish messages with specific routing keys to the direct exchange
   - `consume_from_queue()` -- subscribe to a queue and process messages with a callback
2. Run: `uv run python direct_exchange.py`
3. Observe which messages arrive at which queues based on their routing keys
4. **Key question**: What happens if you publish with a routing key that no queue is bound to? (Answer: the message is silently dropped unless the `mandatory` flag is set)

### Exercise 3: Fanout & Topic Exchanges -- Broadcast and Pattern Matching (~25 min)

Fanout exchanges broadcast to all bound queues (like pub/sub), while topic exchanges enable powerful pattern-based routing. This exercise shows why RabbitMQ's routing model is more flexible than Kafka's simple topic/partition model.

1. Open `app/fanout_topic_exchange.py` and implement the TODO(human) functions:
   - `publish_to_fanout()` -- broadcast messages to all bound queues (routing key ignored)
   - `publish_to_topic()` -- publish messages with multi-segment routing keys (e.g., `"order.created.us"`)
   - `consume_and_display()` -- consume from multiple queues and show which patterns matched
2. Run: `uv run python fanout_topic_exchange.py`
3. Observe: fanout delivers to ALL queues; topic delivers based on `*` and `#` wildcard matches
4. **Key question**: How would you implement a topic exchange binding that receives ALL messages? (Answer: bind with `"#"`)

### Exercise 4: Consumer Acknowledgments, Durability & QoS (~25 min)

Message acknowledgment is what makes RabbitMQ reliable. This exercise teaches the difference between auto-ack and manual ack, how prefetch controls consumer load, and what happens when consumers reject messages.

1. Open `app/consumer_acks.py` and implement the TODO(human) functions:
   - `consume_with_manual_ack()` -- process messages with explicit ack/nack based on content
   - `consume_with_prefetch()` -- set prefetch_count and observe fair dispatch
   - `publish_durable_messages()` -- publish persistent messages with publisher confirms
2. Run: `uv run python consumer_acks.py`
3. Observe: messages that are nack'd with `requeue=True` are redelivered; prefetch limits how many unacked messages the consumer holds
4. **Key question**: Why is `auto_ack=True` dangerous for important messages? What delivery guarantee does manual ack provide?

### Exercise 5: Dead Letter Exchanges and TTL (~20 min)

Dead letter exchanges handle the "failure path" -- what happens when messages expire, are rejected, or overflow. Combined with TTL, DLX enables delayed retry patterns. This is a production-essential pattern for building resilient messaging systems.

1. Open `app/dead_letter.py` and implement the TODO(human) functions:
   - `setup_dead_letter_infrastructure()` -- declare queues with DLX arguments and TTL
   - `publish_and_expire()` -- publish messages that will expire via TTL and land in the DLX queue
   - `reject_to_dead_letter()` -- consume and reject messages, sending them to the DLX
2. Run: `uv run python dead_letter.py`
3. Observe: expired and rejected messages appear in the dead letter queue
4. **Key question**: How would you implement a retry mechanism using DLX + TTL? (Hint: dead letter queue routes back to the original exchange after a delay)

### Phase 6: Reflection (~5 min)

1. Compare RabbitMQ (this practice) with Kafka (practice 003a): When would you choose each?
2. Think about: RabbitMQ's push model vs Kafka's pull model -- which is better for bursty workloads?
3. Consider: How does RabbitMQ's "message removed after ack" model affect system design differently from Kafka's "log retention"?

## Motivation

- **Complements Kafka (003a)**: Understanding both queue-based (RabbitMQ) and log-based (Kafka) messaging enables informed architecture decisions
- **Industry standard**: RabbitMQ is one of the most widely deployed message brokers, used extensively for task queues, microservice communication, and event-driven architectures
- **AMQP protocol**: Learning AMQP gives insight into how message brokers work at the protocol level, transferable to other AMQP brokers
- **Routing flexibility**: RabbitMQ's exchange/binding model is uniquely powerful for complex message routing scenarios that Kafka cannot express natively
- **Production patterns**: Dead letter exchanges, TTL, publisher confirms, and consumer acknowledgments are essential patterns for building reliable messaging systems
- **Foundation for practices 014 (SAGA) and 015 (CQRS)**: Both patterns often use RabbitMQ for inter-service messaging

## References

- [AMQP 0-9-1 Model Explained -- RabbitMQ](https://www.rabbitmq.com/tutorials/amqp-concepts)
- [RabbitMQ Exchanges Documentation](https://www.rabbitmq.com/docs/exchanges)
- [Consumer Acknowledgements and Publisher Confirms](https://www.rabbitmq.com/docs/confirms)
- [Dead Letter Exchanges](https://www.rabbitmq.com/docs/dlx)
- [Time-To-Live and Expiration](https://www.rabbitmq.com/docs/ttl)
- [pika Documentation](https://pika.readthedocs.io/)
- [pika GitHub -- Pure Python AMQP 0-9-1 Client](https://github.com/pika/pika)
- [RabbitMQ Docker Image](https://hub.docker.com/_/rabbitmq)
- [RabbitMQ Tutorials (Python/pika)](https://www.rabbitmq.com/tutorials/tutorial-one-python)
- [Kafka vs RabbitMQ -- AWS Comparison](https://aws.amazon.com/compare/the-difference-between-rabbitmq-and-kafka/)

## Commands

All commands are run from `practice_055_rabbitmq/`.

### Phase 1: Docker & Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start RabbitMQ broker with management UI (detached) |
| `docker compose ps` | Check container status and health |
| `docker compose logs rabbitmq` | View RabbitMQ broker logs |
| `docker compose logs -f rabbitmq` | Follow RabbitMQ broker logs in real time |
| `docker compose down` | Stop and remove the RabbitMQ container |
| `docker compose down -v` | Stop, remove container, and delete volumes (full reset) |

### Phase 2: Python Setup

| Command | Description |
|---------|-------------|
| `cd app && uv sync` | Install Python dependencies from pyproject.toml |

### Phase 3: Exercises

| Command | Description |
|---------|-------------|
| `uv run python setup_infrastructure.py` | Exercise 1: Declare exchanges, queues, and bindings |
| `uv run python direct_exchange.py` | Exercise 2: Publish/consume with direct exchange routing |
| `uv run python fanout_topic_exchange.py` | Exercise 3: Fanout broadcast and topic wildcard routing |
| `uv run python consumer_acks.py` | Exercise 4: Manual acks, prefetch QoS, publisher confirms |
| `uv run python dead_letter.py` | Exercise 5: Dead letter exchanges and TTL expiration |

### Management UI

| Command | Description |
|---------|-------------|
| Open `http://localhost:15672` in browser | RabbitMQ Management UI (login: guest / guest) |

**Note:** Phase 3 Python commands must be run from the `app/` subdirectory (where `pyproject.toml` lives).

## State

`not-started`
