# Practice 003a: Kafka -- Producers & Consumers

## Technologies

- **Apache Kafka 3.9+** -- Distributed event streaming platform (KRaft mode, no ZooKeeper)
- **confluent-kafka** -- High-performance Python client built on librdkafka (C library)
- **Docker / Docker Compose** -- Local single-broker Kafka in KRaft mode

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Theoretical Context

Apache Kafka is a distributed event streaming platform designed for high-throughput, fault-tolerant, and scalable message processing. Originally built at LinkedIn to handle activity streams and operational metrics at massive scale, Kafka is now the de-facto standard for real-time data pipelines, event sourcing, and microservice communication. Unlike traditional message queues (RabbitMQ, Pub/Sub), Kafka is built around an immutable **commit log** that allows consumers to replay messages, making it suitable for both messaging and stream processing.

### Architecture

Kafka organizes data into **topics**, each split into one or more **partitions** for parallelism. A partition is an ordered, immutable sequence of records (the commit log). Each record within a partition has a unique **offset** (sequential ID). Kafka brokers are servers that store partitions and serve read/write requests. Partitions can be replicated across multiple brokers for fault tolerance.

**KRaft mode** (Kafka Raft): Starting with Kafka 3.x, Kafka can run without ZooKeeper. Instead, a subset of brokers form a **Raft consensus group** to manage cluster metadata (topic/partition assignments, configuration). KRaft simplifies deployment and improves scalability — this is the modern Kafka architecture.

### Producers & Consumers

**Producers** write messages to topics. Each message has:
- **Key** (optional): Used to determine which partition the message goes to (hash-based partitioning)
- **Value**: The message payload
- **Timestamp**: When the message was produced

**Consumers** read messages from topics. They track their progress via **offsets**. Kafka does not delete messages on consumption — messages are retained for a configurable period (default 7 days), allowing multiple consumers to read the same data independently.

### Consumer Groups

A **consumer group** is a set of consumers that coordinate to consume partitions of a topic. Each partition is assigned to exactly one consumer in the group (but one consumer can handle multiple partitions). This enables:
- **Horizontal scaling**: Add more consumers to process more partitions in parallel
- **Fault tolerance**: If a consumer dies, its partitions are reassigned to others (**rebalancing**)

Rebalancing is triggered when consumers join/leave or when partition count changes. During rebalancing, consumption pauses briefly.

### Ordering & Delivery Guarantees

- **Ordering**: Kafka guarantees order **within a partition**, not across partitions. Messages with the same key go to the same partition (deterministic).
- **Delivery semantics**:
  - **At-least-once** (default): Consumer might process a message multiple times if it crashes before committing offset.
  - **At-most-once**: Consumer commits offset before processing (risks data loss on crash).
  - **Exactly-once**: Requires transactions (Kafka 0.11+) — complex but possible.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Broker** | A Kafka server that stores and serves messages. |
| **Topic** | A named feed of messages (like a database table). |
| **Partition** | An ordered, immutable log within a topic. Unit of parallelism. |
| **Offset** | Sequential ID of a message within a partition (0, 1, 2, ...). |
| **Producer** | Client that writes messages to topics. |
| **Consumer** | Client that reads messages from topic partitions. |
| **Consumer Group** | Set of consumers sharing partition assignments for a topic. |
| **Rebalancing** | Reassignment of partitions when consumers join/leave a group. |
| **KRaft** | Kafka Raft — metadata managed by Kafka brokers, no ZooKeeper. |

### Ecosystem Context

Kafka competes with Google Pub/Sub (managed, simpler, no replay), RabbitMQ (lower latency, AMQP, no replay), AWS Kinesis (managed, AWS-only, more expensive), and Pulsar (newer, similar to Kafka but with tiered storage). Choose Kafka when you need **high throughput** (millions of messages/sec), **message replay** (consumers can rewind offsets), **stream processing** (Kafka Streams, ksqlDB), and **on-prem or self-managed cloud deployments**. Kafka is the industry standard for event-driven architectures, data integration (CDC with Kafka Connect), and real-time analytics.

## Description

Build an **Event Log System** that demonstrates core Kafka producer/consumer patterns: producing messages with keys and partitions, consuming with manual and automatic offset commits, consumer groups, and partition rebalancing -- all running locally against a single Kafka broker in KRaft mode.

### What you'll learn

1. **Kafka core model** -- brokers, topics, partitions, offsets, segments
2. **KRaft mode** -- Kafka without ZooKeeper: the controller quorum replaces ZK for metadata management
3. **Producers** -- serialization, partitioning (key-based vs round-robin), delivery callbacks, `acks` config
4. **Consumers** -- subscribe, poll loop, manual vs auto commit, `auto.offset.reset`
5. **Consumer groups** -- group coordination, partition assignment, rebalancing when consumers join/leave
6. **Admin API** -- programmatic topic creation with partition and replication config

### Key Kafka concepts (quick reference)

| Concept | Description |
|---------|-------------|
| **Broker** | A Kafka server that stores and serves messages |
| **Topic** | A named feed of messages (like a database table) |
| **Partition** | An ordered, immutable log within a topic. Parallelism unit |
| **Offset** | Sequential ID of a message within a partition (0, 1, 2, ...) |
| **Producer** | Client that writes messages to topics |
| **Consumer** | Client that reads messages from topic partitions |
| **Consumer Group** | Set of consumers sharing partition assignments for a topic |
| **KRaft** | Kafka Raft -- metadata managed by Kafka itself, no ZooKeeper |

## Instructions

### Phase 1: Setup & Infrastructure (~10 min)

1. Start Kafka with `docker compose up -d` from the practice root
2. Verify the broker is healthy: `docker compose logs kafka`
3. Run `uv run python admin.py` to create the practice topics
4. Key question: Why does Kafka use partitions instead of a single log per topic?

### Phase 2: Producer Basics (~25 min)

1. Open `app/producer.py` and implement the TODO(human) functions
2. **User implements:** `produce_event` -- send a single message with key, value, and delivery callback
3. **User implements:** `produce_events_batch` -- send multiple events, flush, and report results
4. Test: `uv run python producer.py` -- verify messages are produced
5. Key question: What does `acks=all` guarantee vs `acks=1`? When would you use each?

### Phase 3: Consumer Basics (~25 min)

1. Open `app/consumer.py` and implement the TODO(human) functions
2. **User implements:** `consume_events` -- poll loop with message processing and manual commit
3. Test: `uv run python consumer.py` -- verify all produced messages are consumed
4. Experiment: run consumer again -- observe no messages (offsets already committed)
5. Key question: What happens if you set `auto.offset.reset` to `earliest` vs `latest` for a new group?

### Phase 4: Consumer Groups & Rebalancing (~25 min)

1. Open `app/consumer_group_demo.py` and implement the TODO(human) functions
2. **User implements:** `run_consumer_worker` -- consumer with rebalance callback
3. **User implements:** `demonstrate_rebalancing` -- launch multiple consumers, observe partition redistribution
4. Test: produce events to a multi-partition topic, then run the demo
5. Key question: What happens to in-flight messages when a rebalance occurs? Why does Kafka revoke partitions before assigning new ones?

### Phase 5: Reflection (~5 min)

1. Compare Kafka vs Pub/Sub (practice 002): push vs pull model, retention, replay capability
2. Discuss: Why is consumer group coordination harder than it looks? (hint: exactly-once vs at-least-once)

## Motivation

- **Industry standard**: Kafka is the dominant event streaming platform in production systems (LinkedIn, Uber, Netflix, Airbnb)
- **Microservice communication**: Core skill for event-driven architectures, CQRS, and saga patterns (practices 014, 015)
- **Data pipelines**: Foundation for real-time data processing (complements Spark practice 011)
- **Career relevance**: Kafka knowledge is consistently among the most requested skills for backend and data engineering roles
- **Complements Pub/Sub (002)**: Understanding both self-hosted (Kafka) and managed (Pub/Sub) messaging enables informed architecture decisions

## References

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [KRaft: Apache Kafka Without ZooKeeper](https://developer.confluent.io/learn/kraft/)
- [confluent-kafka-python API Docs](https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html)
- [confluent-kafka-python GitHub](https://github.com/confluentinc/confluent-kafka-python)
- [Apache Kafka Docker Image](https://hub.docker.com/r/apache/kafka)
- [Kafka Consumer Group Protocol](https://developer.confluent.io/courses/architecture/consumer-group-protocol/)

## Commands

All commands are run from `practice_003a_kafka_producers/`.

### Phase 1: Docker & Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Kafka broker in KRaft mode (detached) |
| `docker compose logs kafka` | View Kafka broker logs to verify healthy startup |
| `docker compose logs -f kafka` | Follow Kafka broker logs in real time |
| `docker compose ps` | Check container status and health |
| `docker compose down` | Stop and remove the Kafka container |
| `docker compose down -v` | Stop, remove container, and delete volumes (full reset) |

### Phase 2: Python Setup & Topic Admin

| Command | Description |
|---------|-------------|
| `cd app && uv sync` | Install Python dependencies from pyproject.toml |
| `uv run python admin.py` | Create practice topics (`events` with 3 partitions, `orders` with 4 partitions) |

### Phase 3: Producer

| Command | Description |
|---------|-------------|
| `uv run python producer.py` | Produce 10 sample events to the `events` topic |

### Phase 4: Consumer

| Command | Description |
|---------|-------------|
| `uv run python consumer.py` | Consume messages from the `events` topic (manual commit, Ctrl+C to stop) |

### Phase 5: Consumer Group Rebalancing Demo

| Command | Description |
|---------|-------------|
| `uv run python consumer_group_demo.py` | Seed 40 messages to `orders` topic, then launch 3 staggered consumers to observe rebalancing |

**Note:** Phase 3-5 Python commands must be run from the `app/` subdirectory (where `pyproject.toml` lives).

## State

`not-started`
