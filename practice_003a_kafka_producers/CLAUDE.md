# Practice 003a: Kafka -- Producers & Consumers

## Technologies

- **Apache Kafka 3.9+** -- Distributed event streaming platform (KRaft mode, no ZooKeeper)
- **confluent-kafka** -- High-performance Python client built on librdkafka (C library)
- **Docker / Docker Compose** -- Local single-broker Kafka in KRaft mode

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

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
