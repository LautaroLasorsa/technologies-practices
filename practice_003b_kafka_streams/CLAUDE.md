# Practice 003b: Kafka Streams & Processing

## Technologies

- **Apache Kafka** -- Distributed event streaming platform (KRaft mode, no ZooKeeper)
- **faust-streaming** -- Python stream processing library inspired by Kafka Streams (async/await, agents, tables, windowing)
- **aiokafka** -- Async Kafka client used internally by Faust
- **Docker Compose** -- Single-node Kafka broker in KRaft mode

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Theoretical Context

### What is Faust and Why Stream Processing?

[Faust](https://faust.readthedocs.io/en/latest/) is a stream processing library that ports the ideas from Kafka Streams to Python. Originally developed at Robinhood to build high-performance distributed systems and real-time data pipelines that process billions of events every day, [Faust Streaming](https://github.com/faust-streaming/faust) (the actively maintained fork) provides both stream processing and event processing without a DSL — it's just Python with async/await. Unlike Kafka Streams (JVM-based), Faust lets Python teams apply the same stream processing patterns (agents, tables, windowing, changelogs) without leaving their language ecosystem.

Stream processing solves the problem of **continuous computation over unbounded data**. Traditional batch systems (like daily ETL jobs) introduce latency — you wait hours for results. Stream processing reacts to events as they arrive, enabling real-time analytics, monitoring, and decision-making. Systems like [Apache Flink](https://nightlies.apache.org/flink/flink-docs-master/docs/concepts/stateful-stream-processing/), Kafka Streams, and Faust are purpose-built for this: they maintain state, handle failures, and guarantee correctness (exactly-once semantics) while processing millions of events per second.

### How Faust Works: Agents, Tables, and Changelogs

Faust's core abstraction is the **agent** (borrowed from the actor model): a long-running async coroutine that consumes events from a Kafka topic, transforms them, and optionally produces to another topic. Agents run concurrently across CPU cores and machines, leveraging Python's async/await for cooperative multitasking and backpressure management. This is fundamentally different from callback-based systems (like traditional Kafka consumers) — `async for event in stream` gives you natural flow control and readable code.

**Stateful processing** is where Faust shines. A **Table** is a sharded, distributed key-value store backed by a Kafka **changelog topic**. When you update `table[key] = value`, Faust writes the change to the changelog, ensuring that if the worker crashes and restarts, it can replay the changelog to rebuild state. This is identical to Kafka Streams' StateStore concept. Tables enable patterns like counting, aggregation, and joins — things impossible with stateless stream transformations. The changelog is partitioned and co-located with the processing tasks, minimizing network hops (data locality principle).

**Windowed tables** extend this to time-based aggregation. A [tumbling window](https://www.ververica.com/stream-processing-with-apache-flink-beginners-guide) divides time into fixed, non-overlapping buckets (e.g., 60-second intervals). Each event falls into exactly one window based on its timestamp. When the window "closes" (stream time advances past it), you can emit a summary (min/max/avg). Faust tracks window state in the table, and the changelog ensures windowed state survives restarts. This is the foundation of time-series analytics in stream processing.

### Exactly-Once Semantics: What It Means and How It Works

By default, stream processors offer **at-least-once** delivery: if a worker crashes mid-processing, it replays from the last committed offset, potentially re-processing some events. For idempotent operations (like max/min), this is fine. But for non-idempotent operations (like counting or updating a database), you get duplicates.

**Exactly-once semantics** (EOS) ensures each event is processed exactly once, even across failures. [Faust's `processing_guarantee="exactly_once"`](https://faust.readthedocs.io/en/latest/userguide/tables.html) uses Kafka's transactional API: it batches table changelog writes and output topic produces into a single atomic transaction, and only commits the consumer offset if the transaction succeeds. If the worker crashes, Kafka aborts the transaction, and the next worker replays from the uncommitted offset — no duplicates escape. The tradeoff is latency (transactions add ~10-100ms per batch) and complexity. EOS is critical for financial systems, billing, and any domain where "approximately correct" isn't acceptable.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Agent** | Async coroutine that consumes from a topic, transforms events, and optionally produces to a sink. The core unit of stream processing in Faust. |
| **Topic** | Kafka topic abstracted as a typed stream. Faust auto-serializes/deserializes Records (typed message schemas). |
| **Record** | Faust's typed message schema (like Pydantic or dataclasses). Fields are validated, and Records serialize to JSON by default. |
| **Table** | Distributed key-value store backed by a Kafka changelog topic. Survives worker restarts via replay. Used for stateful processing (counting, aggregation). |
| **Changelog Topic** | Kafka topic that logs every change to a Table. Enables fault tolerance — if a worker crashes, the next one replays the changelog to rebuild state. |
| **Tumbling Window** | Fixed-size, non-overlapping time buckets (e.g., 60s). Events are assigned to one window based on timestamp. Used for time-bucketed aggregates (e.g., avg temperature per minute). |
| **Sink** | A topic that an agent produces to. If `@app.agent(topic, sink=[output_topic])`, everything the agent `yield`s is sent to the sink. |
| **Exactly-Once** | Processing guarantee where each event affects state/output exactly once, even across failures. Requires Kafka transactions (higher latency, stronger consistency). |
| **Dead-Letter Queue** | A separate topic for malformed/poison events that fail validation. Prevents bad data from blocking the pipeline. |
| **Processing Guarantee** | Either `"at_least_once"` (default, fast, duplicates possible) or `"exactly_once"` (slower, no duplicates, transactional). |

### Ecosystem Context: Faust vs Alternatives

**Faust vs Kafka Streams**: Kafka Streams is the JVM reference implementation — mature, battle-tested, rich ecosystem (KSQL, Spring Cloud Stream). Faust brings the same concepts to Python, trading JVM performance for Python's ecosystem (NumPy, TensorFlow, scikit-learn). Faust is ideal for ML pipelines, data science workflows, and teams already invested in Python.

**Faust vs Spark Structured Streaming**: Spark requires a cluster (YARN/Kubernetes) and uses micro-batch processing (latency in seconds). Faust is lightweight (just a Python process + Kafka), true streaming (sub-second latency), and easier to operationalize for small-to-medium workloads. Spark wins for petabyte-scale batch+streaming hybrid systems.

**Faust vs Flink**: [Flink](https://nightlies.apache.org/flink/flink-docs-master/docs/concepts/stateful-stream-processing/) is the gold standard for stateful stream processing — advanced windowing, event-time watermarks, savepoints. But it's JVM-based, operationally complex (clusters, JobManager, TaskManager), and has a steeper learning curve. Faust is simpler and Python-native, but lacks Flink's sophistication (no hopping windows, no session windows, no CEP). Trade-off: ease-of-use vs feature richness.

**Trade-offs**: Faust's biggest limitation is Python's GIL — CPU-bound transformations don't scale well. For I/O-bound workloads (network calls, database lookups, ML inference via external services), async/await shines. For CPU-heavy transformations, Kafka Streams or Flink's JVM parallelism wins.

## Description

Build a **Sensor Metrics Pipeline** that demonstrates real-time stream processing with Faust: agents that consume-transform-produce, stateful tables for counting, tumbling-window aggregations for time-bucketed metrics, dead-letter routing for malformed events, and exactly-once processing semantics -- all running locally against a Kafka broker in Docker.

### What you'll learn

1. **Faust fundamentals** -- App, agents, topics, Records (typed events), sinks
2. **Consume-transform-produce** -- Agents that read from one topic, transform, and write to another
3. **Stateful processing with tables** -- Distributed key/value stores backed by Kafka changelog topics
4. **Tumbling window aggregations** -- Time-bucketed metrics (e.g., average temperature per sensor per minute)
5. **Dead-letter queue pattern** -- Routing malformed/poison events to a separate topic for inspection
6. **Exactly-once semantics** -- Faust's `processing_guarantee="exactly_once"` and what it means in practice
7. **Faust lifecycle** -- Timers, cron tasks, on_started signals, and the worker CLI

## Instructions

### Phase 1: Setup & Concepts (~10 min)

1. Start Kafka with `docker compose up -d` from the practice root
2. The `app/` directory contains a uv project with faust-streaming already installed
3. Read `app/models.py` (provided) to understand the event schemas: `SensorReading`, `EnrichedReading`, `WindowAggregate`
4. Read `app/config.py` (provided) for topic and app configuration
5. Key question: How does Faust differ from Kafka Streams? (Hint: Faust is a Python library using async/await, Kafka Streams is a JVM library -- same concepts, different runtime)

### Phase 2: Stateless Stream Processing (~25 min)

1. Open `app/agents/enrichment.py`
2. **User implements:** `enrich_readings` agent -- consumes raw `SensorReading` events, validates them, enriches with a computed `status` field (normal/warning/critical based on thresholds), and produces `EnrichedReading` to a sink topic
3. **User implements:** `route_to_dead_letter` -- within the same agent, catch malformed events (e.g., negative temperature, missing sensor_id) and produce them to a dead-letter topic instead
4. Test: run the producer (`uv run python -m app.producer --include-bad`), then the Faust worker (`uv run faust -A app.main worker -l info`), and inspect output topics with `uv run python -m app.inspector enriched-readings`
5. Key question: Why does Faust use `async for event in stream` instead of a callback? (Hint: backpressure, cooperative scheduling)

### Phase 3: Stateful Counting with Tables (~20 min)

1. Open `app/agents/counting.py`
2. **User implements:** `count_by_sensor` agent -- maintains a Faust Table that counts how many readings each sensor_id has produced (total lifetime count)
3. **User implements:** A periodic check (Faust timer) that logs sensors exceeding a threshold count
4. Test: publish events, observe the table state growing
5. Key question: Where is the table data actually stored? (Hint: in-memory + Kafka changelog topic for recovery)

### Phase 4: Windowed Aggregation (~25 min)

1. Open `app/agents/windowing.py`
2. **User implements:** `aggregate_windows` agent -- uses a tumbling window table (60-second windows) to compute per-sensor min/max/avg temperature within each window
3. **User implements:** When a window closes (using Faust's `relative_to_stream()` or `.now()` accessor), produce a `WindowAggregate` summary to an output topic
4. Test: send a burst of readings, wait for the window to close, verify aggregates
5. Key question: What happens to late-arriving events (events whose timestamp falls in an already-closed window)?

### Phase 5: Exactly-Once & Wrap-Up (~15 min)

1. Open `app/main.py` and toggle `processing_guarantee="exactly_once"`
2. Discuss: What does exactly-once mean in Faust? (Hint: idempotent producers + transactional consumer-producer pairs -- not magic)
3. Observe: Faust logs showing transactional commits
4. **User implements:** Expose a simple Faust web endpoint (`@app.page`) that returns the current table state as JSON
5. Key question: Exactly-once adds latency because Faust batches commits -- when is the tradeoff worth it?

## Motivation

- **Stream processing literacy**: Kafka Streams concepts (agents, tables, windowing, changelogs) are universal across Flink, Spark Structured Streaming, and cloud-native equivalents
- **Python-native**: Faust lets you apply stream processing patterns without the JVM -- valuable for Python-heavy teams and rapid prototyping
- **Production patterns**: Dead-letter queues, exactly-once semantics, and windowed aggregations are standard requirements in real-time data pipelines
- **Complements 003a**: Builds on producer/consumer basics with higher-level abstractions for stateful processing

## References

- [Faust Documentation (faust-streaming)](https://faust-streaming.github.io/faust/)
- [Faust Agents](https://faust.readthedocs.io/en/latest/userguide/agents.html)
- [Faust Tables & Windowing](https://faust.readthedocs.io/en/latest/userguide/tables.html)
- [Faust GitHub (faust-streaming fork)](https://github.com/faust-streaming/faust)
- [Kafka Streams Concepts](https://kafka.apache.org/documentation/streams/core-concepts)
- [Uber: Reliable Reprocessing and Dead Letter Queues](https://www.uber.com/blog/reliable-reprocessing/)

## Commands

All commands are run from the practice root (`practice_003b_kafka_streams/`) unless noted otherwise.

### Infrastructure (Docker)

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Kafka broker (KRaft mode, single node) in the background |
| `docker compose down` | Stop and remove the Kafka container |
| `docker compose down -v` | Stop Kafka and delete persisted volume data (clean slate) |
| `docker compose logs -f kafka` | Tail Kafka broker logs |
| `docker compose ps` | Check Kafka container health status |

### Python Setup (run from `app/` subdirectory)

| Command | Description |
|---------|-------------|
| `cd app && uv sync` | Install Python dependencies (faust-streaming, aiokafka) |

### Phase 2: Producer & Faust Worker

| Command | Description |
|---------|-------------|
| `cd app && uv run python -m app.producer` | Produce 30 valid sensor readings (default, 1s interval) |
| `cd app && uv run python -m app.producer --count 50 --interval 0.5` | Produce 50 readings at 0.5s intervals |
| `cd app && uv run python -m app.producer --include-bad` | Produce readings with ~10% malformed events (dead-letter testing) |
| `cd app && uv run faust -A app.main worker -l info` | Start the Faust worker (all agents, info-level logging) |

### Phase 2-4: Topic Inspection

| Command | Description |
|---------|-------------|
| `cd app && uv run python -m app.inspector enriched-readings` | Inspect enriched readings output topic (default: 20 messages) |
| `cd app && uv run python -m app.inspector sensor-dead-letter` | Inspect dead-letter topic for malformed events |
| `cd app && uv run python -m app.inspector window-aggregates` | Inspect windowed aggregation output topic |
| `cd app && uv run python -m app.inspector window-aggregates --limit 5` | Inspect topic with custom message limit |
| `cd app && uv run python -m app.inspector sensor-readings` | Inspect raw sensor readings topic |

### Phase 5: Web Endpoint & Exactly-Once

| Command | Description |
|---------|-------------|
| `cd app && uv run faust -A app.main worker -l info` | Start worker (after toggling `processing_guarantee="exactly_once"` in `main.py`) |
| `curl http://localhost:6066/status/` | Query the Faust web endpoint for current table state (while worker is running) |

## State

`not-started`
