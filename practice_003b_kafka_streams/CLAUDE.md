# Practice 003b: Kafka Streams & Processing

## Technologies

- **Apache Kafka** -- Distributed event streaming platform (KRaft mode, no ZooKeeper)
- **faust-streaming** -- Python stream processing library inspired by Kafka Streams (async/await, agents, tables, windowing)
- **aiokafka** -- Async Kafka client used internally by Faust
- **Docker Compose** -- Single-node Kafka broker in KRaft mode

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

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
