# Practice 003d: Kafka Connect & Change Data Capture

## Technologies

- **Apache Kafka 3.9+** -- Distributed event streaming platform (KRaft mode)
- **Kafka Connect** -- Framework for streaming data between Kafka and external systems
- **Debezium 2.7** -- Open-source CDC platform built on Kafka Connect
- **PostgreSQL 16** -- Source and sink databases with logical replication support
- **Docker / Docker Compose** -- Local multi-container environment (Kafka, Connect, 2x PostgreSQL)
- **confluent-kafka** -- High-performance Python client for consuming CDC events
- **requests** -- HTTP client for Kafka Connect REST API management
- **psycopg2** -- PostgreSQL client for generating source database changes

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Theoretical Context

### What is Change Data Capture (CDC)?

Change Data Capture is a pattern for observing all data changes written to a database and extracting them as a stream of events. Instead of polling the database ("has anything changed?"), CDC taps into the database's own change mechanism -- in PostgreSQL, this is the **Write-Ahead Log (WAL)**. Every INSERT, UPDATE, and DELETE is already recorded in the WAL for crash recovery; CDC simply reads this log and converts it into events that other systems can consume.

CDC solves the **dual-write problem**: when an application writes to a database AND needs to notify other systems (search index, cache, analytics), writing to both is unreliable (what if one fails?). With CDC, you write only to the database, and a separate process (Debezium) reads the change log and publishes events to Kafka. This is both simpler and more reliable.

### Kafka Connect Architecture

[Kafka Connect](https://docs.confluent.io/platform/current/connect/index.html) is a framework for scalable, fault-tolerant streaming of data between Apache Kafka and other systems. It is NOT a single connector -- it's a **runtime that hosts connectors**.

**Core components:**

| Component | Description |
|-----------|-------------|
| **Worker** | A JVM process that runs connectors and tasks. In **standalone mode**, one worker runs everything. In **distributed mode** (used in this practice), multiple workers form a **group** and distribute tasks among themselves. Workers communicate via internal Kafka topics (`_connect-configs`, `_connect-offsets`, `_connect-status`). |
| **Connector** | A logical unit that defines WHERE data comes from/goes to. A connector is either a **source connector** (external system -> Kafka) or a **sink connector** (Kafka -> external system). The connector itself doesn't move data -- it creates **tasks** that do. |
| **Task** | The actual unit of work. A connector creates one or more tasks, each responsible for a subset of the data (e.g., one task per database table, or one per partition). Tasks run inside workers and can be redistributed if a worker fails. |
| **Converter** | Serializes/deserializes data between the connector's internal format and the Kafka message format. Common converters: `JsonConverter` (human-readable, no schema registry), `AvroConverter` (compact, requires Schema Registry), `ProtobufConverter`. |
| **Transform (SMT)** | Single Message Transforms modify records in-flight. Applied after the connector produces a record (source) or before delivering to the connector (sink). Examples: rename fields, route to different topics, extract nested structures. |

**REST API:** Kafka Connect exposes a REST API on port 8083 for all management operations. There is no CLI -- everything is HTTP:
- `GET /connectors` -- list connectors
- `PUT /connectors/{name}/config` -- create or update (idempotent)
- `GET /connectors/{name}/status` -- check health
- `DELETE /connectors/{name}` -- remove connector

**Fault tolerance in distributed mode:** Connector configs are stored in a compacted Kafka topic (`_connect-configs`). If a worker crashes, its tasks are rebalanced to surviving workers. Offset tracking (what has been processed) is stored in `_connect-offsets`, so tasks resume where they left off.

### Debezium CDC Mechanics

[Debezium](https://debezium.io/) is the de-facto standard for database CDC on Kafka Connect. It supports PostgreSQL, MySQL, MongoDB, SQL Server, Oracle, and more.

**How Debezium captures PostgreSQL changes:**

1. **Logical Replication Slot**: Debezium creates a [replication slot](https://www.postgresql.org/docs/current/logicaldecoding.html) in PostgreSQL. A replication slot is a server-side cursor that tracks which WAL position has been read. PostgreSQL guarantees it will NOT delete WAL segments that haven't been consumed by the slot -- this prevents data loss but can cause disk bloat if the consumer is down.

2. **Logical Decoding Plugin**: The slot uses `pgoutput`, PostgreSQL's built-in logical decoding plugin (since v10). It converts binary WAL entries into a structured stream of changes. No extension installation needed.

3. **Publication**: A PostgreSQL publication defines which tables are included in the replication stream. Debezium auto-creates one (configurable via `publication.name` and `publication.autocreate.mode`).

4. **Initial Snapshot**: On first startup (with `snapshot.mode=initial`), Debezium reads all existing rows as "snapshot" events (op=`r`). Then it switches to streaming WAL changes. This ensures the consumer gets a complete picture of the data.

5. **REPLICA IDENTITY**: PostgreSQL's `REPLICA IDENTITY` setting controls what's included in WAL entries for UPDATE/DELETE:
   - `DEFAULT`: Only the primary key (no "before" image for non-PK columns)
   - `FULL`: All columns in both before and after images
   - `NOTHING`: Only the primary key, no before image at all

### Change Event Envelope Structure

Every Debezium CDC event has a standardized [envelope](https://debezium.io/documentation/reference/stable/connectors/postgresql.html#postgresql-events):

```json
{
  "before": { "id": 1, "name": "Laptop", "price": 999.99 },
  "after":  { "id": 1, "name": "Laptop", "price": 899.99 },
  "source": {
    "version": "2.7.0.Final",
    "connector": "postgresql",
    "name": "cdc",
    "ts_ms": 1700000000000,
    "db": "source_db",
    "schema": "public",
    "table": "products",
    "txId": 501,
    "lsn": 23456789
  },
  "op": "u",
  "ts_ms": 1700000001000
}
```

**Operation codes:**

| Code | Meaning | `before` | `after` |
|------|---------|----------|---------|
| `r` | Read (snapshot) | null | row data |
| `c` | Create (INSERT) | null | new row |
| `u` | Update (UPDATE) | old row | new row |
| `d` | Delete (DELETE) | old row | null |

**Key**: The Kafka message key contains the primary key of the affected row (e.g., `{"id": 1}`). This ensures all changes to the same row go to the same Kafka partition, maintaining ordering per row.

### Log Compaction

[Log compaction](https://docs.confluent.io/kafka/design/log_compaction.html) is an alternative to time-based retention. Instead of deleting messages older than `retention.ms`, compaction keeps only the **latest value for each key** and discards older values.

**How it works:**
1. Kafka partitions are stored as ordered **segments** (files on disk). Only the newest segment is "active" (receiving writes).
2. The **log cleaner** background thread identifies closed segments with duplicate keys.
3. It recopies the segment, keeping only the latest record per key, discarding older duplicates.
4. **Tombstones** (messages with a key and null value) mark a key for deletion. After `delete.retention.ms`, the tombstone itself is removed.

**Why this matters for CDC:** Debezium topics use compaction by default. For a table with 1M rows that gets updated frequently, compaction ensures the topic holds at most ~1M messages (current state per row) instead of growing unboundedly. A new consumer can read the compacted topic to reconstruct the full table state.

**Key settings:**
- `cleanup.policy=compact` -- enable compaction
- `segment.ms` / `segment.bytes` -- when to roll a new segment (compaction only runs on closed segments)
- `min.cleanable.dirty.ratio` -- fraction of log that must be "dirty" before compaction triggers
- `delete.retention.ms` -- how long tombstones are kept after compaction

### Single Message Transforms (SMTs)

[SMTs](https://docs.confluent.io/platform/current/connect/transforms/overview.html) are lightweight, in-flight message transformations applied by Kafka Connect. They modify records between the connector and Kafka (source) or between Kafka and the connector (sink).

**Key SMT for CDC: `ExtractNewRecordState`** (Debezium-specific)

Debezium events have a nested envelope (before/after/op/source). Most sink connectors expect flat records. The [ExtractNewRecordState](https://debezium.io/documentation/reference/stable/transformations/event-flattening.html) SMT "unwraps" the envelope:
- For INSERT/UPDATE: extracts the `after` payload as a flat record
- For DELETE: can drop the event, rewrite with a `__deleted` flag, or pass through
- For snapshots: extracts the `after` payload

**Predicates:** Since Kafka 2.6, SMTs support [predicates](https://docs.confluent.io/platform/current/connect/transforms/overview.html) to conditionally apply transforms. Example: only apply a transform to messages matching a topic name pattern.

### Dead Letter Queues (DLQ)

When Kafka Connect encounters a [record it cannot process](https://www.confluent.io/blog/kafka-connect-deep-dive-error-handling-dead-letter-queues/) (deserialization error, transform failure), it can:
- **Fail** (default): Stop the task immediately (`errors.tolerance=none`)
- **Skip**: Log and skip bad records (`errors.tolerance=all`)
- **Route to DLQ**: Send bad records to a separate topic (`errors.deadletterqueue.topic.name`)

DLQ is critical in production CDC pipelines where you can't afford to stop processing because of one malformed event. The DLQ topic can be monitored and bad events reprocessed after fixing.

### Ecosystem Context

Debezium + Kafka Connect is the dominant CDC solution in the Kafka ecosystem. Alternatives:
- **AWS DMS**: Managed CDC for AWS databases (limited to AWS)
- **Airbyte/Fivetran**: Managed ELT platforms with CDC capabilities (higher latency, easier to operate)
- **pg_logical/pglogical**: PostgreSQL-native replication (no Kafka integration)
- **Maxwell's Daemon**: MySQL-only CDC to Kafka (simpler than Debezium for MySQL)

Choose Debezium when you need: database-agnostic CDC, Kafka integration, exactly-once semantics, and fine-grained control over the event stream.

## Description

Build a **CDC pipeline** that captures PostgreSQL changes via Debezium, streams them through Kafka, and replicates them to a second database:

1. **Kafka Connect management** -- interact with the Connect REST API to register, monitor, and manage connectors
2. **Debezium source connector** -- configure PostgreSQL CDC with WAL, replication slots, and snapshots
3. **CDC event consumption** -- parse and display Debezium change event envelopes in real time
4. **JDBC sink connector** -- replicate captured changes to a sink database using the ExtractNewRecordState SMT
5. **Log compaction** -- understand key-based retention and its role in CDC topics

### What you'll learn

1. **Kafka Connect REST API** -- the HTTP control plane for connectors (no CLI exists)
2. **Debezium PostgreSQL CDC** -- WAL, replication slots, pgoutput, snapshots, REPLICA IDENTITY
3. **Change event structure** -- the Debezium envelope (op, before, after, source metadata)
4. **SMTs** -- ExtractNewRecordState for envelope flattening
5. **JDBC sink replication** -- upsert mode, schema evolution, primary key mapping
6. **Log compaction** -- how Kafka retains only the latest value per key

## Instructions

### Phase 1: Infrastructure Setup (~10 min)

1. Start all services with `docker compose up -d` from the practice root
2. Wait for all containers to be healthy: `docker compose ps`
   - Kafka Connect takes the longest (~30-60s) because it waits for Kafka and both PostgreSQL instances
3. Run `cd app && uv sync` to install Python dependencies
4. Run `uv run python connect_manager.py` to verify Connect is running and list available plugins
   - You should see `PostgresConnector` (source) and `JdbcSinkConnector` (sink) in the plugins list
5. Key question: Why does Kafka Connect store its configuration in Kafka topics instead of a config file?

### Phase 2: Kafka Connect REST API (~15 min)

1. Open `app/connect_manager.py` and implement the TODO(human) functions
2. **User implements:** `list_connectors` -- GET /connectors to list registered connectors
3. **User implements:** `create_connector` -- PUT /connectors/{name}/config for idempotent create/update
4. **User implements:** `get_connector_status` -- GET /connectors/{name}/status to check health
5. **User implements:** `delete_connector` -- DELETE /connectors/{name} with proper status handling
6. **User implements:** `wait_for_connector_running` -- polling loop until RUNNING or failure
7. Test: `uv run python connect_manager.py` -- should list plugins and show "(none)" for connectors
8. Key question: Why use PUT instead of POST for creating connectors?

### Phase 3: Debezium Source Connector (~20 min)

1. Open `app/source_connector.py` and implement the TODO(human) functions
2. **User implements:** `build_debezium_config` -- build the connector configuration dict with all required properties (connector.class, database connection, topic.prefix, table.include.list, plugin.name, slot.name, snapshot.mode, etc.)
3. **User implements:** `register_source_connector` -- use connect_manager to register and verify the connector
4. Test: `uv run python source_connector.py` -- connector should reach RUNNING state
5. Verify topics were created: `docker exec kafka-connect-broker /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list` -- look for `cdc.public.products` and `cdc.public.orders`
6. Key question: What happens to WAL segments if Debezium is stopped for a long time? Why is this a production concern?

### Phase 4: CDC Events & Real-Time Changes (~20 min)

1. Open `app/cdc_consumer.py` and implement the TODO(human) functions
2. **User implements:** `parse_cdc_event` -- extract operation type, before/after images, and source metadata from the Debezium envelope
3. **User implements:** `consume_cdc_events` -- Kafka consumer poll loop with JSON deserialization and human-readable output
4. Start the consumer in one terminal: `uv run python cdc_consumer.py`
   - You should immediately see SNAPSHOT events (op='r') for the 5 seed products
5. In a second terminal, run: `uv run python data_changes.py`
   - Watch the CDC consumer display INSERT, UPDATE, and DELETE events in real time
6. Key question: Why does Debezium produce TWO messages for a DELETE (the delete event + a tombstone)?

### Phase 5: JDBC Sink & Database Replication (~20 min)

1. Open `app/sink_connector.py` and implement the TODO(human) functions
2. **User implements:** `build_jdbc_sink_config` -- build the JDBC sink configuration with ExtractNewRecordState SMT, upsert mode, and schema evolution
3. **User implements:** `register_sink_connector` -- register and verify the sink connector
4. Test: `uv run python sink_connector.py` -- sink connector should reach RUNNING state
5. Verify replication in the sink database:
   ```
   docker exec -it postgres-sink psql -U sink_user -d sink_db -c "SELECT * FROM \"cdc.public.products\";"
   ```
6. Run `uv run python data_changes.py` again and verify changes appear in the sink
7. Key question: Why is `insert.mode=upsert` necessary instead of just `insert`?

### Phase 6: Log Compaction (~15 min)

1. Open `app/compaction_demo.py` and implement the TODO(human) functions
2. **User implements:** `create_compacted_topic` -- create a topic with cleanup.policy=compact and aggressive compaction settings
3. **User implements:** `demonstrate_compaction` -- produce duplicate keys, wait for compaction, consume to verify
4. Test: `uv run python compaction_demo.py`
   - First run: may show all messages (compaction hasn't run yet on the active segment)
   - Wait 30+ seconds and run again: should show only latest value per key
5. Key question: Why does Kafka never compact the active (open) segment?

## Motivation

- **Event-driven architectures**: CDC is the foundation for reliable event-driven systems -- it eliminates the dual-write problem where applications must write to both a database and a message queue
- **Data integration**: Kafka Connect + Debezium is the industry standard for streaming database changes to data warehouses, search indexes, caches, and other systems
- **Microservice decomposition**: CDC enables the Outbox Pattern and Strangler Fig Pattern for migrating monoliths to microservices
- **Complements previous practices**: Builds on Kafka fundamentals (003a), directly feeds into SAGA (014), CQRS & Event Sourcing (015), and ETL/ELT (026)
- **Production relevance**: Nearly every organization running Kafka uses Kafka Connect in some form; Debezium is deployed at companies like Airbnb, Shopify, and Zalando

## References

- [Kafka Connect Documentation](https://docs.confluent.io/platform/current/connect/index.html)
- [Kafka Connect Architecture](https://docs.confluent.io/platform/current/connect/design.html)
- [Kafka Connect REST API Reference](https://docs.confluent.io/platform/current/connect/references/restapi.html)
- [Debezium PostgreSQL Connector](https://debezium.io/documentation/reference/stable/connectors/postgresql.html)
- [Debezium JDBC Sink Connector](https://debezium.io/documentation/reference/stable/connectors/jdbc.html)
- [ExtractNewRecordState SMT](https://debezium.io/documentation/reference/stable/transformations/event-flattening.html)
- [Kafka Log Compaction](https://docs.confluent.io/kafka/design/log_compaction.html)
- [Kafka Connect Error Handling & Dead Letter Queues](https://www.confluent.io/blog/kafka-connect-deep-dive-error-handling-dead-letter-queues/)
- [PostgreSQL Logical Decoding](https://www.postgresql.org/docs/current/logicaldecoding.html)

## Commands

All commands are run from `practice_003d_kafka_connect_cdc/`.

### Phase 1: Docker & Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Kafka, Kafka Connect, PostgreSQL source & sink (detached) |
| `docker compose ps` | Check container status and health (all should be "healthy") |
| `docker compose logs kafka-connect` | View Kafka Connect worker logs |
| `docker compose logs -f kafka-connect` | Follow Kafka Connect logs in real time |
| `docker compose logs postgres-source` | View source PostgreSQL logs |
| `docker compose down` | Stop and remove all containers |
| `docker compose down -v` | Stop, remove containers, and delete volumes (full reset) |

### Phase 2: Python Setup & Connect Manager

| Command | Description |
|---------|-------------|
| `cd app && uv sync` | Install Python dependencies from pyproject.toml |
| `uv run python connect_manager.py` | List available connector plugins and registered connectors |

### Phase 3: Source Connector

| Command | Description |
|---------|-------------|
| `uv run python source_connector.py` | Register Debezium PostgreSQL source connector |
| `docker exec kafka-connect-broker /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list` | List all Kafka topics (verify CDC topics were created) |

### Phase 4: CDC Consumer & Data Changes

| Command | Description |
|---------|-------------|
| `uv run python cdc_consumer.py` | Consume and display CDC events from products topic (Ctrl+C to stop) |
| `uv run python data_changes.py` | Generate INSERT/UPDATE/DELETE changes on source database |

### Phase 5: Sink Connector & Verification

| Command | Description |
|---------|-------------|
| `uv run python sink_connector.py` | Register JDBC sink connector for database replication |
| `docker exec -it postgres-sink psql -U sink_user -d sink_db -c "SELECT * FROM \"cdc.public.products\";"` | Query replicated products in sink database |
| `docker exec -it postgres-sink psql -U sink_user -d sink_db -c "SELECT * FROM \"cdc.public.orders\";"` | Query replicated orders in sink database |
| `docker exec -it postgres-source psql -U cdc_user -d source_db -c "SELECT * FROM products;"` | Query source products for comparison |

### Phase 6: Log Compaction

| Command | Description |
|---------|-------------|
| `uv run python compaction_demo.py` | Run log compaction demonstration (produces, waits, then consumes) |

### Debugging

| Command | Description |
|---------|-------------|
| `uv run python connect_manager.py` | Check connector status (run anytime to see all connectors) |
| `docker exec kafka-connect-broker /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic cdc.public.products --from-beginning` | Raw console consumer for CDC topic (JSON output) |
| `docker exec -it postgres-source psql -U cdc_user -d source_db -c "SELECT * FROM pg_replication_slots;"` | Check PostgreSQL replication slots (verify Debezium slot exists) |
| `docker exec -it postgres-source psql -U cdc_user -d source_db -c "SELECT * FROM pg_publication;"` | Check PostgreSQL publications (verify Debezium publication exists) |

**Note:** Phase 2-6 Python commands must be run from the `app/` subdirectory (where `pyproject.toml` lives).

## State

`not-started`
