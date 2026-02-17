# Practice 003c: Kafka Schema Registry & Data Contracts

## Technologies

- **Apache Kafka 3.9+** -- Distributed event streaming platform (KRaft mode, no ZooKeeper)
- **Confluent Schema Registry 7.5** -- Centralized schema management and compatibility enforcement
- **Apache Avro** -- Compact binary serialization format with schema evolution support
- **confluent-kafka[avro]** -- Python client with built-in Avro serializer/deserializer
- **fastavro** -- Fast Avro library for Python (used for low-level inspection)
- **Docker / Docker Compose** -- Local Kafka broker + Schema Registry

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Theoretical Context

### What Problem Does Schema Registry Solve?

In Kafka, messages are opaque byte arrays -- the broker doesn't know or care about the message format. This creates a coordination problem: producers and consumers must agree on the data format. Without enforcement, a producer can change its message structure (add/remove/rename fields) and silently break all downstream consumers. Schema Registry solves this by acting as a **centralized contract authority** that validates schema changes before they reach production.

### Architecture

The Schema Registry is a separate service (not part of Kafka itself) that runs alongside the Kafka cluster. It stores schemas in a special Kafka topic (`_schemas`) for durability, and exposes a REST API on port 8081. The flow:

1. **Producer** registers its schema with the registry before producing (or auto-registers on first produce)
2. **Registry** assigns a globally unique **schema ID** and validates compatibility against previous versions
3. **Producer** encodes each message as: `[magic_byte(1B) | schema_id(4B) | avro_payload(NB)]`
4. **Consumer** reads the schema ID from the message, fetches the writer's schema from the registry (cached locally), and deserializes using Avro's schema resolution rules

### The Confluent Wire Format

Every Avro-serialized message produced by the Confluent serializer has a 5-byte prefix:

```
Byte 0:      Magic byte (always 0x00) -- identifies Confluent encoding
Bytes 1-4:   Schema ID (big-endian unsigned 32-bit int)
Bytes 5+:    Avro binary payload (no Avro header/schema, just data)
```

This is different from standard Avro file format (which embeds the full schema). The wire format keeps messages compact by storing schemas centrally in the registry.

### Avro Serialization

Apache Avro is a row-oriented binary serialization format. Unlike JSON or Protobuf:
- **Schema required for read AND write** -- both writer and reader must have schemas
- **No field tags or names in the binary** -- data is encoded positionally based on schema field order, making it very compact
- **Schema evolution via resolution rules** -- when reader schema differs from writer schema, Avro applies deterministic rules (use defaults, ignore extra fields, etc.)

Avro schema types: `null`, `boolean`, `int`, `long`, `float`, `double`, `bytes`, `string`, `record`, `enum`, `array`, `map`, `union`, `fixed`. Union types like `["null", "string"]` represent optional fields.

### Avro vs Protobuf vs JSON Schema

| Feature | Avro | Protobuf | JSON Schema |
|---------|------|----------|-------------|
| **Encoding** | Binary (compact) | Binary (compact) | Text (JSON, verbose) |
| **Schema in message?** | No (in registry) | No (compiled) | No (in registry) |
| **Schema evolution** | Reader/writer resolution | Field numbering | Validation rules |
| **Schema language** | JSON | .proto IDL | JSON |
| **Code generation** | Optional | Required | Optional |
| **Zero-copy reads** | No | No (but close) | No |
| **Kafka ecosystem** | First-class support | Supported | Supported |

Avro is the default choice in the Confluent/Kafka ecosystem due to its schema evolution model and the fact that schemas are self-describing JSON (no compilation step required).

### Compatibility Modes

The Schema Registry enforces **compatibility rules** when new schema versions are registered. These rules determine what changes are allowed:

| Mode | Rule | Use Case |
|------|------|----------|
| **BACKWARD** (default) | New schema can read old data | Upgrade consumers first, then producers |
| **FORWARD** | Old schema can read new data | Upgrade producers first, then consumers |
| **FULL** | Both BACKWARD and FORWARD | Independent deployment of producers/consumers |
| **BACKWARD_TRANSITIVE** | BACKWARD against ALL prior versions | Stricter: handles skip-version reads |
| **FORWARD_TRANSITIVE** | FORWARD against ALL prior versions | Stricter: handles skip-version reads |
| **FULL_TRANSITIVE** | FULL against ALL prior versions | Strictest: safest for long-lived topics |
| **NONE** | No checks | Development only -- dangerous in production |

**BACKWARD** (default) is the most common because the standard deployment pattern is: deploy new consumers -> verify they work -> deploy new producers.

### Allowed Changes by Compatibility Mode

| Change | BACKWARD | FORWARD | FULL |
|--------|----------|---------|------|
| Add optional field (with default) | Yes | Yes | Yes |
| Remove optional field (with default) | Yes | Yes | Yes |
| Add required field (no default) | **No** | Yes | **No** |
| Remove required field (no default) | Yes | **No** | **No** |
| Rename field | **No** | **No** | **No** |
| Change field type | **No** | **No** | **No** |

### Subject Naming Strategies

A **subject** is the scope under which schemas are versioned in the registry. The naming strategy determines how subjects map to topics:

| Strategy | Subject Name | Schemas per Topic | Use Case |
|----------|-------------|-------------------|----------|
| **TopicNameStrategy** (default) | `<topic>-value` | One | Simple: one event type per topic |
| **RecordNameStrategy** | `<namespace>.<name>` | Multiple | Event sourcing: multiple types per topic |
| **TopicRecordNameStrategy** | `<topic>-<namespace>.<name>` | Multiple | Hybrid: type+topic scoping |

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Subject** | A scope for schema versioning (e.g., "users-value"). Maps to a topic+field combination. |
| **Schema ID** | Globally unique integer identifying a specific schema across all subjects. Embedded in messages. |
| **Version** | Per-subject sequential number (1, 2, 3, ...). A schema can have different versions in different subjects. |
| **Compatibility** | Rules governing what schema changes are allowed when registering a new version. |
| **Wire format** | The 5-byte prefix (magic byte + schema ID) prepended to every Avro message in Kafka. |
| **Writer schema** | The schema used to encode the message (identified by the schema ID in the wire format). |
| **Reader schema** | The schema the consumer expects. Avro resolves differences between writer and reader. |
| **Schema evolution** | The process of changing schemas over time while maintaining compatibility with existing data. |

## Description

Build a **Schema-Governed Event System** that demonstrates the Confluent Schema Registry for enforcing data contracts in Kafka. You will explore the Schema Registry REST API directly, produce/consume Avro-serialized messages, evolve schemas through multiple compatibility modes, and handle multiple event types in a single topic.

### What you'll learn

1. **Schema Registry REST API** -- register schemas, check compatibility, list subjects and versions
2. **Avro serialization** -- produce and consume Avro-encoded messages with automatic schema registration
3. **Wire format** -- parse the 5-byte Confluent prefix to understand schema ID embedding
4. **Compatibility modes** -- BACKWARD, FORWARD, FULL -- what changes each allows and why
5. **Schema evolution** -- evolve schemas safely by adding optional fields with defaults
6. **Multi-schema topics** -- RecordNameStrategy for multiple event types in one topic

## Instructions

### Phase 1: Setup & Infrastructure (~10 min)

1. Start Kafka and Schema Registry: `docker compose up -d` from the practice root
2. Wait for both services to be healthy: `docker compose ps` (check health status)
3. Verify the Schema Registry is running: `curl http://localhost:8081/subjects` (should return `[]`)
4. Install Python dependencies: `cd app && uv sync`
5. Create topics: `uv run python admin.py`

### Phase 2: Schema Registry REST API (~20 min)

1. Open `app/registry_explorer.py` and implement the 4 TODO(human) functions
2. **`list_subjects`** -- This teaches you the most basic registry operation: discovering what schemas exist. Understanding subjects is foundational because every other operation is scoped to a subject. The REST API pattern (GET /subjects) is the same pattern used by monitoring tools and CI pipelines to audit schema state.
3. **`register_schema`** -- This is how schemas enter the registry. The key gotcha here is the double-encoding: the Avro schema must be a JSON *string* inside the request body (not a nested object). Understanding this teaches you how the registry stores schemas internally.
4. **`check_compatibility`** -- The pre-registration compatibility check is the "dry run" that CI/CD pipelines use to prevent breaking changes. The `?verbose=true` parameter gives you actionable error messages.
5. **`get_schema_versions`** -- Version history is your audit trail. When debugging deserialization failures, this tells you exactly how the schema evolved and which version introduced the problem.
6. Run: `uv run python registry_explorer.py`
7. Key question: Why does the registry return a global schema ID (not per-subject)? What advantage does this have for the wire format?

### Phase 3: Avro Producer & Consumer (~20 min)

1. Open `app/avro_producer.py` and implement the 2 TODO(human) functions
2. **`create_avro_producer`** -- Wiring SchemaRegistryClient + AvroSerializer + Producer together teaches you the three-layer architecture: registry (schema storage), serializer (encoding logic), producer (transport). Understanding this layering helps debug issues at each level.
3. **`produce_users`** -- The SerializationContext is key: it tells the serializer which topic and field (key/value) to use for subject naming. Getting this wrong silently registers schemas under the wrong subject.
4. Open `app/avro_consumer.py` and implement the 2 TODO(human) functions
5. **`create_avro_consumer`** -- The consumer mirrors the producer but adds the reader/writer schema concept. Even though we use the same schema here, understanding that the consumer CAN use a different schema is essential for schema evolution.
6. **`consume_users`** -- This is the full round-trip. The deserializer reads the wire format, fetches the writer schema by ID, and applies Avro resolution against the reader schema.
7. Run producer: `uv run python avro_producer.py`
8. Run consumer: `uv run python avro_consumer.py`
9. Key question: Compare the byte size of an Avro-serialized user to its JSON equivalent. Why is Avro more compact?

### Phase 4: Schema Evolution & Compatibility (~25 min)

1. Open `app/schema_evolution.py` and implement the 4 TODO(human) functions
2. **`demonstrate_backward_compatibility`** -- Walk through the default mode: register v1, evolve to v2 (add optional field), see it pass. Then try a breaking change and see the detailed error. This teaches you *why* defaults matter and what "backward" means concretely.
3. **`demonstrate_forward_compatibility`** -- Switch the subject to FORWARD mode and see how the allowed changes differ. This builds intuition for "who upgrades first" -- the core deployment question that compatibility modes answer.
4. **`demonstrate_full_compatibility`** -- FULL mode is the intersection of BACKWARD and FORWARD. Understanding that it's the most restrictive helps you choose the right mode for your deployment strategy.
5. **`analyze_wire_format`** -- Parsing the 5-byte prefix manually demystifies what the serializer/deserializer do under the hood. This is the first thing you inspect when debugging "schema not found" errors.
6. Run: `uv run python schema_evolution.py`
7. Key question: Your team deploys consumers and producers independently with no coordination. Which compatibility mode should you use and why?

### Phase 5: Multi-Schema Topics (~15 min)

1. Open `app/multi_schema_topic.py` and implement the 3 TODO(human) functions
2. **`configure_record_name_strategy`** -- The `subject.name.strategy` config is the key: it changes how the serializer derives the registry subject name. Understanding this unlocks event-sourced architectures where a single topic is the "event log" for an aggregate.
3. **`produce_mixed_events`** -- Producing different event types to the same topic with different serializers shows the strategy in action. Each event type gets its own subject and independent schema evolution.
4. **`consume_mixed_events`** -- The consumer challenge: handling polymorphic events. Without a fixed reader schema, the deserializer uses the writer's schema from the registry. Event dispatch (checking which fields exist) is the pattern used in real event-driven systems.
5. Run: `uv run python multi_schema_topic.py`
6. Key question: What are the trade-offs of one-topic-per-event-type vs multi-type topics?

### Phase 6: Reflection (~5 min)

1. Compare this with practice 003a (plain JSON producers): what do you gain with Schema Registry? What complexity does it add?
2. When would you choose Protobuf over Avro? (Hint: think about code generation vs dynamic schemas)
3. How would you integrate schema compatibility checks into a CI/CD pipeline?

## Motivation

- **Data contracts**: Schema Registry is the industry standard for enforcing contracts between producers and consumers in event-driven architectures. Understanding it is essential for any Kafka-based system.
- **Schema evolution**: Real systems evolve constantly. Learning how to change schemas safely without breaking consumers is a critical production skill.
- **Complements 003a**: Practice 003a taught raw Kafka producers/consumers. This practice adds the data governance layer that production systems require.
- **Career relevance**: Confluent Schema Registry is used at companies like LinkedIn, Netflix, Uber, and most organizations running Kafka in production.
- **Foundation for event sourcing**: Multi-schema topics (RecordNameStrategy) are the building block for CQRS and event sourcing patterns (practices 014, 015).

## References

- [Confluent Schema Registry Documentation](https://docs.confluent.io/platform/current/schema-registry/index.html)
- [Schema Registry REST API Reference](https://docs.confluent.io/platform/current/schema-registry/develop/api.html)
- [Schema Evolution & Compatibility](https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html)
- [Confluent Wire Format](https://docs.confluent.io/platform/current/schema-registry/fundamentals/serdes-develop/index.html#wire-format)
- [Subject Name Strategy](https://docs.confluent.io/platform/current/schema-registry/fundamentals/serdes-develop/index.html#subject-name-strategy)
- [Apache Avro Specification](https://avro.apache.org/docs/current/specification/)
- [confluent-kafka-python Avro Serializer](https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#avroserializer)

## Commands

All commands are run from `practice_003c_kafka_schema_registry/`.

### Phase 1: Docker & Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Kafka (KRaft) and Schema Registry (detached) |
| `docker compose ps` | Check container status and health for both services |
| `docker compose logs kafka` | View Kafka broker logs |
| `docker compose logs schema-registry` | View Schema Registry logs |
| `docker compose logs -f schema-registry` | Follow Schema Registry logs in real time |
| `curl http://localhost:8081/subjects` | Verify Schema Registry is running (returns `[]`) |
| `curl http://localhost:8081/config` | Check global compatibility level (default: BACKWARD) |
| `docker compose down` | Stop and remove containers |
| `docker compose down -v` | Stop, remove containers, and delete volumes (full reset) |

### Phase 2: Python Setup & Topics

| Command | Description |
|---------|-------------|
| `cd app && uv sync` | Install Python dependencies (confluent-kafka[avro], requests, fastavro) |
| `uv run python admin.py` | Create topics: users (3p), sensor-readings (3p), sensor-alerts (3p) |

### Phase 3: Schema Registry REST API

| Command | Description |
|---------|-------------|
| `uv run python registry_explorer.py` | Run the Schema Registry REST API exploration (register, check compat, list versions) |

### Phase 4: Avro Producer & Consumer

| Command | Description |
|---------|-------------|
| `uv run python avro_producer.py` | Produce 10 Avro-serialized user events to the "users" topic |
| `uv run python avro_consumer.py` | Consume and deserialize Avro user events from "users" topic |

### Phase 5: Schema Evolution

| Command | Description |
|---------|-------------|
| `uv run python schema_evolution.py` | Demonstrate BACKWARD, FORWARD, and FULL compatibility modes + wire format analysis |

### Phase 6: Multi-Schema Topics

| Command | Description |
|---------|-------------|
| `uv run python multi_schema_topic.py` | Produce and consume mixed SensorReading + SensorAlert events on same topic |

### Utility: Schema Registry Inspection

| Command | Description |
|---------|-------------|
| `curl http://localhost:8081/subjects` | List all registered subjects |
| `curl http://localhost:8081/subjects/users-value/versions` | List version numbers for users-value subject |
| `curl http://localhost:8081/subjects/users-value/versions/1` | Get schema details for version 1 |
| `curl http://localhost:8081/subjects/users-value/versions/latest` | Get the latest schema version |
| `curl http://localhost:8081/config/users-value` | Get compatibility level for users-value subject |

**Note:** Phase 3-6 Python commands must be run from the `app/` subdirectory (where `pyproject.toml` lives).

## State

`not-started`
