# Practice 003e: Kafka Internals & Performance Tuning

## Technologies

- **Apache Kafka 3.9** -- Distributed event streaming platform (KRaft mode, 3-broker cluster)
- **confluent-kafka** -- High-performance Python client built on librdkafka (C library)
- **Docker / Docker Compose** -- Local 3-broker Kafka cluster in KRaft mode

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Theoretical Context

This practice goes beyond basic produce/consume to explore Kafka's replication protocol, storage internals, and the producer/consumer knobs that determine throughput, latency, and durability. Understanding these internals is what separates "I can use Kafka" from "I can operate and tune Kafka in production."

### Replication Internals

#### ISR (In-Sync Replica Set)

Every partition has a set of **replicas** -- broker IDs that store a copy of the partition log. One replica is the **leader** (handles all reads and writes); the rest are **followers** that continuously fetch from the leader. The **ISR** is the subset of replicas that are "caught up" with the leader within `replica.lag.time.max.ms` (default 30s).

- If a follower falls behind (e.g., slow disk, network partition, broker crash), the controller removes it from the ISR.
- When the follower catches up again, the controller adds it back.
- Only ISR members can be elected leader (by default -- see unclean leader election below).

#### HW (High Watermark) and LEO (Log End Offset)

Each replica tracks two key offsets:

- **LEO (Log End Offset)**: The offset of the *next* message to be written. If a partition has offsets 0-99, LEO = 100.
- **HW (High Watermark)**: The offset up to which *all ISR replicas* have replicated. Consumers can only read up to the HW.

The leader advances its HW when all ISR followers have fetched up to a certain offset. This means:
- **Producer sees**: Message acknowledged when all ISR replicas have written it (with `acks=all`).
- **Consumer sees**: Message visible only after it is replicated to all ISR members (HW advanced past it).
- **ISR shrink effect**: When the ISR shrinks (fewer replicas to wait for), the HW can advance faster, but durability decreases.

#### Leader Epoch

A **leader epoch** is a monotonically increasing integer assigned each time a new leader is elected for a partition. Followers use the leader epoch to detect log divergence after a leader change. If a follower has messages from a higher epoch than the current leader, those messages are truncated (they were written by a stale leader that was fenced). This prevents split-brain scenarios where two brokers both think they are the leader.

#### Unclean Leader Election

By default (`unclean.leader.election.enable=false`), only ISR members can become leader. If ALL ISR members are down, the partition becomes **unavailable** -- no reads or writes. Setting `unclean.leader.election.enable=true` allows an out-of-sync replica to become leader, which restores availability but can **lose committed messages** (the new leader doesn't have them). This is a fundamental availability vs durability trade-off.

### Log Storage Internals

#### Segments and Indexes

A partition's log is stored as a sequence of **segments** on disk. Each segment is a pair of files:
- `.log` -- The actual message data (append-only).
- `.index` -- A sparse offset-to-position index for efficient seeks.
- `.timeindex` -- A sparse timestamp-to-offset index.

Active segment is the one currently being written to. When it reaches `log.segment.bytes` (default 1 GB) or `log.roll.ms`, Kafka rolls to a new segment. Old segments are eligible for deletion (based on `log.retention.hours`, default 168h = 7 days) or compaction.

#### Log Compaction

When `cleanup.policy=compact`, Kafka does NOT delete old segments by time. Instead, it keeps only the **latest value for each key**. The compaction thread periodically reads old segments, discards older entries for duplicate keys, and writes compacted segments. This is essential for:
- Changelog topics (CDC): Keep the latest state per entity.
- KTable backing topics in Kafka Streams.
- Configuration/metadata topics.

Key settings: `min.cleanable.dirty.ratio` (how much dirty data triggers compaction), `segment.ms` (how often to roll segments for compaction eligibility).

### Producer Tuning

#### Idempotent Producers

Setting `enable.idempotence=true` assigns the producer a unique **PID (Producer ID)**. Each message includes the PID and a per-partition **sequence number**. The broker tracks the last sequence per PID per partition and discards duplicates. This eliminates duplicate messages from producer retries (network timeout -> retry -> duplicate write). Idempotence automatically sets `acks=all`, `retries=MAX_INT`, `max.in.flight.requests.per.connection=5`.

#### Transactional Producers

Transactions wrap multiple produce calls in an atomic unit: either all messages are committed (visible to `isolation.level=read_committed` consumers) or all are aborted. The `transactional.id` persists across producer restarts -- if a new producer instance starts with the same ID, the old instance is **fenced** (its in-flight transactions are aborted). This is the foundation of Kafka's exactly-once semantics (EOS).

#### Compression

The producer compresses messages in **batches** before sending. Codec choices:

| Codec | Speed | Ratio | CPU | Best For |
|-------|-------|-------|-----|----------|
| none | -- | 1.0x | none | Baseline |
| snappy | Fast | ~2x | Low | General purpose |
| lz4 | Fastest | ~2x | Lowest | Latency-sensitive |
| zstd | Medium | ~3-4x | Medium | Bandwidth-constrained |
| gzip | Slow | ~3x | High | Archival, not real-time |

Larger batches compress better because the codec has more data to find patterns in.

#### Batching: linger.ms and batch.size

The producer buffers messages per partition. A batch is sent when EITHER:
- The batch reaches `batch.size` bytes (default 16 KB), OR
- `linger.ms` milliseconds have elapsed since the first message in the batch.

`linger.ms=0` (default) means send immediately -- lowest latency, but tiny batches (often 1 message per network request). Setting `linger.ms=10-200` allows the producer to accumulate larger batches, amortizing per-request overhead and improving compression. This is the core **throughput vs latency trade-off** in Kafka producer tuning.

#### acks and min.insync.replicas

These two settings work together:
- `acks=all` (strongest): Leader waits for ALL ISR replicas to acknowledge.
- `min.insync.replicas=N`: The broker rejects writes if ISR < N. With `acks=all` + `min.insync.replicas=2` on a 3-replica topic, data survives any single broker failure but the topic becomes unavailable if 2 brokers are down.

### Consumer Tuning

#### Fetch Parameters

- `fetch.min.bytes`: Minimum bytes the broker accumulates before responding to a fetch. Higher values reduce request rate but add latency.
- `fetch.max.bytes` / `max.partition.fetch.bytes`: Control maximum data per fetch. Tune based on message size and memory constraints.
- `fetch.max.wait.ms`: Maximum time the broker waits to accumulate `fetch.min.bytes`. Default 500ms.

#### Poll Interval and Session Timeout

- `max.poll.interval.ms` (default 5 min): Maximum time between `poll()` calls. If exceeded, the consumer is removed from the group (rebalance). Must be longer than your message processing time.
- `session.timeout.ms` (default 45s): How long the broker waits for heartbeats before declaring the consumer dead. Lower = faster failure detection, but more false positives during GC pauses or network blips.

#### Assignment Strategies

- **RangeAssignor** (default): Assigns contiguous partition ranges per topic. Simple but can be uneven across topics.
- **RoundRobinAssignor**: Distributes partitions one-by-one. More even for multi-topic consumption.
- **CooperativeStickyAssignor**: Incremental cooperative rebalancing -- only revokes partitions that need to move. Consumers keeping their partitions continue consuming during rebalance. This eliminates the "stop-the-world" pause of eager rebalancing.

#### Cooperative vs Eager Rebalancing

- **Eager** (Range, RoundRobin): ALL partitions revoked from ALL consumers, then ALL reassigned. Causes a consumption pause for the entire group.
- **Cooperative** (CooperativeSticky): Only moved partitions are revoked. Other consumers continue uninterrupted. Essential for large consumer groups where rebalances are frequent (scaling, rolling deploys).

#### Static Group Membership (KIP-345)

By setting `group.instance.id`, a consumer gets a stable identity. On disconnect, the coordinator waits for `session.timeout.ms` before rebalancing. If the consumer reconnects with the same `group.instance.id` within that window, it gets its old partitions back with NO rebalance. Critical for Kubernetes rolling deploys and transient failures.

### Monitoring: Consumer Lag

Consumer lag = LEO - committed offset per partition. It is the single most important Kafka operational metric:
- **Increasing lag** = consumers can't keep up with production rate.
- **Sustained lag** = need more consumers, faster processing, or tuning.
- **Lag spikes after deploys** = possible regression in consumer code.

Production systems monitor lag via JMX metrics, Burrow, or the AdminClient API used in this practice.

## Description

Build a **Kafka Performance Lab** that explores replication internals, producer tuning, and consumer optimization on a 3-broker KRaft cluster. You will observe ISR behavior during broker failures, measure the impact of compression codecs and batching parameters, implement idempotent and transactional producers, compare consumer assignment strategies, and build a lag monitor.

### What you'll learn

1. **Replication internals** -- ISR, HW, LEO, leader epoch, min.insync.replicas rejection
2. **Idempotent & transactional producers** -- PID, sequence numbers, exactly-once semantics
3. **Compression codecs** -- snappy, lz4, zstd, gzip trade-offs (throughput vs CPU vs ratio)
4. **Batch tuning** -- linger.ms and batch.size impact on throughput and latency
5. **Consumer tuning** -- fetch params, assignment strategies, cooperative rebalancing
6. **Static membership** -- avoid rebalances during rolling deploys
7. **Lag monitoring** -- compute and display consumer lag per partition

## Instructions

### Phase 1: Cluster Setup & Topic Creation (~10 min)

1. Start the 3-broker cluster: `docker compose up -d` from the practice root.
2. Wait for all 3 brokers to be healthy: `docker compose ps` (all should show "healthy").
3. Install dependencies: `cd app && uv sync`.
4. Create topics: `uv run python admin.py`.
5. Verify topics: `docker exec kafka-internals-1 /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:29092 --describe`.

**Key question**: Why does Kafka need at least 3 brokers for `min.insync.replicas=2` with `replication-factor=3` to be meaningful?

### Phase 2: Replication & ISR Exploration (~20 min)

1. Open `app/replication_explorer.py` and implement the 3 TODO(human) functions.
2. **User implements:** `describe_topic_replicas` -- Query AdminClient for per-partition leader, replicas, ISR. This teaches how to inspect cluster state programmatically, which is essential for monitoring and automation.
3. **User implements:** `monitor_isr_changes` -- Polling loop that detects and logs ISR shrink/expand events. This demonstrates the ISR lifecycle during broker failure and recovery, connecting the theoretical concepts of HW, LEO, and leader epoch to observable behavior.
4. **User implements:** `demonstrate_min_isr_rejection` -- Prove that `acks=all` + `min.insync.replicas=2` rejects writes when ISR < 2. This shows the availability vs durability trade-off in action -- the most critical operational decision in Kafka configuration.
5. Test: `uv run python replication_explorer.py describe` -- see replica state.
6. Test: `uv run python replication_explorer.py monitor` -- in another terminal, run `docker compose stop kafka-internals-2` and observe ISR shrink, then `docker compose start kafka-internals-2` and observe ISR expand.
7. Test: `uv run python replication_explorer.py min-isr` -- follow the prompts to stop 2 brokers and observe the rejection.

### Phase 3: Idempotent & Transactional Producers (~20 min)

1. Open `app/idempotent_producer.py` and implement the 4 TODO(human) functions.
2. **User implements:** `create_idempotent_producer` -- Configure with `enable.idempotence=True`. This teaches the minimal configuration needed and what the broker automatically enforces (acks, retries, max.in.flight).
3. **User implements:** `demonstrate_idempotence` -- Produce and verify no duplicates. This connects PID + sequence number theory to a concrete verification.
4. **User implements:** `create_transactional_producer` -- Configure with `transactional.id` and call `init_transactions()`. This teaches the transaction lifecycle setup, including the fencing mechanism that prevents zombie producers.
5. **User implements:** `atomic_produce` -- Use `begin_transaction()`/`commit_transaction()`/`abort_transaction()`. This demonstrates atomic multi-message writes, the foundation of exactly-once semantics.
6. Test: `uv run python idempotent_producer.py idempotent`.
7. Test: `uv run python idempotent_producer.py transactional`.

### Phase 4: Compression & Batch Tuning (~20 min)

1. Open `app/compression_benchmark.py` and implement the 2 TODO(human) functions.
2. **User implements:** `benchmark_codec` -- Produce with a given `compression.type` and measure throughput. This teaches how to isolate a single variable (codec) and benchmark it, and how Kafka compresses at the batch level (not per-message).
3. **User implements:** `run_compression_comparison` -- Run all codecs and print comparison table. This reveals the real-world performance differences between codecs on your hardware.
4. Test: `uv run python compression_benchmark.py`.
5. Open `app/batch_tuning.py` and implement the 2 TODO(human) functions.
6. **User implements:** `benchmark_batch_config` -- Produce with specific `linger.ms` + `batch.size` and measure throughput. This teaches how batching amortizes per-request overhead and improves compression.
7. **User implements:** `run_batch_comparison` -- Run the test matrix and print comparison table. This quantifies the throughput vs latency trade-off for your workload.
8. Test: `uv run python batch_tuning.py`.

### Phase 5: Consumer Tuning & Assignment Strategies (~20 min)

1. Open `app/consumer_tuning.py` and implement the 3 TODO(human) functions.
2. **User implements:** `benchmark_consumer_config` -- Consume with config overrides and measure throughput. This teaches how fetch parameters affect consumer performance and which knobs matter most.
3. **User implements:** `compare_assignment_strategies` -- Create consumers with Range, RoundRobin, and CooperativeSticky. This shows the different rebalance protocols and assignment patterns in action.
4. **User implements:** `demonstrate_static_membership` -- Use `group.instance.id` to survive a disconnect without rebalance. This is directly applicable to Kubernetes rolling deploys and production resilience.
5. Test: `uv run python consumer_tuning.py benchmark`.
6. Test: `uv run python consumer_tuning.py strategies`.
7. Test: `uv run python consumer_tuning.py static-membership`.

### Phase 6: Lag Monitoring & Reflection (~10 min)

1. Open `app/lag_monitor.py` and implement the 2 TODO(human) functions.
2. **User implements:** `get_consumer_lag` -- Use `list_consumer_group_offsets()` and `get_watermark_offsets()` to compute lag. This teaches the two data sources needed for lag calculation: committed offsets (from the group coordinator) and log-end offsets (from partition leaders).
3. **User implements:** `monitor_lag_loop` -- Continuous lag display. This creates a mini monitoring tool similar to what production systems use.
4. Test: Run `uv run python lag_monitor.py consumer-tuning-group` while running consumer experiments in another terminal.
5. Reflection:
   - Which compression codec would you choose for a latency-sensitive system? For a bandwidth-constrained one?
   - When would you choose `min.insync.replicas=1` vs `=2`?
   - Why is cooperative rebalancing important for large consumer groups?

## Motivation

- **Production readiness**: Knowing how to tune Kafka is the difference between a Kafka cluster that handles 10K msg/s and one that handles 1M msg/s.
- **Operational confidence**: Understanding ISR, HW, and leader election lets you reason about data durability during outages.
- **Interview depth**: "Explain the difference between LEO and HW" and "How does min.insync.replicas interact with acks?" are common senior backend interview questions.
- **Complements 003a**: Practice 003a covered produce/consume basics. This practice covers the internals and tuning that make Kafka reliable and performant at scale.
- **Career relevance**: Kafka tuning is a specialized skill highly valued in data engineering, platform engineering, and backend roles at companies operating event-driven architectures.

## References

- [Apache Kafka Documentation - Replication](https://kafka.apache.org/documentation/#replication)
- [Apache Kafka Documentation - Producer Configs](https://kafka.apache.org/documentation/#producerconfigs)
- [Apache Kafka Documentation - Consumer Configs](https://kafka.apache.org/documentation/#consumerconfigs)
- [Confluent: Exactly-Once Semantics](https://www.confluent.io/blog/exactly-once-semantics-are-possible-heres-how-apache-kafka-does-it/)
- [KIP-345: Static Membership](https://cwiki.apache.org/confluence/display/KAFKA/KIP-345%3A+Introduce+static+membership+protocol+to+reduce+consumer+rebalances)
- [KIP-429: Cooperative Rebalancing](https://cwiki.apache.org/confluence/display/KAFKA/KIP-429%3A+Kafka+Consumer+Incremental+Rebalance+Protocol)
- [confluent-kafka-python API Docs](https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html)

## Commands

All commands are run from `practice_003e_kafka_internals_tuning/`.

### Phase 1: Docker & Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start 3-broker Kafka cluster in KRaft mode (detached) |
| `docker compose ps` | Check all 3 containers are healthy |
| `docker compose logs kafka-1` | View broker 1 logs |
| `docker compose logs -f kafka-1 kafka-2 kafka-3` | Follow logs from all brokers |
| `docker compose down` | Stop and remove all containers |
| `docker compose down -v` | Stop, remove containers, and delete volumes (full reset) |

### Phase 2: Python Setup & Topic Admin

| Command | Description |
|---------|-------------|
| `cd app && uv sync` | Install Python dependencies from pyproject.toml |
| `uv run python admin.py` | Create all practice topics (replication-demo, compression-bench, etc.) |
| `docker exec kafka-internals-1 /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:29092 --describe` | Describe all topics showing partitions, replicas, ISR |

### Phase 3: Replication Explorer

| Command | Description |
|---------|-------------|
| `uv run python replication_explorer.py describe` | Show per-partition leader, replicas, and ISR for replication-demo |
| `uv run python replication_explorer.py monitor` | Continuously monitor ISR changes (Ctrl+C to stop) |
| `uv run python replication_explorer.py min-isr` | Demonstrate min.insync.replicas rejection (interactive) |
| `docker compose stop kafka-internals-2` | Stop broker 2 to trigger ISR shrink (run during monitor) |
| `docker compose start kafka-internals-2` | Restart broker 2 to trigger ISR expand |
| `docker compose stop kafka-internals-2 kafka-internals-3` | Stop 2 brokers to trigger min.insync.replicas rejection |
| `docker compose start kafka-internals-2 kafka-internals-3` | Restart both brokers after min-isr demo |

### Phase 4: Idempotent & Transactional Producers

| Command | Description |
|---------|-------------|
| `uv run python idempotent_producer.py idempotent` | Produce with idempotent producer and verify no duplicates |
| `uv run python idempotent_producer.py transactional` | Produce atomic transaction batches and verify |

### Phase 5: Compression & Batch Tuning

| Command | Description |
|---------|-------------|
| `uv run python compression_benchmark.py` | Benchmark all compression codecs (none, snappy, lz4, zstd, gzip) |
| `uv run python batch_tuning.py` | Benchmark linger.ms + batch.size matrix |

### Phase 6: Consumer Tuning

| Command | Description |
|---------|-------------|
| `uv run python consumer_tuning.py seed` | Seed consumer-tuning topic with 50K test messages |
| `uv run python consumer_tuning.py benchmark` | Benchmark different consumer fetch configurations |
| `uv run python consumer_tuning.py strategies` | Compare Range, RoundRobin, CooperativeSticky assignment |
| `uv run python consumer_tuning.py static-membership` | Demonstrate static membership reconnect without rebalance |

### Phase 7: Lag Monitoring

| Command | Description |
|---------|-------------|
| `uv run python lag_monitor.py consumer-tuning-group` | Monitor consumer lag for consumer-tuning-group (Ctrl+C to stop) |
| `uv run python lag_monitor.py <any-group-id>` | Monitor lag for any consumer group |

**Note:** Phase 3-7 Python commands must be run from the `app/` subdirectory (where `pyproject.toml` lives).

## State

`not-started`
