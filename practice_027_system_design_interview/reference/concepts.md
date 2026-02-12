# System Design Concepts — Quick Reference

Use this as a lookup during exercises. Each concept includes a one-liner, when it matters, and the key trade-off.

---

## Consistency Models

| Model | Guarantee | Use When |
|-------|-----------|----------|
| **Linearizability** | Operations appear instantaneous in real-time order (per-object) | Leader election, distributed locks, financial ledgers |
| **Serializability** | Transactions execute as if serial (global, no recency guarantee) | Banking, inventory management |
| **Strong Consistency** | Linearizability + Serializability (strictest) | Payment processing, auction bidding |
| **Causal Consistency** | Preserves cause-effect ordering only | Social feeds (see reply after original post) |
| **Eventual Consistency** | All replicas converge eventually (no ordering) | DNS, CDN caches, social media likes count |

**Key distinction:** Linearizability is about *recency* (single object). Serializability is about *isolation* (transactions). They're orthogonal.

**Source:** [SystemDesign School — Linearizability vs Serializability](https://systemdesignschool.io/blog/linearizability-vs-serializability)

---

## Distributed Consensus

| Algorithm | Key Idea | Used By |
|-----------|----------|---------|
| **Paxos** | Two-phase (Prepare/Accept), mathematically proven, complex to implement | Google Chubby, Spanner |
| **Raft** | Leader-based with explicit log replication, designed for understandability | etcd, CockroachDB, Consul |
| **ZAB** | Separates leader election from log replication, optimized for throughput | Apache ZooKeeper |

**When you need consensus:** Leader election, distributed locks, configuration management, strongly-consistent replication.

**When you don't:** Eventually consistent systems, cache layers, analytics pipelines.

**Source:** [Paxos vs Raft vs ZAB](https://medium.com/@remisharoon/paxos-vs-raft-vs-zab-a-comprehensive-dive-into-distributed-consensus-protocols-6243a3f6539b)

---

## Database Internals

### Storage Engines

| Engine | Optimized For | Mechanism | Examples |
|--------|--------------|-----------|----------|
| **B-Tree** | Reads | In-place updates, balanced tree, O(log n) lookups | PostgreSQL, MySQL InnoDB |
| **LSM Tree** | Writes | Append-only MemTable → flush to sorted runs → compaction | RocksDB, Cassandra, LevelDB |

**Trade-off:** B-Tree = faster reads, slower writes, more predictable. LSM = faster writes, slower reads (compaction overhead), write amplification.

### Key Mechanisms

- **WAL (Write-Ahead Log):** Log operations before applying. Guarantees durability on crash recovery. Used by virtually all databases.
- **MVCC (Multi-Version Concurrency Control):** Maintain multiple versions of data. Readers never block writers. Used by PostgreSQL, MySQL InnoDB, CockroachDB.
- **Bloom Filter:** Probabilistic data structure. Fast "definitely not in set" checks. Used to skip unnecessary disk reads in LSM trees.

### Sharding Strategies

| Strategy | How | Pro | Con |
|----------|-----|-----|-----|
| **Hash-based** | hash(key) % N | Even distribution | Range queries impossible, resharding is painful |
| **Range-based** | Key ranges per shard | Range queries efficient | Hotspots if keys cluster |
| **Consistent Hashing** | Hash ring, virtual nodes | Minimal resharding on scale | Slightly uneven distribution |
| **Geo-based** | By region/country | Data locality | Uneven shard sizes |

### Replication

| Mode | How | Trade-off |
|------|-----|-----------|
| **Synchronous** | Leader waits for follower ack | Strong consistency, slower writes |
| **Asynchronous** | Leader doesn't wait | Fast writes, potential data loss on leader crash |
| **Semi-synchronous** | Wait for 1 of N followers | Balance: durability + acceptable latency |

---

## Caching Patterns

| Pattern | Flow | Best For |
|---------|------|----------|
| **Cache-Aside** (Lazy) | App checks cache → miss → fetch DB → populate cache | Read-heavy, tolerates stale data |
| **Write-Through** | App writes to cache → cache writes to DB synchronously | Read-heavy, consistency-critical |
| **Write-Behind** (Write-Back) | App writes to cache → cache writes to DB asynchronously | Write-heavy, tolerates temporary inconsistency |
| **Read-Through** | Cache itself fetches from DB on miss | Simplifies app logic, cache as proxy |

### Invalidation Strategies

- **TTL (Time-to-Live):** Simple, eventual consistency. Risk: stale data until expiry.
- **Event-based:** Invalidate on write. Low staleness, complex to implement.
- **Versioning:** Include version in cache key. No invalidation needed, cache naturally rotates.

### Cache Stampede

When a popular cache entry expires, N concurrent requests all miss and hit the DB simultaneously. Mitigations: **lock/lease** (only one request fetches), **early refresh** (refresh before TTL), **stale-while-revalidate** (serve stale, refresh in background).

**Source:** [Azure — Cache-Aside Pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/cache-aside)

---

## Rate Limiting Algorithms

| Algorithm | How | Burst Behavior | Complexity |
|-----------|-----|----------------|------------|
| **Token Bucket** | Tokens refill at fixed rate, requests consume tokens | Allows bursts up to bucket size | Low |
| **Leaky Bucket** | Requests enter bucket, processed at fixed rate | Smooths to constant rate, drops overflow | Low |
| **Fixed Window** | Count requests per time window (e.g., 100/min) | Double burst at window boundaries | Lowest |
| **Sliding Window Log** | Track timestamps, count in rolling window | Accurate, no boundary issues | High (memory) |
| **Sliding Window Counter** | Weighted average of current + previous window | Good approximation, low memory | Low |

**Distributed rate limiting:** Use Redis `INCR` + `EXPIRE` for shared counters across instances. Race condition: use Lua scripts for atomic check-and-increment.

**Source:** [GeeksforGeeks — Rate Limiting Algorithms](https://www.geeksforgeeks.org/system-design/rate-limiting-algorithms-system-design/)

---

## Load Balancing

| Layer | Routes By | Latency | Use Case |
|-------|-----------|---------|----------|
| **L4 (Transport)** | IP + port (5-tuple) | ~tens of μs | High PPS, non-HTTP, first layer |
| **L7 (Application)** | URL, headers, cookies | ~0.5-3ms | Content routing, TLS termination, retries |

**Modern pattern:** L4 in front (speed, DDoS protection) → L7 behind (intelligent routing).

**Algorithms:** Round-robin, least connections, weighted, consistent hashing (for sticky sessions / cache affinity).

**Health checks:** L4 uses TCP connect (3-5s timeout). L7 uses HTTP endpoint validation with outlier ejection.

**Source:** [System Overflow — L4 vs L7](https://www.systemoverflow.com/learn/load-balancing/l4-vs-l7/l4-vs-l7-load-balancing-key-trade-offs-and-when-to-choose-each)

---

## Message Queue Patterns

### Delivery Semantics

| Semantic | Guarantee | Implementation |
|----------|-----------|----------------|
| **At-most-once** | 0 or 1 delivery | Fire-and-forget, no retry |
| **At-least-once** | 1+ deliveries (may duplicate) | Retry + consumer idempotency |
| **Exactly-once** | Exactly 1 processing | Impossible end-to-end; approximate with idempotency + dedup |

### Key Patterns

- **Dead-Letter Queue (DLQ):** Messages that fail N times route to a separate queue for inspection.
- **Backpressure:** Monitor queue depth. If growing: slow producers, scale consumers, or shed load.
- **Ordering:** FIFO within partition (Kafka) or per ordering key (Pub/Sub). Global ordering kills parallelism.
- **Idempotency:** Use message ID + idempotency store (Redis/DB unique constraint) to deduplicate.

**Source:** [The Big Little Guide to Message Queues](https://sudhir.io/the-big-little-guide-to-message-queues)

---

## Real-Time Communication

| Protocol | Direction | Reconnect | Best For |
|----------|-----------|-----------|----------|
| **WebSocket** | Bidirectional | Manual | Chat, gaming, collaborative editing |
| **SSE (Server-Sent Events)** | Server → Client | Automatic | Feeds, notifications, dashboards |
| **Long Polling** | Simulated push via held HTTP | Per-request | Legacy compat, broad proxy support |

### Collaborative Editing

| Approach | How | Trade-off |
|----------|-----|-----------|
| **OT (Operational Transform)** | Transform concurrent operations against each other | Complex, requires central server, deterministic |
| **CRDT (Conflict-free Replicated Data Types)** | Data structures with mathematically guaranteed merge | Simpler, works peer-to-peer, larger metadata overhead |

**Modern preference:** CRDTs (Yjs, Automerge) for new systems. OT (Google Docs) for legacy.

---

## Unique ID Generation

| Scheme | Bits | Sortable | Coordination | Rate |
|--------|------|----------|-------------|------|
| **Snowflake** | 64 | Yes (time) | Worker ID assignment | 4096/ms/worker |
| **UUIDv4** | 128 | No (random) | None | Unlimited |
| **UUIDv7** | 128 | Yes (time) | None | Unlimited |
| **ULID** | 128 | Yes (time) | None | Unlimited |

**Choose Snowflake** when you need compact IDs (64-bit) and can coordinate worker IDs.
**Choose UUIDv7/ULID** when you need sortability without coordination.
**Choose UUIDv4** when you need zero coordination and don't care about ordering.

**Source:** [Authgear — Time-Sortable Identifiers](https://www.authgear.com/post/time-sortable-identifiers-uuidv7-ulid-snowflake)

---

## Reliability Patterns

| Pattern | Purpose |
|---------|---------|
| **Circuit Breaker** | Stop calling a failing service; fail fast instead of waiting. States: Closed → Open → Half-Open. |
| **Bulkhead** | Isolate failures by partitioning resources (thread pools, connection pools). |
| **Retry + Exponential Backoff** | Retry transient failures with increasing delays + jitter. |
| **Graceful Degradation** | Serve stale data, reduce features, return fallback responses under load. |
| **Health Checks** | Liveness (is it running?) vs Readiness (can it serve traffic?). |

## Observability

| Pillar | What | Tool Examples |
|--------|------|--------------|
| **Metrics** | Numeric time-series (QPS, latency p99, error rate) | Prometheus, Datadog |
| **Logs** | Structured event records | ELK, Loki |
| **Traces** | Request flow across services | Jaeger, OpenTelemetry |

**RED method (request-driven):** Rate, Errors, Duration.
**USE method (resource-driven):** Utilization, Saturation, Errors.

**SLI → SLO → SLA:** Service Level Indicator (metric) → Objective (target) → Agreement (contract). Error budget = 1 - SLO.
