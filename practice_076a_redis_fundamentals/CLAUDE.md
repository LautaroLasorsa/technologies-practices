# Practice 076a: Redis Fundamentals -- Data Structures, Pub/Sub & Persistence

## Technologies

- **Redis 7** -- In-memory data structure store used as database, cache, and message broker
- **redis-py** -- Official Python client for Redis (sync mode)
- **Docker / Docker Compose** -- Local Redis server with persistence volumes
- **Redis CLI** -- Interactive command-line interface for inspecting and manipulating data

## Stack

- Python 3.12+ (uv, PEP 723 inline metadata)
- Docker / Docker Compose

## Theoretical Context

### What Is Redis and What Problem Does It Solve?

Redis (Remote Dictionary Server) is an open-source, in-memory data structure store created by Salvatore Sanfilippo in 2009. Unlike traditional databases that read from and write to disk on every operation, Redis keeps the entire dataset in RAM, delivering sub-millisecond response times for reads and writes. This makes Redis the go-to solution when latency matters: session caches, real-time leaderboards, rate limiters, pub/sub messaging, and distributed locks.

Redis is **not just a key-value store**. Its core differentiator is that values are not limited to strings -- Redis natively supports rich data structures (lists, sets, sorted sets, hashes, streams, bitmaps, HyperLogLogs) with atomic operations on each. This means you can perform operations like "add to a sorted set and get the rank" or "push to a list and trim to the last 100 elements" in a single command, without client-side logic or race conditions.

### Architecture: Single-Threaded Event Loop

Redis processes commands using a **single-threaded event loop** (based on `epoll`/`kqueue`). All commands execute sequentially -- there are no locks, no context switches, no race conditions between commands. This design is counterintuitively fast because:

1. **No lock overhead** -- The biggest bottleneck in multi-threaded databases (lock contention) is eliminated entirely.
2. **CPU cache efficiency** -- A single thread keeps data hot in L1/L2 cache.
3. **I/O multiplexing** -- Redis uses non-blocking I/O to handle thousands of concurrent connections on one thread.
4. **Memory-only operations** -- Without disk seeks, each operation completes in microseconds.

Starting with Redis 6.0, **I/O threading** was added: multiple threads handle network reads/writes (parsing client requests and writing responses), but command execution remains single-threaded. This addresses the network I/O bottleneck without sacrificing the simplicity of single-threaded command processing.

### Data Structures

| Structure | Internal Encoding | Key Operations | Typical Use Case |
|-----------|-------------------|----------------|------------------|
| **String** | SDS (Simple Dynamic String), int, embstr | `SET`, `GET`, `INCR`, `DECR`, `MSET`, `MGET` | Caching, counters, session tokens |
| **List** | Quicklist (linked list of ziplists) | `LPUSH`, `RPUSH`, `LPOP`, `RPOP`, `BRPOP`, `LRANGE` | Message queues, activity feeds, recent items |
| **Set** | Hashtable or intset (for small integer-only sets) | `SADD`, `SREM`, `SMEMBERS`, `SINTER`, `SUNION`, `SDIFF` | Tags, unique visitors, set operations |
| **Sorted Set** | Skip list + hashtable | `ZADD`, `ZRANK`, `ZRANGE`, `ZRANGEBYSCORE`, `ZINCRBY` | Leaderboards, priority queues, time-series indexes |
| **Hash** | Ziplist (small) or hashtable (large) | `HSET`, `HGET`, `HMSET`, `HGETALL`, `HINCRBY` | Object storage, user profiles, configuration |
| **Stream** | Radix tree of listpacks | `XADD`, `XREAD`, `XRANGE`, `XGROUP`, `XACK` | Event sourcing, message queues with consumer groups |

**Strings** are the simplest type -- binary-safe sequences up to 512 MB. Redis uses SDS (Simple Dynamic String) internally, which tracks length (O(1) `strlen`) and is binary-safe (can store `\0` bytes). When a string is a valid 64-bit integer, Redis stores it as an int and supports atomic `INCR`/`DECR`.

**Lists** are ordered sequences of strings. Internally, Redis uses a **quicklist** -- a doubly linked list where each node contains a ziplist (compact array). This balances memory efficiency (ziplist packing) with O(1) push/pop at both ends. Lists are ideal for queues: `LPUSH` + `BRPOP` gives you a blocking FIFO queue.

**Sets** are unordered collections of unique strings. Redis automatically switches between an **intset** (compact sorted array of integers) for small all-integer sets and a **hashtable** for general sets. Set operations (`SINTER`, `SUNION`, `SDIFF`) run in O(N*M) where N is the cardinality of the smallest set.

**Sorted Sets** (ZSETs) are the most powerful Redis structure. Each member has an associated floating-point score, and the set is ordered by score. Internally, Redis uses a **skip list** (probabilistic balanced tree, O(log N) insert/lookup) paired with a hashtable (O(1) score lookup by member). This dual structure enables both score-range queries (`ZRANGEBYSCORE`) and member lookups (`ZSCORE`, `ZRANK`) efficiently.

**Hashes** map field names to values within a single key -- like a nested dictionary. Small hashes (< `hash-max-ziplist-entries` entries, each < `hash-max-ziplist-value` bytes) use a **ziplist** (linear scan, but compact and cache-friendly). Larger hashes upgrade to a hashtable. Hashes are the idiomatic way to store objects (like a user profile with name, email, age fields).

**Streams** (added in Redis 5.0) are an append-only log structure modeled after Kafka. Each entry has an auto-generated ID (timestamp-sequence) and contains field-value pairs. Streams support **consumer groups** for distributed processing, making them suitable for event sourcing and reliable message queues -- unlike Pub/Sub, messages persist and can be replayed.

### Key Expiry and TTL

Redis supports per-key TTL (Time To Live). You can set expiry with `EXPIRE key seconds`, `PEXPIRE key milliseconds`, or atomically with `SET key value EX seconds`. When a key expires, Redis removes it using two mechanisms:

1. **Lazy expiration** -- When a client accesses an expired key, Redis checks the TTL and deletes it before returning nil.
2. **Active expiration** -- A background task samples 20 random keys with TTLs 10 times per second. If > 25% are expired, it repeats. This probabilistic approach ensures expired keys don't accumulate without scanning the entire keyspace.

Commands: `TTL key` returns remaining seconds (-1 = no expiry, -2 = key doesn't exist), `PERSIST key` removes the TTL.

### Eviction Policies

When Redis reaches `maxmemory`, it must decide what to evict. The `maxmemory-policy` setting controls this:

| Policy | Description |
|--------|-------------|
| `noeviction` | Return errors on write commands (default). Reads still work. |
| `allkeys-lru` | Evict least recently used keys (approximated LRU). Best general-purpose cache. |
| `allkeys-lfu` | Evict least frequently used keys. Better than LRU for skewed access patterns. |
| `volatile-lru` | Evict LRU keys only among those with a TTL set. |
| `volatile-lfu` | Evict LFU keys only among those with a TTL set. |
| `volatile-ttl` | Evict keys with the shortest remaining TTL first. |
| `allkeys-random` | Evict random keys. Simplest, but suboptimal. |
| `volatile-random` | Evict random keys that have a TTL set. |

Redis uses **approximated LRU/LFU** -- it samples `maxmemory-samples` (default 5) keys and evicts the best candidate among the sample. This is much cheaper than true LRU (which would require a linked list of all keys) and works well in practice.

### Pub/Sub

Redis Pub/Sub is a fire-and-forget messaging system. Publishers send messages to **channels**, and all subscribers currently listening to that channel receive the message instantly. Key characteristics:

- **At-most-once delivery** -- If a subscriber is disconnected when a message is published, that message is lost forever. There is no message persistence, no acknowledgment, no replay.
- **Fan-out** -- One published message is delivered to all subscribers on that channel.
- **Pattern subscriptions** -- `PSUBSCRIBE news.*` matches `news.sports`, `news.tech`, etc.
- **No history** -- Subscribers only receive messages published after they subscribe.

Pub/Sub is ideal for real-time notifications (chat, live updates) where occasional message loss is acceptable. For durable messaging, use Redis Streams (with consumer groups and acknowledgment) or an external system like Kafka.

### Pipelining

Pipelining is a technique to reduce network round-trip time (RTT). Normally, each Redis command requires a full round trip: client sends command, waits for response, then sends the next. With pipelining, the client sends multiple commands without waiting for responses, then reads all responses at once. This can improve throughput by 5-10x for batch operations.

In redis-py, pipelining is implemented via `r.pipeline()`. Commands are buffered locally and sent in a single batch when `pipe.execute()` is called. The server processes them sequentially (still single-threaded) and returns all results at once.

### Transactions (MULTI/EXEC)

Redis transactions group commands into an atomic unit using `MULTI`, `EXEC`, and optionally `WATCH`:

1. `MULTI` -- Start a transaction. Subsequent commands are queued, not executed.
2. Commands are sent and queued (Redis replies `QUEUED` for each).
3. `EXEC` -- Execute all queued commands atomically. No other client can interleave commands between MULTI and EXEC.
4. `DISCARD` -- Abort the transaction and discard queued commands.

**Important**: Redis transactions are NOT rollback-capable. If one command in the transaction fails, the others still execute. The atomicity guarantee is only about **isolation** (no interleaving), not about all-or-nothing semantics.

**WATCH** adds optimistic locking: `WATCH key` monitors a key for changes. If another client modifies the watched key before `EXEC`, the entire transaction aborts (returns nil). This enables check-and-set (CAS) patterns.

In redis-py, `Pipeline` is a transactional pipeline by default (wraps commands in MULTI/EXEC). Use `r.pipeline(transaction=False)` for pure pipelining without MULTI/EXEC.

### Persistence: RDB vs AOF vs Hybrid

Redis offers two persistence mechanisms and a hybrid mode:

**RDB (Redis Database) Snapshots**: At configured intervals (e.g., "save after 3600 seconds if at least 1 key changed"), Redis forks a child process that writes the entire dataset to a compact binary `.rdb` file. The parent continues serving requests using copy-on-write semantics. RDB files are ideal for backups and disaster recovery -- they're compact (50-70% smaller than AOF) and fast to load. Downside: data written between snapshots is lost on crash.

**AOF (Append-Only File)**: Every write command is appended to a log file. Three fsync policies control durability:
- `appendfsync always` -- fsync after every command. Slowest, safest (at most one command lost).
- `appendfsync everysec` -- fsync every second. Good compromise (at most ~1 second of data lost). **Recommended for production.**
- `appendfsync no` -- Let the OS decide when to flush. Fastest, least safe.

AOF files grow over time. Redis periodically **rewrites** the AOF: it creates a new file by replaying the current dataset (not the old AOF), producing a minimal set of commands. This runs in a forked child process.

**Hybrid persistence** (Redis 4.0+, enabled by default since 7.0): When rewriting the AOF, Redis writes an RDB snapshot as the preamble followed by AOF commands for changes since the snapshot. Recovery loads the RDB portion first (fast), then replays the AOF tail (recent changes). This combines RDB's fast recovery with AOF's minimal data loss.

| Aspect | RDB | AOF | Hybrid |
|--------|-----|-----|--------|
| **Data loss** | Minutes (between snapshots) | ~1 second (with `everysec`) | ~1 second |
| **File size** | Compact binary | Larger (command log) | Compact + small tail |
| **Recovery speed** | Fast (binary load) | Slow (replay commands) | Fast (RDB preamble + small replay) |
| **I/O impact** | Spike during fork | Continuous (append) | Moderate |

### Ecosystem Context

Redis competes with **Memcached** (simpler, multi-threaded, no persistence -- choose for pure caching), **KeyDB** (multi-threaded Redis fork), **Dragonfly** (modern Redis-compatible, multi-threaded, claims 25x throughput), **Valkey** (community fork after Redis license change to SSPL in 2024), and **etcd** (distributed KV for coordination, not caching). Managed offerings include **AWS ElastiCache**, **Azure Cache for Redis**, **GCP Memorystore**, and **Redis Cloud** (by Redis Ltd).

Choose Redis when you need: sub-millisecond latency, rich data structures (not just GET/SET), pub/sub or streams, distributed locking, or a combination of caching + messaging in one system. For pure caching with simple keys, Memcached may suffice. For durable event streaming, Kafka is more appropriate. For distributed consensus and configuration, use etcd or ZooKeeper.

## Description

Build a series of **standalone Python scripts** that explore Redis fundamentals hands-on. Each script focuses on a specific Redis capability, progressing from basic key-value operations to pub/sub messaging, pipelining, transactions, and persistence configuration. All scripts connect to a local Redis 7 container via Docker Compose.

### What you'll learn

1. **String operations** -- SET/GET, atomic counters (INCR/DECR), MSET/MGET batch operations
2. **Lists as queues** -- LPUSH/RPUSH, LPOP/RPOP, blocking pops (BRPOP) for consumer patterns
3. **Sets and sorted sets** -- Unique collections, set operations (SINTER/SUNION), leaderboard pattern with ZADD/ZRANK
4. **Hashes** -- Object storage pattern (HSET/HGET/HGETALL), field-level atomic increments
5. **Key expiry and TTL** -- Setting expiry, checking TTL, persistence with PERSIST
6. **Pub/Sub** -- Publishing messages, subscribing to channels, pattern subscriptions
7. **Pipelining** -- Batch operations with reduced round trips, performance comparison
8. **Transactions** -- MULTI/EXEC atomic command groups, WATCH for optimistic locking
9. **Persistence** -- Inspecting RDB/AOF configuration, triggering manual saves, comparing config options

## Instructions

### Phase 1: Setup & Verify (~5 min)

1. Start Redis with `docker compose up -d`
2. Verify Redis is running: `docker compose exec redis redis-cli PING` (should return `PONG`)
3. Explore the Redis CLI briefly: `docker compose exec redis redis-cli` then try `SET hello world`, `GET hello`, `DEL hello`

### Phase 2: Strings & Counters (~15 min)

1. Open `01_strings.py` and implement the TODO(human) functions
2. **User implements:** `basic_string_operations()` -- SET, GET, MSET, MGET, and conditional SET (NX/XX flags). This teaches the most fundamental Redis operations and how SET can behave like both "insert" and "upsert" depending on flags.
3. **User implements:** `atomic_counters()` -- INCR, DECR, INCRBY, INCRBYFLOAT for thread-safe counters. This demonstrates why Redis counters are superior to read-modify-write patterns in application code -- atomicity is guaranteed by the single-threaded execution model.
4. Run: `uv run 01_strings.py`
5. Key question: Why is INCR atomic in Redis but not in a typical database? What makes the single-threaded model an advantage here?

### Phase 3: Lists as Queues (~15 min)

1. Open `02_lists.py` and implement the TODO(human) functions
2. **User implements:** `list_as_queue()` -- LPUSH/RPOP for FIFO queue, RPUSH/LPOP for stack-like behavior, LRANGE to peek without consuming. This teaches the fundamental queue/deque pattern that underpins Redis-based task queues (like Celery and ARQ).
3. **User implements:** `blocking_consumer()` -- BRPOP for blocking queue consumption with timeout. This is the Redis equivalent of Kafka's `poll()` -- the consumer blocks until a message arrives or the timeout expires, avoiding busy-waiting loops.
4. Run: `uv run 02_lists.py`
5. Key question: Why use BRPOP instead of polling with RPOP in a loop? What's the performance difference?

### Phase 4: Sets & Sorted Sets (~15 min)

1. Open `03_sets_and_sorted_sets.py` and implement the TODO(human) functions
2. **User implements:** `set_operations()` -- SADD, SMEMBERS, SINTER, SUNION, SDIFF, SRANDMEMBER. This teaches Redis set theory operations -- useful for tagging systems, permissions, and finding common elements across collections.
3. **User implements:** `leaderboard()` -- ZADD, ZINCRBY, ZRANGE/ZREVRANGE with scores, ZRANK/ZREVRANK. This is one of the most common Redis patterns in production -- think gaming leaderboards, trending content, priority queues. The skip list internals give O(log N) operations for all rank/score queries.
4. Run: `uv run 03_sets_and_sorted_sets.py`
5. Key question: Why does a sorted set use both a skip list AND a hashtable internally? What would you lose with only one?

### Phase 5: Hashes (~10 min)

1. Open `04_hashes.py` and implement the TODO(human) functions
2. **User implements:** `hash_as_object()` -- HSET (multi-field), HGET, HGETALL, HDEL, HEXISTS, HINCRBY. This teaches the idiomatic way to store structured objects in Redis -- one hash per entity (e.g., `user:1001`) with fields for each attribute. Compared to serializing JSON into a string key, hashes allow partial reads/writes and field-level atomic increments.
3. Run: `uv run 04_hashes.py`
4. Key question: When would you use a hash vs serializing an object to a JSON string? What are the trade-offs?

### Phase 6: Key Expiry & TTL (~10 min)

1. Open `05_expiry.py` and implement the TODO(human) functions
2. **User implements:** `expiry_and_ttl()` -- SET with EX/PX, EXPIRE, TTL, PTTL, PERSIST. This teaches Redis's key lifecycle management, which is the foundation of all caching strategies. Understanding expiry is critical for avoiding stale data and managing memory.
3. Run: `uv run 05_expiry.py`
4. Key question: What happens if you SET a key that already has a TTL without specifying a new TTL? Does the old TTL survive?

### Phase 7: Pub/Sub (~15 min)

1. Open `06_pubsub.py` and implement the TODO(human) functions
2. **User implements:** `publisher()` -- connect to Redis and PUBLISH messages to a channel. This teaches the fire-and-forget model: the publisher doesn't know (or care) how many subscribers are listening.
3. **User implements:** `subscriber()` -- SUBSCRIBE to a channel, enter a message loop, handle different message types (subscribe confirmation, actual messages, unsubscribe). This teaches the subscription lifecycle and the blocking nature of a subscribed Redis client.
4. Run: `uv run 06_pubsub.py` (launches both publisher and subscriber in threads)
5. Key question: What happens if you publish to a channel with no subscribers? Is the message stored anywhere?

### Phase 8: Pipelining (~10 min)

1. Open `07_pipelining.py` and implement the TODO(human) functions
2. **User implements:** `pipelining_demo()` -- compare individual commands vs pipelined batch: SET 1000 keys individually, then SET 1000 keys in a pipeline, measure and compare wall-clock time. This demonstrates the dramatic performance difference (5-10x) that pipelining provides by eliminating per-command network round trips.
3. Run: `uv run 07_pipelining.py`
4. Key question: Does pipelining change the order in which commands execute on the server? Can a pipelined command read the result of a previous pipelined command?

### Phase 9: Transactions (MULTI/EXEC) (~15 min)

1. Open `08_transactions.py` and implement the TODO(human) functions
2. **User implements:** `basic_transaction()` -- MULTI/EXEC to atomically transfer a "balance" between two keys (decrement one, increment the other). This teaches that MULTI/EXEC guarantees isolation (no interleaving) but NOT rollback -- if one command fails, others still execute.
3. **User implements:** `optimistic_locking()` -- WATCH a key, read its value, start a MULTI/EXEC to conditionally update it. If another client modifies the watched key, the transaction aborts. This is Redis's version of CAS (compare-and-swap) -- essential for race-condition-free updates.
4. Run: `uv run 08_transactions.py`
5. Key question: How does Redis WATCH differ from database row-level locking? What happens under high contention?

### Phase 10: Persistence Exploration (~10 min)

1. Open `09_persistence.py` and implement the TODO(human) functions
2. **User implements:** `explore_persistence()` -- Use CONFIG GET to inspect persistence settings (save schedule, AOF config, dbfilename), trigger a manual BGSAVE, check LASTSAVE timestamp, and inspect INFO persistence stats. This teaches how to verify and tune Redis durability settings -- critical for production deployments where data loss tolerance varies.
3. Run: `uv run 09_persistence.py`
4. Key question: If you need zero data loss, can Redis alone guarantee it? What combination of settings gets closest?

## Motivation

- **Industry ubiquitous**: Redis is used by virtually every major tech company (Twitter, GitHub, Snapchat, StackOverflow, Airbnb) as the caching and real-time data layer. It consistently ranks among the most-used databases.
- **Foundation for advanced patterns**: Practice 076b (distributed locking, Lua scripts, cache patterns), 052 (resilience patterns with Redis), and 078 (distributed rate limiting) all depend on Redis fundamentals.
- **Production essential**: Understanding data structure selection, expiry policies, persistence trade-offs, and pipelining is critical for designing performant systems -- these are common interview and design review topics.
- **Complements messaging practices**: Pub/Sub and streams contrast with Kafka (003a-003e), RabbitMQ (055), and Pub/Sub (002) -- understanding Redis's niche in the messaging landscape enables informed architecture decisions.

## References

- [Redis Documentation](https://redis.io/docs/latest/)
- [Redis Data Types](https://redis.io/docs/latest/develop/data-types/)
- [Redis Persistence](https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/)
- [Redis Pub/Sub](https://redis.io/docs/latest/develop/pubsub/)
- [Redis Pipelining](https://redis.io/docs/latest/develop/use/pipelining/)
- [Redis Transactions](https://redis.io/docs/latest/develop/interact/transactions/)
- [Redis Key Eviction](https://redis.io/docs/latest/develop/reference/eviction/)
- [redis-py Documentation](https://redis-py.readthedocs.io/en/stable/)
- [redis-py Pipelines and Transactions](https://redis.io/docs/latest/develop/clients/redis-py/transpipe/)
- [Redis Internals: Data Structures](https://redis.io/technology/data-structures/)

## Commands

All commands are run from the `practice_076a_redis_fundamentals/` folder root.

### Phase 1: Setup & Verify

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Redis 7 container in detached mode |
| `docker compose ps` | Check Redis container status and health |
| `docker compose logs redis` | View Redis server startup logs |
| `docker compose exec redis redis-cli PING` | Verify Redis is responding (should return PONG) |
| `docker compose exec redis redis-cli` | Open interactive Redis CLI session |

### Phase 2: Strings & Counters

| Command | Description |
|---------|-------------|
| `uv run 01_strings.py` | Run string operations and atomic counter exercises |

### Phase 3: Lists as Queues

| Command | Description |
|---------|-------------|
| `uv run 02_lists.py` | Run list/queue operations and blocking consumer exercises |

### Phase 4: Sets & Sorted Sets

| Command | Description |
|---------|-------------|
| `uv run 03_sets_and_sorted_sets.py` | Run set operations and leaderboard exercises |

### Phase 5: Hashes

| Command | Description |
|---------|-------------|
| `uv run 04_hashes.py` | Run hash object storage exercises |

### Phase 6: Key Expiry & TTL

| Command | Description |
|---------|-------------|
| `uv run 05_expiry.py` | Run key expiry and TTL management exercises |

### Phase 7: Pub/Sub

| Command | Description |
|---------|-------------|
| `uv run 06_pubsub.py` | Run pub/sub publisher and subscriber demo (threaded) |

### Phase 8: Pipelining

| Command | Description |
|---------|-------------|
| `uv run 07_pipelining.py` | Run pipelining benchmark comparing individual vs batched commands |

### Phase 9: Transactions

| Command | Description |
|---------|-------------|
| `uv run 08_transactions.py` | Run MULTI/EXEC transaction and WATCH optimistic locking exercises |

### Phase 10: Persistence Exploration

| Command | Description |
|---------|-------------|
| `uv run 09_persistence.py` | Inspect and explore Redis persistence configuration |

### Redis CLI Inspection (useful during any phase)

| Command | Description |
|---------|-------------|
| `docker compose exec redis redis-cli INFO server` | Show Redis server info (version, uptime, config) |
| `docker compose exec redis redis-cli INFO memory` | Show memory usage statistics |
| `docker compose exec redis redis-cli INFO persistence` | Show RDB/AOF persistence status |
| `docker compose exec redis redis-cli INFO keyspace` | Show per-database key counts |
| `docker compose exec redis redis-cli DBSIZE` | Count total keys in current database |
| `docker compose exec redis redis-cli KEYS '*'` | List all keys (development only -- never in production) |
| `docker compose exec redis redis-cli MONITOR` | Real-time stream of all commands being processed (Ctrl+C to stop) |
| `docker compose exec redis redis-cli CONFIG GET save` | Show RDB snapshot schedule |
| `docker compose exec redis redis-cli CONFIG GET appendonly` | Check if AOF is enabled |

### Cleanup

| Command | Description |
|---------|-------------|
| `docker compose down` | Stop and remove Redis container |
| `docker compose down -v` | Stop container and delete persistence volume (full reset) |
| `python clean.py` | Full cleanup: Docker volumes, Python caches |

## State

`not-started`
