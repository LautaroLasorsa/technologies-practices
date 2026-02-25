# Practice 076b: Redis Advanced -- Distributed Locking, Cache Patterns & Lua Scripts

## Technologies

- **Redis 7.x** -- In-memory data structure store used as database, cache, and message broker
- **Redis Lua Scripting** -- Server-side atomic script execution via EVAL/EVALSHA
- **Redis Streams** -- Append-only log data structure with consumer group support
- **redis-py** -- Python Redis client with async support
- **Docker Compose** -- Local Redis instance for development

## Stack

- Python 3.12+ (PEP 723 inline script metadata)
- Docker / Docker Compose

## Theoretical Context

### Distributed Locking with Redis

In distributed systems, multiple processes or services often need exclusive access to a shared resource -- a database row, a file, an external API with rate limits, or a critical section of business logic. A **distributed lock** coordinates this access across processes that do not share memory.

Redis implements distributed locking using the `SET key value NX EX seconds` command, which atomically sets a key only if it does not exist (NX = Not eXists) and assigns a time-to-live (EX = Expire). This single command provides both mutual exclusion (only one client succeeds) and deadlock prevention (the TTL ensures the lock is eventually released even if the holder crashes).

**The unlock problem:** Simply calling `DEL lock_key` to release a lock is dangerous -- if the lock has already expired and been acquired by another client, `DEL` removes the other client's lock. The solution is **atomic check-and-delete** using a Lua script:

```lua
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
else
    return 0
end
```

Each client generates a unique token (UUID) when acquiring the lock and stores it as the value. On release, the Lua script verifies ownership before deleting. Because Redis executes Lua scripts atomically (no other command can interleave), this is safe from race conditions.

**The Redlock Algorithm:** For systems requiring higher reliability, Antirez (Salvatore Sanfilippo, Redis creator) proposed the [Redlock algorithm](https://redis.io/docs/latest/develop/clients/patterns/distributed-locks/), which acquires locks across N independent Redis instances (typically N=5). A lock is considered acquired when the majority (N/2 + 1) of instances grant it within a validity window. This tolerates individual Redis node failures. However, Martin Kleppmann [critiqued Redlock](https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html) arguing that clock drift and process pauses can still break safety guarantees, and that **fencing tokens** (monotonically increasing IDs checked by the resource itself) are the only truly safe approach. For efficiency-only locks (where occasional double-execution is tolerable), single-instance SET NX EX is sufficient and simpler.

| Concept | Description |
|---------|-------------|
| **SET NX EX** | Atomic "set if not exists with expiry" -- the fundamental lock primitive |
| **Lock token** | A unique UUID stored as the lock value, used to verify ownership on release |
| **Lua unlock script** | Atomic check-and-delete that prevents releasing another client's lock |
| **Fencing token** | Monotonically increasing ID that the protected resource validates, defending against stale lock holders |
| **Redlock** | Multi-instance locking algorithm requiring majority quorum; trades simplicity for fault tolerance |
| **TTL/lease** | Time-to-live on the lock key; prevents deadlocks from crashed clients but creates a correctness window |

### Cache Patterns

Caching reduces latency and load on primary data stores by keeping frequently accessed data in fast memory (Redis). Four fundamental patterns govern how cache and database interact:

**Cache-Aside (Lazy Loading):** The application manages the cache directly. On read: check cache first; on miss, fetch from DB, store in cache, return. On write: update DB, then invalidate or update cache. This is the most common pattern because: (1) the cache only contains data that is actually requested, (2) if Redis fails the application falls back to the DB (degraded performance, not failure), and (3) it works with any database. The downside is that every cache miss incurs the latency of a DB read plus a cache write.

**Read-Through:** The cache layer itself is responsible for loading data on a miss. The application always reads from the cache; the cache fetches from the DB transparently. Simplifies application code but tightly couples the cache implementation to the data source.

**Write-Through:** Every write goes to both cache and database synchronously. The write completes only after both are updated. Guarantees strong consistency between cache and DB but adds write latency. Best for systems where reads are far more frequent than writes and consistency is critical (financial data, user sessions).

**Write-Behind (Write-Back):** Writes go to the cache immediately; the cache asynchronously flushes to the database after a delay or batch. Maximizes write throughput but risks data loss if Redis crashes before flushing. Best for analytics, metrics, and activity feeds where brief inconsistency is acceptable.

| Pattern | Consistency | Read Latency | Write Latency | Failure Impact |
|---------|-------------|-------------|---------------|----------------|
| **Cache-Aside** | Eventual | Miss: high, Hit: low | Low (DB only) | Degrades to DB-only |
| **Read-Through** | Eventual | Miss: high (auto-fill), Hit: low | Low (DB only) | Fails if cache is required |
| **Write-Through** | Strong | Hit: low | High (cache + DB) | Write fails if either fails |
| **Write-Behind** | Eventual (async lag) | Hit: low | Very low (cache only) | Data loss if cache crashes |

### Cache Stampede Prevention

A **cache stampede** (thundering herd) occurs when a popular cache key expires and many concurrent requests simultaneously attempt to recompute and cache the value, overwhelming the backend. Three mitigation strategies exist:

1. **Locking:** Only one process recomputes; others wait or serve stale data. Simple but adds latency for waiting clients and complexity for distributed lock management.

2. **Probabilistic Early Recomputation (XFetch):** Each request independently decides whether to recompute *before* the key expires, with probability increasing as expiry approaches. The formula from [Vattani et al.](https://github.com/internetarchive/xfetch):

```
should_recompute = (now - delta * beta * ln(random())) >= expiry
```

Where `delta` is the time the last recomputation took, `beta` is a tuning parameter (default 1.0), and `expiry` is when the cached value expires. As `now` approaches `expiry`, the probability of recomputation increases. The exponential distribution ensures that statistically, exactly one process recomputes early -- no coordination needed.

3. **Stale-While-Revalidate:** Serve the expired value immediately while triggering a background refresh. Requires storing both the value and a "soft TTL" (serve stale after this) alongside the "hard TTL" (evict after this).

### Lua Scripting in Redis

Redis executes Lua scripts atomically -- no other Redis command can run between steps of a script. This makes Lua the tool for implementing multi-step operations that must be consistent: rate limiters, locks, conditional updates, and transactions that span multiple keys.

**EVAL vs EVALSHA:** `EVAL script numkeys key1 ... arg1 ...` sends the full script text on every call. `EVALSHA sha1 numkeys key1 ... arg1 ...` sends only the SHA-1 hash of a previously loaded script (via `SCRIPT LOAD`). In production, use `SCRIPT LOAD` + `EVALSHA` to avoid retransmitting large scripts on every call.

**Key rules for Lua scripts:**
- Access keys via `KEYS[n]` (1-indexed), arguments via `ARGV[n]`
- All keys a script touches must be passed in `KEYS` -- this allows Redis Cluster to verify all keys hash to the same slot
- Scripts should be short and fast; Redis is single-threaded, so a slow script blocks all other operations
- Use `redis.call()` for commands that should propagate errors, `redis.pcall()` for error-safe calls
- `redis.log(redis.LOG_WARNING, "message")` for debugging scripts in the Redis log

**Common Lua patterns:**
- **Atomic compare-and-swap:** Read a value, check a condition, update if satisfied
- **Rate limiting:** Calculate token refill, check capacity, decrement -- all in one round trip
- **Conditional expiry:** SET a value only if its current TTL is below a threshold
- **Batch operations:** Perform multiple related updates atomically without MULTI/EXEC overhead

### Redis Streams

Redis Streams (introduced in Redis 5.0) are an append-only log data structure, conceptually similar to Apache Kafka topics but embedded in Redis. Each stream entry has an auto-generated ID (timestamp-based: `<millisecondsTime>-<sequenceNumber>`) and contains a set of field-value pairs.

**Consumer Groups** enable multiple consumers to cooperatively process a stream, with each message delivered to exactly one consumer in the group (similar to Kafka consumer groups). Key commands:

| Command | Purpose |
|---------|---------|
| `XADD stream * field value ...` | Append an entry to the stream; `*` auto-generates the ID |
| `XLEN stream` | Get the number of entries in the stream |
| `XRANGE stream start end [COUNT n]` | Read entries by ID range |
| `XREAD COUNT n BLOCK ms STREAMS stream id` | Read new entries (blocking or non-blocking) |
| `XGROUP CREATE stream group id` | Create a consumer group starting from the given ID |
| `XREADGROUP GROUP group consumer COUNT n BLOCK ms STREAMS stream >` | Read new entries for this consumer group; `>` means undelivered messages only |
| `XACK stream group id ...` | Acknowledge processing of entries (removes from Pending Entries List) |
| `XPENDING stream group` | Inspect unacknowledged entries (useful for monitoring and recovery) |

**Pending Entries List (PEL):** When a consumer reads a message via `XREADGROUP`, Redis adds it to that consumer's PEL. The message stays in the PEL until the consumer sends `XACK`. If a consumer crashes, its pending entries can be claimed by another consumer using `XCLAIM` or `XAUTOCLAIM` -- this provides at-least-once delivery semantics. The PEL is the mechanism that prevents message loss: unacknowledged messages are not forgotten.

**Streams vs Pub/Sub:** Pub/Sub is fire-and-forget (messages lost if no subscriber is connected). Streams persist messages and support replay from any position. Streams vs Kafka: Streams are simpler (no brokers, partitions, or ZooKeeper) but limited to a single Redis node's memory and throughput. Use Streams for moderate-throughput in-process event routing; use Kafka for high-throughput cross-service event streaming.

### Sliding Window Rate Limiting with Sorted Sets

Redis sorted sets (ZSET) provide an elegant implementation of sliding window rate limiting. Each request is recorded as a member with its timestamp as the score. To check the rate:

1. `ZREMRANGEBYSCORE key 0 (now - window_size)` -- Remove entries outside the window
2. `ZCARD key` -- Count entries in the window
3. If count < limit: `ZADD key now unique_id` -- Record this request
4. `EXPIRE key window_size` -- Set TTL for cleanup

This provides a true sliding window with no boundary-spike problem (unlike fixed windows). The trade-off is O(log N) per operation and higher memory usage (one entry per request vs a single counter).

For atomicity in concurrent environments, wrap these commands in a Lua script so the check-and-add is indivisible.

## Description

Build a **Redis Advanced Patterns Lab** that implements distributed locking, cache patterns, Lua scripting, and Redis Streams from scratch. Each exercise is a standalone Python script exploring a specific pattern:

1. **Distributed Lock** -- Implement `SET NX EX` acquire + Lua-based safe release; test with concurrent workers
2. **Cache-Aside with TTL** -- Implement the cache-aside pattern with a simulated slow database
3. **Cache Stampede Prevention** -- Implement the XFetch probabilistic early recomputation algorithm
4. **Lua Rate Limiter** -- Implement an atomic sliding window rate limiter as a Lua script
5. **Lua Compare-and-Swap** -- Implement atomic conditional updates with Lua
6. **Redis Streams** -- Implement a producer/consumer system with consumer groups and acknowledgment

### Architecture

```
  Exercise Scripts (Python)
       |
       v
  +----------+     Lua Scripts (atomic operations)
  |  Redis   |  <-- EVAL/EVALSHA
  |  7.x     |  <-- Streams: XADD/XREADGROUP/XACK
  |  (Docker) |  <-- Sorted Sets: ZADD/ZREMRANGEBYSCORE
  +----------+
```

### What you'll learn

1. **Distributed locking** -- SET NX EX, ownership tokens, atomic Lua unlock
2. **Cache-aside pattern** -- Read-through flow, TTL management, invalidation
3. **Stampede prevention** -- Probabilistic early recomputation (XFetch algorithm)
4. **Lua scripting** -- EVAL, EVALSHA, SCRIPT LOAD, atomic multi-step operations
5. **Rate limiting** -- Sliding window with sorted sets, atomic Lua enforcement
6. **Redis Streams** -- XADD, XREADGROUP, XACK, consumer groups, pending entry recovery

## Instructions

### Phase 1: Setup & Infrastructure (~5 min)

1. Start Redis: `docker compose up -d`
2. Verify Redis is running: `docker compose ps` (should show healthy)
3. Test connectivity: `docker compose exec redis redis-cli PING` (should return PONG)
4. Key question: Why use Redis 7.x specifically? What features does it add over 6.x? (Hint: Redis Functions, improved ACLs, but for this practice the main benefit is Lua 5.1 scripting stability)

### Phase 2: Distributed Locking (~20 min)

1. Open `src/01_distributed_lock.py` and study the structure
2. **`acquire_lock()`** -- This teaches the fundamental distributed lock primitive: SET NX EX. The NX flag provides mutual exclusion (only one client succeeds), and the EX TTL prevents deadlocks if the holder crashes. The unique token (UUID) stored as the value is critical -- without it, you cannot safely release the lock because you cannot distinguish your lock from another client's. This is the single most important Redis pattern for distributed systems.
3. **`release_lock()`** -- This teaches why DEL is insufficient for lock release and why Lua scripts are necessary. If your lock expires and another client acquires it, a plain DEL would delete their lock. The Lua script atomically checks ownership (GET == your token) and deletes only if you still own it. This is the canonical example of "why you need Lua in Redis."
4. **`demo_concurrent_workers()`** -- This demonstrates the lock under contention. Multiple asyncio tasks compete for the same lock, and the counters show that exactly one task holds it at a time. Without the lock, the shared counter would have race conditions.
5. Run: `uv run src/01_distributed_lock.py`
6. Key question: What happens if a lock holder's operation takes longer than the TTL? How would you implement lock renewal/extension?

### Phase 3: Cache-Aside Pattern (~15 min)

1. Open `src/02_cache_aside.py` and study the structure
2. **`cache_aside_get()`** -- This implements the most common caching pattern: check Redis first, on miss fetch from the "database" (simulated), store in Redis with TTL, return. The TTL is critical -- without it, stale data lives forever. This teaches the read path of cache-aside and why it is the safest default caching strategy (Redis failure degrades to slower DB reads, not data loss).
3. **`cache_aside_invalidate()`** -- This teaches cache invalidation on writes. When the underlying data changes, the cache entry must be deleted (not updated) to avoid write-ordering issues. Delete-on-write is simpler and safer than update-on-write because it avoids the "what if the DB write succeeds but the cache update fails" problem.
4. **`demo_cache_performance()`** -- This provides concrete latency numbers: cache hit vs cache miss. Seeing a 100x difference in latency builds intuition for why caching matters and what TTL values make sense.
5. Run: `uv run src/02_cache_aside.py`
6. Key question: If you have two services both writing to the same database table and both using cache-aside, how do you keep their caches consistent?

### Phase 4: Cache Stampede Prevention (~20 min)

1. Open `src/03_cache_stampede.py` and study the structure
2. **`xfetch_get()`** -- This implements the XFetch probabilistic early recomputation algorithm. Instead of all clients stampeding when a key expires, each client independently decides whether to recompute based on an exponential probability distribution. As the key approaches expiry, the probability increases, so statistically one client recomputes just before expiration. This teaches a coordination-free solution to cache stampede -- no locks, no leader election, just probability theory.
3. **`demo_stampede_comparison()`** -- This contrasts naive cache-aside (stampede on expiry) with XFetch (smooth recomputation). By simulating many concurrent clients hitting the same key near its expiry time, you see the difference: naive triggers N database calls, XFetch triggers approximately 1.
4. Run: `uv run src/03_cache_stampede.py`
5. Key question: What happens if beta is set very high (e.g., 10)? Very low (e.g., 0.1)? How does delta (recomputation time) affect the algorithm?

### Phase 5: Lua Rate Limiter (~20 min)

1. Open `src/04_lua_rate_limiter.py` and study the structure
2. **`SlidingWindowLimiter.__init__()`** -- This teaches SCRIPT LOAD + EVALSHA. Instead of sending the full Lua script on every rate-limit check, you load it once and call it by SHA hash. This is the production pattern for Lua scripts in Redis.
3. **`SLIDING_WINDOW_LUA`** -- This is the core Lua script. It atomically: (a) removes entries outside the window with ZREMRANGEBYSCORE, (b) counts remaining entries with ZCARD, (c) conditionally adds a new entry with ZADD, and (d) sets a TTL with EXPIRE. All four steps execute atomically because Redis runs Lua scripts single-threaded. This teaches you why Lua is essential for rate limiting -- without atomicity, concurrent requests could all pass the count check before any of them record their entry.
4. **`demo_rate_limiting()`** -- This sends bursts of requests and shows the limiter in action. The output shows allowed/denied decisions with timestamps, making the sliding window behavior visible.
5. Run: `uv run src/04_lua_rate_limiter.py`
6. Key question: How does this sliding window approach compare to the token bucket algorithm from practice 052? When would you choose one over the other?

### Phase 6: Lua Compare-and-Swap (~15 min)

1. Open `src/05_lua_compare_and_swap.py` and study the structure
2. **`CAS_LUA`** -- This Lua script implements atomic compare-and-swap: read the current value, compare with the expected value, write the new value only if they match. This is the foundation of optimistic concurrency control. It teaches you that Redis GET + conditional SET is NOT safe without Lua (another client can modify the value between your GET and SET).
3. **`compare_and_swap()`** -- The Python wrapper that calls the Lua script. Understanding the interface (key, expected, desired) maps to the CAS primitive used in lock-free programming (compare_exchange in C++/Rust).
4. **`demo_optimistic_counter()`** -- This uses CAS in a retry loop to implement a lock-free counter. Multiple concurrent tasks increment the counter, and CAS retries handle conflicts. Comparing the CAS counter result with the expected total verifies correctness.
5. Run: `uv run src/05_lua_compare_and_swap.py`
6. Key question: When would you prefer CAS over distributed locking? What are the trade-offs in terms of contention and throughput?

### Phase 7: Redis Streams with Consumer Groups (~25 min)

1. Open `src/06_redis_streams.py` and study the structure
2. **`produce_events()`** -- This teaches XADD, which appends entries to a stream with auto-generated IDs. Each entry is a set of field-value pairs (like a lightweight record). The `*` ID lets Redis generate a timestamp-based ID, ensuring ordering.
3. **`create_consumer_group()`** -- This teaches XGROUP CREATE, which sets up a consumer group starting from a specific stream position. The `$` ID means "only new messages," while `0` means "all existing messages." The `mkstream=True` parameter creates the stream if it does not exist -- a common gotcha when setting up consumer groups.
4. **`consume_events()`** -- This teaches XREADGROUP with consumer groups. The `>` ID means "give me only undelivered messages." After processing, XACK removes the message from the Pending Entries List (PEL). This is the at-least-once delivery mechanism: if a consumer crashes before XACK, the message stays in the PEL and can be reclaimed.
5. **`demo_consumer_groups()`** -- This runs multiple consumers in the same group processing a shared stream. Each message is delivered to exactly one consumer (load balancing). The output shows which consumer processed which message, demonstrating the consumer group partitioning.
6. Run: `uv run src/06_redis_streams.py`
7. Key question: How would you handle a consumer that crashes mid-processing? What role does XPENDING + XCLAIM play in recovery?

### Phase 8: Reflection (~5 min)

1. Compare Redis distributed locking with etcd/ZooKeeper-based locking. When is Redis "good enough"?
2. How would cache-aside + XFetch work in a Redis Cluster (multiple shards)?
3. When would you choose Redis Streams over Kafka? Over Redis Pub/Sub?

## Motivation

- **Production essential**: Distributed locking and caching are used in virtually every backend system. Understanding the pitfalls (unsafe unlock, stampede, stale data) prevents production outages.
- **Lua scripting mastery**: Lua scripts in Redis are the tool for atomic multi-step operations. Rate limiters, locks, and conditional updates all require Lua for correctness under concurrency.
- **Complements practice 052**: Practice 052 implements rate limiting as a pattern; this practice implements it as a Redis Lua script, showing the distributed version.
- **Complements practice 076a**: Practice 076a covers Redis fundamentals (data types, persistence, pub/sub); this practice builds advanced patterns on top of that foundation.
- **Cache patterns are universal**: Every caching system (Memcached, CDN, browser cache) follows the same patterns. Learning them with Redis builds transferable knowledge.
- **Interview staple**: "Design a rate limiter," "implement a distributed lock," and "explain cache stampede" are frequent system design interview questions.

## References

- [Redis Distributed Locks](https://redis.io/docs/latest/develop/clients/patterns/distributed-locks/)
- [Martin Kleppmann: How to do distributed locking](https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html)
- [Redis Lua Scripting](https://redis.io/docs/latest/develop/programmability/eval-intro/)
- [Redis Streams Introduction](https://redis.io/docs/latest/develop/data-types/streams/)
- [AWS Caching Patterns with Redis](https://docs.aws.amazon.com/whitepapers/latest/database-caching-strategies-using-redis/caching-patterns.html)
- [XFetch: Optimal Probabilistic Cache Stampede Prevention](https://github.com/internetarchive/xfetch)
- [Cache Stampede (Wikipedia)](https://en.wikipedia.org/wiki/Cache_stampede)
- [Redis Sorted Set Rate Limiting](https://redis.io/tutorials/howtos/ratelimiting/)
- [redis-py Documentation](https://redis-py.readthedocs.io/en/stable/)

## Commands

All commands are run from `practice_076b_redis_advanced/`.

### Phase 1: Docker & Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Redis 7.x (detached) |
| `docker compose ps` | Check Redis container status and health |
| `docker compose logs redis` | View Redis container logs |
| `docker compose exec redis redis-cli PING` | Verify Redis connectivity (should return PONG) |
| `docker compose exec redis redis-cli INFO server` | Show Redis server info (version, mode, uptime) |
| `docker compose exec redis redis-cli MONITOR` | Live stream of all commands hitting Redis (useful for debugging) |
| `docker compose down` | Stop and remove containers |
| `docker compose down -v` | Stop, remove containers, and delete volumes (full reset) |

### Phase 2: Distributed Locking

| Command | Description |
|---------|-------------|
| `uv run src/01_distributed_lock.py` | Run distributed lock exercise (acquire, release, concurrent workers) |

### Phase 3: Cache-Aside Pattern

| Command | Description |
|---------|-------------|
| `uv run src/02_cache_aside.py` | Run cache-aside exercise (hit/miss latency, invalidation) |

### Phase 4: Cache Stampede Prevention

| Command | Description |
|---------|-------------|
| `uv run src/03_cache_stampede.py` | Run stampede prevention exercise (naive vs XFetch comparison) |

### Phase 5: Lua Rate Limiter

| Command | Description |
|---------|-------------|
| `uv run src/04_lua_rate_limiter.py` | Run Lua-based sliding window rate limiter exercise |

### Phase 6: Lua Compare-and-Swap

| Command | Description |
|---------|-------------|
| `uv run src/05_lua_compare_and_swap.py` | Run Lua compare-and-swap exercise (optimistic concurrency) |

### Phase 7: Redis Streams

| Command | Description |
|---------|-------------|
| `uv run src/06_redis_streams.py` | Run Redis Streams exercise (producer, consumer groups, XACK) |

### Inspection & Debugging

| Command | Description |
|---------|-------------|
| `docker compose exec redis redis-cli` | Open interactive Redis CLI |
| `docker compose exec redis redis-cli KEYS '*'` | List all keys currently in Redis |
| `docker compose exec redis redis-cli TYPE keyname` | Check the type of a specific key |
| `docker compose exec redis redis-cli TTL keyname` | Check remaining TTL of a key |
| `docker compose exec redis redis-cli XINFO STREAM streamname` | Inspect a stream's metadata (length, groups, first/last entry) |
| `docker compose exec redis redis-cli XINFO GROUPS streamname` | List consumer groups on a stream |
| `docker compose exec redis redis-cli SCRIPT EXISTS sha1` | Check if a Lua script is cached by SHA |
| `docker compose exec redis redis-cli SCRIPT FLUSH` | Clear all cached Lua scripts |
| `docker compose exec redis redis-cli FLUSHDB` | Delete all keys in current database (useful between exercises) |

## State

`not-started`
