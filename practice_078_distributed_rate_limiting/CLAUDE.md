# Practice 078: Distributed Rate Limiting -- Token Bucket, Sliding Window & Redis Lua

## Technologies

- **Rate Limiting Algorithms** -- Fixed window, sliding window log, sliding window counter, token bucket, leaky bucket
- **Redis** -- In-memory data store for shared rate limit state across distributed replicas
- **Redis Lua Scripts** -- Atomic server-side scripting for race-condition-free rate limiting
- **FastAPI** -- HTTP API framework with middleware-based rate limiting
- **Docker Compose** -- Multi-replica orchestration to demonstrate distributed behavior

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Theoretical Context

### Why Rate Limiting Matters

Rate limiting controls how many requests a client can make to a service within a given time period. Without it, a single misbehaving client (or attacker) can overwhelm a service, causing degraded performance or outages for all users. Rate limiting is a fundamental building block of API design, DDoS protection, and resource fairness in multi-tenant systems.

In a **single-process** application, rate limiting is straightforward: keep counters in memory. In a **distributed system** with multiple application replicas behind a load balancer, the challenge is that each replica sees only a fraction of the traffic. If each replica maintains independent counters, a client can exceed the global limit by spreading requests across replicas. Shared state is required -- and Redis is the industry-standard solution because it offers sub-millisecond latency and atomic operations.

This is a **top-5 system design interview question**. Interviewers expect you to discuss algorithm trade-offs, distributed consistency, Redis implementation details, and where rate limiting sits in system architecture (API gateway vs application layer vs service mesh).

### The Five Rate Limiting Algorithms

#### 1. Fixed Window Counter

The simplest algorithm. Divide time into fixed intervals (e.g., 1-minute windows). Maintain a counter per client per window. Increment on each request; reject when the counter exceeds the limit.

**Redis pattern:** `INCR key` + `EXPIRE key window_size` (set TTL only on the first request in a window).

**Pros:** Minimal memory (one counter per client). O(1) per request. Simple to implement and understand.

**Cons:** The **boundary spike problem**. A client can send `limit` requests at second 59 of window N and `limit` requests at second 0 of window N+1, effectively getting 2x the rate in a 2-second span. This is a well-known weakness that interviewers specifically ask about.

```
Window 1: [0s -------- 59s] | Window 2: [60s --------]
                    100 reqs |  100 reqs
                    ^^^^^^^^^^^^^^^^
                    200 reqs in 2 seconds (limit was 100/min)
```

#### 2. Sliding Window Log

Maintain a log (sorted set) of timestamps for every request. To check if a new request is allowed: remove entries older than `now - window_size`, count remaining entries, compare against limit.

**Redis pattern:** `ZREMRANGEBYSCORE key -inf (now - window)` + `ZCARD key` + `ZADD key now member`. Uses sorted sets where scores are timestamps.

**Pros:** Perfectly accurate -- no boundary spike problem. The window slides continuously with each request.

**Cons:** Memory-intensive. For a user making 1000 req/sec, you store 1000 entries per second. At scale (millions of users), this becomes prohibitive. Also, the `ZREMRANGEBYSCORE` + `ZCARD` + `ZADD` sequence must be atomic.

#### 3. Sliding Window Counter

A hybrid that approximates the sliding window using two fixed-window counters (current and previous). The estimated count is:

```
estimated = prev_count * (1 - elapsed_fraction) + current_count
```

Where `elapsed_fraction` is how far into the current window we are. For example, if we are 70% through the current minute, we weight the previous minute's count by 30% and add the current minute's full count.

**Redis pattern:** Two `INCR` keys (one per window) + arithmetic in application code or Lua.

**Pros:** Memory-efficient (two counters per client, like fixed window). Eliminates most of the boundary spike problem. Good balance of accuracy vs cost.

**Cons:** Approximate -- assumes requests in the previous window were evenly distributed, which may not be true. Still, in practice, this approximation is close enough for most rate limiting use cases.

#### 4. Token Bucket

A bucket holds tokens up to a maximum capacity. Tokens are added at a fixed **refill rate** (e.g., 10/second). Each request consumes one or more tokens. If insufficient tokens, the request is rejected.

**Key insight:** Tokens are not added by a background timer. Instead, each request calculates how many tokens *should have been added* since the last request (**lazy refill**). This eliminates the need for timers and makes the algorithm stateless between requests.

**Redis pattern:** Redis hash (`tokens`, `last_refill` fields) + Lua script for atomic read-calculate-write.

**Pros:** Allows **burst traffic** up to bucket capacity while enforcing a long-term average rate. This is exactly what most APIs want: tolerate short spikes (webhook bursts, page loads) but cap sustained abuse. Two tunable parameters: `capacity` (burst size) and `refill_rate` (sustained rate).

**Cons:** More complex to implement correctly (the Lua script must handle refill math atomically). Slightly more memory than fixed window (two fields per key vs one counter).

```
capacity=20, refill_rate=10/sec

Time 0.0s: tokens=20 (full bucket)
  Burst of 20 requests -> all pass, tokens=0
Time 0.5s: tokens=5 (refilled 10*0.5)
  5 requests -> all pass, tokens=0
Time 1.0s: tokens=5 (refilled 10*0.5)
  Steady 10 req/sec -> sustainable indefinitely
```

#### 5. Leaky Bucket

Requests enter a FIFO queue (the "bucket"). The queue drains at a fixed rate. If the queue is full, new requests are dropped. Unlike token bucket which *rejects* excess requests, leaky bucket *delays* them (up to queue capacity).

**Redis pattern:** `LPUSH` to add, `RPOP` to drain at fixed intervals (requires a timer or Lua-based drain).

**Pros:** Produces a perfectly smooth output rate -- ideal for network traffic shaping or APIs that must never spike.

**Cons:** Adds latency (requests wait in queue). More complex to implement in a stateless/distributed way because it requires a drain mechanism. Less commonly used for API rate limiting (token bucket is preferred).

### Algorithm Comparison

| Algorithm | Memory/Client | Burst Tolerance | Accuracy | Complexity | Best For |
|-----------|--------------|-----------------|----------|------------|----------|
| Fixed Window | O(1) -- 1 counter | Boundary spikes | Low | Very low | Simple abuse prevention |
| Sliding Window Log | O(N) -- N entries | None | Perfect | Medium | Strict per-user quotas |
| Sliding Window Counter | O(1) -- 2 counters | Minimal | High (approx) | Low | Production API limits |
| Token Bucket | O(1) -- 2 fields | Yes (configurable) | High | Medium | API gateways, bursty traffic |
| Leaky Bucket | O(N) -- queue | Queue absorbs burst | Perfect | High | Traffic shaping, smoothing |

### Why Redis Lua Scripts?

Rate limiting requires **atomic read-modify-write** operations. Consider the token bucket without atomicity:

```
Replica A: reads tokens=5
Replica B: reads tokens=5       (same value!)
Replica A: writes tokens=4      (consumed 1)
Replica B: writes tokens=4      (consumed 1, but should be 3!)
```

Both replicas read the same token count and both decrement independently, allowing 2 requests through when only 1 token should have been consumed. This is a classic **race condition**.

**Redis Lua scripts solve this** because they execute atomically -- no other command can interleave while a Lua script runs. The entire read-calculate-write cycle happens in a single atomic operation. This is equivalent to a database transaction but with sub-millisecond latency.

**EVAL vs EVALSHA:** `EVAL` sends the full Lua script text on every call. `EVALSHA` sends only the SHA1 hash after the script is loaded once with `SCRIPT LOAD`. In production, always use EVALSHA to save bandwidth. The Python `redis` library handles this automatically via `register_script()`.

**Redis MULTI/EXEC** (pipelines with `transaction=True`) provide atomicity for a fixed sequence of commands, but they cannot branch on intermediate results. You cannot do "read token count, then conditionally decrement" in a MULTI block. Lua scripts can.

### Distributed Rate Limiting Architecture

```
                Load Balancer
               /      |      \
          Replica1  Replica2  Replica3
               \      |      /
              Redis (shared state)
```

All replicas check the same Redis keys. Because Lua scripts execute atomically in Redis (single-threaded command processing), there are no race conditions even under high concurrency.

**Where to apply rate limiting:**

| Layer | Example | Pros | Cons |
|-------|---------|------|------|
| **API Gateway** (Nginx, Kong, Envoy) | Global per-IP limit before requests reach your app | Lowest latency, catches abuse earliest | Limited to simple rules (IP, path) |
| **Application middleware** (this practice) | Per-user, per-endpoint, tiered limits | Rich context (user tier, API key), flexible rules | Adds latency to every request |
| **Service mesh** (Istio, Envoy sidecar) | Per-service rate limiting | No code changes, infrastructure-level | Requires mesh infrastructure |

Production systems often combine layers: API gateway for coarse DDoS protection, application middleware for business-logic limits.

### System Design Interview Angle

When asked "Design a rate limiter" in an interview, cover these points:

1. **Requirements clarification:** Per-user? Per-IP? Per-endpoint? Global? What rate? (e.g., 100 req/min per user)
2. **Algorithm choice:** Token bucket for most APIs (handles bursts). Sliding window counter for strict limits. Fixed window only if simplicity trumps accuracy.
3. **Distributed state:** Redis as central counter store. Lua scripts for atomicity. Key design: `ratelimit:{user_id}:{endpoint}`.
4. **Failure modes:** What happens when Redis is down? Options: fail-open (allow all), fail-closed (reject all), local fallback (in-memory limiter per replica). Most systems fail-open to avoid self-inflicted outages.
5. **Performance:** Rate limit check adds ~1ms (Redis round-trip). For 100K req/sec, Redis handles this easily (single instance does ~100K ops/sec for simple commands; Lua scripts are somewhat slower).
6. **Response headers:** `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, `Retry-After` (HTTP 429 response).
7. **Race conditions:** Why naive INCR-then-check fails, and how Lua scripts fix it.

### References

- Redis. ["Build 5 Rate Limiters with Redis."](https://redis.io/tutorials/howtos/ratelimiting/) Redis Tutorials.
- [Cloudflare Blog: How We Built Rate Limiting](https://blog.cloudflare.com/counting-things-a-lot-of-different-things/)
- [Stripe: Rate Limiters and Load Shedders](https://stripe.com/blog/rate-limiters)
- [Google Cloud: Rate Limiting Strategies](https://cloud.google.com/architecture/rate-limiting-strategies-techniques)
- [System Design Primer: Rate Limiter](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-rate-limiter)
- [FreeCodeCamp: Build Rate Limiting with Redis and Lua](https://www.freecodecamp.org/news/build-rate-limiting-system-using-redis-and-lua/)
- [Halodoc: Redis and Lua Powered Sliding Window](https://blogs.halodoc.io/taming-the-traffic-redis-and-lua-powered-sliding-window-rate-limiter-in-action/)
- [AlgoMaster: Rate Limiting Algorithms Explained with Code](https://blog.algomaster.io/p/rate-limiting-algorithms-explained-with-code)
- Microsoft. ["Rate Limiting Pattern."](https://learn.microsoft.com/en-us/azure/architecture/patterns/rate-limiting) Azure Architecture Center.

## Description

Build a **Distributed Rate Limiting System** with five rate limiting algorithms, all backed by Redis for distributed state. A FastAPI service deployed as **multiple replicas** demonstrates that rate limits are enforced globally (not per-replica). You implement:

1. **Fixed Window Counter** -- `INCR` + `EXPIRE` pattern (the simplest, with the boundary spike flaw)
2. **Sliding Window Log** -- Redis sorted sets (`ZADD`, `ZREMRANGEBYSCORE`, `ZCARD`) via Lua script
3. **Sliding Window Counter** -- Hybrid of two fixed windows with weighted approximation
4. **Token Bucket** -- Full Lua script with lazy refill (the industry standard)
5. **Comparison benchmark** -- Side-by-side test of all algorithms under burst and steady traffic
6. **FastAPI middleware** -- Rate limiting applied as middleware with proper HTTP 429 responses
7. **Multi-replica test** -- Proof that limits hold across replicas hitting the same Redis

### Architecture

```
  Client (test scripts / curl)
         |
    Load Balancer (Nginx)
    /         |         \
 Replica 1  Replica 2  Replica 3   (FastAPI, each with rate limit middleware)
    \         |         /
        Redis (shared state)
        - Token bucket: HASH + Lua
        - Sliding window: ZSET + Lua
        - Fixed window: STRING + INCR
```

### What you'll learn

1. **Five rate limiting algorithms** -- Implementation, trade-offs, and when to use each
2. **Redis Lua scripting** -- Atomic operations, EVAL/EVALSHA, script registration
3. **Redis data structures for rate limiting** -- Strings (INCR), Hashes (HGET/HSET), Sorted Sets (ZADD/ZRANGEBYSCORE)
4. **Distributed consistency** -- Why naive implementations break under concurrency
5. **FastAPI middleware** -- Applying rate limits transparently with proper HTTP response headers
6. **Multi-replica validation** -- Proving that distributed rate limiting works across instances
7. **System design interview preparation** -- Articulating trade-offs and architecture decisions

## Instructions

### Phase 1: Setup & Infrastructure (~5 min)

1. Start Redis and the Nginx load balancer with `docker compose up -d redis nginx`
2. Verify Redis: `docker compose exec redis redis-cli PING` should return `PONG`
3. Install dependencies: `uv sync`
4. Key question: Why can't each application replica maintain its own in-memory rate limit counter?

### Phase 2: Fixed Window Counter (~15 min)

1. Open `src/01_fixed_window.py` and study the structure
2. **User implements:** The Lua script for atomic `INCR` + conditional `EXPIRE` -- this teaches why you cannot use separate `INCR` and `EXPIRE` commands (race condition between them)
3. **User implements:** The `allow()` method that calls the Lua script and tracks metrics
4. **User implements:** The `demo_fixed_window()` function that demonstrates the boundary spike problem -- send requests right at the boundary of two windows to show the 2x burst
5. Run: `uv run python src/01_fixed_window.py`
6. Key question: Can you think of a scenario where the boundary spike actually matters? (Hint: think billing APIs, database writes)

### Phase 3: Sliding Window Log (~20 min)

1. Open `src/02_sliding_window_log.py` and study the structure
2. **User implements:** The Lua script that atomically prunes expired entries (`ZREMRANGEBYSCORE`), counts remaining (`ZCARD`), and conditionally adds the new request (`ZADD`) -- all in one atomic operation. This is the first exercise where you see why `MULTI/EXEC` is insufficient (you need conditional logic after the `ZCARD`).
3. **User implements:** The `allow()` method and the demo function
4. Run: `uv run python src/02_sliding_window_log.py`
5. Key question: If you have 10 million users each making 100 req/min, how much memory does the sorted set approach consume? Is this practical?

### Phase 4: Sliding Window Counter (~15 min)

1. Open `src/03_sliding_window_counter.py` and study the structure
2. **User implements:** The Lua script that reads two fixed-window counters and computes the weighted estimate. This teaches the approximation technique that most production rate limiters use.
3. **User implements:** The `allow()` method and demo
4. Run: `uv run python src/03_sliding_window_counter.py`
5. Key question: Under what traffic pattern does the sliding window counter give the worst approximation?

### Phase 5: Token Bucket (~20 min) -- Core of the practice

1. Open `src/04_token_bucket.py` and study the structure
2. Review the Lua script template in `lua/token_bucket.lua`
3. **User implements:** The full Lua script -- read state, calculate refill, try to consume, save state. This is the most important exercise: the lazy refill calculation is the core insight of the token bucket algorithm.
4. **User implements:** The `allow()` method with metrics tracking
5. **User implements:** The demo showing burst behavior vs sustained rate
6. Run: `uv run python src/04_token_bucket.py`
7. Key question: How would you set `capacity` and `refill_rate` for an API that should handle 100 req/sec sustained with occasional bursts of up to 500?

### Phase 6: Algorithm Comparison (~10 min)

1. Open `src/05_comparison.py` and study the structure
2. **User implements:** The benchmark function that sends identical traffic patterns through all four algorithms and collects allowed/rejected counts per second
3. Run: `uv run python src/05_comparison.py`
4. Examine the comparison output -- which algorithm allowed the most burst traffic? Which was strictest?

### Phase 7: FastAPI Middleware & Multi-Replica (~15 min)

1. Open `src/06_fastapi_app.py` and study the structure
2. **User implements:** The `RateLimitMiddleware` class that intercepts requests, checks the rate limiter, and returns 429 with proper headers when rejected
3. Start three replicas behind Nginx: `docker compose up --build -d`
4. Run the multi-replica test: `uv run python src/07_multi_replica_test.py`
5. Observe that the total allowed requests across all replicas stays within the global limit
6. Key question: What should the middleware do if Redis is unreachable? Fail-open or fail-closed?

## Motivation

- **Top system design interview question**: "Design a rate limiter" appears in interviews at Google, Meta, Amazon, Stripe, and Cloudflare. This practice gives you implementation-level depth that most candidates lack.
- **Production essential**: Every API needs rate limiting. Understanding the algorithms (not just configuring a library) is critical for choosing the right approach and debugging edge cases.
- **Redis mastery**: Lua scripting, sorted sets, atomic operations, and key expiry are Redis skills that transfer to caching, distributed locking, and session management.
- **Complements practice 052 (Resilience Patterns)**: Practice 052 covers rate limiting as one of three patterns; this practice goes deep on rate limiting specifically with five algorithms and distributed deployment.
- **Distributed systems fundamentals**: Shared state, atomicity, race conditions, and consistency across replicas are core distributed systems concepts demonstrated through a concrete, practical problem.

## Commands

All commands are run from `practice_078_distributed_rate_limiting/`.

### Phase 1: Infrastructure Setup

| Command | Description |
|---------|-------------|
| `docker compose up -d redis nginx` | Start Redis and Nginx load balancer (detached) |
| `docker compose up -d` | Start all services including 3 FastAPI replicas |
| `docker compose ps` | Check status of all containers |
| `docker compose exec redis redis-cli PING` | Verify Redis is responding |
| `docker compose exec redis redis-cli INFO memory` | Check Redis memory usage |
| `docker compose logs redis` | View Redis container logs |

### Phase 1: Python Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies (redis, fastapi, uvicorn, httpx, matplotlib) |

### Phases 2-5: Individual Algorithm Exercises

| Command | Description |
|---------|-------------|
| `uv run python src/01_fixed_window.py` | Run fixed window counter exercise (requires Redis) |
| `uv run python src/02_sliding_window_log.py` | Run sliding window log exercise (requires Redis) |
| `uv run python src/03_sliding_window_counter.py` | Run sliding window counter exercise (requires Redis) |
| `uv run python src/04_token_bucket.py` | Run token bucket exercise (requires Redis) |

### Phase 6: Algorithm Comparison

| Command | Description |
|---------|-------------|
| `uv run python src/05_comparison.py` | Run side-by-side benchmark of all four algorithms |

### Phase 7: FastAPI & Multi-Replica

| Command | Description |
|---------|-------------|
| `docker compose up --build -d` | Build and start all services (3 FastAPI replicas + Redis + Nginx) |
| `docker compose logs -f api` | Follow FastAPI replica logs |
| `docker compose logs -f nginx` | Follow Nginx load balancer logs |
| `uv run python src/07_multi_replica_test.py` | Run multi-replica rate limiting test |
| `curl http://localhost:8080/health` | Health check via load balancer |
| `curl http://localhost:8080/api/data` | Request data via load balancer (rate-limited) |
| `curl -H "X-API-Key: user-123" http://localhost:8080/api/data` | Request with explicit API key for per-user limiting |

### Inspection & Debugging

| Command | Description |
|---------|-------------|
| `docker compose exec redis redis-cli KEYS "ratelimit:*"` | List all rate limit keys in Redis |
| `docker compose exec redis redis-cli HGETALL "ratelimit:tokenbucket:user-123"` | Inspect token bucket state for a specific user |
| `docker compose exec redis redis-cli ZRANGE "ratelimit:slidingwindow:user-123" 0 -1 WITHSCORES` | Inspect sliding window entries for a user |
| `docker compose exec redis redis-cli DBSIZE` | Count total keys in Redis |
| `docker compose exec redis redis-cli FLUSHDB` | Clear all rate limit keys (reset state) |

### Cleanup

| Command | Description |
|---------|-------------|
| `docker compose down` | Stop and remove all containers |
| `docker compose down -v` | Stop, remove containers, and delete Redis data volume |

## State

`not-started`
