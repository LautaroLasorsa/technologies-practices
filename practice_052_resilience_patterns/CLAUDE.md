# Practice 052: Resilience Patterns — Circuit Breaker, Bulkhead & Rate Limiting

## Technologies

- **Circuit Breaker Pattern** — State-machine-based fault isolation (popularized by Michael Nygard in *Release It!*)
- **Bulkhead Pattern** — Semaphore/pool-based dependency isolation (from ship hull compartment design)
- **Rate Limiting** — Token bucket and sliding window algorithms for request throttling
- **Redis** — In-memory data store used for distributed rate limiting state
- **asyncio** — Python's native async runtime for concurrent I/O
- **aiohttp** — Async HTTP client/server library for simulating unreliable services

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose (Redis)

## Theoretical Context

### Why Resilience Patterns Matter

In a microservices architecture, failures are not exceptional — they are routine. Network partitions, overloaded services, garbage collection pauses, deployment rollouts, database connection exhaustion, and hardware faults all occur regularly at scale. Without resilience patterns, a single slow or failing dependency can cascade through the entire system: threads block waiting for timeouts, connection pools drain, memory fills with queued requests, and eventually every service in the call chain becomes unresponsive. This is known as **cascading failure** — one component's failure propagates upstream and downstream until the entire system collapses.

Resilience patterns address this by accepting that failures will happen and designing systems to **degrade gracefully** rather than fail completely. The three core patterns — circuit breaker, bulkhead, and rate limiting — each address a different failure vector: circuit breakers prevent repeated calls to broken dependencies, bulkheads isolate resource consumption per dependency, and rate limiters protect services from overload. Used together, they form a layered defense that keeps a system partially operational even when individual components fail.

The key insight behind all resilience patterns is the difference between **failing fast** and **failing slow**. A service that returns an error in 5ms is far less dangerous than one that hangs for 30 seconds. Fast failure frees resources immediately; slow failure ties them up, creating backpressure that propagates through the system. Every resilience pattern ultimately converts slow failures into fast ones.

### Circuit Breaker Pattern

The circuit breaker pattern was popularized by Michael Nygard in his book *Release It!* (2007, 2nd edition 2018) and formalized by Martin Fowler. The name comes from electrical circuit breakers: when current (failures) exceeds a safe threshold, the breaker trips to protect the circuit (system) from damage.

A circuit breaker is a **finite state machine** with three states:

| State | Behavior | Transitions |
|-------|----------|-------------|
| **CLOSED** | Requests pass through normally. The breaker monitors success/failure counts. | -> OPEN when failure count >= threshold within a time window |
| **OPEN** | All requests are **rejected immediately** (fail fast) without calling the downstream service. | -> HALF_OPEN after a recovery timeout expires |
| **HALF_OPEN** | A limited number of **probe requests** are allowed through to test if the downstream service has recovered. | -> CLOSED if probe requests succeed (service recovered) |
|  |  | -> OPEN if any probe request fails (service still broken) |

**State transition details:**

1. **CLOSED -> OPEN**: The breaker maintains a failure counter (or sliding window of recent call outcomes). When failures exceed a configured threshold — for example, 5 consecutive failures, or >50% failure rate in the last 60 seconds — the breaker "trips" to OPEN. This prevents further requests from reaching the already-struggling downstream service.

2. **OPEN -> HALF_OPEN**: After a configured recovery timeout (e.g., 30 seconds), the breaker transitions to HALF_OPEN. During this timeout, all calls are rejected instantly. The timeout gives the downstream service time to recover (restart, clear its queue, scale up, etc.).

3. **HALF_OPEN -> CLOSED**: In HALF_OPEN, the breaker allows a small number of test requests (e.g., 3). If all succeed, the breaker assumes the downstream service has recovered and transitions back to CLOSED, resuming normal operation.

4. **HALF_OPEN -> OPEN**: If any test request in HALF_OPEN fails, the breaker trips back to OPEN immediately. The downstream service has not yet recovered, so the breaker extends the protection period.

**Why circuit breakers matter:**
- **Prevents cascading failure**: Without a circuit breaker, callers continue sending requests to a broken service, each one consuming a thread/connection while waiting for a timeout. This drains the caller's resources and propagates the failure upstream.
- **Gives recovery time**: An overwhelmed service receiving 10,000 requests/second cannot recover if requests keep arriving. The circuit breaker cuts traffic to zero during the OPEN state, giving the service breathing room.
- **Provides fast feedback**: Instead of waiting 30 seconds for a timeout, callers get an immediate `CircuitOpenError` in <1ms. This lets them execute fallback logic (serve cached data, return degraded response, route to a backup service).
- **Monitors health**: The breaker's state is a real-time indicator of downstream health. Dashboards can show which breakers are OPEN, alerting operators to problems.

**Configuration trade-offs:**
- **Failure threshold too low** -> breaker trips on transient errors (false positives), reducing availability
- **Failure threshold too high** -> breaker trips too late, allowing cascading failure to begin
- **Recovery timeout too short** -> breaker enters HALF_OPEN before downstream has recovered, causing flapping
- **Recovery timeout too long** -> service stays isolated longer than necessary, reducing availability

### Bulkhead Pattern

The bulkhead pattern takes its name from ship hull design. Ships are divided into watertight compartments (bulkheads) so that if one compartment floods, the ship stays afloat — the damage is contained. In software, bulkheads isolate different parts of a system so that a failure in one dependency does not exhaust shared resources and starve other dependencies.

**The problem bulkheads solve:** Consider a service that calls three downstream APIs: a user service, a payment service, and a notification service. All three share a common HTTP connection pool of 100 connections. If the payment service becomes slow (responding in 10 seconds instead of 100ms), all 100 connections gradually become occupied waiting for payment responses. Now user service and notification service calls also fail — not because they are broken, but because there are no free connections. A single slow dependency has taken down the entire service.

**Implementation approaches:**

| Approach | Mechanism | Granularity |
|----------|-----------|-------------|
| **Semaphore-based** | `asyncio.Semaphore(N)` limits concurrent calls to a dependency | Per-dependency concurrency limit |
| **Thread pool-based** | Separate `ThreadPoolExecutor` per dependency | Per-dependency thread isolation |
| **Connection pool-based** | Separate HTTP client/connection pool per dependency | Per-dependency connection isolation |
| **Process-based** | Separate process/container per dependency group | Strongest isolation (memory, CPU) |

In async Python, the semaphore-based approach is the most natural: each dependency gets an `asyncio.Semaphore` that limits how many concurrent calls can be in flight. When the semaphore is full, additional calls either wait (with a timeout) or are rejected immediately with a `BulkheadFullError`.

**Key design decisions:**
- **Max concurrent calls per dependency**: Too low → artificial bottleneck under normal load. Too high → insufficient isolation. Rule of thumb: set to the dependency's expected throughput × expected latency × safety factor.
- **Wait timeout**: How long to wait for a semaphore slot. Zero (fail immediately) gives the fastest feedback. A short timeout (e.g., 500ms) allows brief bursts. A long timeout defeats the purpose.
- **What to do when rejected**: Return cached data, use a default value, propagate the error, or queue for later retry.

### Rate Limiting

Rate limiting controls the rate at which requests are accepted, protecting services from overload. Unlike circuit breakers (which react to downstream failures) and bulkheads (which limit concurrency), rate limiters proactively cap throughput before problems occur.

**Four common algorithms:**

**1. Token Bucket** — The most widely used algorithm. A bucket holds tokens up to a maximum capacity. Tokens are added at a fixed rate (e.g., 10 tokens/second). Each request consumes one token. If the bucket is empty, the request is rejected. This allows **burst traffic** up to the bucket capacity while maintaining a long-term average rate.

```
Bucket capacity: 20 tokens
Refill rate: 10 tokens/second

Time 0: bucket = 20 (full)
Burst of 20 requests: all pass, bucket = 0
Next 2 seconds: 0 requests arrive, bucket refills to 20
Steady 10 req/sec: all pass, bucket stays near 10
```

**2. Sliding Window** — Count requests in a window of the last N seconds. Each request checks: "how many requests occurred in the last N seconds?" If the count exceeds the limit, reject. Uses Redis sorted sets (ZRANGEBYSCORE) for distributed implementation. Provides smoother rate limiting than fixed windows — no boundary spike problem.

**3. Fixed Window** — Divide time into fixed intervals (e.g., 1-minute buckets). Count requests per interval. Simple but has a boundary problem: a client can send `limit` requests at second 59 of one window and `limit` requests at second 0 of the next window, effectively doubling the rate.

**4. Leaky Bucket** — Requests enter a FIFO queue (the bucket). The queue drains at a fixed rate. If the queue is full, new requests are dropped. Produces a perfectly smooth output rate but adds latency (queuing). Used in network traffic shaping.

| Algorithm | Burst Tolerance | Smoothness | Complexity | Redis Pattern |
|-----------|----------------|------------|------------|---------------|
| Token Bucket | Yes (up to capacity) | Good (long-term average) | Medium | HASH + Lua script |
| Sliding Window | No | Excellent | Medium | Sorted Set (ZSET) |
| Fixed Window | Boundary spike | Fair | Low | INCR + EXPIRE |
| Leaky Bucket | No (queued) | Perfect | Medium | List (LPUSH/RPOP) |

**Distributed rate limiting with Redis:** In-memory rate limiters only work for single-process applications. In a distributed system with multiple replicas, rate state must be shared. Redis is the standard choice because: (1) it is fast enough for the read-modify-write cycle on every request, (2) Lua scripts provide atomic execution of multi-step logic, and (3) TTL-based key expiry handles cleanup automatically. The token bucket Lua script atomically calculates elapsed time, refills tokens, and tries to consume — all in a single round trip.

### Retry with Exponential Backoff

Retries complement circuit breakers. When a request fails with a transient error (503 Service Unavailable, connection timeout, rate limited), retrying after a delay often succeeds. **Exponential backoff** increases the delay between retries: 1s, 2s, 4s, 8s, etc. **Jitter** adds a random offset to prevent the **thundering herd** problem — when many clients retry at exactly the same time after a service restart, causing it to fail again immediately.

```
retry_delay = min(base_delay * (2 ** attempt) + random(0, jitter), max_delay)
```

**When to retry**: transient errors (503, 429, connection reset, timeout). **When NOT to retry**: permanent errors (400 Bad Request, 404 Not Found, 401 Unauthorized). Retries should be bounded (max 3-5 attempts) and integrated with the circuit breaker — when the breaker is OPEN, retries are pointless and should be skipped.

### Timeout Patterns

Two distinct timeouts apply to every remote call:

- **Connection timeout**: Maximum time to establish the TCP connection. If the server is unreachable (firewall, DNS failure), this fires. Typically 1-5 seconds.
- **Read timeout**: Maximum time to wait for a response after the connection is established. If the server is processing slowly, this fires. Typically 5-30 seconds depending on the operation.

**Cascading timeouts**: In a call chain A -> B -> C, A's timeout must be greater than B's, which must be greater than C's. If A has a 5-second timeout and B has a 10-second timeout, A will time out before B gets a response — and B's work is wasted.

### Ecosystem and Real-World Tools

| Tool | Language | Notes |
|------|----------|-------|
| **Netflix Hystrix** | Java | The original circuit breaker library (2012). Now deprecated but historically influential. Introduced bulkhead, circuit breaker, and fallback patterns to the mainstream. |
| **resilience4j** | Java | Hystrix's spiritual successor. Lightweight, functional-style API. Provides CircuitBreaker, Bulkhead, RateLimiter, Retry, TimeLimiter as composable decorators. |
| **Polly** | .NET | Comprehensive resilience library with circuit breaker, retry, bulkhead, timeout, fallback policies. |
| **tenacity** | Python | Retry library with exponential backoff, jitter, and stop conditions. Does not include circuit breaker or bulkhead. |
| **aiobreaker** | Python | Async circuit breaker for Python. Lightweight but limited to circuit breaking only. |
| **Envoy Proxy** | C++ (sidecar) | Service mesh sidecar that implements circuit breaking, rate limiting, retries, and timeouts at the infrastructure level — no application code changes needed. |
| **Istio** | Go (control plane) | Service mesh built on Envoy that provides resilience policies via configuration (DestinationRule, VirtualService). |

In production, resilience patterns are often applied at **two levels**: application-level (libraries like resilience4j, Polly) for fine-grained control, and infrastructure-level (Envoy, Istio) for cross-cutting policies. This practice implements them at the application level to build deep understanding of the mechanics.

### References

- Nygard, Michael T. *Release It! Design and Deploy Production-Ready Software*. 2nd ed., Pragmatic Bookshelf, 2018.
- Fowler, Martin. ["CircuitBreaker."](https://martinfowler.com/bliki/CircuitBreaker.html) martinfowler.com, 2014.
- Microsoft. ["Circuit Breaker pattern."](https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker) Azure Architecture Center.
- Microsoft. ["Bulkhead pattern."](https://learn.microsoft.com/en-us/azure/architecture/patterns/bulkhead) Azure Architecture Center.
- Redis. ["Build 5 Rate Limiters with Redis."](https://redis.io/tutorials/howtos/ratelimiting/) Redis Tutorials.
- resilience4j. ["Getting Started."](https://resilience4j.readme.io/docs/getting-started) resilience4j documentation.

## Description

Build a **Resilience Testing Lab** that implements the core resilience patterns and composes them into a layered defense. An unreliable HTTP service (simulated with aiohttp) serves as the target, with configurable failure modes (healthy, degraded, down). You implement:

1. **Circuit Breaker** — A state machine (CLOSED/OPEN/HALF_OPEN) that monitors downstream failures and fails fast when the downstream is broken. Three small focused TODOs: the time-based `state` auto-transition, `_handle_success`, and `_handle_failure`.
2. **Bulkhead** — Semaphore-based dependency isolation that prevents a slow dependency from starving resources for other dependencies. One focused TODO: the `call()` method with timeout-acquire + try/finally release.
3. **Rate Limiter (provided)** — A working Redis token-bucket limiter is provided as a finished component. Rate limiting algorithms (fixed window, sliding window log/counter, token bucket) and their Lua scripts are covered in depth in practice 078, so this practice does not re-implement them.
4. **Combined Resilience Layer** — Stack rate limiter, bulkhead, and circuit breaker into a single `ResilientClient`. One focused TODO: the composition in `call()` mapping each layer's exception to a rejection reason.

All demos, load tests, and plotting are scaffolded — you implement the core pattern mechanics, not the measurement harness.

### Architecture

```
                    +-------------------+
  Requests ------->| Rate Limiter      |  (Redis-backed, rejects over-limit)
                    +--------+----------+
                             |
                    +--------v----------+
                    | Bulkhead          |  (semaphore, rejects when pool full)
                    +--------+----------+
                             |
                    +--------v----------+
                    | Circuit Breaker   |  (state machine, rejects when OPEN)
                    +--------+----------+
                             |
                    +--------v----------+
                    | Downstream Call   |  (HTTP to unreliable service)
                    +-------------------+
```

Each layer can independently reject a request. The order matters: rate limiting first (cheapest check, no I/O), then bulkhead (limits concurrency), then circuit breaker (checks downstream health), then the actual call.

### What you'll learn

1. **Circuit breaker state machine** — Implementing the three-state FSM with threshold-based transitions
2. **Semaphore-based bulkhead** — Using `asyncio.Semaphore` with timeouts for dependency isolation
3. **Distributed rate limiting** — Atomic Redis operations with Lua scripts for token bucket; sorted sets for sliding window
4. **Pattern composition** — How resilience patterns layer and interact (which order, how metrics flow)
5. **Failure simulation** — Designing controllable failure modes for testing resilience logic

## Instructions

### Phase 1: Setup & Infrastructure (~5 min)

1. Start Redis: `docker compose up -d`
2. Verify Redis: `docker compose ps` (redis container should be running)
3. Install Python dependencies: `uv sync`
4. Start the unreliable service in a dedicated terminal: `uv run python -m src._00_unreliable_service`
5. Quick sanity check: `curl http://localhost:8089/health` → `{"status": "ok"}`

### Phase 2: Circuit Breaker (~25 min)

1. Open `src/_01_circuit_breaker.py`
2. **TODO 1 — `state` property**: Implement the lazy, time-based `OPEN → HALF_OPEN` auto-transition. This is how real circuit breakers (resilience4j, Polly) avoid background timers — every state read checks the clock.
3. **TODO 2 — `_handle_success()`**: CLOSED resets the failure counter; HALF_OPEN counts successful probes and closes when enough have succeeded. This is how a breaker *recovers*.
4. **TODO 3 — `_handle_failure()`**: CLOSED increments the failure counter and trips to OPEN at threshold; HALF_OPEN goes back to OPEN on any probe failure. This is how a breaker *trips* and *re-trips*.
5. The `call()` method is scaffolded — it reads state, rejects fast when OPEN, and dispatches to your two handlers. Read it to see how the three TODOs fit together.
6. Run: `uv run python -m src._01_circuit_breaker` (with the unreliable service running)
7. Watch the state transitions print as the breaker moves CLOSED → OPEN → HALF_OPEN → CLOSED.
8. Key question: If the downstream service needs 10s to restart, what `recovery_timeout` would you choose and why?

### Phase 3: Bulkhead (~15 min)

1. Open `src/_02_bulkhead.py`
2. **TODO — `Bulkhead.call()`**: Acquire the semaphore with `asyncio.wait_for(..., timeout=max_wait_time)`. On `TimeoutError`, record a rejection and raise `BulkheadFullError`. Execute `func` in a `try/finally` so the slot is always released. Track `peak_concurrent` and `call_latencies`.
3. The isolation demo (with vs without bulkhead, on a degraded service) is scaffolded.
4. Run: `uv run python -m src._02_bulkhead`
5. Compare the printed metrics — `fast_api` average latency should be dramatically lower when bulkheads isolate the fast dependency from the slow one.
6. Key question: How would you decide `max_concurrent` for a production bulkhead?

### Phase 4: Rate Limiter (provided, ~5 min)

1. Open `src/_03_rate_limiter.py` — skim it, there are **no TODOs**. This module provides a finished `TokenBucketRateLimiter` backed by Redis + Lua for exercise 4 to compose with.
2. Rate limiting algorithms and their atomic Lua scripts are covered in depth in **practice 078** (fixed window, sliding window log, sliding window counter, token bucket) — don't re-implement them here.
3. Sanity check (optional): `uv run python -m src._03_rate_limiter` runs a burst through the bucket to confirm Redis is reachable.

### Phase 5: Combined Resilience Layer (~20 min)

1. Open `src/_04_combined_resilience.py`
2. **TODO — `ResilientClient.call()`**: Compose the three layers in cost-ascending order: `rate_limiter.require()` first, then `bulkhead.call(circuit_breaker.call, func, *args, **kwargs)`. Map each layer's exception to a rejection reason (`"rate_limit"`, `"bulkhead"`, `"circuit_open"`, `"error"`) via `self._record(...)`, then re-raise. On success, record and return the value.
3. The load test (mode-timeline, RPS generator, dashboard plot) is scaffolded.
4. Run: `uv run python -m src._04_combined_resilience`
5. Inspect `plots/combined_resilience_dashboard.png` — scatter of outcomes over time, circuit breaker state timeline, cumulative outcomes, and a rejection pie.
6. Key question: In what order should you tune the three patterns? Which parameters affect each other?

### Phase 6: Reflection (~5 min)

1. Application-level resilience vs service-mesh resilience (Envoy/Istio) — when would you choose each?
2. How would you test resilience in CI/CD? (chaos engineering, fault injection)
3. What production metrics indicate a resilience pattern is activating?

## Motivation

- **Production essential**: Every production microservice needs resilience patterns. Understanding them from first principles (not just configuring a library) is critical for debugging failures and tuning parameters.
- **Complements practice 014 (SAGA)**: SAGAs handle business-level failure recovery; resilience patterns handle infrastructure-level failure isolation. Together they form a complete failure management strategy.
- **Industry standard**: Netflix (Hystrix), AWS (circuit breaker in AppMesh), Azure (resilience patterns documentation), and Google (SRE practices) all rely on these patterns.
- **Interview relevance**: System design interviews frequently ask about cascading failures, circuit breakers, and rate limiting. Implementing them from scratch demonstrates deep understanding.
- **Foundation for service mesh**: Understanding what Envoy/Istio do under the hood (circuit breaking, rate limiting, retries) makes you effective at configuring and debugging service mesh policies.

## Commands

All commands are run from `practice_052_resilience_patterns/`.

### Phase 1: Docker & Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Redis (detached) |
| `docker compose ps` | Check Redis container status |
| `docker compose logs redis` | View Redis logs |
| `docker compose down` | Stop and remove containers |
| `docker compose down -v` | Stop, remove containers, and delete volumes (full reset) |

### Phase 2: Python Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies (redis, aiohttp, matplotlib) |

### Phase 3: Unreliable Service (run in dedicated terminal)

| Command | Description |
|---------|-------------|
| `uv run python -m src._00_unreliable_service` | Start the simulated unreliable HTTP service on port 8089 |
| `curl http://localhost:8089/health` | Check service health (always responds) |
| `curl http://localhost:8089/api/data` | Call the unreliable endpoint (subject to current failure mode) |
| `curl -X POST http://localhost:8089/admin/mode/healthy` | Switch service to healthy mode (100% success, fast responses) |
| `curl -X POST http://localhost:8089/admin/mode/degraded` | Switch service to degraded mode (50% errors, slow responses) |
| `curl -X POST http://localhost:8089/admin/mode/down` | Switch service to down mode (100% errors) |

### Phase 4: Exercises

| Command | Description |
|---------|-------------|
| `uv run python -m src._01_circuit_breaker` | Run circuit breaker exercise (requires unreliable service running) |
| `uv run python -m src._02_bulkhead` | Run bulkhead exercise (requires unreliable service running) |
| `uv run python -m src._03_rate_limiter` | Sanity check the provided token-bucket limiter (requires Redis running) |
| `uv run python -m src._04_combined_resilience` | Run combined resilience exercise (requires both unreliable service and Redis) |

## State

`not-started`
