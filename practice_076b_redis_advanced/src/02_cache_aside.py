# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "redis>=5.0",
# ]
# ///
"""Cache-Aside Pattern with Redis.

Demonstrates the most common caching strategy: check Redis first, on miss
fetch from the "database" (simulated), store in Redis with TTL, return.
Also covers cache invalidation on writes.

Run with Redis running (docker compose up -d):
    uv run src/02_cache_aside.py
"""

from __future__ import annotations

import asyncio
import json
import time

import redis.asyncio as aioredis


# ── Configuration ────────────────────────────────────────────────────

REDIS_URL = "redis://localhost:6379/0"
CACHE_PREFIX = "cache:user:"
DEFAULT_TTL_SECONDS = 10

# Simulated "database" -- in production this would be PostgreSQL, etc.
FAKE_DATABASE: dict[str, dict] = {
    "user:1": {"id": 1, "name": "Alice", "email": "alice@example.com", "role": "admin"},
    "user:2": {"id": 2, "name": "Bob", "email": "bob@example.com", "role": "user"},
    "user:3": {"id": 3, "name": "Charlie", "email": "charlie@example.com", "role": "user"},
    "user:4": {"id": 4, "name": "Diana", "email": "diana@example.com", "role": "moderator"},
    "user:5": {"id": 5, "name": "Eve", "email": "eve@example.com", "role": "user"},
}


# ── Simulated slow database ─────────────────────────────────────────


async def fake_db_query(key: str) -> dict | None:
    """Simulate a slow database query (100-200ms latency)."""
    await asyncio.sleep(0.15)  # Simulate DB latency
    return FAKE_DATABASE.get(key)


async def fake_db_update(key: str, data: dict) -> None:
    """Simulate a database write."""
    await asyncio.sleep(0.1)  # Simulate DB write latency
    FAKE_DATABASE[key] = data


# ── TODO(human): Implement these functions ───────────────────────────


async def cache_aside_get(
    r: aioredis.Redis,
    user_id: int,
    ttl: int = DEFAULT_TTL_SECONDS,
) -> tuple[dict | None, bool]:
    """Fetch a user using the cache-aside pattern.

    # ── Exercise Context ──────────────────────────────────────────────────
    # Cache-aside (also called "lazy loading") is the most common caching
    # strategy because it is the safest default:
    #   - Only requested data enters the cache (no wasted memory)
    #   - If Redis fails, the application falls back to the DB (degraded
    #     performance, not data loss)
    #   - The application has full control over what gets cached and for
    #     how long
    #
    # The read path is:
    #   1. Check Redis for the key
    #   2. If HIT: return cached data (fast path, ~1ms)
    #   3. If MISS: query the database (~150ms), store result in Redis
    #      with a TTL, return the data
    #
    # The TTL is critical: without it, stale data lives forever. But
    # choosing the right TTL is a trade-off: too short = more DB load,
    # too long = more stale reads.
    # ──────────────────────────────────────────────────────────────────────

    TODO(human): Implement the cache-aside read pattern.

    Steps:
      1. Build the cache key:
             cache_key = f"{CACHE_PREFIX}{user_id}"

      2. Try to read from Redis:
             cached = await r.get(cache_key)

      3. If cached is not None (cache HIT):
         a. Deserialize the JSON string back to a dict:
                data = json.loads(cached)
         b. Return (data, True) -- True means "cache hit"

      4. If cached is None (cache MISS):
         a. Fetch from the "database":
                data = await fake_db_query(f"user:{user_id}")
         b. If data is not None, store it in Redis with a TTL:
                await r.set(cache_key, json.dumps(data), ex=ttl)
            - json.dumps() serializes the dict to a JSON string
            - ex=ttl sets the expiration in seconds
         c. Return (data, False) -- False means "cache miss"

    Args:
        r: Async Redis client.
        user_id: The user ID to fetch.
        ttl: Cache TTL in seconds.

    Returns:
        A tuple of (user_data_or_None, was_cache_hit).

    Why json.dumps/loads?
      Redis values are strings (or bytes). You must serialize structured
      data to store it. JSON is the simplest format for this. In production,
      consider msgpack or pickle for better performance.

    Docs: https://redis-py.readthedocs.io/en/stable/commands.html#redis.commands.core.CoreCommands.get
    """
    raise NotImplementedError("TODO(human): Implement cache_aside_get")


async def cache_aside_invalidate(
    r: aioredis.Redis,
    user_id: int,
    new_data: dict,
) -> None:
    """Update the database and invalidate the cache entry.

    # ── Exercise Context ──────────────────────────────────────────────────
    # Cache invalidation on writes is one of the "two hard things in
    # computer science." The cache-aside pattern uses DELETE (not UPDATE)
    # on the cache key when the underlying data changes. Why?
    #
    # DELETE is safer than UPDATE because of this failure scenario:
    #   1. Service A updates the DB (succeeds)
    #   2. Service A updates the cache (fails -- network blip)
    #   Result: DB has new data, cache has OLD data = stale reads
    #
    # With DELETE instead:
    #   1. Service A updates the DB (succeeds)
    #   2. Service A deletes the cache key (even if this fails, the TTL
    #      will eventually expire the stale entry)
    #   Result: Next read triggers a cache miss and loads fresh data
    #
    # This "invalidate, don't update" principle is a fundamental caching
    # best practice.
    # ──────────────────────────────────────────────────────────────────────

    TODO(human): Implement cache invalidation on write.

    Steps:
      1. Write the new data to the "database":
             await fake_db_update(f"user:{user_id}", new_data)

      2. Delete the cache entry (invalidate):
             cache_key = f"{CACHE_PREFIX}{user_id}"
             await r.delete(cache_key)

    Note: We delete AFTER the DB write succeeds. If the DB write fails,
    we do NOT invalidate the cache (the cached data is still correct).
    The order matters: DB first, then cache.

    Args:
        r: Async Redis client.
        user_id: The user ID being updated.
        new_data: The new user data to write.
    """
    raise NotImplementedError("TODO(human): Implement cache_aside_invalidate")


# ── Demonstration (boilerplate) ──────────────────────────────────────


async def demo_cache_performance() -> None:
    """Demonstrate cache-aside pattern with performance metrics."""
    r = aioredis.from_url(REDIS_URL, decode_responses=True)

    try:
        # Clean up
        for key in await r.keys(f"{CACHE_PREFIX}*"):
            await r.delete(key)

        print("=== Cache-Aside Pattern ===\n")

        # --- Test 1: Cache miss (cold cache) ---
        print("--- Test 1: Cache Miss (cold cache) ---")
        t0 = time.perf_counter()
        data, hit = await cache_aside_get(r, 1)
        elapsed_miss = (time.perf_counter() - t0) * 1000
        print(f"  User 1: {data}")
        print(f"  Cache hit: {hit} (expected: False)")
        print(f"  Latency: {elapsed_miss:.1f}ms (includes DB query)")

        # --- Test 2: Cache hit (warm cache) ---
        print("\n--- Test 2: Cache Hit (warm cache) ---")
        t0 = time.perf_counter()
        data, hit = await cache_aside_get(r, 1)
        elapsed_hit = (time.perf_counter() - t0) * 1000
        print(f"  User 1: {data}")
        print(f"  Cache hit: {hit} (expected: True)")
        print(f"  Latency: {elapsed_hit:.1f}ms (Redis only)")

        speedup = elapsed_miss / elapsed_hit if elapsed_hit > 0 else float("inf")
        print(f"\n  Speedup: {speedup:.0f}x faster on cache hit")

        # --- Test 3: Multiple users ---
        print("\n--- Test 3: Multiple users (mix of hits and misses) ---")
        hit_count = 0
        miss_count = 0
        for uid in [1, 2, 3, 1, 2, 4, 1, 5, 3, 2]:
            _, hit = await cache_aside_get(r, uid)
            if hit:
                hit_count += 1
            else:
                miss_count += 1
        print(f"  Hits: {hit_count}, Misses: {miss_count}")
        print(f"  Hit rate: {hit_count / (hit_count + miss_count) * 100:.0f}%")

        # --- Test 4: Cache invalidation ---
        print("\n--- Test 4: Cache Invalidation ---")
        _, hit = await cache_aside_get(r, 1)
        print(f"  Before update - Cache hit: {hit} (expected: True)")

        new_data = {"id": 1, "name": "Alice", "email": "alice_new@example.com", "role": "superadmin"}
        await cache_aside_invalidate(r, 1, new_data)
        print(f"  Updated user 1 and invalidated cache")

        data, hit = await cache_aside_get(r, 1)
        print(f"  After update - Cache hit: {hit} (expected: False, cache was invalidated)")
        print(f"  Data reflects update: email={data['email']}, role={data['role']}")

        # --- Test 5: Non-existent user ---
        print("\n--- Test 5: Non-existent user ---")
        data, hit = await cache_aside_get(r, 999)
        print(f"  User 999: {data} (expected: None)")
        print(f"  Cache hit: {hit} (expected: False)")

        # --- Test 6: TTL verification ---
        print("\n--- Test 6: TTL verification ---")
        await cache_aside_get(r, 2, ttl=2)
        ttl_remaining = await r.ttl(f"{CACHE_PREFIX}2")
        print(f"  TTL for user 2: {ttl_remaining}s (expected: ~2)")

    finally:
        await r.aclose()


async def main() -> None:
    print("=" * 60)
    print(" Practice 076b: Cache-Aside Pattern")
    print("=" * 60)
    print()
    await demo_cache_performance()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
