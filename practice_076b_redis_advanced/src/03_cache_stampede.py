# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "redis>=5.0",
# ]
# ///
"""Cache Stampede Prevention with XFetch.

Demonstrates the probabilistic early recomputation algorithm (XFetch) that
prevents cache stampedes without locks or coordination. Compares naive
cache-aside (which stampedes on expiry) with XFetch (which smoothly
recomputes before expiry).

Run with Redis running (docker compose up -d):
    uv run src/03_cache_stampede.py
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time

import redis.asyncio as aioredis


# ── Configuration ────────────────────────────────────────────────────

REDIS_URL = "redis://localhost:6379/0"
CACHE_KEY = "stampede:demo:expensive_result"
# We use a short TTL for demonstration purposes; in production this
# might be 60-300 seconds.
CACHE_TTL = 3  # seconds

# XFetch metadata keys (stored alongside the cached value)
META_SUFFIX_DELTA = ":delta"    # Time the last recomputation took
META_SUFFIX_EXPIRY = ":expiry"  # Unix timestamp when the value expires


# ── Simulated expensive computation ─────────────────────────────────

# Tracks how many times the "database" was queried (to measure stampede)
db_query_count = 0


async def expensive_computation(key: str) -> str:
    """Simulate an expensive database query or computation (~200ms)."""
    global db_query_count
    db_query_count += 1
    await asyncio.sleep(0.2)  # Simulate expensive work
    return json.dumps({
        "key": key,
        "result": f"computed_at_{time.time():.3f}",
        "query_number": db_query_count,
    })


# ── TODO(human): Implement this function ─────────────────────────────


async def xfetch_get(
    r: aioredis.Redis,
    key: str,
    ttl: int,
    recompute_fn,
    beta: float = 1.0,
) -> tuple[str, bool]:
    """Fetch a value using the XFetch probabilistic early recomputation algorithm.

    # ── Exercise Context ──────────────────────────────────────────────────
    # A cache stampede occurs when a popular key expires and N concurrent
    # clients all simultaneously fetch from the database to rebuild the
    # cache. If N=1000 and the DB query takes 200ms, you get 1000 identical
    # queries running in parallel -- this can easily take down a database.
    #
    # The XFetch algorithm (Vattani et al., presented at RedisConf 2017)
    # solves this WITHOUT locks or coordination. Each client independently
    # decides whether to recompute based on a probabilistic formula:
    #
    #   should_recompute = (now - delta * beta * ln(random())) >= expiry
    #
    # Where:
    #   - delta: time the last recomputation took (seconds)
    #   - beta: tuning parameter (default 1.0; higher = earlier recompute)
    #   - expiry: Unix timestamp when the cached value expires
    #   - random(): uniform random in (0, 1)
    #   - ln(): natural logarithm
    #
    # As `now` approaches `expiry`, the probability that the inequality
    # holds increases. The exponential distribution ensures that
    # statistically, exactly ONE client recomputes slightly before expiry.
    #
    # Key insight: `-ln(random())` follows an exponential distribution
    # with mean 1. Multiplied by `delta * beta`, it creates a "recompute
    # window" proportional to how expensive the computation is. Expensive
    # computations (large delta) trigger earlier recomputation.
    #
    # Benefits over locking:
    #   - No coordination needed (no distributed lock overhead)
    #   - No waiting (every client gets a response immediately)
    #   - Self-tuning (delta adapts to actual computation cost)
    #
    # Reference: https://github.com/internetarchive/xfetch
    # ──────────────────────────────────────────────────────────────────────

    TODO(human): Implement the XFetch algorithm.

    Steps:
      1. Read the cached value and its metadata from Redis:
             cached_value = await r.get(key)
             delta_str = await r.get(key + META_SUFFIX_DELTA)
             expiry_str = await r.get(key + META_SUFFIX_EXPIRY)

         - cached_value: the actual cached data (or None on first call)
         - delta: how long the last recomputation took (float seconds)
         - expiry: Unix timestamp when this cache entry expires

      2. Decide whether to recompute:

         a. If cached_value is None (cache miss):
            -> Always recompute (no cached data exists)

         b. If cached_value exists, apply the XFetch formula:
                now = time.time()
                delta = float(delta_str) if delta_str else 0.0
                expiry = float(expiry_str) if expiry_str else now
                random_val = random.random()

                # Guard against log(0)
                if random_val == 0:
                    random_val = 1e-10

                should_recompute = (now - delta * beta * math.log(random_val)) >= expiry

            -> If should_recompute is False, return (cached_value, True)
               (cache hit, no recomputation needed)

      3. If recomputing (either cache miss or probabilistic trigger):
         a. Record the start time:
                start = time.time()

         b. Call the recompute function:
                new_value = await recompute_fn(key)

         c. Calculate delta (how long recomputation took):
                new_delta = time.time() - start

         d. Calculate the new expiry timestamp:
                new_expiry = time.time() + ttl

         e. Store the value and metadata in Redis using a pipeline
            (for efficiency -- sends all commands in one round trip):
                async with r.pipeline() as pipe:
                    pipe.set(key, new_value, ex=ttl)
                    pipe.set(key + META_SUFFIX_DELTA, str(new_delta), ex=ttl)
                    pipe.set(key + META_SUFFIX_EXPIRY, str(new_expiry), ex=ttl)
                    await pipe.execute()

         f. Return (new_value, False) -- False means "recomputed"

    Args:
        r: Async Redis client.
        key: The cache key.
        ttl: Time-to-live in seconds for the cache entry.
        recompute_fn: Async callable that takes a key and returns the value.
        beta: XFetch tuning parameter. Higher = earlier probabilistic
              recomputation. Default 1.0 is optimal for most workloads.

    Returns:
        A tuple of (value, was_cache_hit).

    Hints:
      - math.log() is the natural logarithm (ln)
      - random.random() returns a float in [0, 1)
      - The pipeline is not strictly necessary (individual SET calls work)
        but is good practice -- fewer round trips to Redis
      - In production, you might store delta and expiry as fields in a
        Redis Hash instead of separate keys
    """
    raise NotImplementedError("TODO(human): Implement xfetch_get")


# ── Naive cache-aside (for comparison) ───────────────────────────────


async def naive_cache_get(
    r: aioredis.Redis,
    key: str,
    ttl: int,
    recompute_fn,
) -> tuple[str, bool]:
    """Standard cache-aside: on miss or expiry, recompute."""
    cached = await r.get(key)
    if cached is not None:
        return cached, True

    value = await recompute_fn(key)
    await r.set(key, value, ex=ttl)
    return value, False


# ── Demonstration (boilerplate) ──────────────────────────────────────


async def demo_stampede_comparison() -> None:
    """Compare naive cache-aside vs XFetch under simulated stampede."""
    r = aioredis.from_url(REDIS_URL, decode_responses=True)

    try:
        print("=== Cache Stampede Prevention: Naive vs XFetch ===\n")

        # --- Experiment 1: Naive cache-aside stampede ---
        print("--- Experiment 1: Naive Cache-Aside ---")
        print(f"  Simulating {20} concurrent clients hitting an expired key...\n")

        # Clear cache and reset counter
        global db_query_count
        await r.delete("naive:demo")
        db_query_count = 0

        # Populate cache, then let it expire
        await r.set("naive:demo", "seed_value", ex=1)
        print("  Seeded cache with TTL=1s. Waiting for expiry...")
        await asyncio.sleep(1.5)

        # Stampede: 20 concurrent requests hit an expired key
        db_query_count = 0
        tasks = [naive_cache_get(r, "naive:demo", CACHE_TTL, expensive_computation) for _ in range(20)]
        results = await asyncio.gather(*tasks)

        hits = sum(1 for _, hit in results if hit)
        misses = sum(1 for _, hit in results if not hit)
        print(f"  Results: {hits} hits, {misses} misses")
        print(f"  Database queries triggered: {db_query_count}")
        print(f"  --> STAMPEDE: {db_query_count} redundant DB calls!")

        # --- Experiment 2: XFetch prevents stampede ---
        print("\n--- Experiment 2: XFetch Probabilistic Early Recomputation ---")
        print(f"  Same scenario: {20} concurrent clients, but with XFetch...\n")

        # Seed cache with XFetch metadata
        await r.delete(CACHE_KEY, CACHE_KEY + META_SUFFIX_DELTA, CACHE_KEY + META_SUFFIX_EXPIRY)
        db_query_count = 0

        # Initial population
        val, _ = await xfetch_get(r, CACHE_KEY, CACHE_TTL, expensive_computation, beta=1.0)
        initial_queries = db_query_count
        print(f"  Initial cache fill: {initial_queries} DB query (expected: 1)")

        # Wait until close to expiry (but not past it)
        print(f"  Waiting {CACHE_TTL - 0.5}s (near expiry but not expired)...")
        await asyncio.sleep(CACHE_TTL - 0.5)

        # Near-expiry burst: XFetch should trigger ~1 early recomputation
        db_query_count = 0
        tasks = [xfetch_get(r, CACHE_KEY, CACHE_TTL, expensive_computation, beta=1.0) for _ in range(20)]
        results = await asyncio.gather(*tasks)

        recomputed = sum(1 for _, hit in results if not hit)
        served_from_cache = sum(1 for _, hit in results if hit)
        print(f"  Results: {served_from_cache} served from cache, {recomputed} triggered recomputation")
        print(f"  Database queries triggered: {db_query_count}")
        print(f"  --> XFetch: only ~{db_query_count} DB call(s) instead of {20}")

        # --- Experiment 3: Beta tuning ---
        print("\n--- Experiment 3: Beta Tuning ---")
        for beta in [0.5, 1.0, 2.0, 5.0]:
            await r.delete(CACHE_KEY, CACHE_KEY + META_SUFFIX_DELTA, CACHE_KEY + META_SUFFIX_EXPIRY)
            db_query_count = 0

            # Seed
            await xfetch_get(r, CACHE_KEY, CACHE_TTL, expensive_computation, beta=beta)
            db_query_count = 0

            # Wait until 80% of TTL has elapsed
            await asyncio.sleep(CACHE_TTL * 0.8)

            tasks = [xfetch_get(r, CACHE_KEY, CACHE_TTL, expensive_computation, beta=beta) for _ in range(10)]
            await asyncio.gather(*tasks)

            print(f"  beta={beta:.1f}: {db_query_count} recomputations from 10 clients at 80% TTL")

    finally:
        await r.aclose()


async def main() -> None:
    print("=" * 60)
    print(" Practice 076b: Cache Stampede Prevention (XFetch)")
    print("=" * 60)
    print()
    await demo_stampede_comparison()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
