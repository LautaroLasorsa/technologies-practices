"""Exercise 04: Token Bucket Rate Limiter

The industry-standard rate limiting algorithm. A bucket holds tokens up to
a maximum capacity. Tokens refill at a constant rate. Each request consumes
tokens. If insufficient tokens, the request is rejected.

The key insight: tokens are refilled lazily (calculated on each request based
on elapsed time), not by a background timer. This makes the algorithm
stateless between requests and easy to implement in Redis with a Lua script.

This is the CORE exercise of the practice.

Run (requires Redis from docker-compose):
    uv run python src/04_token_bucket.py
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import redis.asyncio as aioredis

from src.common import (
    REDIS_URL,
    RateLimiter,
    create_redis_client,
    load_lua_script,
    print_header,
    print_metrics,
)


# ---------------------------------------------------------------------------
# Token Bucket Rate Limiter
# ---------------------------------------------------------------------------


class TokenBucketRateLimiter(RateLimiter):
    """Token bucket rate limiter backed by Redis + Lua.

    Parameters
    ----------
    name:
        Human-readable name for this limiter.
    redis_client:
        An async Redis client.
    capacity:
        Maximum tokens the bucket can hold (= max burst size).
    refill_rate:
        Tokens added per second (= sustained request rate).
    key_prefix:
        Redis key prefix for namespacing.
    """

    def __init__(
        self,
        name: str,
        redis_client: aioredis.Redis,
        capacity: int = 20,
        refill_rate: float = 10.0,
        key_prefix: str = "ratelimit:tokenbucket",
    ) -> None:
        super().__init__(name)
        self.redis = redis_client
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.key_prefix = key_prefix
        self._script: Any | None = None

    async def _ensure_script(self) -> None:
        """Register the Lua script with Redis (done once, cached via SHA)."""
        if self._script is None:
            lua_source = load_lua_script("token_bucket")
            self._script = self.redis.register_script(lua_source)

    async def allow(self, key: str, cost: int = 1) -> bool:
        """Check if the request is allowed under the token bucket algorithm.

        Returns True if allowed (tokens consumed), False if rejected.
        """
        # TODO(human): Implement the token bucket allow() method.
        #
        # WHY THIS MATTERS:
        # The token bucket is the most important rate limiting algorithm
        # to know. It is used by AWS API Gateway, Stripe, Cloudflare,
        # GitHub, and most production API rate limiters. The lazy refill
        # technique (calculating tokens based on elapsed time rather than
        # using a timer) is the core insight that makes it practical for
        # distributed systems.
        #
        # WHAT TO DO:
        #
        # 1. Ensure the Lua script is registered:
        #    await self._ensure_script()
        #
        # 2. Build the Redis key:
        #    redis_key = f"{self.key_prefix}:{key}"
        #
        # 3. Call the Lua script:
        #    result = await self._script(
        #        keys=[redis_key],
        #        args=[self.capacity, self.refill_rate, time.time(), cost],
        #    )
        #
        #    The script returns [allowed (0 or 1), tokens_remaining (string)].
        #    tokens_remaining is a string because Lua returns floats as
        #    strings through Redis.
        #
        # 4. Interpret:
        #    allowed = result[0] == 1
        #    tokens_remaining = float(result[1])  # for debugging/logging
        #
        # 5. Record and return:
        #    self.metrics.record(allowed)
        #    return allowed
        #
        # DEBUGGING TIP: Print tokens_remaining to watch the bucket
        # drain during bursts and refill during pauses. You should see:
        #   - Burst: tokens decrease by `cost` per request
        #   - Pause: tokens increase (capped at capacity)
        #   - Steady at refill_rate: tokens stay near a constant level
        raise NotImplementedError("TODO(human): implement token bucket allow()")

    async def close(self) -> None:
        """Close the Redis connection."""
        await self.redis.aclose()


# ---------------------------------------------------------------------------
# Demo: Token bucket burst behavior
# ---------------------------------------------------------------------------


async def demo_token_bucket() -> None:
    """Demonstrate the token bucket's burst tolerance and refill behavior.

    Three phases:
    1. Burst -- exhaust the bucket quickly (shows burst tolerance)
    2. Pause -- wait for refill (shows lazy refill in action)
    3. Steady -- send at the refill rate (shows sustainable throughput)
    """
    # TODO(human): Implement the token bucket demo.
    #
    # WHY THIS MATTERS:
    # This demo makes the token bucket's two key properties visible:
    # (a) burst tolerance (the bucket absorbs short spikes), and
    # (b) sustained rate enforcement (long-term throughput = refill_rate).
    # These are the exact properties that make it the industry standard.
    #
    # STEP-BY-STEP PLAN:
    #
    # 1. CONNECT TO REDIS:
    #    client = await create_redis_client()
    #
    # 2. CREATE THE LIMITER:
    #    limiter = TokenBucketRateLimiter(
    #        name="token_bucket_demo",
    #        redis_client=client,
    #        capacity=15,       # can absorb a burst of 15
    #        refill_rate=5.0,   # sustained rate: 5 req/sec
    #        key_prefix="ratelimit:demo:tokenbucket",
    #    )
    #
    # 3. CLEAN UP stale keys.
    #
    # 4. PHASE A -- Burst:
    #    print("\n--- Phase A: Burst (20 requests as fast as possible) ---")
    #    for i in range(20):
    #        allowed = await limiter.allow("demo-user")
    #        status = "PASS" if allowed else "REJECT"
    #        print(f"  Request {i+1:2d}: {status}")
    #
    #    Expected: first 15 pass (capacity), last 5 rejected.
    #    print(f"\n  Burst result: {limiter.metrics.allowed_requests} passed, "
    #          f"{limiter.metrics.rejected_requests} rejected")
    #    print("  (Capacity=15, so 15 passed and 5 were rejected)")
    #
    # 5. PHASE B -- Pause and refill:
    #    print("\n--- Phase B: Pause 3 seconds (bucket refills) ---")
    #    await asyncio.sleep(3.0)
    #
    #    After 3s at 5 tokens/sec, the bucket has 15 tokens (capped at capacity).
    #    result = await limiter.allow("demo-user")
    #    print(f"  After 3s pause: allowed={result}")
    #    print("  (Bucket refilled: 3s * 5 tokens/s = 15 tokens)")
    #
    # 6. PHASE C -- Steady traffic at refill rate:
    #    print("\n--- Phase C: Steady traffic (5 req/sec for 4 seconds) ---")
    #    steady_allowed = 0
    #    steady_rejected = 0
    #    for i in range(20):
    #        allowed = await limiter.allow("demo-user")
    #        if allowed:
    #            steady_allowed += 1
    #        else:
    #            steady_rejected += 1
    #        await asyncio.sleep(0.2)  # 5 req/sec = 200ms interval
    #
    #    print(f"  Steady result: {steady_allowed} passed, {steady_rejected} rejected")
    #    print("  (At refill rate, all requests should pass)")
    #
    # 7. PRINT SUMMARY:
    #    print_metrics(limiter)
    #
    # 8. CLEANUP:
    #    await limiter.close()
    raise NotImplementedError("TODO(human): implement demo_token_bucket()")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print_header("Exercise 04: Token Bucket Rate Limiter")
    print("Prerequisites:")
    print("  - Redis running (docker compose up -d redis)")
    print()
    asyncio.run(demo_token_bucket())


if __name__ == "__main__":
    main()
