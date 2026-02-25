"""Exercise 02: Sliding Window Log Rate Limiter

Maintains a sorted set of request timestamps per client. To check if a
request is allowed: prune entries outside the window, count remaining,
and conditionally add the new entry. Provides PERFECT accuracy (no
boundary spike problem) at the cost of O(N) memory per client.

Implementation uses a Lua script because the prune-count-conditionally-add
sequence requires branching on an intermediate result (the count after
pruning), which Redis MULTI/EXEC cannot do.

Run (requires Redis from docker-compose):
    uv run python src/02_sliding_window_log.py
"""

from __future__ import annotations

import asyncio
import os
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
# Sliding Window Log Rate Limiter
# ---------------------------------------------------------------------------


class SlidingWindowLogRateLimiter(RateLimiter):
    """Sliding window log backed by Redis sorted sets + Lua.

    Each request is stored as a member in a sorted set with the timestamp
    as the score. The Lua script atomically prunes old entries, counts
    current entries, and conditionally adds the new request.

    Parameters
    ----------
    name:
        Human-readable name for this limiter.
    redis_client:
        An async Redis client.
    max_requests:
        Maximum requests allowed within the window.
    window_seconds:
        Size of the sliding window in seconds.
    key_prefix:
        Redis key prefix for namespacing.
    """

    def __init__(
        self,
        name: str,
        redis_client: aioredis.Redis,
        max_requests: int = 10,
        window_seconds: float = 60.0,
        key_prefix: str = "ratelimit:slidinglog",
    ) -> None:
        super().__init__(name)
        self.redis = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix
        self._script: Any | None = None
        self._counter = 0  # for unique member IDs

    async def _ensure_script(self) -> None:
        """Register the Lua script with Redis."""
        if self._script is None:
            lua_source = load_lua_script("sliding_window_log")
            self._script = self.redis.register_script(lua_source)

    def _make_member(self, now: float) -> str:
        """Generate a unique member ID for the sorted set.

        Members must be unique (sorted sets deduplicate by member, not
        by score). We use timestamp + PID + counter to guarantee uniqueness
        even under high concurrency from multiple replicas.
        """
        self._counter += 1
        return f"{now}:{os.getpid()}:{self._counter}"

    async def allow(self, key: str, cost: int = 1) -> bool:
        """Check if the request is allowed under the sliding window log algorithm.

        Returns True if allowed, False if rejected.
        """
        # TODO(human): Implement the sliding window log allow() method.
        #
        # WHY THIS MATTERS:
        # This exercise teaches you how to use Redis sorted sets for
        # perfectly accurate rate limiting. The sorted set is one of
        # Redis's most powerful data structures -- it supports O(log N)
        # insertion, O(log N + M) range deletion, and O(1) cardinality.
        # These operations map perfectly to the sliding window algorithm.
        #
        # WHAT TO DO:
        #
        # 1. Ensure the Lua script is registered:
        #    await self._ensure_script()
        #
        # 2. Build the Redis key and generate a unique member:
        #    redis_key = f"{self.key_prefix}:{key}"
        #    now = time.time()
        #    member = self._make_member(now)
        #
        # 3. Call the Lua script:
        #    result = await self._script(
        #        keys=[redis_key],
        #        args=[self.max_requests, self.window_seconds, now, member],
        #    )
        #
        #    The script returns [allowed (0 or 1), current_count].
        #
        # 4. Interpret and record:
        #    allowed = result[0] == 1
        #    self.metrics.record(allowed)
        #
        # 5. Return the result.
        #
        # NOTE ON COST:
        # For simplicity, cost > 1 is not supported in the sorted set
        # approach (each entry represents exactly one request). If you
        # needed weighted requests, you'd either add multiple entries
        # or switch to the token bucket algorithm.
        raise NotImplementedError("TODO(human): implement sliding window log allow()")

    async def close(self) -> None:
        """Close the Redis connection."""
        await self.redis.aclose()


# ---------------------------------------------------------------------------
# Demo: Sliding window accuracy
# ---------------------------------------------------------------------------


async def demo_sliding_window_log() -> None:
    """Demonstrate the sliding window log -- no boundary spike.

    Sends the same traffic pattern as the fixed window demo to show
    that the sliding window correctly limits across boundaries.
    """
    # TODO(human): Implement the sliding window log demo.
    #
    # WHY THIS MATTERS:
    # This demo directly contrasts with the fixed window demo (Exercise 01).
    # By sending the same boundary-spike traffic pattern, you'll see that
    # the sliding window log correctly rejects requests that the fixed
    # window allowed. This makes the accuracy difference concrete.
    #
    # STEP-BY-STEP PLAN:
    #
    # 1. CONNECT TO REDIS:
    #    client = await create_redis_client()
    #    print("Connected to Redis")
    #
    # 2. CREATE THE LIMITER:
    #    limiter = SlidingWindowLogRateLimiter(
    #        name="sliding_log_demo",
    #        redis_client=client,
    #        max_requests=10,
    #        window_seconds=5.0,
    #        key_prefix="ratelimit:demo:slidinglog",
    #    )
    #
    # 3. CLEAN UP:
    #    keys = await client.keys("ratelimit:demo:slidinglog:*")
    #    if keys:
    #        await client.delete(*keys)
    #
    # 4. PHASE A -- Steady traffic (same as fixed window):
    #    print("\n--- Phase A: Steady traffic (12 requests) ---")
    #    for i in range(12):
    #        allowed = await limiter.allow("demo-user")
    #        status = "PASS" if allowed else "REJECT"
    #        print(f"  Request {i+1:2d}: {status}")
    #        await asyncio.sleep(0.1)
    #
    # 5. PHASE B -- Boundary test:
    #    Wait for the window to expire so we start fresh:
    #    print("\n  Waiting for window to expire...")
    #    await asyncio.sleep(5.0)
    #
    #    Rapid burst of 15 requests:
    #    print("\n--- Phase B: Rapid burst (15 requests in ~0.3s) ---")
    #    limiter.metrics = RateLimiterMetrics()
    #    for i in range(15):
    #        allowed = await limiter.allow("burst-user")
    #        status = "PASS" if allowed else "REJECT"
    #        print(f"  Request {i+1:2d}: {status}")
    #        await asyncio.sleep(0.02)
    #
    #    Expected: exactly 10 pass, 5 rejected. No boundary spike.
    #    print(f"\n  Result: {limiter.metrics.allowed_requests} allowed, "
    #          f"{limiter.metrics.rejected_requests} rejected")
    #    print("  No boundary spike -- sliding window is perfectly accurate!")
    #
    # 6. MEMORY ANALYSIS:
    #    Check how many members are in the sorted set:
    #    key = "ratelimit:demo:slidinglog:burst-user"
    #    count = await client.zcard(key)
    #    print(f"\n  Sorted set members for burst-user: {count}")
    #    print("  (Each request = 1 member. At scale, this adds up!)")
    #
    # 7. CLEANUP:
    #    print_metrics(limiter)
    #    await limiter.close()
    raise NotImplementedError("TODO(human): implement demo_sliding_window_log()")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print_header("Exercise 02: Sliding Window Log Rate Limiter")
    print("Prerequisites:")
    print("  - Redis running (docker compose up -d redis)")
    print()
    asyncio.run(demo_sliding_window_log())


if __name__ == "__main__":
    main()
