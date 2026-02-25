"""Exercise 01: Fixed Window Counter Rate Limiter

The simplest rate limiting algorithm. Divide time into fixed-size windows
(e.g., 1-minute buckets), maintain a counter per client per window, and
reject requests when the counter exceeds the limit.

Implementation uses a Lua script for atomic INCR + conditional EXPIRE.
The Redis key includes the window timestamp, so each window gets its own
key that auto-expires via TTL.

Run (requires Redis from docker-compose):
    uv run python src/01_fixed_window.py
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
# Fixed Window Rate Limiter
# ---------------------------------------------------------------------------


class FixedWindowRateLimiter(RateLimiter):
    """Fixed window counter backed by Redis + Lua.

    Parameters
    ----------
    name:
        Human-readable name for this limiter.
    redis_client:
        An async Redis client.
    max_requests:
        Maximum requests allowed per window.
    window_seconds:
        Window duration in seconds.
    key_prefix:
        Redis key prefix for namespacing.
    """

    def __init__(
        self,
        name: str,
        redis_client: aioredis.Redis,
        max_requests: int = 10,
        window_seconds: int = 60,
        key_prefix: str = "ratelimit:fixedwindow",
    ) -> None:
        super().__init__(name)
        self.redis = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix
        self._script: Any | None = None

    async def _ensure_script(self) -> None:
        """Register the Lua script with Redis (done once, cached via SHA)."""
        if self._script is None:
            lua_source = load_lua_script("fixed_window")
            self._script = self.redis.register_script(lua_source)

    def _window_key(self, key: str) -> str:
        """Build the Redis key for the current fixed window.

        The key includes the window start timestamp, so each window gets
        its own counter that auto-expires.  For a 60-second window at
        time 1706745637, the window start is 1706745600.
        """
        window_start = int(time.time()) // self.window_seconds * self.window_seconds
        return f"{self.key_prefix}:{key}:{window_start}"

    async def allow(self, key: str, cost: int = 1) -> bool:
        """Check if the request is allowed under the fixed window algorithm.

        Returns True if allowed (counter incremented), False if rejected.
        """
        # TODO(human): Implement the fixed window allow() method.
        #
        # WHY THIS MATTERS:
        # This is your first Redis Lua script integration. The pattern here
        # (register script once, call with keys + args, interpret result)
        # is the same pattern used by ALL the rate limiters in this practice.
        # Getting it right here makes the remaining exercises easier.
        #
        # WHAT TO DO:
        #
        # 1. Ensure the Lua script is registered:
        #    await self._ensure_script()
        #
        # 2. Build the Redis key for the current window:
        #    redis_key = self._window_key(key)
        #
        # 3. Call the Lua script:
        #    result = await self._script(
        #        keys=[redis_key],
        #        args=[self.max_requests, self.window_seconds],
        #    )
        #
        #    The script returns [allowed (0 or 1), current_count].
        #    `result` is a list: result[0] is the allowed flag, result[1]
        #    is the current counter value.
        #
        # 4. Interpret the result:
        #    allowed = result[0] == 1
        #
        # 5. Update metrics:
        #    self.metrics.record(allowed)
        #
        # 6. Return True if allowed, False if rejected.
        #
        # HINT: The Lua script returns integers. In Python, Redis returns
        # these as `int` type, so `result[0] == 1` works directly.
        raise NotImplementedError("TODO(human): implement fixed window allow()")

    async def close(self) -> None:
        """Close the Redis connection."""
        await self.redis.aclose()


# ---------------------------------------------------------------------------
# Demo: Fixed window with boundary spike demonstration
# ---------------------------------------------------------------------------


async def demo_fixed_window() -> None:
    """Demonstrate the fixed window rate limiter and its boundary spike flaw.

    Sends requests in two phases:
    1. Steady traffic within a single window (should mostly pass)
    2. Burst at the boundary of two windows (shows the 2x spike problem)
    """
    # TODO(human): Implement the fixed window demo.
    #
    # WHY THIS MATTERS:
    # The boundary spike problem is THE critical weakness of fixed window
    # counters. Every system design interview about rate limiting asks about
    # it. Seeing it happen concretely (not just in theory) builds real
    # understanding. You'll also verify that the Lua script works correctly.
    #
    # STEP-BY-STEP PLAN:
    #
    # 1. CONNECT TO REDIS:
    #    client = await create_redis_client()
    #    print("Connected to Redis")
    #
    # 2. CREATE THE LIMITER with a short window for testing:
    #    limiter = FixedWindowRateLimiter(
    #        name="fixed_window_demo",
    #        redis_client=client,
    #        max_requests=10,
    #        window_seconds=5,    # 5-second windows for quick demo
    #        key_prefix="ratelimit:demo:fixedwindow",
    #    )
    #
    # 3. CLEAN UP STALE KEYS from previous runs:
    #    Use a pattern delete or unique run ID in the prefix.
    #    Simple approach: await client.flushdb()  (only for demo!)
    #    Or: iterate and delete keys matching the prefix.
    #
    # 4. PHASE A -- Steady traffic within one window:
    #    print("\n--- Phase A: Steady traffic (10 requests in 5s window) ---")
    #    for i in range(12):
    #        allowed = await limiter.allow("demo-user")
    #        status = "PASS" if allowed else "REJECT"
    #        print(f"  Request {i+1:2d}: {status}")
    #        await asyncio.sleep(0.1)
    #
    #    Expected: first 10 pass, last 2 rejected.
    #
    # 5. PHASE B -- Boundary spike demonstration:
    #    print("\n--- Phase B: Boundary spike (burst at window edge) ---")
    #    Reset metrics: limiter.metrics = RateLimiterMetrics()
    #
    #    Wait until we're near the end of a window:
    #    now = time.time()
    #    window_end = (int(now) // 5 + 1) * 5  # next 5-second boundary
    #    wait_time = window_end - now - 0.5     # arrive 0.5s before boundary
    #    if wait_time > 0:
    #        print(f"  Waiting {wait_time:.1f}s until window boundary...")
    #        await asyncio.sleep(wait_time)
    #
    #    Send 10 requests just before the boundary:
    #    print("  Sending 10 requests just before window boundary...")
    #    for i in range(10):
    #        allowed = await limiter.allow("spike-user")
    #        await asyncio.sleep(0.02)  # 20ms between requests
    #
    #    Wait for boundary to pass:
    #    await asyncio.sleep(1.0)
    #
    #    Send 10 more requests just after the boundary:
    #    print("  Sending 10 requests just after window boundary...")
    #    for i in range(10):
    #        allowed = await limiter.allow("spike-user")
    #        await asyncio.sleep(0.02)
    #
    #    All 20 should pass! The user got 20 requests in ~1.5 seconds
    #    even though the limit is 10 per 5 seconds.
    #    print(f"\n  Boundary spike result: {limiter.metrics.allowed_requests} "
    #          f"requests allowed in ~1.5 seconds (limit is 10 per 5s)")
    #    print("  This is the boundary spike problem!")
    #
    # 6. PRINT SUMMARY:
    #    print_metrics(limiter)
    #
    # 7. CLEANUP:
    #    await limiter.close()
    raise NotImplementedError("TODO(human): implement demo_fixed_window()")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print_header("Exercise 01: Fixed Window Counter Rate Limiter")
    print("Prerequisites:")
    print("  - Redis running (docker compose up -d redis)")
    print()
    asyncio.run(demo_fixed_window())


if __name__ == "__main__":
    main()
