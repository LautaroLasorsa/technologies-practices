"""Exercise 03: Sliding Window Counter Rate Limiter

A hybrid algorithm that approximates the sliding window using two fixed-window
counters (current and previous). Memory-efficient (O(1) per client) with
accuracy close to the full sliding window log.

The weighted formula:
    estimated = prev_count * (1 - elapsed_fraction) + current_count

Where elapsed_fraction is how far we are into the current window (0.0 to 1.0).

Run (requires Redis from docker-compose):
    uv run python src/03_sliding_window_counter.py
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
# Sliding Window Counter Rate Limiter
# ---------------------------------------------------------------------------


class SlidingWindowCounterRateLimiter(RateLimiter):
    """Sliding window counter backed by Redis + Lua.

    Uses two fixed-window counters (current and previous) with a weighted
    formula to approximate the sliding window count. Best balance of
    accuracy and memory efficiency.

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
        key_prefix: str = "ratelimit:swcounter",
    ) -> None:
        super().__init__(name)
        self.redis = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix
        self._script: Any | None = None

    async def _ensure_script(self) -> None:
        """Register the Lua script with Redis."""
        if self._script is None:
            lua_source = load_lua_script("sliding_window_counter")
            self._script = self.redis.register_script(lua_source)

    def _window_keys(self, key: str) -> tuple[str, str, float]:
        """Calculate the current and previous window keys, plus elapsed fraction.

        Returns
        -------
        current_key:
            Redis key for the current window.
        prev_key:
            Redis key for the previous window.
        elapsed_fraction:
            How far into the current window we are (0.0 to 1.0).
        """
        now = time.time()
        current_window = int(now) // self.window_seconds
        elapsed_in_window = now - (current_window * self.window_seconds)
        elapsed_fraction = elapsed_in_window / self.window_seconds

        current_key = f"{self.key_prefix}:{key}:{current_window}"
        prev_key = f"{self.key_prefix}:{key}:{current_window - 1}"

        return current_key, prev_key, elapsed_fraction

    async def allow(self, key: str, cost: int = 1) -> bool:
        """Check if the request is allowed under the sliding window counter.

        Returns True if allowed, False if rejected.
        """
        # TODO(human): Implement the sliding window counter allow() method.
        #
        # WHY THIS MATTERS:
        # The sliding window counter is what most production rate limiters
        # actually use. It combines the memory efficiency of fixed window
        # (two counters) with the accuracy of sliding window (weighted
        # approximation). Understanding this hybrid approach is essential
        # for system design discussions.
        #
        # WHAT TO DO:
        #
        # 1. Ensure the Lua script is registered:
        #    await self._ensure_script()
        #
        # 2. Get window keys and elapsed fraction:
        #    current_key, prev_key, elapsed_fraction = self._window_keys(key)
        #
        # 3. Call the Lua script:
        #    result = await self._script(
        #        keys=[current_key, prev_key],
        #        args=[self.max_requests, self.window_seconds, elapsed_fraction],
        #    )
        #
        #    The script returns [allowed (0 or 1), estimated_count (string)].
        #    Note: estimated_count is a string because Lua returns floats
        #    as strings through Redis.
        #
        # 4. Interpret:
        #    allowed = result[0] == 1
        #    estimated = float(result[1])  # parse the string
        #
        # 5. Record and return:
        #    self.metrics.record(allowed)
        #    return allowed
        #
        # DEBUGGING TIP: Print the estimated count to see the weighted
        # formula in action. You should see the previous window's
        # contribution decrease as elapsed_fraction increases.
        raise NotImplementedError("TODO(human): implement sliding window counter allow()")

    async def close(self) -> None:
        """Close the Redis connection."""
        await self.redis.aclose()


# ---------------------------------------------------------------------------
# Demo: Sliding window counter approximation
# ---------------------------------------------------------------------------


async def demo_sliding_window_counter() -> None:
    """Demonstrate the sliding window counter's weighted approximation.

    Shows how the previous window's count gradually loses influence as
    the current window progresses, and compares accuracy against the
    fixed window approach.
    """
    # TODO(human): Implement the sliding window counter demo.
    #
    # WHY THIS MATTERS:
    # This demo shows the approximation in action. By printing the
    # estimated count at different points in the window, you'll see
    # the weight shifting from previous to current. This builds intuition
    # for when the approximation is good (uniform traffic) vs bad
    # (spiky traffic concentrated at one end of a window).
    #
    # STEP-BY-STEP PLAN:
    #
    # 1. CONNECT TO REDIS:
    #    client = await create_redis_client()
    #
    # 2. CREATE THE LIMITER with a short window:
    #    limiter = SlidingWindowCounterRateLimiter(
    #        name="sw_counter_demo",
    #        redis_client=client,
    #        max_requests=10,
    #        window_seconds=5,
    #        key_prefix="ratelimit:demo:swcounter",
    #    )
    #
    # 3. CLEAN UP stale keys.
    #
    # 4. PHASE A -- Fill a window, then observe decay:
    #    print("\n--- Phase A: Fill window, then watch weight decay ---")
    #
    #    Send 8 requests to fill the current window partially:
    #    for i in range(8):
    #        await limiter.allow("demo-user")
    #        await asyncio.sleep(0.05)
    #
    #    Now wait for the window to roll over:
    #    print("  Sent 8 requests. Waiting for window rollover...")
    #    await asyncio.sleep(5.0)
    #
    #    In the new window, send requests and print the estimated count:
    #    print("  New window -- watching previous window's weight decay:")
    #    for i in range(8):
    #        current_key, prev_key, elapsed_frac = limiter._window_keys("demo-user")
    #        prev_count_raw = await client.get(prev_key)
    #        prev_count = int(prev_count_raw) if prev_count_raw else 0
    #        weight = 1 - elapsed_frac
    #        weighted = prev_count * weight
    #        allowed = await limiter.allow("demo-user")
    #        print(f"  t+{elapsed_frac*5:.1f}s: prev_weight={weight:.2f} "
    #              f"weighted_prev={weighted:.1f} allowed={allowed}")
    #        await asyncio.sleep(0.6)
    #
    # 5. PRINT SUMMARY:
    #    print_metrics(limiter)
    #
    # 6. CLEANUP:
    #    await limiter.close()
    raise NotImplementedError("TODO(human): implement demo_sliding_window_counter()")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print_header("Exercise 03: Sliding Window Counter Rate Limiter")
    print("Prerequisites:")
    print("  - Redis running (docker compose up -d redis)")
    print()
    asyncio.run(demo_sliding_window_counter())


if __name__ == "__main__":
    main()
