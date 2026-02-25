# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "redis>=5.0",
# ]
# ///
"""Sliding Window Rate Limiter with Lua Scripts.

Demonstrates atomic rate limiting using a Redis Lua script that operates
on sorted sets. The Lua script ensures the check-count-add sequence is
indivisible, preventing concurrent requests from bypassing the limit.

Run with Redis running (docker compose up -d):
    uv run src/04_lua_rate_limiter.py
"""

from __future__ import annotations

import asyncio
import time
import uuid

import redis.asyncio as aioredis


# ── Configuration ────────────────────────────────────────────────────

REDIS_URL = "redis://localhost:6379/0"

# Lua script for atomic sliding window rate limiting.
#
# Uses a Redis sorted set (ZSET) where:
#   - Each member is a unique request ID (prevents deduplication)
#   - Each score is the request timestamp in seconds (float)
#
# The script atomically:
#   1. Removes entries outside the sliding window (ZREMRANGEBYSCORE)
#   2. Counts entries remaining in the window (ZCARD)
#   3. If count < limit: adds the new entry (ZADD) and sets TTL (EXPIRE)
#   4. Returns 1 (allowed) or 0 (denied)
#
# KEYS[1] = the rate limit key (e.g., "ratelimit:user:42")
# ARGV[1] = current timestamp (seconds, float)
# ARGV[2] = window size in seconds
# ARGV[3] = maximum requests allowed in the window
# ARGV[4] = unique request ID (for the ZADD member)
SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local request_id = ARGV[4]

-- Step 1: Remove entries outside the sliding window.
-- Entries with score < (now - window) are expired.
redis.call("ZREMRANGEBYSCORE", key, 0, now - window)

-- Step 2: Count how many entries remain in the window.
local count = redis.call("ZCARD", key)

-- Step 3: If under the limit, record this request.
if count < limit then
    redis.call("ZADD", key, now, request_id)
    -- Set TTL slightly longer than window to auto-cleanup
    redis.call("EXPIRE", key, math.ceil(window) + 1)
    return 1  -- allowed
else
    return 0  -- denied
end
"""


# ── TODO(human): Implement this class ────────────────────────────────


class SlidingWindowLimiter:
    """Atomic sliding window rate limiter backed by Redis Lua scripts.

    Uses SCRIPT LOAD + EVALSHA for efficiency: the Lua script is loaded
    once and executed by SHA hash on each request, avoiding retransmission
    of the script text.
    """

    def __init__(
        self,
        r: aioredis.Redis,
        window_seconds: float,
        max_requests: int,
        key_prefix: str = "ratelimit:",
    ) -> None:
        """Initialize the rate limiter.

        # ── Exercise Context ──────────────────────────────────────────────
        # This teaches the SCRIPT LOAD + EVALSHA pattern, which is the
        # production-standard way to use Lua scripts in Redis. Instead of
        # sending the full script text on every EVAL call (wasteful for
        # large or frequently-called scripts), you:
        #   1. Load the script once with SCRIPT LOAD, which returns a SHA1
        #   2. Call EVALSHA with the SHA1 on every request
        #
        # EVALSHA is faster because it avoids parsing and compiling the
        # script on every call -- Redis caches compiled scripts by SHA1.
        # If the script is evicted from the cache (SCRIPT FLUSH, restart),
        # EVALSHA returns a NOSCRIPT error, and you fall back to EVAL.
        # ─────────────────────────────────────────────────────────────────

        TODO(human): Store parameters and load the Lua script.

        Steps:
          1. Store the instance attributes:
                 self.r = r
                 self.window_seconds = window_seconds
                 self.max_requests = max_requests
                 self.key_prefix = key_prefix

          2. Initialize self.script_sha to None:
                 self.script_sha: str | None = None

             The SHA will be set when _ensure_script() is called.

        Args:
            r: Async Redis client.
            window_seconds: Size of the sliding window in seconds.
            max_requests: Maximum allowed requests per window.
            key_prefix: Prefix for rate limit keys in Redis.
        """
        raise NotImplementedError("TODO(human): Implement __init__")

    async def _ensure_script(self) -> str:
        """Load the Lua script if not already cached; return the SHA.

        TODO(human): Implement script loading with SCRIPT LOAD.

        Steps:
          1. If self.script_sha is not None, return it (already loaded).

          2. Load the script:
                 self.script_sha = await self.r.script_load(SLIDING_WINDOW_LUA)

             SCRIPT LOAD sends the script to Redis, which compiles it and
             returns the SHA1 hash. The script stays in Redis's script
             cache until SCRIPT FLUSH or server restart.

          3. Return self.script_sha.

        Returns:
            The SHA1 hash of the loaded Lua script.

        Docs: https://redis-py.readthedocs.io/en/stable/commands.html#redis.commands.core.CoreCommands.script_load
        """
        raise NotImplementedError("TODO(human): Implement _ensure_script")

    async def is_allowed(self, identifier: str) -> bool:
        """Check if a request from the given identifier is allowed.

        # ── Exercise Context ──────────────────────────────────────────────
        # This is where the Lua script is actually executed via EVALSHA.
        # The script runs atomically on the Redis server, which means:
        #   - No other Redis command can interleave between the ZREMRANGE,
        #     ZCARD, and ZADD steps
        #   - Two concurrent requests cannot both pass the count check
        #     and both get added (which would exceed the limit)
        #
        # This atomicity is WHY Lua scripts exist for rate limiting.
        # Without Lua, you would need MULTI/EXEC transactions or external
        # locking, both of which are more complex and less performant.
        #
        # The unique request_id (UUID) ensures each ZADD creates a
        # distinct member in the sorted set. If we used the timestamp
        # as the member, two requests at the exact same microsecond
        # would be deduplicated (ZADD is a set operation).
        # ─────────────────────────────────────────────────────────────────

        TODO(human): Execute the rate limiting Lua script via EVALSHA.

        Steps:
          1. Ensure the script is loaded:
                 sha = await self._ensure_script()

          2. Build the rate limit key:
                 key = f"{self.key_prefix}{identifier}"

          3. Get the current time:
                 now = time.time()

          4. Generate a unique request ID:
                 request_id = f"{now}:{uuid.uuid4().hex[:8]}"

          5. Execute the script via EVALSHA:
                 result = await self.r.evalsha(
                     sha,
                     1,           # number of KEYS arguments
                     key,         # KEYS[1]
                     str(now),    # ARGV[1]: current timestamp
                     str(self.window_seconds),  # ARGV[2]: window size
                     str(self.max_requests),     # ARGV[3]: limit
                     request_id,  # ARGV[4]: unique request ID
                 )

             The Lua script returns 1 (allowed) or 0 (denied).

          6. Return True if result == 1, False otherwise.

        Args:
            identifier: The entity being rate-limited (e.g., user ID, IP).

        Returns:
            True if the request is allowed, False if rate-limited.

        Note on EVALSHA vs EVAL:
          EVALSHA only works if the script is in Redis's cache. If Redis
          restarts, the cache is lost. In production, handle NOSCRIPT
          errors by falling back to EVAL (redis-py can do this automatically
          with registered scripts, but here we use raw EVALSHA for learning).
        """
        raise NotImplementedError("TODO(human): Implement is_allowed")


# ── Demonstration (boilerplate) ──────────────────────────────────────


async def demo_rate_limiting() -> None:
    """Demonstrate the sliding window rate limiter."""
    r = aioredis.from_url(REDIS_URL, decode_responses=True)

    try:
        # Clean up
        for key in await r.keys("ratelimit:*"):
            await r.delete(key)

        print("=== Sliding Window Rate Limiter (Lua Script) ===\n")

        # --- Test 1: Basic rate limiting ---
        print("--- Test 1: Basic Rate Limiting (5 req/10s window) ---")
        limiter = SlidingWindowLimiter(r, window_seconds=10, max_requests=5)

        for i in range(8):
            allowed = await limiter.is_allowed("user:alice")
            status = "ALLOWED" if allowed else "DENIED"
            print(f"  Request {i + 1}: {status}")

        # --- Test 2: Per-user isolation ---
        print("\n--- Test 2: Per-User Isolation ---")
        limiter2 = SlidingWindowLimiter(r, window_seconds=10, max_requests=3)

        for i in range(4):
            alice = await limiter2.is_allowed("user:alice2")
            bob = await limiter2.is_allowed("user:bob")
            print(f"  Round {i + 1}: Alice={'OK' if alice else 'DENIED'}, Bob={'OK' if bob else 'DENIED'}")

        # --- Test 3: Window sliding ---
        print("\n--- Test 3: Window Sliding (2 req/2s window) ---")
        limiter3 = SlidingWindowLimiter(r, window_seconds=2, max_requests=2)

        r1 = await limiter3.is_allowed("user:charlie")
        r2 = await limiter3.is_allowed("user:charlie")
        r3 = await limiter3.is_allowed("user:charlie")
        print(f"  3 rapid requests: {r1}, {r2}, {r3} (expected: True, True, False)")

        print(f"  Waiting 2.5s for window to slide...")
        await asyncio.sleep(2.5)

        r4 = await limiter3.is_allowed("user:charlie")
        print(f"  After window slides: {r4} (expected: True)")

        # --- Test 4: Burst simulation ---
        print("\n--- Test 4: Burst Simulation (10 req/5s window) ---")
        limiter4 = SlidingWindowLimiter(r, window_seconds=5, max_requests=10)
        allowed_count = 0
        denied_count = 0

        for _ in range(25):
            if await limiter4.is_allowed("user:burst"):
                allowed_count += 1
            else:
                denied_count += 1

        print(f"  25 rapid requests: {allowed_count} allowed, {denied_count} denied")
        print(f"  Expected: 10 allowed, 15 denied")

        # --- Test 5: Verify sorted set contents ---
        print("\n--- Test 5: Inspecting Redis Sorted Set ---")
        members = await r.zrange("ratelimit:user:burst", 0, -1, withscores=True)
        print(f"  Entries in sorted set: {len(members)}")
        if members:
            oldest_ts = members[0][1]
            newest_ts = members[-1][1]
            print(f"  Time span: {newest_ts - oldest_ts:.3f}s")

    finally:
        await r.aclose()


async def main() -> None:
    print("=" * 60)
    print(" Practice 076b: Lua-Based Sliding Window Rate Limiter")
    print("=" * 60)
    print()
    await demo_rate_limiting()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
