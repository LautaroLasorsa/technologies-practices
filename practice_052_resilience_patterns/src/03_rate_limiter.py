"""Exercise 03: Distributed Rate Limiting with Redis

Implement two rate limiting algorithms backed by Redis for shared state:

1. Token Bucket -- allows burst traffic up to bucket capacity while
   maintaining a long-term average rate.  Uses a Lua script for atomicity.
2. Sliding Window -- counts requests in a rolling time window.  Uses Redis
   sorted sets (ZSET) with MULTI/EXEC pipelines for atomicity.

Both algorithms store state in Redis so they work correctly across multiple
application replicas (distributed rate limiting).

Run (requires Redis from docker-compose):
    uv run python src/03_rate_limiter.py
"""

from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import redis.asyncio as aioredis


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RateLimitExceededError(Exception):
    """Raised when a request is rejected by the rate limiter."""

    def __init__(self, limiter_name: str, retry_after: float | None = None) -> None:
        self.limiter_name = limiter_name
        self.retry_after = retry_after
        msg = f"Rate limit exceeded for '{limiter_name}'."
        if retry_after is not None:
            msg += f" Retry after {retry_after:.2f}s."
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class RateLimiterMetrics:
    """Counters for monitoring rate limiter behavior."""

    total_requests: int = 0
    allowed_requests: int = 0
    rejected_requests: int = 0
    timestamps_allowed: list[float] = field(default_factory=list)
    timestamps_rejected: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------


class RateLimiter(ABC):
    """Base class for rate limiter implementations."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.metrics = RateLimiterMetrics()

    @abstractmethod
    async def allow(self, key: str, cost: int = 1) -> bool:
        """Check if a request is allowed.  Returns True if allowed, False if rejected.

        Parameters
        ----------
        key : str
            The rate limit key (e.g., client IP, API key, user ID).
        cost : int
            Number of tokens/units this request consumes (default 1).
        """
        ...

    async def require(self, key: str, cost: int = 1) -> None:
        """Like allow(), but raises RateLimitExceededError on rejection."""
        if not await self.allow(key, cost):
            raise RateLimitExceededError(self.name)

    @abstractmethod
    async def close(self) -> None:
        """Release resources (Redis connections)."""
        ...


# ---------------------------------------------------------------------------
# Token Bucket Rate Limiter (Redis + Lua)
# ---------------------------------------------------------------------------

# The Lua script that runs atomically inside Redis.
# This is the standard approach used in production rate limiters.
#
# Keys: KEYS[1] = the rate limit key (e.g., "ratelimit:user:123")
# Args: ARGV[1] = bucket capacity (max tokens)
#        ARGV[2] = refill rate (tokens per second)
#        ARGV[3] = now (current timestamp as float, seconds)
#        ARGV[4] = cost (tokens to consume for this request)
#
# The script stores two fields in a Redis hash:
#   - "tokens": current number of tokens (float)
#   - "last_refill": timestamp of last refill (float)
#
# Returns: 1 if allowed, 0 if rejected

TOKEN_BUCKET_LUA_SCRIPT = """
-- TODO(human): Write the Lua script for atomic token bucket rate limiting.
--
-- WHY THIS MATTERS:
-- In a distributed system with multiple application replicas, each checking
-- the same rate limit, you CANNOT do read-then-write in two separate Redis
-- commands.  Between the read (GET tokens) and the write (SET tokens - 1),
-- another replica could also read the same value and also decrement, allowing
-- double the intended rate.  Lua scripts execute atomically in Redis --
-- no other command can interleave.  This is the same reason SQL databases
-- use transactions.
--
-- THE ALGORITHM (token bucket):
--
-- 1. READ CURRENT STATE:
--    local tokens = tonumber(redis.call('HGET', KEYS[1], 'tokens'))
--    local last_refill = tonumber(redis.call('HGET', KEYS[1], 'last_refill'))
--    (If the key doesn't exist, tokens and last_refill will be nil)
--
-- 2. PARSE ARGUMENTS:
--    local capacity = tonumber(ARGV[1])
--    local refill_rate = tonumber(ARGV[2])
--    local now = tonumber(ARGV[3])
--    local cost = tonumber(ARGV[4])
--
-- 3. INITIALIZE IF NEW KEY:
--    if tokens == nil then
--        tokens = capacity  (start with a full bucket)
--        last_refill = now
--    end
--
-- 4. REFILL TOKENS BASED ON ELAPSED TIME:
--    local elapsed = now - last_refill
--    tokens = math.min(capacity, tokens + elapsed * refill_rate)
--    last_refill = now
--
--    This is the key insight: tokens are not refilled by a background timer.
--    Instead, each request calculates how many tokens SHOULD have been added
--    since the last request.  This is lazy evaluation -- no timers needed.
--
-- 5. TRY TO CONSUME:
--    if tokens >= cost then
--        tokens = tokens - cost
--        redis.call('HSET', KEYS[1], 'tokens', tostring(tokens))
--        redis.call('HSET', KEYS[1], 'last_refill', tostring(last_refill))
--        redis.call('EXPIRE', KEYS[1], math.ceil(capacity / refill_rate) * 2)
--        return 1  -- allowed
--    else
--        redis.call('HSET', KEYS[1], 'tokens', tostring(tokens))
--        redis.call('HSET', KEYS[1], 'last_refill', tostring(last_refill))
--        redis.call('EXPIRE', KEYS[1], math.ceil(capacity / refill_rate) * 2)
--        return 0  -- rejected
--    end
--
--    The EXPIRE sets a TTL so the key auto-deletes if the client disappears.
--    Without it, Redis would accumulate stale rate limit keys forever.
--
-- HINT: Use tostring() when writing floats to Redis hash fields, because
-- Redis hash values are strings.  Use tonumber() when reading them back.
--
return 0
"""


class TokenBucketRateLimiter(RateLimiter):
    """Token bucket rate limiter backed by Redis.

    Parameters
    ----------
    name:
        Human-readable name for this limiter.
    redis_client:
        An async Redis client (redis.asyncio.Redis).
    capacity:
        Maximum tokens the bucket can hold (= max burst size).
    refill_rate:
        Tokens added per second (= sustained request rate).
    key_prefix:
        Redis key prefix to namespace rate limit keys.
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
        """Register the Lua script with Redis (done once, cached)."""
        if self._script is None:
            self._script = self.redis.register_script(TOKEN_BUCKET_LUA_SCRIPT)

    async def allow(self, key: str, cost: int = 1) -> bool:
        """Check if the request is allowed under the token bucket algorithm.

        Returns True if allowed (tokens consumed), False if rejected.
        """
        # TODO(human): Implement the token bucket allow() method.
        #
        # WHY THIS MATTERS:
        # This method is the Python-side interface to the Lua script.  It
        # prepares the arguments, calls the script, and interprets the result.
        # The Lua script does the heavy lifting (atomicity, refill math), but
        # this method handles metrics tracking and the Python-Redis interface.
        #
        # WHAT TO DO:
        # 1. Call await self._ensure_script() to register the Lua script.
        #
        # 2. Build the Redis key:
        #        redis_key = f"{self.key_prefix}:{key}"
        #
        # 3. Call the Lua script:
        #        result = await self._script(
        #            keys=[redis_key],
        #            args=[self.capacity, self.refill_rate, time.time(), cost],
        #        )
        #
        # 4. Interpret the result:
        #    - result == 1: allowed.  Record in metrics.
        #    - result == 0: rejected.  Record in metrics.
        #
        # 5. Update metrics:
        #    - metrics.total_requests += 1
        #    - If allowed: metrics.allowed_requests += 1
        #      metrics.timestamps_allowed.append(time.time())
        #    - If rejected: metrics.rejected_requests += 1
        #      metrics.timestamps_rejected.append(time.time())
        #
        # 6. Return True if allowed, False if rejected.
        #
        # HINT: The Lua script returns an integer (1 or 0).  Redis returns this
        # as an int in Python, so `result == 1` works directly.
        raise NotImplementedError("TODO(human): implement token bucket allow()")

    async def close(self) -> None:
        """Close the Redis connection."""
        await self.redis.aclose()


# ---------------------------------------------------------------------------
# Sliding Window Rate Limiter (Redis Sorted Sets)
# ---------------------------------------------------------------------------


class SlidingWindowRateLimiter(RateLimiter):
    """Sliding window rate limiter backed by Redis sorted sets.

    Each request adds an entry to a sorted set with the timestamp as the
    score.  To check if a request is allowed: remove entries outside the
    window, count remaining entries, and compare against the limit.

    Parameters
    ----------
    name:
        Human-readable name for this limiter.
    redis_client:
        An async Redis client (redis.asyncio.Redis).
    max_requests:
        Maximum requests allowed within the window.
    window_seconds:
        Size of the sliding window in seconds.
    key_prefix:
        Redis key prefix to namespace rate limit keys.
    """

    def __init__(
        self,
        name: str,
        redis_client: aioredis.Redis,
        max_requests: int = 10,
        window_seconds: float = 1.0,
        key_prefix: str = "ratelimit:slidingwindow",
    ) -> None:
        super().__init__(name)
        self.redis = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix

    async def allow(self, key: str, cost: int = 1) -> bool:
        """Check if the request is allowed under the sliding window algorithm.

        Returns True if allowed, False if rejected.
        """
        # TODO(human): Implement the sliding window allow() method using Redis
        # sorted sets and a pipeline for atomicity.
        #
        # WHY THIS MATTERS:
        # The sliding window algorithm provides smoother rate limiting than a
        # fixed window (no boundary-spike problem) and is simpler to reason
        # about than the token bucket (no refill rate, just "N requests per
        # M seconds").  Using Redis sorted sets is the standard distributed
        # implementation.
        #
        # THE ALGORITHM:
        #
        # 1. BUILD THE KEY AND TIMESTAMPS:
        #    redis_key = f"{self.key_prefix}:{key}"
        #    now = time.time()
        #    window_start = now - self.window_seconds
        #
        # 2. USE A REDIS PIPELINE for atomicity (all commands execute together):
        #    async with self.redis.pipeline(transaction=True) as pipe:
        #
        #        a. REMOVE EXPIRED ENTRIES (outside the window):
        #           pipe.zremrangebyscore(redis_key, "-inf", window_start)
        #           This prunes entries older than window_start.  Without this,
        #           the sorted set would grow unboundedly.
        #
        #        b. COUNT CURRENT ENTRIES (requests in the window):
        #           pipe.zcard(redis_key)
        #           This tells us how many requests are in the current window.
        #
        #        c. ADD THE NEW REQUEST (optimistically):
        #           member = f"{now}:{id(asyncio.current_task())}"
        #           pipe.zadd(redis_key, {member: now})
        #           The member must be unique (hence the task id suffix).
        #           The score is the timestamp, used for window pruning.
        #
        #        d. SET TTL (auto-cleanup):
        #           pipe.expire(redis_key, int(self.window_seconds) + 1)
        #
        #        e. EXECUTE THE PIPELINE:
        #           results = await pipe.execute()
        #
        # 3. INTERPRET RESULTS:
        #    - results[0] = number of entries removed (from zremrangebyscore)
        #    - results[1] = count of entries BEFORE adding the new one (from zcard)
        #    - results[2] = whether the new entry was added (from zadd)
        #    - results[3] = TTL was set (from expire)
        #
        #    current_count = results[1]  (count BEFORE adding this request)
        #
        # 4. DECIDE:
        #    if current_count < self.max_requests:
        #        -> allowed (the entry we added stays)
        #    else:
        #        -> rejected.  But we already added the entry!  We must remove it:
        #           await self.redis.zrem(redis_key, member)
        #
        # 5. UPDATE METRICS (same as token bucket: total, allowed/rejected, timestamps)
        #
        # 6. RETURN True if allowed, False if rejected.
        #
        # SUBTLETY: We add the entry optimistically BEFORE checking the count.
        # This is because the pipeline executes atomically -- no other client
        # can interleave between the ZCARD and the ZADD.  But if we're over
        # the limit, we must clean up after ourselves.  An alternative is to
        # use a Lua script (like the token bucket) to avoid the extra ZREM.
        #
        # HINT for unique member IDs:
        #   import os; member = f"{now}:{os.getpid()}:{id(asyncio.current_task())}"
        raise NotImplementedError("TODO(human): implement sliding window allow()")

    async def close(self) -> None:
        """Close the Redis connection."""
        await self.redis.aclose()


# ---------------------------------------------------------------------------
# Demo: compare token bucket vs sliding window
# ---------------------------------------------------------------------------


async def demo_rate_limiting() -> None:
    """Send bursts of requests through both rate limiters and compare behavior.

    The token bucket allows bursts (up to capacity) while maintaining a
    long-term average rate.  The sliding window enforces a stricter per-window
    count with no burst tolerance.

    Saves comparison plots to plots/.
    """
    # TODO(human): Implement the rate limiting comparison demo.
    #
    # WHY THIS MATTERS:
    # Understanding the behavioral difference between token bucket and sliding
    # window is essential for choosing the right algorithm.  The token bucket
    # is better for APIs that need to handle legitimate traffic spikes (e.g.,
    # webhook bursts, batch uploads).  The sliding window is better when you
    # want a strict, smooth rate limit (e.g., preventing abuse, protecting a
    # fragile downstream).  This demo makes that difference visible.
    #
    # STEP-BY-STEP PLAN:
    #
    # 1. CONNECT TO REDIS:
    #    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    #    redis_client = aioredis.from_url(redis_url)
    #    await redis_client.ping()  (verify connection)
    #    print("Connected to Redis")
    #
    # 2. CREATE BOTH LIMITERS with the same effective rate:
    #    token_bucket = TokenBucketRateLimiter(
    #        name="token_bucket",
    #        redis_client=redis_client,
    #        capacity=15,       # allows burst of 15
    #        refill_rate=5.0,   # sustained rate: 5 req/sec
    #    )
    #    sliding_window = SlidingWindowRateLimiter(
    #        name="sliding_window",
    #        redis_client=redis_client,
    #        max_requests=5,    # 5 requests per 1-second window
    #        window_seconds=1.0,
    #    )
    #
    # 3. CLEAR ANY STALE KEYS (from previous runs):
    #    Use redis_client.delete() to remove test keys, or use unique key
    #    prefixes per run (e.g., include a timestamp).
    #
    # 4. TEST PATTERN -- three phases:
    #
    #    Phase A -- Burst: send 20 requests as fast as possible
    #    Phase B -- Pause: wait 3 seconds (let token bucket refill)
    #    Phase C -- Steady: send 1 request every 150ms for 5 seconds
    #
    #    For each phase, iterate and call:
    #        tb_allowed = await token_bucket.allow("test_client")
    #        sw_allowed = await sliding_window.allow("test_client")
    #        print(f"  #{i}: TB={'OK' if tb_allowed else 'REJECT'}  "
    #              f"SW={'OK' if sw_allowed else 'REJECT'}")
    #
    # 5. PRINT SUMMARY:
    #    - Token bucket: allowed / rejected / total
    #    - Sliding window: allowed / rejected / total
    #    - Highlight the difference: TB allowed more during burst phase
    #
    # 6. GENERATE COMPARISON PLOT:
    #    - Create a plots/ directory: os.makedirs("plots", exist_ok=True)
    #    - Use matplotlib to create a timeline plot:
    #      x-axis = time (seconds since start)
    #      y-axis = cumulative allowed requests
    #      Two lines: token bucket (allows burst) vs sliding window (smooth)
    #    - Save to plots/rate_limiter_comparison.png
    #    - Print "Plot saved to plots/rate_limiter_comparison.png"
    #
    # 7. CLEANUP:
    #    await token_bucket.close()
    #    await sliding_window.close()
    #    (Don't close redis_client twice -- the limiters close it)
    #
    # HINT for the timeline plot:
    #   import matplotlib.pyplot as plt
    #   fig, ax = plt.subplots(figsize=(12, 6))
    #   tb_times = [t - start for t in token_bucket.metrics.timestamps_allowed]
    #   sw_times = [t - start for t in sliding_window.metrics.timestamps_allowed]
    #   ax.step(tb_times, range(1, len(tb_times)+1), label="Token Bucket", where="post")
    #   ax.step(sw_times, range(1, len(sw_times)+1), label="Sliding Window", where="post")
    #   ax.set_xlabel("Time (s)")
    #   ax.set_ylabel("Cumulative Allowed Requests")
    #   ax.legend()
    #   fig.savefig("plots/rate_limiter_comparison.png", dpi=150)
    raise NotImplementedError("TODO(human): implement demo_rate_limiting()")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Exercise 03: Distributed Rate Limiting")
    print("=" * 60)
    print()
    print("Prerequisites:")
    print("  - Redis running (docker compose up -d)")
    print()
    asyncio.run(demo_rate_limiting())


if __name__ == "__main__":
    main()
