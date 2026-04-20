"""Provided: Token Bucket Rate Limiter (used by exercise _04).

No exercises here -- rate limiting algorithms (token bucket, sliding
window, fixed window) are covered in depth in practice 078, including the
atomic Lua script implementation. This module just imports cleanly so the
combined-resilience exercise has a working rate limiter to compose with.

If you want to revisit the Lua script or compare algorithms, see:
  practice_078_distributed_rate_limiting/lua/token_bucket.lua
  practice_078_distributed_rate_limiting/src/_04_token_bucket.py

Run standalone (sanity check, requires Redis):
    uv run python -m src._03_rate_limiter
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

import redis.asyncio as aioredis


class RateLimitExceededError(Exception):
    def __init__(self, limiter_name: str) -> None:
        self.limiter_name = limiter_name
        super().__init__(f"Rate limit exceeded for '{limiter_name}'")


@dataclass
class RateLimiterMetrics:
    total_requests: int = 0
    allowed_requests: int = 0
    rejected_requests: int = 0
    timestamps_allowed: list[float] = field(default_factory=list)
    timestamps_rejected: list[float] = field(default_factory=list)


# Atomic token bucket: refill based on elapsed time, then try to consume.
# Same algorithm as practice 078's lua/token_bucket.lua.
_TOKEN_BUCKET_LUA = """
local capacity    = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now         = tonumber(ARGV[3])
local cost        = tonumber(ARGV[4])

local tokens      = tonumber(redis.call('HGET', KEYS[1], 'tokens'))
local last_refill = tonumber(redis.call('HGET', KEYS[1], 'last_refill'))

if tokens == nil then
    tokens = capacity
    last_refill = now
end

local elapsed = math.max(0, now - last_refill)
tokens = math.min(capacity, tokens + elapsed * refill_rate)
last_refill = now

local allowed = 0
if tokens >= cost then
    tokens = tokens - cost
    allowed = 1
end

redis.call('HSET', KEYS[1], 'tokens', tostring(tokens))
redis.call('HSET', KEYS[1], 'last_refill', tostring(last_refill))
redis.call('EXPIRE', KEYS[1], math.ceil(capacity / refill_rate) * 2)

return allowed
"""


class TokenBucketRateLimiter:
    """Redis-backed token bucket limiter (provided complete)."""

    def __init__(
        self,
        name: str,
        redis_client: aioredis.Redis,
        capacity: int = 20,
        refill_rate: float = 10.0,
        key_prefix: str = "ratelimit:tokenbucket",
    ) -> None:
        self.name = name
        self.redis = redis_client
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.key_prefix = key_prefix
        self.metrics = RateLimiterMetrics()
        self._script: Any = redis_client.register_script(_TOKEN_BUCKET_LUA)

    async def allow(self, key: str, cost: int = 1) -> bool:
        redis_key = f"{self.key_prefix}:{key}"
        result = await self._script(
            keys=[redis_key],
            args=[self.capacity, self.refill_rate, time.time(), cost],
        )
        allowed = result == 1
        self.metrics.total_requests += 1
        now = time.time()
        if allowed:
            self.metrics.allowed_requests += 1
            self.metrics.timestamps_allowed.append(now)
        else:
            self.metrics.rejected_requests += 1
            self.metrics.timestamps_rejected.append(now)
        return allowed

    async def require(self, key: str, cost: int = 1) -> None:
        if not await self.allow(key, cost):
            raise RateLimitExceededError(self.name)

    async def close(self) -> None:
        await self.redis.aclose()


# -- Sanity-check demo (scaffolded) ---------------------------------------


async def _demo() -> None:
    url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    client = aioredis.from_url(url)
    await client.ping()
    limiter = TokenBucketRateLimiter(
        name="demo", redis_client=client, capacity=5, refill_rate=2.0,
    )
    print("Token bucket: capacity=5, refill=2/s")
    print("Sending a burst of 10 requests as fast as possible:")
    for i in range(10):
        ok = await limiter.allow("demo-client")
        print(f"  req {i + 1}: {'ALLOW' if ok else 'REJECT'}")
    print(f"\nAllowed={limiter.metrics.allowed_requests}  "
          f"Rejected={limiter.metrics.rejected_requests}")
    await limiter.close()


def main() -> None:
    print("Sanity check: token bucket (requires Redis)\n")
    asyncio.run(_demo())


if __name__ == "__main__":
    main()
