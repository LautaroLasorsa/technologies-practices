"""Shared utilities for rate limiter exercises.

Contains the abstract base class, metrics, exceptions, and Redis connection
helpers used by all rate limiter implementations.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import redis.asyncio as aioredis


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
LUA_DIR = Path(__file__).parent.parent / "lua"


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

    def record(self, allowed: bool) -> None:
        """Record a request result."""
        now = time.time()
        self.total_requests += 1
        if allowed:
            self.allowed_requests += 1
            self.timestamps_allowed.append(now)
        else:
            self.rejected_requests += 1
            self.timestamps_rejected.append(now)

    def summary(self) -> str:
        """Return a one-line summary."""
        return (
            f"total={self.total_requests}  "
            f"allowed={self.allowed_requests}  "
            f"rejected={self.rejected_requests}"
        )


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
        """Return True if the request is allowed, False if rejected."""
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
# Redis helpers
# ---------------------------------------------------------------------------


async def create_redis_client(url: str = REDIS_URL) -> aioredis.Redis:
    """Create and verify an async Redis connection."""
    client = aioredis.from_url(url, decode_responses=False)
    await client.ping()
    return client


def load_lua_script(name: str) -> str:
    """Load a Lua script from the lua/ directory."""
    path = LUA_DIR / f"{name}.lua"
    return path.read_text(encoding="utf-8")


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    print()


def print_metrics(limiter: RateLimiter) -> None:
    """Print limiter metrics in a formatted block."""
    m = limiter.metrics
    print(f"  [{limiter.name}] {m.summary()}")
