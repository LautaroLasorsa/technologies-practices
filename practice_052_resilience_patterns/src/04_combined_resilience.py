"""Exercise 04: Combined Resilience Layer

Stack all three patterns -- rate limiter, bulkhead, circuit breaker -- into a
single ResilientClient that applies them in sequence:

    Request -> Rate Limiter -> Bulkhead -> Circuit Breaker -> Downstream Call

Each layer can independently reject a request.  The order matters:
  1. Rate limiter first (cheapest check, Redis lookup only, no concurrency slot)
  2. Bulkhead second (limits concurrency, acquires a semaphore slot)
  3. Circuit breaker third (checks downstream health, may fail fast)
  4. Actual downstream call last (most expensive, involves network I/O)

A load test drives realistic traffic through the combined layer while the
unreliable service cycles through failure modes, producing a dashboard of
metrics and rejection reasons.

Run (with 00_unreliable_service.py running and Redis started):
    uv run python src/04_combined_resilience.py
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import aiohttp
import redis.asyncio as aioredis

# Import from previous exercises -- these should already be implemented
from src.01_circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitBreakerCallError,
)
from src.02_bulkhead import Bulkhead, BulkheadFullError
from src.03_rate_limiter import (
    TokenBucketRateLimiter,
    RateLimitExceededError,
)


# ---------------------------------------------------------------------------
# Request result tracking
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    """Result of a single request through the resilience layer."""

    timestamp: float
    latency_ms: float
    success: bool
    rejection_reason: str | None = None  # "rate_limit", "bulkhead", "circuit_open", "error"
    error_message: str | None = None


@dataclass
class LoadTestMetrics:
    """Aggregate metrics from a load test run."""

    results: list[RequestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def total_requests(self) -> int:
        return len(self.results)

    @property
    def successful_requests(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def rejected_by_rate_limit(self) -> int:
        return sum(1 for r in self.results if r.rejection_reason == "rate_limit")

    @property
    def rejected_by_bulkhead(self) -> int:
        return sum(1 for r in self.results if r.rejection_reason == "bulkhead")

    @property
    def rejected_by_circuit(self) -> int:
        return sum(1 for r in self.results if r.rejection_reason == "circuit_open")

    @property
    def failed_with_error(self) -> int:
        return sum(1 for r in self.results if r.rejection_reason == "error")

    @property
    def avg_latency_ms(self) -> float:
        latencies = [r.latency_ms for r in self.results if r.success]
        return sum(latencies) / len(latencies) if latencies else 0.0

    @property
    def p99_latency_ms(self) -> float:
        latencies = sorted(r.latency_ms for r in self.results if r.success)
        if not latencies:
            return 0.0
        idx = int(len(latencies) * 0.99)
        return latencies[min(idx, len(latencies) - 1)]


# ---------------------------------------------------------------------------
# Resilient Client
# ---------------------------------------------------------------------------


class ResilientClient:
    """Composes rate limiter, bulkhead, and circuit breaker into a layered defense.

    The call chain is:
        rate_limiter.require(key) -> bulkhead.call() -> circuit_breaker.call() -> func()

    Each layer can independently reject the request.  The ResilientClient tracks
    which layer rejected each request for observability.

    Parameters
    ----------
    rate_limiter:
        A TokenBucketRateLimiter (or any RateLimiter with require()).
    bulkhead:
        A Bulkhead (semaphore-based concurrency limiter).
    circuit_breaker:
        A CircuitBreaker (state machine for downstream health).
    rate_limit_key:
        The key to use for rate limiting (e.g., client IP, API key).
    """

    def __init__(
        self,
        rate_limiter: TokenBucketRateLimiter,
        bulkhead: Bulkhead,
        circuit_breaker: CircuitBreaker,
        rate_limit_key: str = "default",
    ) -> None:
        self.rate_limiter = rate_limiter
        self.bulkhead = bulkhead
        self.circuit_breaker = circuit_breaker
        self.rate_limit_key = rate_limit_key
        self.load_test_metrics = LoadTestMetrics()

    async def call(
        self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute *func* through the full resilience stack.

        Order: rate limiter -> bulkhead -> circuit breaker -> func

        Returns the function's return value on success.
        Records every call result (success or rejection) in load_test_metrics.

        Raises the original rejection exception so the caller can handle it
        (e.g., return a cached response, serve a degraded page, etc.).
        """
        # TODO(human): Implement the combined resilience call chain.
        #
        # WHY THIS MATTERS:
        # This is where resilience patterns compose.  In production, you rarely
        # use just one pattern -- you layer them.  The order matters:
        #   - Rate limiter FIRST: cheapest check (single Redis call), rejects
        #     before consuming any local resources.
        #   - Bulkhead SECOND: limits concurrency, acquires a semaphore slot.
        #     If the rate limiter already rejected, we don't waste a slot.
        #   - Circuit breaker THIRD: checks downstream health.  If the bulkhead
        #     already rejected, we don't even evaluate the circuit state.
        #   - Actual call LAST: most expensive (network I/O, downstream processing).
        #
        # Each layer wrapping the next is like an onion: the outermost layer
        # (rate limiter) peels off the most requests, so the inner layers
        # handle progressively fewer.  This is efficient and mirrors how
        # production systems like resilience4j, Polly, and Envoy compose their
        # resilience decorators.
        #
        # THE ALGORITHM:
        #
        # start = time.time()
        #
        # 1. RATE LIMITING CHECK:
        #    try:
        #        await self.rate_limiter.require(self.rate_limit_key)
        #    except RateLimitExceededError:
        #        -> Record RequestResult with rejection_reason="rate_limit"
        #        -> Re-raise the exception
        #
        # 2. BULKHEAD + CIRCUIT BREAKER + FUNCTION CALL:
        #    The bulkhead wraps the circuit breaker call, which wraps the
        #    actual function.  Use lambdas or nested calls:
        #
        #    try:
        #        result = await self.bulkhead.call(
        #            self.circuit_breaker.call, func, *args, **kwargs
        #        )
        #    except BulkheadFullError:
        #        -> Record RequestResult with rejection_reason="bulkhead"
        #        -> Re-raise
        #    except CircuitOpenError:
        #        -> Record RequestResult with rejection_reason="circuit_open"
        #        -> Re-raise
        #    except CircuitBreakerCallError as e:
        #        -> Record RequestResult with rejection_reason="error",
        #           error_message=str(e.original)
        #        -> Re-raise
        #    except Exception as e:
        #        -> Record RequestResult with rejection_reason="error",
        #           error_message=str(e)
        #        -> Re-raise
        #
        # 3. ON SUCCESS:
        #    -> Record RequestResult with success=True
        #    -> Return result
        #
        # HOW TO RECORD:
        #   latency_ms = (time.time() - start) * 1000
        #   self.load_test_metrics.results.append(RequestResult(
        #       timestamp=start,
        #       latency_ms=latency_ms,
        #       success=True/False,
        #       rejection_reason=...,
        #       error_message=...,
        #   ))
        #
        # IMPORTANT:
        # The bulkhead.call() takes a callable and its arguments.  Passing
        # self.circuit_breaker.call as the callable and func, *args, **kwargs
        # as ITS arguments means the bulkhead acquires a semaphore slot, then
        # inside that slot, the circuit breaker evaluates the call.  This
        # nesting is key to correct composition.
        #
        # HINT for clean nesting:
        #   await self.bulkhead.call(self.circuit_breaker.call, func, *args, **kwargs)
        #   This passes circuit_breaker.call as the function to bulkhead.call,
        #   and func, *args, **kwargs are forwarded as its arguments.
        raise NotImplementedError("TODO(human): implement resilient client call()")


# ---------------------------------------------------------------------------
# Helper: call the unreliable service
# ---------------------------------------------------------------------------

SERVICE_URL = "http://localhost:8089"


async def call_unreliable_service(session: aiohttp.ClientSession) -> dict[str, Any]:
    """Make an HTTP GET to the unreliable service's /api/data endpoint."""
    async with session.get(
        f"{SERVICE_URL}/api/data", timeout=aiohttp.ClientTimeout(total=5)
    ) as resp:
        body = await resp.json()
        if resp.status >= 500:
            raise aiohttp.ClientResponseError(
                request_info=resp.request_info,
                history=resp.history,
                status=resp.status,
                message=body.get("error", "Server error"),
            )
        return body


async def switch_mode(session: aiohttp.ClientSession, mode: str) -> None:
    """Switch the unreliable service to a different failure mode."""
    async with session.post(f"{SERVICE_URL}/admin/mode/{mode}") as resp:
        result = await resp.json()
        print(f"  [Service] Mode changed: {result['previous_mode']} -> {result['current_mode']}")


# ---------------------------------------------------------------------------
# Load test
# ---------------------------------------------------------------------------


async def load_test() -> None:
    """Drive realistic traffic through the ResilientClient while the service
    cycles through failure modes.

    Timeline:
      0-5s:   healthy   -- most requests succeed
      5-15s:  degraded  -- circuit breaker may trip, bulkhead fills
      15-20s: down      -- circuit breaker trips, all rejected fast
      20-30s: healthy   -- circuit breaker recovers, traffic resumes

    Sends ~10 requests/second throughout.  Collects per-request results for
    analysis and plotting.
    """
    # TODO(human): Implement the load test that exercises all resilience layers.
    #
    # WHY THIS MATTERS:
    # Individual pattern tests are useful, but the real test is how patterns
    # behave together under realistic conditions.  This load test simulates
    # a production scenario: steady traffic, a degradation event, a full
    # outage, and recovery.  The per-request metrics show you exactly what
    # happened: how many requests were served, how many were rejected, WHICH
    # LAYER rejected them, and what the user-visible latency was at each point.
    #
    # STEP-BY-STEP PLAN:
    #
    # 1. SETUP INFRASTRUCTURE:
    #    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    #    redis_client = aioredis.from_url(redis_url)
    #    await redis_client.ping()
    #
    # 2. CREATE THE RESILIENCE COMPONENTS:
    #    rate_limiter = TokenBucketRateLimiter(
    #        name="load_test_rl",
    #        redis_client=redis_client,
    #        capacity=15,        # burst allowance
    #        refill_rate=10.0,   # sustained 10 req/sec
    #    )
    #    bulkhead = Bulkhead(
    #        name="load_test_bh",
    #        max_concurrent=8,   # max 8 in-flight calls
    #        max_wait_time=1.0,  # wait up to 1s for a slot
    #    )
    #    circuit_breaker = CircuitBreaker(
    #        name="load_test_cb",
    #        failure_threshold=5,
    #        recovery_timeout=8.0,
    #        half_open_max_calls=3,
    #    )
    #
    # 3. COMPOSE INTO RESILIENT CLIENT:
    #    client = ResilientClient(
    #        rate_limiter=rate_limiter,
    #        bulkhead=bulkhead,
    #        circuit_breaker=circuit_breaker,
    #        rate_limit_key="load_test",
    #    )
    #
    # 4. DEFINE THE TIMELINE (mode changes):
    #    timeline = [
    #        (0,  "healthy"),
    #        (5,  "degraded"),
    #        (15, "down"),
    #        (20, "healthy"),
    #    ]
    #    test_duration = 30  # total seconds
    #
    # 5. RUN THE LOAD TEST:
    #    async with aiohttp.ClientSession() as session:
    #
    #        a. Start a background task that changes modes at the scheduled times:
    #           async def mode_scheduler():
    #               for delay, mode in timeline:
    #                   await asyncio.sleep(delay - (time.time() - start))
    #                   await switch_mode(session, mode)
    #
    #        b. Start a request generator that sends ~10 req/sec:
    #           async def request_generator():
    #               while time.time() - start < test_duration:
    #                   try:
    #                       await client.call(call_unreliable_service, session)
    #                   except (RateLimitExceededError, BulkheadFullError,
    #                           CircuitOpenError, CircuitBreakerCallError, Exception):
    #                       pass  # already recorded in metrics
    #                   await asyncio.sleep(0.1)  # ~10 req/sec
    #
    #        c. Run both concurrently:
    #           start = time.time()
    #           client.load_test_metrics.start_time = start
    #           await asyncio.gather(mode_scheduler(), request_generator())
    #           client.load_test_metrics.end_time = time.time()
    #
    # 6. PRINT RESULTS DASHBOARD:
    #    metrics = client.load_test_metrics
    #    Print a formatted summary:
    #
    #    === Load Test Results ===
    #    Duration: {end - start:.1f}s
    #    Total requests: {total}
    #    Successful:     {success} ({success/total*100:.1f}%)
    #    Rejected by:
    #      Rate limiter:    {rl_count}
    #      Bulkhead:        {bh_count}
    #      Circuit breaker: {cb_count}
    #      Downstream error:{err_count}
    #    Latency (successful):
    #      Average: {avg:.1f}ms
    #      P99:     {p99:.1f}ms
    #
    # 7. GENERATE DASHBOARD PLOT:
    #    Create a 2x2 subplot figure:
    #
    #    Top-left: Timeline of request outcomes (scatter plot)
    #      - x = time, y = latency, color = outcome (green=success, red=error,
    #        orange=rate_limit, blue=bulkhead, purple=circuit_open)
    #
    #    Top-right: Cumulative requests by outcome (stacked area or step plot)
    #
    #    Bottom-left: Circuit breaker state over time (step plot showing
    #      CLOSED/HALF_OPEN/OPEN as 0/1/2)
    #
    #    Bottom-right: Rejection breakdown (pie chart)
    #
    #    Save to plots/combined_resilience_dashboard.png
    #
    # 8. CLEANUP:
    #    await redis_client.aclose()
    #
    # HINT for the mode_scheduler timing:
    #   The timeline has absolute times (0s, 5s, 15s, 20s).  To wait the right
    #   amount, calculate: sleep_time = target_time - (time.time() - start)
    #   Use max(0, sleep_time) to handle any clock drift.
    #
    # HINT for generating the circuit breaker state timeline:
    #   transitions = circuit_breaker.metrics.state_transitions
    #   Build a step function: at each transition timestamp, the state changes.
    #   Map states to numbers: CLOSED=0, HALF_OPEN=1, OPEN=2
    raise NotImplementedError("TODO(human): implement load_test()")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Exercise 04: Combined Resilience Layer")
    print("=" * 60)
    print()
    print("Prerequisites:")
    print("  - 00_unreliable_service.py running on port 8089")
    print("  - Redis running (docker compose up -d)")
    print()
    asyncio.run(load_test())


if __name__ == "__main__":
    main()
