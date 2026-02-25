"""Exercise 02: Bulkhead Pattern -- Semaphore-Based Dependency Isolation

The bulkhead pattern isolates dependencies so that a slow or failing dependency
cannot consume all shared resources (threads, connections, semaphores) and starve
other dependencies.

In async Python the natural implementation is an asyncio.Semaphore per dependency.
Each semaphore caps the number of concurrent in-flight calls to that dependency.
When the semaphore is full, additional callers either wait (with a timeout) or are
rejected immediately.

This exercise contrasts "with bulkhead" vs "without bulkhead" to show how
isolation prevents a slow dependency from degrading fast dependencies.

Run (with 00_unreliable_service.py running in another terminal):
    uv run python src/02_bulkhead.py
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import aiohttp


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BulkheadFullError(Exception):
    """Raised when a call is rejected because the bulkhead has no free slots."""

    def __init__(self, name: str, max_concurrent: int) -> None:
        self.name = name
        self.max_concurrent = max_concurrent
        super().__init__(
            f"Bulkhead '{name}' is full ({max_concurrent}/{max_concurrent} slots in use). "
            f"Request rejected."
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class BulkheadMetrics:
    """Counters for monitoring bulkhead behavior."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    peak_concurrent: int = 0
    total_wait_time: float = 0.0
    call_latencies: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bulkhead
# ---------------------------------------------------------------------------


class Bulkhead:
    """Semaphore-based bulkhead for async dependency isolation.

    Parameters
    ----------
    name:
        Human-readable name for this bulkhead (used in errors and logs).
    max_concurrent:
        Maximum number of calls that can be in-flight simultaneously.
        When all slots are in use, new calls are rejected or wait.
    max_wait_time:
        Maximum seconds to wait for a semaphore slot before raising
        BulkheadFullError.  Set to 0 for immediate rejection (fail fast).
    """

    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_wait_time: float = 0.0,
    ) -> None:
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_wait_time = max_wait_time

        self.metrics = BulkheadMetrics()

        # -- internal state --
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrent)
        self._active_calls: int = 0

    @property
    def active_calls(self) -> int:
        """Number of calls currently in-flight inside this bulkhead."""
        return self._active_calls

    @property
    def available_slots(self) -> int:
        """Number of free slots in the semaphore."""
        return self.max_concurrent - self._active_calls

    async def call(
        self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute *func* through the bulkhead.

        Attempts to acquire a semaphore slot.  If max_wait_time > 0, waits
        up to that duration.  If the slot cannot be acquired, raises
        BulkheadFullError.  Otherwise, executes the function and releases the
        slot on completion.

        Returns the function's return value on success.
        Raises the original exception on failure (after releasing the slot).
        Raises BulkheadFullError when no slot is available within max_wait_time.
        """
        # TODO(human): Implement the bulkhead call method using asyncio.Semaphore.
        #
        # WHY THIS MATTERS:
        # This is the core of the bulkhead pattern.  The semaphore limits how
        # many concurrent calls can be in-flight to a specific dependency.
        # When a downstream service becomes slow, its calls hold semaphore slots
        # longer, but because each dependency has its OWN semaphore, other
        # dependencies are not affected.  Without this isolation, all dependencies
        # compete for a shared pool, and one slow dependency can starve the rest.
        #
        # THE ALGORITHM:
        #
        # 1. TRY TO ACQUIRE THE SEMAPHORE (with timeout):
        #    - Record wait_start = time.time()
        #    - If max_wait_time <= 0 (immediate rejection mode):
        #        Use self._semaphore.acquire() but only if it can be acquired
        #        WITHOUT blocking.  The trick: check _semaphore.locked() -- if
        #        it's locked (all permits taken), the semaphore is full.
        #        BUT _semaphore.locked() is only True when the count is 0, which
        #        isn't quite right for counting semaphores.
        #
        #        Better approach: use asyncio.wait_for(self._semaphore.acquire(),
        #        timeout=0.001) -- effectively instant.  If it times out, raise
        #        BulkheadFullError.
        #
        #    - If max_wait_time > 0:
        #        Use asyncio.wait_for(self._semaphore.acquire(),
        #        timeout=self.max_wait_time).  If it times out, raise
        #        BulkheadFullError.
        #
        #    - On TimeoutError: increment metrics.rejected_calls, raise
        #      BulkheadFullError(self.name, self.max_concurrent)
        #
        #    - Record wait_time = time.time() - wait_start
        #    - Add wait_time to metrics.total_wait_time
        #
        # 2. TRACK CONCURRENCY:
        #    - Increment self._active_calls
        #    - Update metrics.peak_concurrent = max(peak_concurrent, _active_calls)
        #
        # 3. EXECUTE THE FUNCTION (in a try/finally to always release):
        #    - call_start = time.time()
        #    - try:
        #        result = await func(*args, **kwargs)
        #        metrics.successful_calls += 1
        #        return result
        #      except Exception:
        #        metrics.failed_calls += 1
        #        raise  (let the original exception propagate)
        #      finally:
        #        latency = time.time() - call_start
        #        metrics.call_latencies.append(latency)
        #        metrics.total_calls += 1
        #        self._active_calls -= 1
        #        self._semaphore.release()
        #
        # IMPORTANT SUBTLETY:
        # The try/finally around the function execution is critical.  If the
        # function raises an exception (or times out), the semaphore slot MUST
        # be released.  Otherwise, slots leak and the bulkhead gradually fills
        # up, eventually rejecting all calls permanently.  This is analogous to
        # connection pool leaks in database programming.
        #
        # HINT:
        #   try:
        #       await asyncio.wait_for(self._semaphore.acquire(), timeout=...)
        #   except asyncio.TimeoutError:
        #       self.metrics.rejected_calls += 1
        #       raise BulkheadFullError(self.name, self.max_concurrent)
        raise NotImplementedError("TODO(human): implement bulkhead call()")


# ---------------------------------------------------------------------------
# Helpers: simulated dependencies
# ---------------------------------------------------------------------------

SERVICE_URL = "http://localhost:8089"


async def call_fast_api(session: aiohttp.ClientSession) -> dict[str, Any]:
    """Simulate a fast dependency (always healthy endpoint)."""
    start = time.time()
    async with session.get(
        f"{SERVICE_URL}/health", timeout=aiohttp.ClientTimeout(total=2)
    ) as resp:
        result = await resp.json()
        result["latency_ms"] = round((time.time() - start) * 1000, 1)
        return result


async def call_slow_api(session: aiohttp.ClientSession) -> dict[str, Any]:
    """Simulate a slow dependency (degraded endpoint with high latency)."""
    start = time.time()
    async with session.get(
        f"{SERVICE_URL}/api/data", timeout=aiohttp.ClientTimeout(total=10)
    ) as resp:
        body = await resp.json()
        if resp.status >= 500:
            raise aiohttp.ClientResponseError(
                request_info=resp.request_info,
                history=resp.history,
                status=resp.status,
                message=body.get("error", "Server error"),
            )
        body["latency_ms"] = round((time.time() - start) * 1000, 1)
        return body


async def switch_mode(session: aiohttp.ClientSession, mode: str) -> None:
    """Switch the unreliable service to a different failure mode."""
    async with session.post(f"{SERVICE_URL}/admin/mode/{mode}") as resp:
        result = await resp.json()
        print(f"  [Service] Mode changed: {result['previous_mode']} -> {result['current_mode']}")


# ---------------------------------------------------------------------------
# Demo: Bulkhead isolation
# ---------------------------------------------------------------------------


async def demo_bulkhead_isolation() -> None:
    """Contrast dependency isolation with and without bulkheads.

    Scenario:
      - "fast_api" calls /health (always fast, ~10ms)
      - "slow_api" calls /api/data in degraded mode (50% errors, 500-2000ms)
      - Both share the same aiohttp session (simulating a shared connection pool)

    Without bulkhead: launch many concurrent calls to both APIs.  The slow_api
    calls hog shared resources, increasing fast_api latency.

    With bulkhead: each API gets its own semaphore.  slow_api can only use N
    concurrent slots.  fast_api calls remain fast because they have their own
    isolated pool.
    """
    # TODO(human): Implement the bulkhead isolation demo.
    #
    # WHY THIS MATTERS:
    # The whole point of bulkheads is isolation.  This demo makes that benefit
    # tangible by measuring fast_api latency with and without isolation.  Seeing
    # the numbers shows that without a bulkhead, a slow dependency's impact is
    # not limited to itself -- it degrades everything.  With a bulkhead, the
    # blast radius is contained.
    #
    # STEP-BY-STEP PLAN:
    #
    # 1. PUT THE SERVICE IN DEGRADED MODE:
    #    async with aiohttp.ClientSession() as session:
    #        await switch_mode(session, "degraded")
    #
    # 2. DEFINE THE WORKLOAD:
    #    - num_fast_calls = 20   (to the fast /health endpoint)
    #    - num_slow_calls = 20   (to the degraded /api/data endpoint)
    #
    # 3. RUN WITHOUT BULKHEAD:
    #    - Print "--- Without Bulkhead (shared resource pool) ---"
    #    - Create tasks for all calls:
    #        fast_tasks = [call_fast_api(session) for _ in range(num_fast_calls)]
    #        slow_tasks = [call_slow_api(session) for _ in range(num_slow_calls)]
    #    - Time the execution:
    #        start = time.time()
    #        results = await asyncio.gather(
    #            *fast_tasks, *slow_tasks, return_exceptions=True
    #        )
    #        elapsed = time.time() - start
    #    - Separate fast results (first num_fast_calls) from slow results
    #    - For fast results, extract latency_ms from successful ones
    #    - Print: total time, average fast_api latency, count of errors
    #    - Store these metrics for comparison
    #
    # 4. RUN WITH BULKHEAD:
    #    - Print "--- With Bulkhead (isolated resource pools) ---"
    #    - Create two bulkheads:
    #        fast_bulkhead = Bulkhead("fast_api", max_concurrent=10, max_wait_time=1.0)
    #        slow_bulkhead = Bulkhead("slow_api", max_concurrent=5, max_wait_time=1.0)
    #      (slow_api gets fewer slots -- it's the unreliable one)
    #    - Create tasks using bulkhead.call():
    #        fast_tasks = [
    #            fast_bulkhead.call(call_fast_api, session)
    #            for _ in range(num_fast_calls)
    #        ]
    #        slow_tasks = [
    #            slow_bulkhead.call(call_slow_api, session)
    #            for _ in range(num_slow_calls)
    #        ]
    #    - Time and gather as before (with return_exceptions=True)
    #    - Extract fast_api latencies and compute average
    #    - Print: total time, avg fast_api latency, bulkhead rejections
    #    - Print bulkhead metrics: peak_concurrent, rejected_calls
    #
    # 5. COMPARISON SUMMARY:
    #    - Print a comparison table showing:
    #        Metric                | Without BH | With BH
    #        ----------------------|------------|--------
    #        Total time            | ...        | ...
    #        Avg fast_api latency  | ...        | ...
    #        Fast API errors       | ...        | ...
    #        Slow API rejections   | N/A        | ...
    #
    #    The key insight: with bulkheads, fast_api latency should be
    #    significantly lower because slow_api calls cannot monopolize
    #    all concurrency slots.
    #
    # 6. RESTORE SERVICE MODE:
    #    await switch_mode(session, "healthy")
    #
    # HINTS:
    # - Use asyncio.gather(*tasks, return_exceptions=True) to avoid one
    #   exception canceling all tasks.
    # - To extract latency from results, check isinstance(result, dict)
    #   (exceptions will be Exception instances in the results list).
    # - time.time() around the gather gives wall-clock time, which captures
    #   the concurrency behavior.
    raise NotImplementedError("TODO(human): implement demo_bulkhead_isolation()")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Exercise 02: Bulkhead Pattern")
    print("=" * 60)
    print()
    print("Prerequisites:")
    print("  - 00_unreliable_service.py running on port 8089")
    print()
    asyncio.run(demo_bulkhead_isolation())


if __name__ == "__main__":
    main()
