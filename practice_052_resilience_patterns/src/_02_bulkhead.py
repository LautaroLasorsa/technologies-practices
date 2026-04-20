"""Exercise 02: Bulkhead Pattern -- Semaphore Isolation.

Implement the single core method: `Bulkhead.call()`. It must acquire a
semaphore slot (with an optional wait timeout), execute the callable,
and release the slot no matter what — including on exceptions.

The comparison demo (with vs. without bulkhead) is scaffolded for you.

Run (with _00_unreliable_service running):
    uv run python -m src._02_bulkhead
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import aiohttp


class BulkheadFullError(Exception):
    def __init__(self, name: str, max_concurrent: int) -> None:
        self.name = name
        self.max_concurrent = max_concurrent
        super().__init__(f"Bulkhead '{name}' full ({max_concurrent} slots in use)")


@dataclass
class BulkheadMetrics:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    peak_concurrent: int = 0
    call_latencies: list[float] = field(default_factory=list)


class Bulkhead:
    """Semaphore-based isolation for a single dependency.

    `max_concurrent` caps in-flight calls.  If `max_wait_time` is 0, full
    bulkhead rejects immediately; otherwise callers wait up to that many
    seconds for a slot.
    """

    def __init__(self, name: str, max_concurrent: int = 10, max_wait_time: float = 0.0) -> None:
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_wait_time = max_wait_time
        self.metrics = BulkheadMetrics()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_calls = 0

    @property
    def active_calls(self) -> int:
        return self._active_calls

    # -- TODO ------------------------------------------------------------------

    async def call(
        self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute *func* inside the bulkhead.

        Acquire a semaphore slot (waiting at most `max_wait_time` seconds).
        On timeout, increment `rejected_calls` and raise `BulkheadFullError`.
        Update `_active_calls`/`peak_concurrent` around the call.  Use
        try/finally so the slot is released even on exceptions.  Append the
        call latency to `call_latencies` and update success/failed counters.
        """
        # TODO(human): implement the bulkhead call.
        # Hints:
        #   timeout = max(self.max_wait_time, 0.001)  # 0 means "fail fast"
        #   try:
        #       await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
        #   except asyncio.TimeoutError:
        #       self.metrics.rejected_calls += 1
        #       raise BulkheadFullError(self.name, self.max_concurrent)
        #   # then track concurrency, execute func in try/finally, release semaphore
        raise NotImplementedError("TODO(human): implement Bulkhead.call")


# -- HTTP helpers (scaffolded) --------------------------------------------

SERVICE_URL = "http://localhost:8089"


async def call_fast_api(session: aiohttp.ClientSession) -> dict[str, Any]:
    start = time.time()
    async with session.get(
        f"{SERVICE_URL}/health", timeout=aiohttp.ClientTimeout(total=2)
    ) as resp:
        body = await resp.json()
        body["latency_ms"] = (time.time() - start) * 1000
        return body


async def call_slow_api(session: aiohttp.ClientSession) -> dict[str, Any]:
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
                message=body.get("error", "server error"),
            )
        body["latency_ms"] = (time.time() - start) * 1000
        return body


async def switch_mode(session: aiohttp.ClientSession, mode: str) -> None:
    async with session.post(f"{SERVICE_URL}/admin/mode/{mode}") as resp:
        result = await resp.json()
        print(f"  [service] {result['previous_mode']} -> {result['current_mode']}")


# -- Demo (scaffolded) ----------------------------------------------------


def _fast_latencies(results: list[Any]) -> list[float]:
    return [r["latency_ms"] for r in results if isinstance(r, dict) and "latency_ms" in r]


def _count_exceptions(results: list[Any]) -> int:
    return sum(1 for r in results if isinstance(r, Exception))


async def _run_without_bulkhead(session: aiohttp.ClientSession, fast_n: int, slow_n: int) -> dict[str, Any]:
    print("\n--- Without bulkhead (shared resource pool) ---")
    start = time.time()
    fast_tasks = [call_fast_api(session) for _ in range(fast_n)]
    slow_tasks = [call_slow_api(session) for _ in range(slow_n)]
    results = await asyncio.gather(*fast_tasks, *slow_tasks, return_exceptions=True)
    elapsed = time.time() - start

    fast_results = results[:fast_n]
    latencies = _fast_latencies(fast_results)
    return {
        "elapsed": elapsed,
        "fast_avg_ms": statistics.mean(latencies) if latencies else 0.0,
        "fast_p99_ms": max(latencies) if latencies else 0.0,
        "fast_errors": _count_exceptions(fast_results),
    }


async def _run_with_bulkhead(session: aiohttp.ClientSession, fast_n: int, slow_n: int) -> dict[str, Any]:
    print("\n--- With bulkhead (isolated pools) ---")
    fast_bh = Bulkhead("fast_api", max_concurrent=10, max_wait_time=1.0)
    slow_bh = Bulkhead("slow_api", max_concurrent=5, max_wait_time=1.0)

    start = time.time()
    fast_tasks = [fast_bh.call(call_fast_api, session) for _ in range(fast_n)]
    slow_tasks = [slow_bh.call(call_slow_api, session) for _ in range(slow_n)]
    results = await asyncio.gather(*fast_tasks, *slow_tasks, return_exceptions=True)
    elapsed = time.time() - start

    fast_results = results[:fast_n]
    latencies = _fast_latencies(fast_results)
    return {
        "elapsed": elapsed,
        "fast_avg_ms": statistics.mean(latencies) if latencies else 0.0,
        "fast_p99_ms": max(latencies) if latencies else 0.0,
        "fast_errors": _count_exceptions(fast_results),
        "slow_rejections": slow_bh.metrics.rejected_calls,
        "slow_peak": slow_bh.metrics.peak_concurrent,
    }


def _print_comparison(without: dict[str, Any], with_: dict[str, Any]) -> None:
    print("\n--- Comparison ---")
    print(f"  {'metric':25} {'without':>12} {'with':>12}")
    print(f"  {'total elapsed (s)':25} {without['elapsed']:>12.2f} {with_['elapsed']:>12.2f}")
    print(f"  {'fast_api avg (ms)':25} {without['fast_avg_ms']:>12.0f} {with_['fast_avg_ms']:>12.0f}")
    print(f"  {'fast_api p99 (ms)':25} {without['fast_p99_ms']:>12.0f} {with_['fast_p99_ms']:>12.0f}")
    print(f"  {'fast_api errors':25} {without['fast_errors']:>12} {with_['fast_errors']:>12}")
    print(f"  {'slow_api rejections':25} {'n/a':>12} {with_['slow_rejections']:>12}")


async def demo_bulkhead_isolation() -> None:
    fast_n, slow_n = 20, 20
    async with aiohttp.ClientSession() as session:
        await switch_mode(session, "degraded")
        without = await _run_without_bulkhead(session, fast_n, slow_n)
        with_ = await _run_with_bulkhead(session, fast_n, slow_n)
        await switch_mode(session, "healthy")
    _print_comparison(without, with_)


def main() -> None:
    print("Exercise 02: Bulkhead Pattern")
    print("Prereq: _00_unreliable_service running on :8089\n")
    asyncio.run(demo_bulkhead_isolation())


if __name__ == "__main__":
    main()
