"""Exercise 04: Combined Resilience Layer.

Compose rate limiter -> bulkhead -> circuit breaker -> downstream call
into a single `ResilientClient.call()`. This is the only TODO in the
file; the timeline-based load test and dashboard plot are scaffolded.

Run (with _00_unreliable_service running and Redis started):
    uv run python -m src._04_combined_resilience
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import aiohttp
import redis.asyncio as aioredis

from src._01_circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerCallError,
    CircuitOpenError,
    CircuitState,
)
from src._02_bulkhead import Bulkhead, BulkheadFullError
from src._03_rate_limiter import (
    RateLimitExceededError,
    TokenBucketRateLimiter,
)


@dataclass
class RequestResult:
    timestamp: float
    latency_ms: float
    success: bool
    rejection_reason: str | None = None  # rate_limit|bulkhead|circuit_open|error
    error_message: str | None = None


@dataclass
class LoadTestMetrics:
    results: list[RequestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    def count(self, reason: str | None) -> int:
        return sum(1 for r in self.results if r.rejection_reason == reason)

    @property
    def successes(self) -> int:
        return sum(1 for r in self.results if r.success)


class ResilientClient:
    """Layered resilience: rate limit -> bulkhead -> circuit breaker -> call."""

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

    def _record(
        self, start: float, success: bool, reason: str | None = None, err_msg: str | None = None,
    ) -> None:
        self.load_test_metrics.results.append(RequestResult(
            timestamp=start,
            latency_ms=(time.time() - start) * 1000,
            success=success,
            rejection_reason=reason,
            error_message=err_msg,
        ))

    # -- TODO ------------------------------------------------------------------

    async def call(
        self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        """Run *func* through the full stack.

        Order (cheapest first): rate_limiter.require -> bulkhead.call
        wrapping circuit_breaker.call wrapping *func*. Map each layer's
        exception to a `rejection_reason` via `self._record(...)`, then
        re-raise so the caller sees which layer rejected.  On success,
        record with success=True and return the value.
        """
        start = time.time()


        try:
            await self.rate_limiter.require(self.rate_limit_key)
            result = await self.bulkhead.call(self.circuit_breaker.call,func,*args,**kwargs)
            self._record(start, True, None, None)
            return result
        except Exception as e:

            reason:str|None = None
            match e:
                case RateLimitExceededError():
                    reason = "rate_limit"
                case CircuitOpenError():
                    reason = "circuit_open"
                case BulkheadFullError():
                    reason = "bulkhead"
                case _:
                    reason = "error"
            self._record(start, False, reason, str(e))
            raise

# -- HTTP helpers (scaffolded) --------------------------------------------

SERVICE_URL = "http://localhost:8089"


async def call_unreliable_service(session: aiohttp.ClientSession) -> dict[str, Any]:
    async with session.get(
        f"{SERVICE_URL}/api/data", timeout=aiohttp.ClientTimeout(total=5)
    ) as resp:
        body = await resp.json()
        if resp.status >= 500:
            raise aiohttp.ClientResponseError(
                request_info=resp.request_info,
                history=resp.history,
                status=resp.status,
                message=body.get("error", "server error"),
            )
        return body


async def switch_mode(session: aiohttp.ClientSession, mode: str) -> None:
    async with session.post(f"{SERVICE_URL}/admin/mode/{mode}") as resp:
        result = await resp.json()
        print(f"  [service] {result['previous_mode']} -> {result['current_mode']}")


# -- Load test (scaffolded) -----------------------------------------------


def _build_client(redis_client: aioredis.Redis) -> ResilientClient:
    return ResilientClient(
        rate_limiter=TokenBucketRateLimiter(
            name="load_test_rl", redis_client=redis_client, capacity=15, refill_rate=10.0,
        ),
        bulkhead=Bulkhead("load_test_bh", max_concurrent=8, max_wait_time=1.0),
        circuit_breaker=CircuitBreaker(
            name="load_test_cb", failure_threshold=5, recovery_timeout=8.0, half_open_max_calls=3,
        ),
        rate_limit_key="load_test",
    )


async def _schedule_modes(
    session: aiohttp.ClientSession, timeline: list[tuple[float, str]], start: float,
) -> None:
    for delay, mode in timeline:
        wait = max(0.0, delay - (time.time() - start))
        await asyncio.sleep(wait)
        await switch_mode(session, mode)


async def _generate_load(
    client: ResilientClient,
    session: aiohttp.ClientSession,
    start: float,
    duration: float,
    rps: float,
) -> None:
    interval = 1.0 / rps
    while time.time() - start < duration:
        try:
            await client.call(call_unreliable_service, session)
        except Exception:
            pass  # recorded in metrics
        await asyncio.sleep(interval)


def _print_results(client: ResilientClient) -> None:
    m = client.load_test_metrics
    total = len(m.results)
    if total == 0:
        print("No results.")
        return
    latencies = sorted(r.latency_ms for r in m.results if r.success)
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0
    avg = sum(latencies) / len(latencies) if latencies else 0.0
    print("\n=== Load Test Results ===")
    print(f"  duration         : {m.end_time - m.start_time:.1f}s")
    print(f"  total requests   : {total}")
    print(f"  successful       : {m.successes} ({m.successes / total * 100:.1f}%)")
    print(f"  rate_limit       : {m.count('rate_limit')}")
    print(f"  bulkhead         : {m.count('bulkhead')}")
    print(f"  circuit_open     : {m.count('circuit_open')}")
    print(f"  error            : {m.count('error')}")
    print(f"  avg latency (ok) : {avg:.1f}ms")
    print(f"  p99 latency (ok) : {p99:.1f}ms")


def _save_dashboard(client: ResilientClient, path: str) -> None:
    import matplotlib.pyplot as plt

    m = client.load_test_metrics
    t0 = m.start_time
    colors = {
        None: "green",
        "rate_limit": "orange",
        "bulkhead": "blue",
        "circuit_open": "purple",
        "error": "red",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    for reason, color in colors.items():
        xs = [r.timestamp - t0 for r in m.results if r.rejection_reason == reason]
        ys = [r.latency_ms for r in m.results if r.rejection_reason == reason]
        label = "success" if reason is None else reason
        ax.scatter(xs, ys, s=10, alpha=0.6, color=color, label=label)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("latency (ms)")
    ax.set_yscale("log")
    ax.set_title("Request outcomes over time")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 1]
    transitions = client.circuit_breaker.metrics.state_transitions
    state_map = {CircuitState.CLOSED: 0, CircuitState.HALF_OPEN: 1, CircuitState.OPEN: 2}
    if transitions:
        ts = [t - t0 for t, _, _ in transitions]
        ys = [state_map[new] for _, _, new in transitions]
        ts = [0.0] + ts
        ys = [0] + ys
        ax.step(ts, ys, where="post", color="black")
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([0, 1, 2], ["CLOSED", "HALF_OPEN", "OPEN"])
    ax.set_xlabel("time (s)")
    ax.set_title("Circuit breaker state")

    ax = axes[1, 0]
    cumulative: dict[str | None, list[int]] = {k: [] for k in colors}
    running = {k: 0 for k in colors}
    times = []
    for r in sorted(m.results, key=lambda x: x.timestamp):
        running[r.rejection_reason] += 1
        times.append(r.timestamp - t0)
        for k in cumulative:
            cumulative[k].append(running[k])
    for reason, color in colors.items():
        label = "success" if reason is None else reason
        ax.plot(times, cumulative[reason], color=color, label=label)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cumulative count")
    ax.set_title("Cumulative outcomes")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1, 1]
    counts = {k: running[k] for k in colors if running[k] > 0}
    labels = ["success" if k is None else k for k in counts]
    ax.pie(counts.values(), labels=labels, colors=[colors[k] for k in counts], autopct="%1.0f%%")
    ax.set_title("Rejection breakdown")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"  dashboard: {path}")


async def load_test() -> None:
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    redis_client = aioredis.from_url(redis_url)
    await redis_client.ping()

    client = _build_client(redis_client)
    timeline = [(0.0, "healthy"), (5.0, "degraded"), (15.0, "down"), (20.0, "healthy")]
    duration = 30.0
    rps = 10.0

    async with aiohttp.ClientSession() as session:
        start = time.time()
        client.load_test_metrics.start_time = start
        await asyncio.gather(
            _schedule_modes(session, timeline, start),
            _generate_load(client, session, start, duration, rps),
        )
        client.load_test_metrics.end_time = time.time()

    _print_results(client)
    _save_dashboard(client, "plots/combined_resilience_dashboard.png")
    await redis_client.aclose()


def main() -> None:
    print("Exercise 04: Combined Resilience Layer")
    print("Prereq: _00_unreliable_service running + Redis up\n")
    asyncio.run(load_test())


if __name__ == "__main__":
    main()
