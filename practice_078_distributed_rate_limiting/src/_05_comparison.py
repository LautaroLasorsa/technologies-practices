"""Exercise 05: Algorithm Comparison Benchmark

Sends identical traffic patterns through all four rate limiting algorithms
and compares their behavior side-by-side. Generates a comparison plot
showing allowed requests over time for each algorithm.

Run (requires Redis from docker-compose):
    uv run python src/_05_comparison.py
"""

from __future__ import annotations

import asyncio
import os
import time

from src.common import (
    RateLimiter,
    RateLimiterMetrics,
    create_redis_client,
    print_header,
    print_metrics,
)
from src._01_fixed_window import FixedWindowRateLimiter
from src._02_sliding_window_log import SlidingWindowLogRateLimiter
from src._03_sliding_window_counter import SlidingWindowCounterRateLimiter
from src._04_token_bucket import TokenBucketRateLimiter


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


async def run_benchmark() -> None:
    """Run identical traffic through all four algorithms and compare.

    Traffic pattern:
    - Phase 1: Burst of 25 requests in ~0.5s
    - Phase 2: Pause for 3 seconds
    - Phase 3: Steady 8 req/sec for 5 seconds (40 requests)
    - Phase 4: Another burst of 15 requests
    """
    client = await create_redis_client()

    fixed_window = FixedWindowRateLimiter(
        name="Fixed Window",
        redis_client=client,
        max_requests=10,
        window_seconds=1,
        key_prefix="ratelimit:bench:fw",
    )
    sliding_log = SlidingWindowLogRateLimiter(
        name="Sliding Log",
        redis_client=client,
        max_requests=10,
        window_seconds=1.0,
        key_prefix="ratelimit:bench:sl",
    )
    sliding_counter = SlidingWindowCounterRateLimiter(
        name="Sliding Counter",
        redis_client=client,
        max_requests=10,
        window_seconds=1,
        key_prefix="ratelimit:bench:sc",
    )
    token_bucket = TokenBucketRateLimiter(
        name="Token Bucket",
        redis_client=client,
        capacity=10,
        refill_rate=10.0,
        key_prefix="ratelimit:bench:tb",
    )
    limiters = [fixed_window, sliding_log, sliding_counter, token_bucket]

    await client.flushdb()

    start_time = time.time()

    print("Phase 1: Burst of 25 requests...")
    for i in range(25):
        for limiter in limiters:
            await limiter.allow("bench-user")
        await asyncio.sleep(0.02)

    print("Phase 2: Pause (3 seconds)...")
    await asyncio.sleep(3.0)

    print("Phase 3: Steady 8 req/sec for 5 seconds...")
    for i in range(40):
        for limiter in limiters:
            await limiter.allow("bench-user")
        await asyncio.sleep(0.125)

    print("Phase 4: Final burst of 15 requests...")
    for i in range(15):
        for limiter in limiters:
            await limiter.allow("bench-user")
        await asyncio.sleep(0.02)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Algorithm':<20} {'Allowed':>8} {'Rejected':>8} {'Total':>8}")
    print("-" * 48)
    for limiter in limiters:
        m = limiter.metrics
        print(f"{limiter.name:<20} {m.allowed_requests:>8} "
              f"{m.rejected_requests:>8} {m.total_requests:>8}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    for limiter, color in zip(limiters, colors):
        times = [t - start_time for t in limiter.metrics.timestamps_allowed]
        if times:
            ax.step(times, range(1, len(times) + 1),
                    label=f"{limiter.name} ({limiter.metrics.allowed_requests})",
                    color=color, linewidth=2, where="post")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Cumulative Allowed Requests")
    ax.set_title("Rate Limiter Algorithm Comparison")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("plots/algorithm_comparison.png", dpi=150)
    print("\nPlot saved to plots/algorithm_comparison.png")

    for limiter in limiters:
        await limiter.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print_header("Exercise 05: Algorithm Comparison Benchmark")
    print("Prerequisites:")
    print("  - Redis running (docker compose up -d redis)")
    print("  - Exercises 01-04 implemented (imports their classes)")
    print()
    asyncio.run(run_benchmark())


if __name__ == "__main__":
    main()
