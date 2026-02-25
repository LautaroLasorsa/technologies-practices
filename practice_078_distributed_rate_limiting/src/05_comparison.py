"""Exercise 05: Algorithm Comparison Benchmark

Sends identical traffic patterns through all four rate limiting algorithms
and compares their behavior side-by-side. Generates a comparison plot
showing allowed requests over time for each algorithm.

Run (requires Redis from docker-compose):
    uv run python src/05_comparison.py
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
from src.01_fixed_window import FixedWindowRateLimiter
from src.02_sliding_window_log import SlidingWindowLogRateLimiter
from src.03_sliding_window_counter import SlidingWindowCounterRateLimiter
from src.04_token_bucket import TokenBucketRateLimiter


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
    # TODO(human): Implement the algorithm comparison benchmark.
    #
    # WHY THIS MATTERS:
    # Understanding the BEHAVIORAL differences between algorithms is just
    # as important as understanding their implementation. This benchmark
    # makes those differences concrete and visible. In a system design
    # interview, being able to say "I benchmarked all four algorithms and
    # here's how they differ under burst vs steady traffic" demonstrates
    # deep practical understanding.
    #
    # STEP-BY-STEP PLAN:
    #
    # 1. CONNECT TO REDIS:
    #    client = await create_redis_client()
    #
    # 2. CREATE ALL FOUR LIMITERS with equivalent effective rates:
    #    All limiters target ~10 requests per 1-second window.
    #
    #    fixed_window = FixedWindowRateLimiter(
    #        name="Fixed Window",
    #        redis_client=client,
    #        max_requests=10,
    #        window_seconds=1,
    #        key_prefix="ratelimit:bench:fw",
    #    )
    #    sliding_log = SlidingWindowLogRateLimiter(
    #        name="Sliding Log",
    #        redis_client=client,
    #        max_requests=10,
    #        window_seconds=1.0,
    #        key_prefix="ratelimit:bench:sl",
    #    )
    #    sliding_counter = SlidingWindowCounterRateLimiter(
    #        name="Sliding Counter",
    #        redis_client=client,
    #        max_requests=10,
    #        window_seconds=1,
    #        key_prefix="ratelimit:bench:sc",
    #    )
    #    token_bucket = TokenBucketRateLimiter(
    #        name="Token Bucket",
    #        redis_client=client,
    #        capacity=15,        # burst tolerance
    #        refill_rate=10.0,   # sustained rate
    #        key_prefix="ratelimit:bench:tb",
    #    )
    #
    #    limiters = [fixed_window, sliding_log, sliding_counter, token_bucket]
    #
    # 3. CLEAN UP stale keys:
    #    for prefix in ["ratelimit:bench:fw", "ratelimit:bench:sl",
    #                    "ratelimit:bench:sc", "ratelimit:bench:tb"]:
    #        keys = await client.keys(f"{prefix}:*")
    #        if keys:
    #            await client.delete(*keys)
    #
    # 4. RUN TRAFFIC PATTERN through all limiters simultaneously:
    #
    #    start_time = time.time()
    #
    #    Phase 1 -- Burst:
    #    print("Phase 1: Burst of 25 requests...")
    #    for i in range(25):
    #        for limiter in limiters:
    #            await limiter.allow("bench-user")
    #        await asyncio.sleep(0.02)
    #
    #    Phase 2 -- Pause:
    #    print("Phase 2: Pause (3 seconds)...")
    #    await asyncio.sleep(3.0)
    #
    #    Phase 3 -- Steady at 8 req/sec:
    #    print("Phase 3: Steady 8 req/sec for 5 seconds...")
    #    for i in range(40):
    #        for limiter in limiters:
    #            await limiter.allow("bench-user")
    #        await asyncio.sleep(0.125)
    #
    #    Phase 4 -- Final burst:
    #    print("Phase 4: Final burst of 15 requests...")
    #    for i in range(15):
    #        for limiter in limiters:
    #            await limiter.allow("bench-user")
    #        await asyncio.sleep(0.02)
    #
    # 5. PRINT RESULTS:
    #    print("\n" + "=" * 60)
    #    print("RESULTS")
    #    print("=" * 60)
    #    print(f"{'Algorithm':<20} {'Allowed':>8} {'Rejected':>8} {'Total':>8}")
    #    print("-" * 48)
    #    for limiter in limiters:
    #        m = limiter.metrics
    #        print(f"{limiter.name:<20} {m.allowed_requests:>8} "
    #              f"{m.rejected_requests:>8} {m.total_requests:>8}")
    #
    # 6. GENERATE COMPARISON PLOT:
    #    import matplotlib
    #    matplotlib.use("Agg")
    #    import matplotlib.pyplot as plt
    #
    #    os.makedirs("plots", exist_ok=True)
    #    fig, ax = plt.subplots(figsize=(14, 7))
    #
    #    For each limiter, plot cumulative allowed requests over time:
    #    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    #    for limiter, color in zip(limiters, colors):
    #        times = [t - start_time for t in limiter.metrics.timestamps_allowed]
    #        if times:
    #            ax.step(times, range(1, len(times) + 1),
    #                    label=f"{limiter.name} ({limiter.metrics.allowed_requests})",
    #                    color=color, linewidth=2, where="post")
    #
    #    ax.set_xlabel("Time (seconds)")
    #    ax.set_ylabel("Cumulative Allowed Requests")
    #    ax.set_title("Rate Limiter Algorithm Comparison")
    #    ax.legend(loc="upper left")
    #    ax.grid(True, alpha=0.3)
    #    fig.tight_layout()
    #    fig.savefig("plots/algorithm_comparison.png", dpi=150)
    #    print("\nPlot saved to plots/algorithm_comparison.png")
    #
    # 7. CLEANUP:
    #    for limiter in limiters:
    #        await limiter.close()
    raise NotImplementedError("TODO(human): implement run_benchmark()")


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
