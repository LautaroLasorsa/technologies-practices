"""Exercise 07: Multi-Replica Rate Limiting Test

Sends concurrent requests through the Nginx load balancer to multiple
FastAPI replicas and verifies that the GLOBAL rate limit is enforced
correctly (not per-replica).

This is the capstone exercise: it proves that distributed rate limiting
works in practice. Without shared Redis state, each replica would allow
its own quota, effectively tripling the rate limit with 3 replicas.

Prerequisites:
    docker compose up --build -d   (starts Redis, 3 API replicas, Nginx)

Run:
    uv run python src/_07_multi_replica_test.py
"""

from __future__ import annotations

import asyncio
import time

import httpx

from src.common import print_header


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8080"  # Nginx load balancer
TOTAL_REQUESTS = 50
CONCURRENCY = 10  # concurrent tasks sending requests
API_KEY = "test-user-multi-replica"


# ---------------------------------------------------------------------------
# Multi-Replica Test
# ---------------------------------------------------------------------------


async def multi_replica_test() -> None:
    """Send concurrent requests through the load balancer and verify limits.

    The test sends TOTAL_REQUESTS requests as fast as possible with
    CONCURRENCY parallel workers. If rate limiting is working correctly
    across replicas, the number of allowed (200) responses should be
    close to the configured capacity, not capacity * num_replicas.
    """
    # Verify services are running
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{BASE_URL}/health", timeout=5.0)
            print(f"Health check: {resp.json()}")
        except httpx.ConnectError:
            print(f"ERROR: Cannot connect to load balancer at {BASE_URL}. "
                  "Is docker compose up?")
            return

    # Flush rate limit state
    import redis.asyncio as aioredis
    redis_client = aioredis.from_url("redis://localhost:6379")
    await redis_client.flushdb()
    await redis_client.aclose()
    print("Flushed Redis rate limit state")

    # Send requests concurrently
    results: list[tuple[int, str, float]] = []
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def send_request(client: httpx.AsyncClient, idx: int) -> None:
        async with semaphore:
            start = time.time()
            resp = await client.get(
                f"{BASE_URL}/api/data",
                headers={"X-API-Key": API_KEY},
                timeout=10.0,
            )
            latency = (time.time() - start) * 1000
            data = resp.json()
            replica = data.get("replica", "unknown")
            results.append((resp.status_code, replica, latency))

    async with httpx.AsyncClient() as client:
        print(f"\nSending {TOTAL_REQUESTS} requests with "
              f"concurrency={CONCURRENCY}...")
        start = time.time()
        tasks = [send_request(client, i) for i in range(TOTAL_REQUESTS)]
        await asyncio.gather(*tasks)
        elapsed = time.time() - start
        print(f"Completed in {elapsed:.2f}s")

    # Analyze results
    allowed = sum(1 for s, _, _ in results if s == 200)
    rejected = sum(1 for s, _, _ in results if s == 429)
    errors = sum(1 for s, _, _ in results if s not in (200, 429))

    from collections import Counter
    replica_counts = Counter(r for _, r, _ in results)

    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"  Total requests:  {TOTAL_REQUESTS}")
    print(f"  Allowed (200):   {allowed}")
    print(f"  Rejected (429):  {rejected}")
    if errors:
        print(f"  Errors:          {errors}")
    print(f"\n  Requests per replica:")
    for replica, count in sorted(replica_counts.items()):
        print(f"    {replica}: {count} requests")

    latencies = [lat for _, _, lat in results]
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    print(f"\n  Avg latency: {avg_latency:.1f}ms")
    print(f"  Max latency: {max_latency:.1f}ms")

    # Validate distributed behavior
    print(f"\n{'='*50}")
    print("VALIDATION")
    print(f"{'='*50}")
    capacity = 20  # matches docker-compose.yml RATE_LIMIT_CAPACITY

    if allowed <= capacity + 5:
        print(f"  PASS: {allowed} allowed <= {capacity}+5 (global limit)")
        print("  Rate limiting is correctly distributed across replicas!")
    else:
        print(f"  FAIL: {allowed} allowed > {capacity}+5")
        print("  Rate limiting may NOT be working across replicas!")

    if len(replica_counts) > 1:
        print(f"  PASS: Requests were distributed across "
              f"{len(replica_counts)} replicas")
    else:
        print("  WARNING: All requests went to a single replica")
        print("  (Nginx may need more connections to distribute)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print_header("Exercise 07: Multi-Replica Rate Limiting Test")
    print("Prerequisites:")
    print("  - docker compose up --build -d  (3 API replicas + Redis + Nginx)")
    print()
    asyncio.run(multi_replica_test())


if __name__ == "__main__":
    main()
