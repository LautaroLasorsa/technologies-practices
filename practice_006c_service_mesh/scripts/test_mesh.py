"""Load test for the service mesh.

Sends requests through the mesh and reports metrics:
  - Success/error rates
  - Latency distribution
  - Envoy stats (retries, circuit breaker activations)

Usage:
  python scripts/test_mesh.py
  python scripts/test_mesh.py --requests 50 --concurrency 5
"""

import argparse
import statistics
import sys
import time

import httpx

FRONTEND_URL = "http://localhost:10000"
FRONTEND_ADMIN = "http://localhost:9901"
BACKEND_ADMIN = "http://localhost:9902"


def send_requests(n: int, concurrency: int) -> list[dict]:
    """Send n requests to the frontend, collecting results."""
    results = []

    with httpx.Client(timeout=10.0) as client:
        for i in range(n):
            start = time.monotonic()
            try:
                resp = client.get(f"{FRONTEND_URL}/api/data")
                elapsed_ms = (time.monotonic() - start) * 1000
                results.append({
                    "request": i + 1,
                    "status": resp.status_code,
                    "latency_ms": round(elapsed_ms, 2),
                    "success": 200 <= resp.status_code < 300,
                })
            except httpx.RequestError as exc:
                elapsed_ms = (time.monotonic() - start) * 1000
                results.append({
                    "request": i + 1,
                    "status": 0,
                    "latency_ms": round(elapsed_ms, 2),
                    "success": False,
                    "error": str(exc),
                })

    return results


def print_results(results: list[dict]) -> None:
    """Print summary statistics."""
    total = len(results)
    successes = sum(1 for r in results if r["success"])
    failures = total - successes
    latencies = [r["latency_ms"] for r in results]

    print("\n" + "=" * 60)
    print("Service Mesh Load Test Results")
    print("=" * 60)
    print(f"  Total requests:  {total}")
    print(f"  Successful:      {successes} ({100 * successes / total:.1f}%)")
    print(f"  Failed:          {failures} ({100 * failures / total:.1f}%)")
    print()
    print(f"  Latency (ms):")
    print(f"    Min:    {min(latencies):.2f}")
    print(f"    Max:    {max(latencies):.2f}")
    print(f"    Mean:   {statistics.mean(latencies):.2f}")
    print(f"    Median: {statistics.median(latencies):.2f}")
    if len(latencies) >= 2:
        print(f"    StdDev: {statistics.stdev(latencies):.2f}")
    print()

    # Status code breakdown
    status_counts: dict[int, int] = {}
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    print("  Status codes:")
    for status, count in sorted(status_counts.items()):
        label = "connection error" if status == 0 else str(status)
        print(f"    {label}: {count}")
    print()


def print_envoy_stats() -> None:
    """Fetch and print relevant Envoy stats."""
    print("-" * 60)
    print("Envoy Statistics (frontend sidecar)")
    print("-" * 60)

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{FRONTEND_ADMIN}/stats")
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                patterns = [
                    "upstream_rq_total",
                    "upstream_rq_2xx",
                    "upstream_rq_5xx",
                    "upstream_rq_retry",
                    "upstream_rq_retry_success",
                    "upstream_cx_total",
                    "circuit_breakers",
                    "upstream_rq_pending_overflow",
                ]
                for line in lines:
                    if any(p in line for p in patterns):
                        # Only show backend_cluster stats (skip frontend_app)
                        if "backend_cluster" in line or "circuit" in line:
                            print(f"  {line}")
            print()
    except httpx.RequestError:
        print("  (Could not reach Envoy admin interface)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Service mesh load test")
    parser.add_argument("--requests", "-n", type=int, default=20, help="Number of requests")
    parser.add_argument("--concurrency", "-c", type=int, default=1, help="Concurrency level (unused, sequential)")
    args = parser.parse_args()

    # Check connectivity
    print("Checking mesh connectivity...")
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{FRONTEND_URL}/health")
            if resp.status_code != 200:
                print(f"ERROR: Frontend health check failed: {resp.status_code}")
                sys.exit(1)
            print(f"  Frontend:  OK ({resp.json()})")

            resp = client.get(f"{FRONTEND_URL}/api/data")
            if resp.status_code != 200:
                print(f"  WARNING: Backend via mesh returned {resp.status_code}")
            else:
                print(f"  Mesh flow: OK (frontend -> envoy -> backend)")
    except httpx.RequestError as exc:
        print(f"ERROR: Cannot reach frontend: {exc}")
        print("Make sure 'docker compose up' is running.")
        sys.exit(1)

    # Run load test
    print(f"\nSending {args.requests} requests through the mesh...")
    results = send_requests(args.requests, args.concurrency)
    print_results(results)
    print_envoy_stats()

    print("=" * 60)
    print("Tip: Set backend to degraded mode and re-run to test retries:")
    print(f'  curl -X POST {FRONTEND_URL}/api/admin/mode/degraded')
    print(f"  python scripts/test_mesh.py -n {args.requests}")
    print()
    print("Check Envoy admin UI:")
    print(f"  Frontend Envoy stats:  {FRONTEND_ADMIN}/stats")
    print(f"  Frontend Envoy clusters: {FRONTEND_ADMIN}/clusters")
    print("=" * 60)


if __name__ == "__main__":
    main()
