"""End-to-end test for the Task Processing Pipeline.

Run this after `docker compose up --build` to verify the full pipeline works.

Usage:
    pip install httpx   (or: pip install -r scripts/requirements.txt)
    python scripts/test_pipeline.py
"""

from __future__ import annotations

import sys
import time

import httpx

BASE_URL = "http://localhost"
API_URL = f"{BASE_URL}/api"
DASHBOARD_URL = f"{BASE_URL}/dashboard"

NUM_TASKS = 5
POLL_TIMEOUT = 60  # seconds


def submit_tasks(client: httpx.Client) -> list[str]:
    """Submit NUM_TASKS tasks and return their IDs."""
    task_ids: list[str] = []
    for i in range(NUM_TASKS):
        resp = client.post(
            f"{API_URL}/tasks",
            json={
                "name": f"test-task-{i}",
                "payload": {"index": i, "data": f"sample-{i}"},
                "priority": i % 3,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        task_ids.append(data["task_id"])
        print(f"  Submitted task {i+1}/{NUM_TASKS}: {data['task_id']}")
    return task_ids


def wait_for_completion(client: httpx.Client, task_ids: list[str]) -> bool:
    """Poll until all tasks are completed or failed, or timeout."""
    start = time.time()
    remaining = set(task_ids)

    while remaining and (time.time() - start) < POLL_TIMEOUT:
        for task_id in list(remaining):
            resp = client.get(f"{API_URL}/tasks/{task_id}")
            if resp.status_code == 200:
                status = resp.json().get("status", "")
                if status in ("completed", "failed"):
                    remaining.discard(task_id)
                    print(f"  Task {task_id[:8]}... -> {status}")
        if remaining:
            time.sleep(2)

    return len(remaining) == 0


def check_dashboard(client: httpx.Client) -> dict:
    """Fetch stats from the dashboard."""
    resp = client.get(f"{DASHBOARD_URL}/stats")
    resp.raise_for_status()
    return resp.json()


def check_stats(client: httpx.Client) -> dict:
    """Fetch stats from the API gateway."""
    resp = client.get(f"{API_URL}/stats")
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    print("=" * 60)
    print("Task Pipeline -- End-to-End Test")
    print("=" * 60)

    with httpx.Client(timeout=10.0) as client:
        # 1. Health checks
        print("\n[1] Checking service health...")
        for name, url in [("API Gateway", f"{API_URL}/health"), ("Dashboard", f"{DASHBOARD_URL}/health")]:
            try:
                resp = client.get(url)
                status = resp.json().get("status", "unknown")
                print(f"  {name}: {status}")
            except httpx.ConnectError:
                print(f"  {name}: UNREACHABLE -- is docker compose up?")
                sys.exit(1)

        # 2. Submit tasks
        print(f"\n[2] Submitting {NUM_TASKS} tasks...")
        task_ids = submit_tasks(client)

        # 3. Wait for processing
        print("\n[3] Waiting for tasks to be processed...")
        all_done = wait_for_completion(client, task_ids)
        if not all_done:
            print(f"  TIMEOUT: Not all tasks finished within {POLL_TIMEOUT}s")

        # 4. Final stats
        print("\n[4] Pipeline stats (API Gateway):")
        api_stats = check_stats(client)
        for key, value in api_stats.items():
            print(f"  {key}: {value}")

        print("\n[5] Dashboard stats:")
        dash_stats = check_dashboard(client)
        for key, value in dash_stats.items():
            print(f"  {key}: {value}")

        # 6. Summary
        print("\n" + "=" * 60)
        completed = api_stats.get("total_completed", 0)
        failed = api_stats.get("total_failed", 0)
        total = completed + failed
        print(f"Result: {completed} completed, {failed} failed out of {total} processed")
        if total >= NUM_TASKS:
            print("SUCCESS: All tasks were processed.")
        else:
            print(f"WARNING: Expected {NUM_TASKS} tasks processed, got {total}.")
        print("=" * 60)


if __name__ == "__main__":
    main()
