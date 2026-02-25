# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "redis>=5.0",
# ]
# ///
"""Distributed Locking with Redis.

Demonstrates the SET NX EX pattern for mutual exclusion across distributed
processes, and the critical Lua-based unlock that prevents releasing another
client's lock.

Run with Redis running (docker compose up -d):
    uv run src/01_distributed_lock.py
"""

from __future__ import annotations

import asyncio
import time
import uuid

import redis.asyncio as aioredis


# ── Configuration ────────────────────────────────────────────────────

REDIS_URL = "redis://localhost:6379/0"
LOCK_KEY = "practice:076b:distributed_lock"
LOCK_TTL_SECONDS = 5

# Lua script for safe lock release.
# This script atomically checks that the lock is still owned by the caller
# (by comparing the stored value with the caller's token) and only then
# deletes the key. Without this atomicity guarantee, a plain DEL could
# remove a lock that was already acquired by a different client after our
# TTL expired.
UNLOCK_LUA = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
else
    return 0
end
"""


# ── TODO(human): Implement these functions ───────────────────────────


async def acquire_lock(
    r: aioredis.Redis,
    lock_key: str,
    token: str,
    ttl_seconds: int,
    retry_interval: float = 0.1,
    max_retries: int = 50,
) -> bool:
    """Attempt to acquire a distributed lock with retries.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches the fundamental distributed lock primitive using
    # Redis SET with NX (Not eXists) and EX (Expire) flags. NX ensures only
    # one client can set the key (mutual exclusion). EX sets a TTL so the
    # lock auto-releases if the holder crashes (deadlock prevention). The
    # token (UUID) stored as the value enables safe release later -- without
    # it, you cannot verify ownership on unlock.
    #
    # This is the single most important Redis pattern for distributed systems.
    # Every distributed lock library (python-redis-lock, pottery, etc.) uses
    # this exact primitive under the hood.
    # ──────────────────────────────────────────────────────────────────────

    TODO(human): Implement distributed lock acquisition with retry loop.

    Steps:
      1. In a loop (up to max_retries):
         a. Try to acquire the lock:
                result = await r.set(lock_key, token, nx=True, ex=ttl_seconds)
            - nx=True: SET only if the key does NOT exist (mutual exclusion)
            - ex=ttl_seconds: automatically expire the key after ttl_seconds
            - token: a unique UUID string identifying this lock holder

         b. If result is True (lock acquired), return True immediately.

         c. If result is None/False (lock held by someone else), sleep for
            retry_interval seconds and try again:
                await asyncio.sleep(retry_interval)

      2. If all retries are exhausted, return False (failed to acquire).

    Args:
        r: Async Redis client.
        lock_key: The Redis key used as the lock.
        token: A unique identifier for this lock holder (UUID).
        ttl_seconds: Lock expiration time in seconds.
        retry_interval: Seconds to wait between retry attempts.
        max_retries: Maximum number of acquisition attempts.

    Returns:
        True if the lock was acquired, False otherwise.

    Hints:
      - r.set() returns True on success, None on NX failure
      - The retry loop simulates "spinning" on the lock, like a spinlock
        in systems programming but with async sleep instead of busy-wait
      - In production, add jitter to retry_interval to prevent thundering
        herd when many clients retry simultaneously

    Docs: https://redis-py.readthedocs.io/en/stable/commands.html#redis.commands.core.CoreCommands.set
    """
    raise NotImplementedError("TODO(human): Implement acquire_lock")


async def release_lock(
    r: aioredis.Redis,
    lock_key: str,
    token: str,
) -> bool:
    """Safely release a distributed lock using a Lua script.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches why a plain DEL command is UNSAFE for lock
    # release and why Lua scripts are necessary for atomic check-and-delete.
    #
    # The danger scenario:
    #   1. Client A acquires lock with TTL=5s
    #   2. Client A's operation takes 6 seconds (longer than TTL)
    #   3. Lock expires automatically after 5s
    #   4. Client B acquires the (now free) lock
    #   5. Client A finishes and calls DEL -- this deletes Client B's lock!
    #
    # The Lua script solves this by atomically checking that the stored
    # token matches the caller's token before deleting. Since Redis executes
    # Lua scripts atomically (no interleaving), the check-and-delete is
    # indivisible.
    #
    # This is the canonical example of "why Lua exists in Redis." You will
    # see this exact pattern in every Redis locking library and in the
    # official Redis documentation.
    # ──────────────────────────────────────────────────────────────────────

    TODO(human): Implement safe lock release using the UNLOCK_LUA script.

    Steps:
      1. Execute the UNLOCK_LUA script via r.eval():
             result = await r.eval(UNLOCK_LUA, 1, lock_key, token)
         Parameters:
           - UNLOCK_LUA: the Lua script source code (string)
           - 1: number of KEYS arguments (we have one key: lock_key)
           - lock_key: becomes KEYS[1] inside the Lua script
           - token: becomes ARGV[1] inside the Lua script

         The Lua script does:
           - GET KEYS[1] (reads the current lock value)
           - Compares it with ARGV[1] (our token)
           - If they match: DEL KEYS[1] and return 1 (success)
           - If they don't match: return 0 (not our lock)

      2. Return True if result == 1 (we owned and released the lock),
         False otherwise (lock was not ours or did not exist).

    Args:
        r: Async Redis client.
        lock_key: The Redis key used as the lock.
        token: The unique token used when acquiring this lock.

    Returns:
        True if the lock was successfully released, False otherwise.

    Why eval() and not just GET + DEL?
      Because between your GET and DEL, another client could acquire the
      lock. The Lua script executes both atomically -- no interleaving.

    Docs: https://redis-py.readthedocs.io/en/stable/commands.html#redis.commands.core.CoreCommands.eval
    """
    raise NotImplementedError("TODO(human): Implement release_lock")


# ── Demonstration (boilerplate) ──────────────────────────────────────


async def worker(
    worker_id: int,
    r: aioredis.Redis,
    shared_counter: dict[str, int],
    results: list[str],
) -> None:
    """Simulate a worker that needs exclusive access to a shared resource."""
    token = str(uuid.uuid4())
    acquired = await acquire_lock(r, LOCK_KEY, token, LOCK_TTL_SECONDS)

    if not acquired:
        results.append(f"  Worker {worker_id}: FAILED to acquire lock")
        return

    results.append(f"  Worker {worker_id}: ACQUIRED lock (token={token[:8]}...)")

    # Critical section: read-modify-write on shared counter
    current = shared_counter["value"]
    await asyncio.sleep(0.05)  # Simulate work
    shared_counter["value"] = current + 1

    released = await release_lock(r, LOCK_KEY, token)
    results.append(
        f"  Worker {worker_id}: RELEASED lock (success={released}), "
        f"counter={shared_counter['value']}"
    )


async def demo_concurrent_workers() -> None:
    """Demonstrate distributed lock with concurrent workers."""
    r = aioredis.from_url(REDIS_URL, decode_responses=True)

    try:
        # Clean up any stale lock
        await r.delete(LOCK_KEY)

        print("=== Distributed Lock: Concurrent Workers ===\n")
        print(f"Lock key: {LOCK_KEY}")
        print(f"Lock TTL: {LOCK_TTL_SECONDS}s")
        print()

        # --- Test 1: Sequential correctness ---
        print("--- Test 1: 5 concurrent workers with lock ---")
        shared_counter: dict[str, int] = {"value": 0}
        results: list[str] = []

        workers = [
            worker(i, r, shared_counter, results) for i in range(5)
        ]
        await asyncio.gather(*workers)

        for line in results:
            print(line)
        print(f"\n  Final counter value: {shared_counter['value']} (expected: 5)")
        print(f"  Correct: {shared_counter['value'] == 5}")

        # --- Test 2: Lock ownership verification ---
        print("\n--- Test 2: Ownership verification ---")
        token_a = str(uuid.uuid4())
        token_b = str(uuid.uuid4())

        acquired_a = await acquire_lock(r, LOCK_KEY, token_a, LOCK_TTL_SECONDS)
        print(f"  Client A acquired: {acquired_a} (token={token_a[:8]}...)")

        # Client B tries to release Client A's lock -- should fail
        released_b = await release_lock(r, LOCK_KEY, token_b)
        print(f"  Client B tried to release A's lock: {released_b} (should be False)")

        # Client A releases its own lock -- should succeed
        released_a = await release_lock(r, LOCK_KEY, token_a)
        print(f"  Client A released own lock: {released_a} (should be True)")

        # --- Test 3: TTL expiry ---
        print("\n--- Test 3: TTL-based auto-release ---")
        token_c = str(uuid.uuid4())
        short_ttl = 1
        acquired_c = await acquire_lock(r, LOCK_KEY, token_c, short_ttl)
        print(f"  Client C acquired with TTL={short_ttl}s: {acquired_c}")

        ttl = await r.ttl(LOCK_KEY)
        print(f"  TTL remaining: {ttl}s")

        print(f"  Waiting {short_ttl + 0.5}s for lock to expire...")
        await asyncio.sleep(short_ttl + 0.5)

        exists = await r.exists(LOCK_KEY)
        print(f"  Lock still exists after TTL: {bool(exists)} (should be False)")

    finally:
        await r.aclose()


async def main() -> None:
    print("=" * 60)
    print(" Practice 076b: Distributed Locking with Redis")
    print("=" * 60)
    print()
    await demo_concurrent_workers()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
