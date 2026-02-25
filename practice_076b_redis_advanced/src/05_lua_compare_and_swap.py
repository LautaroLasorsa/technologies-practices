# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "redis>=5.0",
# ]
# ///
"""Atomic Compare-and-Swap with Lua Scripts.

Demonstrates optimistic concurrency control using a Lua script that
atomically reads, compares, and conditionally writes a Redis key. This
is the Redis equivalent of the CAS (compare_exchange) instruction in
lock-free programming.

Run with Redis running (docker compose up -d):
    uv run src/05_lua_compare_and_swap.py
"""

from __future__ import annotations

import asyncio

import redis.asyncio as aioredis


# ── Configuration ────────────────────────────────────────────────────

REDIS_URL = "redis://localhost:6379/0"

# Lua script for atomic compare-and-swap.
#
# Reads the current value of KEYS[1], compares it with ARGV[1] (expected).
# If they match, sets KEYS[1] to ARGV[2] (desired) and returns 1 (success).
# If they don't match, returns 0 (conflict) and does NOT modify the key.
#
# If the key does not exist and ARGV[1] is the empty string "", the CAS
# succeeds (allows initializing a key that doesn't exist yet).
#
# KEYS[1] = the key to compare-and-swap
# ARGV[1] = expected current value (or "" for "key does not exist")
# ARGV[2] = desired new value
CAS_LUA = """
local key = KEYS[1]
local expected = ARGV[1]
local desired = ARGV[2]

local current = redis.call("GET", key)

-- Handle the "key does not exist" case:
-- If expected is "" and current is false (nil), that's a match.
if current == false then
    if expected == "" then
        redis.call("SET", key, desired)
        return 1
    else
        return 0
    end
end

-- Normal case: compare current with expected
if current == expected then
    redis.call("SET", key, desired)
    return 1
else
    return 0
end
"""


# ── TODO(human): Implement these functions ───────────────────────────


async def compare_and_swap(
    r: aioredis.Redis,
    key: str,
    expected: str,
    desired: str,
) -> bool:
    """Perform an atomic compare-and-swap on a Redis key.

    # ── Exercise Context ──────────────────────────────────────────────────
    # Compare-and-swap (CAS) is the fundamental building block of lock-free
    # concurrency. It is the Redis equivalent of:
    #   - C++: std::atomic::compare_exchange_strong()
    #   - Rust: AtomicU64::compare_exchange()
    #   - Java: AtomicReference.compareAndSet()
    #
    # The pattern: read the current value, compute the new value, attempt
    # to write it only if nobody else modified it in the meantime. If the
    # value changed (conflict), retry from the beginning.
    #
    # Why not just GET + SET?
    #   Between your GET and SET, another client can modify the value.
    #   Your SET would overwrite their change -- a lost update. The Lua
    #   script prevents this by making the read-compare-write atomic.
    #
    # CAS vs distributed locking:
    #   - CAS: optimistic (assume no conflict, retry if wrong). Best when
    #     conflicts are rare. No lock overhead, no deadlock risk.
    #   - Locking: pessimistic (assume conflict, hold exclusive access).
    #     Best when conflicts are frequent. Adds lock acquisition latency.
    # ──────────────────────────────────────────────────────────────────────

    TODO(human): Execute the CAS Lua script and return the result.

    Steps:
      1. Execute the CAS_LUA script via r.eval():
             result = await r.eval(CAS_LUA, 1, key, expected, desired)

         Parameters:
           - CAS_LUA: the Lua script source code
           - 1: number of KEYS arguments
           - key: becomes KEYS[1] in the Lua script
           - expected: becomes ARGV[1] (the value we expect to find)
           - desired: becomes ARGV[2] (the value we want to write)

      2. Return True if result == 1 (CAS succeeded),
         False if result == 0 (CAS failed -- value was different).

    Args:
        r: Async Redis client.
        key: The Redis key to update.
        expected: The value we expect the key to currently hold.
                  Use "" (empty string) to match "key does not exist."
        desired: The value to set if the current value matches expected.

    Returns:
        True if the swap succeeded, False if there was a conflict.

    Docs: https://redis-py.readthedocs.io/en/stable/commands.html#redis.commands.core.CoreCommands.eval
    """
    raise NotImplementedError("TODO(human): Implement compare_and_swap")


async def cas_increment(
    r: aioredis.Redis,
    key: str,
    max_retries: int = 100,
) -> int:
    """Increment an integer stored in Redis using CAS in a retry loop.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise uses CAS to build a lock-free counter -- the same
    # pattern used in lock-free data structures (Michael-Scott queue,
    # Treiber stack, etc.). The retry loop is called a "CAS loop" or
    # "optimistic retry loop."
    #
    # The pattern:
    #   1. Read the current value
    #   2. Compute the new value (current + 1)
    #   3. CAS(expected=current, desired=new)
    #   4. If CAS fails (another client modified the value), go to step 1
    #
    # This is functionally equivalent to Redis INCR, but demonstrates
    # the CAS pattern that works for ANY arbitrary update (not just +1).
    # In practice, you would use INCR for simple counters; CAS is for
    # complex conditional updates where INCR is insufficient.
    #
    # The max_retries bound prevents infinite loops under extreme
    # contention. In production, consider exponential backoff between
    # retries for high-contention keys.
    # ──────────────────────────────────────────────────────────────────────

    TODO(human): Implement a CAS-based increment loop.

    Steps:
      1. Loop up to max_retries times:

         a. Read the current value:
                current_str = await r.get(key)

         b. Determine the current integer value:
                if current_str is None:
                    current_val = 0
                    expected = ""     # "key does not exist" sentinel
                else:
                    current_val = int(current_str)
                    expected = current_str

         c. Compute the desired new value:
                desired = str(current_val + 1)

         d. Attempt the CAS:
                success = await compare_and_swap(r, key, expected, desired)

         e. If success is True, return current_val + 1 (the new value).

         f. If success is False (conflict), continue the loop (retry).

      2. If all retries exhausted, raise RuntimeError:
             raise RuntimeError(f"CAS increment failed after {max_retries} retries")

    Args:
        r: Async Redis client.
        key: The Redis key holding the integer counter.
        max_retries: Maximum number of CAS attempts before giving up.

    Returns:
        The new counter value after incrementing.

    Raises:
        RuntimeError: If max_retries is exceeded (extreme contention).
    """
    raise NotImplementedError("TODO(human): Implement cas_increment")


# ── Demonstration (boilerplate) ──────────────────────────────────────


async def demo_compare_and_swap() -> None:
    """Demonstrate CAS operations and lock-free counter."""
    r = aioredis.from_url(REDIS_URL, decode_responses=True)

    try:
        print("=== Atomic Compare-and-Swap (Lua Script) ===\n")

        # --- Test 1: Basic CAS ---
        print("--- Test 1: Basic CAS Operations ---")
        key = "cas:test:basic"
        await r.delete(key)

        # CAS on non-existent key (expected="" means "does not exist")
        ok = await compare_and_swap(r, key, "", "hello")
        print(f"  CAS '' -> 'hello': {ok} (expected: True)")

        val = await r.get(key)
        print(f"  Current value: {val}")

        # CAS with correct expectation
        ok = await compare_and_swap(r, key, "hello", "world")
        print(f"  CAS 'hello' -> 'world': {ok} (expected: True)")

        # CAS with wrong expectation (conflict)
        ok = await compare_and_swap(r, key, "hello", "oops")
        print(f"  CAS 'hello' -> 'oops': {ok} (expected: False, value is 'world')")

        val = await r.get(key)
        print(f"  Final value: {val} (expected: 'world')")

        # --- Test 2: CAS-based lock-free counter ---
        print("\n--- Test 2: Lock-Free Counter (CAS Loop) ---")
        counter_key = "cas:test:counter"
        await r.delete(counter_key)

        # Sequential increments
        for i in range(5):
            new_val = await cas_increment(r, counter_key)
            print(f"  Increment {i + 1}: counter = {new_val}")

        # --- Test 3: Concurrent CAS increments ---
        print("\n--- Test 3: Concurrent CAS Increments (20 tasks) ---")
        concurrent_key = "cas:test:concurrent"
        await r.delete(concurrent_key)

        num_tasks = 20
        tasks = [cas_increment(r, concurrent_key) for _ in range(num_tasks)]
        results = await asyncio.gather(*tasks)

        final_val = await r.get(concurrent_key)
        print(f"  {num_tasks} concurrent increments completed")
        print(f"  Final counter value: {final_val} (expected: {num_tasks})")
        print(f"  Correct: {int(final_val) == num_tasks}")
        print(f"  All results: {sorted(results)}")

        # --- Test 4: CAS for conditional update ---
        print("\n--- Test 4: Conditional Update (status transitions) ---")
        status_key = "cas:test:status"
        await r.set(status_key, "pending")

        # Only transition if current status is "pending"
        ok = await compare_and_swap(r, status_key, "pending", "processing")
        print(f"  pending -> processing: {ok} (expected: True)")

        # Duplicate transition attempt (already "processing")
        ok = await compare_and_swap(r, status_key, "pending", "processing")
        print(f"  pending -> processing (again): {ok} (expected: False)")

        # Valid transition from current state
        ok = await compare_and_swap(r, status_key, "processing", "completed")
        print(f"  processing -> completed: {ok} (expected: True)")

        val = await r.get(status_key)
        print(f"  Final status: {val}")

    finally:
        await r.aclose()


async def main() -> None:
    print("=" * 60)
    print(" Practice 076b: Lua Compare-and-Swap")
    print("=" * 60)
    print()
    await demo_compare_and_swap()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
