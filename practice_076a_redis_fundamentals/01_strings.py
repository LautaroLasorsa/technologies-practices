# /// script
# requires-python = ">=3.12"
# dependencies = ["redis>=5.0"]
# ///
"""Redis Strings & Atomic Counters.

Demonstrates:
  - SET / GET / DEL basics
  - Conditional SET with NX (only if not exists) and XX (only if exists)
  - MSET / MGET for batch key-value operations
  - Atomic INCR / DECR / INCRBY / INCRBYFLOAT for thread-safe counters

Run after starting Redis:
    uv run 01_strings.py
"""

from __future__ import annotations

import redis


def get_client() -> redis.Redis:
    """Create a Redis client connected to the local Docker container.

    decode_responses=True makes redis-py return Python str instead of bytes.
    This is more convenient for most use cases (you don't need .decode()
    on every GET), but means you can't store/retrieve raw binary data.
    """
    return redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


# -- Boilerplate helpers --------------------------------------------------


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def cleanup(r: redis.Redis, prefix: str) -> None:
    """Delete all keys matching a prefix (for clean demo runs)."""
    for key in r.scan_iter(f"{prefix}*"):
        r.delete(key)


# -- TODO(human): Implement these functions --------------------------------


def basic_string_operations(r: redis.Redis) -> None:
    """Exercise 1: Basic string GET/SET and batch operations.

    TODO(human): Implement this function.

    This exercise teaches the most fundamental Redis operations. Every Redis
    interaction starts with strings -- they are the building block for
    caching, session storage, and configuration values.

    Steps:

    1. SET a key "greeting" to the value "Hello, Redis!":
           r.set("greeting", "Hello, Redis!")
       SET always succeeds (creates or overwrites). It returns True on success.

    2. GET the key "greeting" and print it:
           value = r.get("greeting")
       GET returns the value as a string (because decode_responses=True),
       or None if the key doesn't exist.

    3. Conditional SET with NX (Not eXists):
           result = r.set("greeting", "Overwritten?", nx=True)
       NX makes SET behave like "insert only" -- it only sets the key if
       it does NOT already exist. Returns True if set, None if key existed.
       Print the result and then GET "greeting" again to confirm it wasn't
       overwritten.

    4. Conditional SET with XX (only if eXists):
           result = r.set("greeting", "Updated!", xx=True)
       XX makes SET behave like "update only" -- it only sets the key if
       it ALREADY exists. Returns True if set, None if key didn't exist.
       Print the result and GET "greeting" to confirm it was updated.

    5. Try XX on a non-existent key:
           result = r.set("nonexistent", "value", xx=True)
       Print the result (should be None) to demonstrate XX fails on new keys.

    6. Batch SET with MSET -- set multiple keys at once:
           r.mset({"city": "Buenos Aires", "country": "Argentina", "lang": "Python"})
       MSET is atomic -- all keys are set in a single operation. There is
       no MSET equivalent of NX/XX (use MSETNX for all-or-nothing NX).

    7. Batch GET with MGET -- retrieve multiple keys at once:
           values = r.mget("city", "country", "lang", "nonexistent")
       MGET returns a list of values in the same order as the keys.
       Non-existent keys return None in the list. Print the results.

    8. DEL a key:
           deleted_count = r.delete("greeting")
       DEL returns the number of keys that were removed (0 if key didn't exist).
       Print the count, then GET "greeting" to confirm it's gone (returns None).

    Docs:
      - SET: https://redis.io/docs/latest/commands/set/
      - GET: https://redis.io/docs/latest/commands/get/
      - MSET: https://redis.io/docs/latest/commands/mset/
      - MGET: https://redis.io/docs/latest/commands/mget/
      - DEL: https://redis.io/docs/latest/commands/del/
    """
    raise NotImplementedError("TODO(human): implement basic_string_operations")


def atomic_counters(r: redis.Redis) -> None:
    """Exercise 2: Atomic counters with INCR/DECR.

    TODO(human): Implement this function.

    This exercise teaches why Redis counters are powerful. In a typical
    database or application server, incrementing a counter requires:
      1. READ the current value
      2. ADD 1 in application code
      3. WRITE the new value back
    If two clients do this concurrently, one increment can be lost (lost
    update problem). Redis's INCR is atomic because the single-threaded
    event loop executes the read-modify-write as one indivisible operation.

    Steps:

    1. SET a counter key "page_views" to "0":
           r.set("page_views", 0)
       Even though we pass an int, Redis stores it as the string "0".
       Redis detects that the string is a valid 64-bit integer and stores
       it in an optimized integer encoding internally.

    2. INCR the counter 5 times in a loop, printing the return value each time:
           for i in range(5):
               new_val = r.incr("page_views")
               print(f"  INCR #{i+1}: {new_val}")
       INCR atomically increments by 1 and returns the NEW value.
       If the key doesn't exist, INCR initializes it to 0 first, then
       increments (so you don't need to SET before INCR).

    3. DECR the counter twice:
           new_val = r.decr("page_views")
       DECR atomically decrements by 1. Print the result after each call.

    4. INCRBY -- increment by an arbitrary integer:
           new_val = r.incrby("page_views", 10)
       Atomically adds 10 to the counter. There's also DECRBY for subtraction.
       Print the result.

    5. INCRBYFLOAT -- increment by a floating-point number:
           new_val = r.incrbyfloat("page_views", 1.5)
       After this, the value is no longer a pure integer -- it's stored as
       a string representation of the float (e.g., "16.5"). Print the result.
       Note: There is no DECRBYFLOAT; use a negative value with INCRBYFLOAT.

    6. Try INCR on a non-numeric string:
           r.set("name", "Alice")
       Then try:
           try:
               r.incr("name")
           except redis.ResponseError as e:
               print(f"  Error: {e}")
       This demonstrates that INCR only works on values that are valid
       integers. Redis returns an error: "value is not an integer or out
       of range".

    Docs:
      - INCR: https://redis.io/docs/latest/commands/incr/
      - INCRBY: https://redis.io/docs/latest/commands/incrby/
      - INCRBYFLOAT: https://redis.io/docs/latest/commands/incrbyfloat/
    """
    raise NotImplementedError("TODO(human): implement atomic_counters")


# -- Orchestration (boilerplate) -------------------------------------------


def main() -> None:
    r = get_client()
    r.ping()
    print("Connected to Redis!")

    # Clean up keys from previous runs
    cleanup(r, "greeting")
    cleanup(r, "city")
    cleanup(r, "country")
    cleanup(r, "lang")
    cleanup(r, "nonexistent")
    cleanup(r, "page_views")
    cleanup(r, "name")

    section("Exercise 1: Basic String Operations")
    basic_string_operations(r)

    section("Exercise 2: Atomic Counters")
    atomic_counters(r)

    print("\n--- All string exercises completed ---")


if __name__ == "__main__":
    main()
