# /// script
# requires-python = ">=3.12"
# dependencies = ["redis>=5.0"]
# ///
"""Redis Key Expiry & TTL Management.

Demonstrates:
  - SET with EX (seconds) and PX (milliseconds) for atomic set+expire
  - EXPIRE / PEXPIRE to add TTL to existing keys
  - TTL / PTTL to inspect remaining time
  - PERSIST to remove a TTL
  - Observing key expiration in real time

Run after starting Redis:
    uv run 05_expiry.py
"""

from __future__ import annotations

import time

import redis


def get_client() -> redis.Redis:
    return redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def cleanup(r: redis.Redis, prefix: str) -> None:
    for key in r.scan_iter(f"{prefix}*"):
        r.delete(key)


# -- TODO(human): Implement these functions --------------------------------


def expiry_and_ttl(r: redis.Redis) -> None:
    """Exercise: Key expiry, TTL inspection, and PERSIST.

    TODO(human): Implement this function.

    Key expiry is the foundation of all caching in Redis. Every cache
    entry should have a TTL to prevent stale data and unbounded memory
    growth. Understanding how TTLs work -- and the subtle behaviors around
    overwriting keys -- is critical for production caching.

    Redis expires keys using two mechanisms:
      1. Lazy expiration: when a client reads an expired key, Redis
         checks the TTL and deletes it before returning nil.
      2. Active expiration: a background task samples 20 random keys
         with TTLs 10 times per second. If >25% are expired, it repeats.

    Steps:

    1. SET a key with an expiry using the EX parameter:
           r.set("session:abc123", "user-data-here", ex=10)
       EX sets the TTL in seconds. The SET and EXPIRE happen atomically
       in a single command. Print "Key set with 10s TTL".

    2. Check the remaining TTL with TTL:
           remaining = r.ttl("session:abc123")
       TTL returns:
         - Positive integer: seconds remaining
         - -1: key exists but has no expiry
         - -2: key does not exist
       Print the remaining TTL.

    3. Check with millisecond precision using PTTL:
           remaining_ms = r.pttl("session:abc123")
       PTTL returns milliseconds remaining. Print it.

    4. SET a key with millisecond TTL:
           r.set("flash:msg", "quick!", px=500)
       PX sets the TTL in milliseconds (here, 500ms = half a second).
       Print PTTL, then sleep 0.6 seconds and try to GET the key:
           time.sleep(0.6)
           value = r.get("flash:msg")
       Print the value (should be None -- the key expired).

    5. Add a TTL to an existing key using EXPIRE:
           r.set("permanent_key", "some value")
           ttl_before = r.ttl("permanent_key")
       Print ttl_before (should be -1, meaning no expiry).
       Then add a 5-second TTL:
           r.expire("permanent_key", 5)
           ttl_after = r.ttl("permanent_key")
       Print ttl_after.

    6. Remove a TTL with PERSIST:
           r.persist("permanent_key")
           ttl_now = r.ttl("permanent_key")
       PERSIST makes the key permanent again (removes the TTL).
       Print ttl_now (should be -1 again).

    7. Demonstrate the overwrite-removes-TTL behavior:
           r.set("cached:item", "v1", ex=30)
           print(f"  TTL after initial SET: {r.ttl('cached:item')}")
           r.set("cached:item", "v2")
           print(f"  TTL after plain SET (no EX): {r.ttl('cached:item')}")
       IMPORTANT: A plain SET (without EX/PX/KEEPTTL) REMOVES the TTL!
       This is a common production bug -- updating a cached value without
       re-specifying the TTL makes it permanent. Print both TTLs to
       demonstrate.

       To preserve the TTL when updating, use SET with KEEPTTL:
           r.set("cached:item2", "v1", ex=30)
           print(f"  TTL before update: {r.ttl('cached:item2')}")
           r.set("cached:item2", "v2", keepttl=True)
           print(f"  TTL after KEEPTTL update: {r.ttl('cached:item2')}")

    8. Watch a key expire in real time:
           r.set("countdown", "watching...", ex=3)
       Loop checking TTL every 0.5 seconds until the key is gone:
           while True:
               ttl = r.ttl("countdown")
               value = r.get("countdown")
               print(f"  TTL={ttl}, exists={value is not None}")
               if ttl == -2:
                   print("  Key has expired and been removed!")
                   break
               time.sleep(0.5)

    Docs:
      - SET (EX/PX/KEEPTTL): https://redis.io/docs/latest/commands/set/
      - TTL: https://redis.io/docs/latest/commands/ttl/
      - EXPIRE: https://redis.io/docs/latest/commands/expire/
      - PERSIST: https://redis.io/docs/latest/commands/persist/
    """
    raise NotImplementedError("TODO(human): implement expiry_and_ttl")


# -- Orchestration (boilerplate) -------------------------------------------


def main() -> None:
    r = get_client()
    r.ping()
    print("Connected to Redis!")

    cleanup(r, "session:")
    cleanup(r, "flash:")
    cleanup(r, "permanent_key")
    cleanup(r, "cached:")
    cleanup(r, "countdown")

    section("Exercise: Key Expiry & TTL")
    expiry_and_ttl(r)

    print("\n--- All expiry exercises completed ---")


if __name__ == "__main__":
    main()
