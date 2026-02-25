# /// script
# requires-python = ">=3.12"
# dependencies = ["redis>=5.0"]
# ///
"""Redis Hashes -- Object Storage Pattern.

Demonstrates:
  - HSET (single and multi-field), HGET, HGETALL
  - HDEL, HEXISTS, HLEN
  - HINCRBY / HINCRBYFLOAT for field-level atomic increments
  - Hash as the idiomatic Redis pattern for storing structured objects

Run after starting Redis:
    uv run 04_hashes.py
"""

from __future__ import annotations

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


def hash_as_object(r: redis.Redis) -> None:
    """Exercise: Redis Hash for structured object storage.

    TODO(human): Implement this function.

    Hashes are the idiomatic way to represent objects in Redis. Instead of
    serializing a user profile to JSON and storing it as a string under one
    key, you store each field separately in a hash. This gives you:

      - Partial reads: HGET one field without loading the entire object
      - Partial writes: HSET one field without rewriting the entire object
      - Atomic field increments: HINCRBY on a counter field (e.g., "login_count")
      - Memory efficiency: small hashes use ziplist encoding, which is very compact

    The trade-off: hashes don't support nested structures. A hash field's
    value is always a string. For nested objects, you either flatten the
    structure (e.g., "address:city", "address:zip") or serialize sub-objects
    to JSON strings.

    Steps:

    1. Create a user profile with multiple fields using HSET:
           r.hset("user:1001", mapping={
               "name": "Alice",
               "email": "alice@example.com",
               "age": "30",
               "city": "Buenos Aires",
               "login_count": "0",
           })
       HSET with mapping= sets multiple fields at once (equivalent to the
       deprecated HMSET). Returns the number of NEW fields created (not
       updated). Print the return value.

    2. Get a single field with HGET:
           name = r.hget("user:1001", "name")
       Returns the field value as a string, or None if the field doesn't
       exist. Print the name.

    3. Get all fields and values with HGETALL:
           profile = r.hgetall("user:1001")
       Returns a Python dict {field: value}. All values are strings.
       Print the entire profile.

    4. Check if a field exists with HEXISTS:
           has_email = r.hexists("user:1001", "email")
           has_phone = r.hexists("user:1001", "phone")
       Returns True/False. Print both results.

    5. Get the number of fields with HLEN:
           field_count = r.hlen("user:1001")
       O(1) operation. Print it.

    6. Atomic field increment with HINCRBY:
           new_count = r.hincrby("user:1001", "login_count", 1)
       HINCRBY atomically increments the integer value of a hash field.
       Returns the new value. Increment login_count 3 times in a loop
       and print the result each time.

    7. Float increment with HINCRBYFLOAT:
           r.hset("user:1001", "balance", "100.00")
           new_balance = r.hincrbyfloat("user:1001", "balance", -25.50)
       HINCRBYFLOAT works on float-valued fields. Print the new balance.

    8. Delete a field with HDEL:
           deleted = r.hdel("user:1001", "city")
       Returns the number of fields removed. Print it. Then HGETALL
       again to see the updated profile.

    9. Get only the field names with HKEYS, and only values with HVALS:
           fields = r.hkeys("user:1001")
           values = r.hvals("user:1001")
       Print both. HKEYS and HVALS are O(N) where N is the number of fields.

    10. Create a second user and demonstrate independence:
            r.hset("user:1002", mapping={
                "name": "Bob",
                "email": "bob@example.com",
                "age": "25",
                "login_count": "0",
            })
        GET user:1002's name and user:1001's name side by side to show
        that each hash key is a separate namespace.

    Docs:
      - HSET: https://redis.io/docs/latest/commands/hset/
      - HGETALL: https://redis.io/docs/latest/commands/hgetall/
      - HINCRBY: https://redis.io/docs/latest/commands/hincrby/
    """
    raise NotImplementedError("TODO(human): implement hash_as_object")


# -- Orchestration (boilerplate) -------------------------------------------


def main() -> None:
    r = get_client()
    r.ping()
    print("Connected to Redis!")

    cleanup(r, "user:")

    section("Exercise: Hash as Object Storage")
    hash_as_object(r)

    print("\n--- All hash exercises completed ---")


if __name__ == "__main__":
    main()
