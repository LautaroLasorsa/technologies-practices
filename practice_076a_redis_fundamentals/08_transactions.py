# /// script
# requires-python = ">=3.12"
# dependencies = ["redis>=5.0"]
# ///
"""Redis Transactions -- MULTI/EXEC and Optimistic Locking with WATCH.

Demonstrates:
  - MULTI/EXEC for atomic command groups (isolation, not rollback)
  - WATCH for optimistic locking (CAS pattern)
  - The difference between transactions and pipelining
  - What happens when a watched key is modified by another client

Run after starting Redis:
    uv run 08_transactions.py
"""

from __future__ import annotations

import threading
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


def basic_transaction(r: redis.Redis) -> None:
    """Exercise 1: Atomic balance transfer with MULTI/EXEC.

    TODO(human): Implement this function.

    Redis transactions (MULTI/EXEC) guarantee ISOLATION: no other client
    can execute commands between your MULTI and EXEC. All queued commands
    execute as an atomic unit. However, they do NOT guarantee ROLLBACK:
    if one command fails (e.g., wrong type), others still execute.

    In redis-py, Pipeline is transactional by default (wraps in MULTI/EXEC).
    Use r.pipeline() for a transaction, r.pipeline(transaction=False)
    for pure pipelining.

    Scenario: Transfer 100 units from account A to account B atomically.
    Without a transaction, another client could read the intermediate state
    (A debited but B not yet credited).

    Steps:

    1. Set up two accounts:
           r.set("account:A", 500)
           r.set("account:B", 300)
       Print initial balances.

    2. Perform an atomic transfer using a transaction:
           pipe = r.pipeline()  # transaction=True by default
           pipe.decrby("account:A", 100)
           pipe.incrby("account:B", 100)
           results = pipe.execute()
       pipe.execute() sends MULTI, then both commands, then EXEC.
       Results is a list: [new_balance_A, new_balance_B].
       Print both results.

    3. Verify the balances:
           balance_a = r.get("account:A")
           balance_b = r.get("account:B")
       Print both. The total should still be 800 (500 + 300).

    4. Demonstrate that FAILED commands don't roll back others:
           r.set("account:C", "not-a-number")
           pipe = r.pipeline()
           pipe.incr("account:A")     # This will succeed
           pipe.incr("account:C")     # This will FAIL (not an integer)
           pipe.incr("account:B")     # This will still succeed!
           try:
               results = pipe.execute()
           except redis.ResponseError:
               # redis-py raises on the first error in the results
               pass

       To handle per-command errors without raising, use:
           pipe = r.pipeline()
           pipe.incr("account:A")
           pipe.incr("account:C")  # will error
           pipe.incr("account:B")
           results = pipe.execute(raise_on_error=False)
           for i, result in enumerate(results):
               if isinstance(result, Exception):
                   print(f"  Command {i} failed: {result}")
               else:
                   print(f"  Command {i} result: {result}")

       IMPORTANT: account:A and account:B were STILL incremented even
       though account:C failed. This is NOT like SQL transactions.
       Redis transactions guarantee isolation, not atomicity in the
       ACID sense.

    Docs:
      - Transactions: https://redis.io/docs/latest/develop/interact/transactions/
      - redis-py Pipelines: https://redis.io/docs/latest/develop/clients/redis-py/transpipe/
    """
    raise NotImplementedError("TODO(human): implement basic_transaction")


def optimistic_locking(r: redis.Redis) -> None:
    """Exercise 2: WATCH for optimistic locking (CAS pattern).

    TODO(human): Implement this function.

    WATCH implements optimistic locking: you monitor one or more keys,
    and if ANY of them are modified by another client before your EXEC,
    the entire transaction aborts (EXEC returns None / raises WatchError).

    This is like compare-and-swap (CAS) in concurrent programming:
      1. Read the current value (after WATCH)
      2. Compute the new value in your application
      3. Write the new value (in MULTI/EXEC)
      4. If someone else changed the value in step 1-3, retry

    redis-py provides a convenience method `transaction()` that handles
    the retry loop for you.

    Scenario: Safely increment a counter by reading it, adding 1, and
    setting the new value -- but ONLY if no one else changed it in between.

    Steps:

    1. Set up a shared counter:
           r.set("shared_counter", 100)

    2. Demonstrate successful WATCH + MULTI/EXEC (no contention):
       Use redis-py's transaction() convenience method:

           def increment(pipe: redis.client.Pipeline) -> None:
               # This function is called inside a WATCH context.
               # pipe is already watching "shared_counter".
               current = int(pipe.get("shared_counter"))
               new_value = current + 1
               # pipe.multi() switches from WATCH mode to MULTI mode.
               pipe.multi()
               pipe.set("shared_counter", new_value)

           r.transaction(increment, "shared_counter")

       r.transaction(func, *watch_keys) does:
         a. WATCH the specified keys
         b. Call your function with the pipeline
         c. EXEC the transaction
         d. If WatchError, retry from (a) automatically

       Print the counter value after the transaction.

    3. Demonstrate WATCH detecting a conflict:
       Start a background thread that modifies "shared_counter" while
       we're in the middle of a transaction.

       The boilerplate below provides a _conflicting_writer() function
       that changes "shared_counter" after a short delay.

       Use a manual WATCH approach to show the conflict:
           pipe = r.pipeline()
           pipe.watch("shared_counter")
           current = int(pipe.get("shared_counter"))
           print(f"  Read current value: {current}")
           # Now wait so the background thread can modify the key
           time.sleep(1.5)
           try:
               pipe.multi()
               pipe.set("shared_counter", current + 1)
               result = pipe.execute()
               print(f"  Transaction succeeded: {result}")
           except redis.WatchError:
               print("  WatchError! Another client modified the key.")
               print("  Transaction was aborted -- no commands executed.")
           finally:
               pipe.reset()

    4. Verify the final counter value:
           final = r.get("shared_counter")
       Print it and explain which write won.

    Docs:
      - WATCH: https://redis.io/docs/latest/commands/watch/
      - redis-py transaction(): https://redis-py.readthedocs.io/en/stable/advanced_features.html
    """
    raise NotImplementedError("TODO(human): implement optimistic_locking")


# -- Boilerplate helpers ---------------------------------------------------


def _conflicting_writer(delay: float) -> None:
    """Background thread that modifies shared_counter after a delay."""
    time.sleep(delay)
    r2 = get_client()
    r2.set("shared_counter", 999)
    print("  [background] Set shared_counter to 999")


# -- Orchestration (boilerplate) -------------------------------------------


def main() -> None:
    r = get_client()
    r.ping()
    print("Connected to Redis!")

    cleanup(r, "account:")
    cleanup(r, "shared_counter")

    section("Exercise 1: Basic Transaction (MULTI/EXEC)")
    basic_transaction(r)

    section("Exercise 2: Optimistic Locking (WATCH)")
    # Start the conflicting writer for the WATCH demo
    conflict_thread = threading.Thread(
        target=_conflicting_writer,
        args=(0.5,),
        daemon=True,
    )
    conflict_thread.start()
    optimistic_locking(r)
    conflict_thread.join(timeout=5)

    print("\n--- All transaction exercises completed ---")


if __name__ == "__main__":
    main()
