# /// script
# requires-python = ">=3.12"
# dependencies = ["redis>=5.0"]
# ///
"""Redis Pipelining -- Batch Operations for Performance.

Demonstrates:
  - Individual commands vs pipelined batch: wall-clock time comparison
  - How pipelining reduces network round trips
  - Reading pipeline results

Run after starting Redis:
    uv run 07_pipelining.py
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


def pipelining_demo(r: redis.Redis) -> None:
    """Exercise: Compare individual commands vs pipelined batch.

    TODO(human): Implement this function.

    Without pipelining, each Redis command follows this pattern:
      1. Client sends command over TCP
      2. Client waits for response (network round trip: ~0.1-1ms local, ~1-10ms remote)
      3. Client parses response
      4. Repeat for next command

    With pipelining, the client sends ALL commands without waiting for
    responses, then reads ALL responses at once. This eliminates the
    per-command round-trip latency, giving 5-10x improvement for batch
    operations.

    Analogy: Pipelining is like putting 100 letters in the mailbox at once
    instead of going to the mailbox 100 separate times.

    IMPORTANT: Pipelined commands are NOT atomic (unlike MULTI/EXEC).
    Other clients can interleave commands between your pipelined ones.
    If you need atomicity, use a transaction (MULTI/EXEC pipeline).

    Steps:

    1. Run 1000 individual SET commands (no pipeline) and measure time:
           start = time.perf_counter()
           for i in range(1000):
               r.set(f"bench:individual:{i}", f"value-{i}")
           elapsed_individual = time.perf_counter() - start
       Print the elapsed time.

    2. Run 1000 SET commands via a pipeline and measure time:
           start = time.perf_counter()
           pipe = r.pipeline(transaction=False)
           for i in range(1000):
               pipe.set(f"bench:pipelined:{i}", f"value-{i}")
           results = pipe.execute()
           elapsed_pipelined = time.perf_counter() - start

       r.pipeline(transaction=False) creates a pure pipeline (no MULTI/EXEC).
       Each pipe.set() just buffers the command locally -- nothing is sent yet.
       pipe.execute() sends ALL buffered commands in one batch and returns
       a list of results (one per command, in order).

       Print the elapsed time.

    3. Calculate and print the speedup:
           speedup = elapsed_individual / elapsed_pipelined
           print(f"  Speedup: {speedup:.1f}x faster with pipelining")

    4. Demonstrate reading results from a pipeline:
           pipe = r.pipeline(transaction=False)
           pipe.get("bench:pipelined:0")
           pipe.get("bench:pipelined:1")
           pipe.get("bench:pipelined:999")
           pipe.dbsize()
           results = pipe.execute()
       pipe.execute() returns results in the same order as the commands.
       Print each result:
           print(f"  First key:  {results[0]}")
           print(f"  Second key: {results[1]}")
           print(f"  Last key:   {results[2]}")
           print(f"  DB size:    {results[3]}")

    5. Pipeline with mixed command types:
           pipe = r.pipeline(transaction=False)
           pipe.set("pipe:counter", 0)
           pipe.incr("pipe:counter")
           pipe.incr("pipe:counter")
           pipe.incr("pipe:counter")
           pipe.get("pipe:counter")
           results = pipe.execute()
       Print all results to see the return value of each command:
         - SET returns True
         - Each INCR returns the new value (1, 2, 3)
         - GET returns the final value as a string ("3")
       This demonstrates that even though commands execute sequentially
       on the server, within a pipeline you CANNOT use the result of
       one command as input to another (they're all queued at once).

    Docs:
      - Pipelining: https://redis.io/docs/latest/develop/use/pipelining/
      - redis-py Pipeline: https://redis.io/docs/latest/develop/clients/redis-py/transpipe/
    """
    raise NotImplementedError("TODO(human): implement pipelining_demo")


# -- Orchestration (boilerplate) -------------------------------------------


def main() -> None:
    r = get_client()
    r.ping()
    print("Connected to Redis!")

    cleanup(r, "bench:")
    cleanup(r, "pipe:")

    section("Exercise: Pipelining Performance Comparison")
    pipelining_demo(r)

    # Cleanup benchmark keys
    print("\n  Cleaning up benchmark keys...")
    pipe = r.pipeline(transaction=False)
    for i in range(1000):
        pipe.delete(f"bench:individual:{i}")
        pipe.delete(f"bench:pipelined:{i}")
    pipe.delete("pipe:counter")
    pipe.execute()
    print("  Done.")

    print("\n--- All pipelining exercises completed ---")


if __name__ == "__main__":
    main()
