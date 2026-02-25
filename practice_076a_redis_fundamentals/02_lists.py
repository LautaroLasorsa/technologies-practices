# /// script
# requires-python = ">=3.12"
# dependencies = ["redis>=5.0"]
# ///
"""Redis Lists -- Queues and Blocking Consumers.

Demonstrates:
  - LPUSH / RPUSH to add elements
  - LPOP / RPOP to remove elements
  - LRANGE to peek at list contents without consuming
  - LLEN for list length
  - BRPOP for blocking queue consumption (the consumer pattern)

Run after starting Redis:
    uv run 02_lists.py
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


def list_as_queue(r: redis.Redis) -> None:
    """Exercise 1: List operations for queue and stack patterns.

    TODO(human): Implement this function.

    Redis lists are doubly-linked (internally quicklists), giving O(1)
    push/pop at both ends. This makes them perfect for:
      - FIFO queue: LPUSH (enqueue at head) + RPOP (dequeue from tail)
      - LIFO stack: LPUSH (push) + LPOP (pop from same end)
      - Capped lists: LPUSH + LTRIM to keep only the N most recent items

    Steps:

    1. Create a FIFO queue by pushing tasks from the left:
           r.lpush("tasks", "task-1", "task-2", "task-3")
       LPUSH inserts elements at the HEAD (left) of the list. When passing
       multiple values, they are inserted left-to-right, so the list
       becomes: [task-3, task-2, task-1] (last argument is the new head).
       LPUSH returns the new length of the list. Print it.

    2. Peek at the entire list without consuming:
           items = r.lrange("tasks", 0, -1)
       LRANGE returns elements from index `start` to `stop` (inclusive).
       0 = first element, -1 = last element. This is O(N) for the range.
       Print the items to see the order.

    3. Get the list length:
           length = r.llen("tasks")
       LLEN is O(1) because Redis stores the length. Print it.

    4. Consume from the queue (FIFO) using RPOP:
           task = r.rpop("tasks")
       RPOP removes and returns the TAIL (right) element. Since we pushed
       at the head, RPOP gives us the oldest item first (FIFO order).
       Pop all items in a loop until RPOP returns None (empty list):
           while (task := r.rpop("tasks")) is not None:
               print(f"  Processing: {task}")

    5. Demonstrate RPUSH + LPOP (equivalent FIFO, just from the other direction):
           r.rpush("jobs", "job-A", "job-B", "job-C")
       RPUSH inserts at the TAIL. The list becomes: [job-A, job-B, job-C].
       Use LPOP to consume from the head:
           job = r.lpop("jobs")
       Pop all and print to confirm FIFO order.

    6. Demonstrate LRANGE for pagination (without consuming):
           r.rpush("logs", "log-1", "log-2", "log-3", "log-4", "log-5")
       Get only the first 3 elements:
           page = r.lrange("logs", 0, 2)
       Get elements 3-4 (second page):
           page2 = r.lrange("logs", 3, 4)
       Print both pages.

    Docs:
      - LPUSH: https://redis.io/docs/latest/commands/lpush/
      - RPOP: https://redis.io/docs/latest/commands/rpop/
      - LRANGE: https://redis.io/docs/latest/commands/lrange/
      - LLEN: https://redis.io/docs/latest/commands/llen/
    """
    raise NotImplementedError("TODO(human): implement list_as_queue")


def blocking_consumer(r: redis.Redis) -> None:
    """Exercise 2: Blocking queue consumption with BRPOP.

    TODO(human): Implement this function.

    BRPOP is the blocking version of RPOP. It waits until an element
    is available in the list OR the timeout expires. This is the foundation
    of Redis-based task queues (Celery, ARQ, Sidekiq) -- the worker blocks
    on BRPOP instead of busy-polling with RPOP in a loop.

    Advantages over polling:
      - No CPU waste: the client is blocked at the TCP level, not spinning.
      - Instant response: as soon as a message arrives, the client wakes up.
      - Multiple lists: BRPOP can watch multiple lists and returns from the
        first one that gets an element (priority queue pattern).

    Steps:

    1. Start a background thread that acts as a producer:
       (This is provided in the boilerplate below -- it pushes 3 items
       to "work_queue" with 1-second delays between them.)

    2. In the main thread, consume items using BRPOP in a loop:
           while True:
               result = r.brpop("work_queue", timeout=5)
               if result is None:
                   print("  Timeout -- no more items. Stopping.")
                   break
               list_name, value = result
               print(f"  Received from '{list_name}': {value}")

       BRPOP returns a tuple (list_name, value) when an item is available,
       or None when the timeout expires. The list_name is included because
       BRPOP can watch multiple lists: r.brpop(["high", "medium", "low"], timeout=5)
       and it tells you which list the item came from.

       The timeout is in seconds. Use timeout=0 for infinite blocking
       (careful in production -- the connection stays open indefinitely).

    Note: The producer thread is started for you in the boilerplate below.
    You only need to implement the consumer loop.

    Docs:
      - BRPOP: https://redis.io/docs/latest/commands/brpop/
      - BLPOP: https://redis.io/docs/latest/commands/blpop/
    """
    raise NotImplementedError("TODO(human): implement blocking_consumer")


# -- Orchestration (boilerplate) -------------------------------------------


def _producer(r: redis.Redis, queue: str, items: list[str], delay: float) -> None:
    """Background producer that pushes items with a delay."""
    time.sleep(0.5)  # Let the consumer start first
    for item in items:
        print(f"  [producer] Pushing: {item}")
        r.lpush(queue, item)
        time.sleep(delay)
    print("  [producer] Done producing.")


def main() -> None:
    r = get_client()
    r.ping()
    print("Connected to Redis!")

    cleanup(r, "tasks")
    cleanup(r, "jobs")
    cleanup(r, "logs")
    cleanup(r, "work_queue")

    section("Exercise 1: List as Queue (FIFO/LIFO)")
    list_as_queue(r)

    section("Exercise 2: Blocking Consumer (BRPOP)")
    # Start a producer thread that will push items while the consumer blocks
    producer_thread = threading.Thread(
        target=_producer,
        args=(get_client(), "work_queue", ["msg-1", "msg-2", "msg-3"], 1.0),
        daemon=True,
    )
    producer_thread.start()
    blocking_consumer(r)
    producer_thread.join(timeout=10)

    print("\n--- All list exercises completed ---")


if __name__ == "__main__":
    main()
