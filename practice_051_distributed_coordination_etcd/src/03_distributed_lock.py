"""Exercise 3: Distributed Locking with Transactions.

This exercise builds a distributed lock from etcd's atomic transaction API.
The lock uses compare-and-swap (CAS): a transaction checks whether the lock
key exists, and only creates it if it doesn't. This guarantees mutual
exclusion — only one holder at a time.

The lock is backed by a lease, so if the holder crashes, the lock
automatically releases after the TTL expires. No permanent deadlocks.

This is the same CAS pattern used in:
- ZooKeeper recipes (ephemeral sequential nodes)
- DynamoDB conditional writes
- Redis SETNX (but without consensus — Redlock adds it)
- etcd's own concurrency.Mutex (Go client)
"""

import threading
import time

import etcd3


ETCD_HOST = "localhost"
ETCD_PORT = 2379

LOCK_KEY = "/locks/shared-resource"
COUNTER_KEY = "/state/counter"


def create_client() -> etcd3.Etcd3Client:
    """Create an etcd client connected to the first cluster node."""
    return etcd3.client(host=ETCD_HOST, port=ETCD_PORT)


def cleanup(client: etcd3.Etcd3Client) -> None:
    """Delete lock and counter keys to start fresh."""
    client.delete_prefix("/locks/")
    client.delete_prefix("/state/")


# TODO(human): Implement the EtcdLock class
#
# A distributed lock built on etcd transactions and leases. The lock provides
# mutual exclusion across multiple processes/threads: only one holder can
# hold the lock at any time, and the lock auto-releases if the holder crashes.
#
# The class needs three pieces of state:
#   - client: the etcd3 client
#   - lock_key: the key used as the lock (e.g., "/locks/shared-resource")
#   - holder_id: a unique identifier for this lock holder (e.g., "worker-1")
#   - lease: the lease backing the lock (created during acquire)
#
# Implement these methods:
#
# __init__(self, client, lock_key, holder_id, ttl_secs=10):
#     Store the parameters. Set lease to None initially.
#
# acquire(self, timeout_secs=30.0) -> bool:
#     Try to acquire the lock using a CAS transaction. The algorithm:
#
#     1. Create a lease with the given TTL: self.lease = client.lease(ttl).
#        The lease ensures the lock auto-releases if we crash.
#
#     2. Execute a transaction:
#        Compare:  [client.transactions.version(lock_key) == 0]
#                  (version == 0 means the key does NOT exist)
#        Success:  [client.transactions.put(lock_key, holder_id, lease)]
#                  (create the lock key with our ID, backed by the lease)
#        Failure:  [client.transactions.get(lock_key)]
#                  (key already exists — someone else holds the lock)
#
#     3. If the transaction succeeded (result[0] is True): we hold the lock.
#        Print a message and return True.
#
#     4. If the transaction failed: the lock is held by someone else.
#        Extract the current holder from the failure response.
#        Print who holds it, then watch the lock_key for deletion.
#        Use client.watch(lock_key) and wait for a DeleteEvent.
#        When the key is deleted (lock released), retry from step 1.
#
#     5. Repeat until the lock is acquired or timeout_secs is exceeded.
#        Return False if timeout is reached.
#
#     This is the standard CAS-based lock acquisition pattern. The watch-and-
#     retry avoids busy-spinning — the thread sleeps until the lock is released.
#
# release(self) -> None:
#     Release the lock by deleting the lock key and revoking the lease.
#     1. client.delete(lock_key)
#     2. self.lease.revoke()
#     3. Print a release message.
#     Handle the case where the lease or key may already be gone (e.g.,
#     if the lock expired before explicit release).
#
# Use:
#   client.transaction(compare=[...], success=[...], failure=[...])
#     -> returns (succeeded: bool, responses: list)
#   client.transactions.version(key) == 0   (key does not exist)
#   client.transactions.put(key, value, lease)
#   client.transactions.get(key)
#   client.watch(key) -> (events_iterator, cancel)
#   etcd3.events.DeleteEvent for detecting lock release
#
# Class signature:
#   class EtcdLock:
#       def __init__(self, client: etcd3.Etcd3Client, lock_key: str,
#                    holder_id: str, ttl_secs: int = 10) -> None
#       def acquire(self, timeout_secs: float = 30.0) -> bool
#       def release(self) -> None
class EtcdLock:
    def __init__(
        self,
        client: etcd3.Etcd3Client,
        lock_key: str,
        holder_id: str,
        ttl_secs: int = 10,
    ) -> None:
        raise NotImplementedError("TODO(human): Implement EtcdLock.__init__")

    def acquire(self, timeout_secs: float = 30.0) -> bool:
        raise NotImplementedError("TODO(human): Implement EtcdLock.acquire")

    def release(self) -> None:
        raise NotImplementedError("TODO(human): Implement EtcdLock.release")


# TODO(human): Implement run_lock_contention
#
# Spawn multiple threads (simulating distributed workers) that all compete
# for the same lock. Each worker acquires the lock, reads a counter from
# etcd, increments it, writes it back, and releases the lock.
#
# Without the lock, concurrent read-modify-write would cause lost updates
# (classic race condition). With the lock, the final counter must exactly
# equal num_workers — proving mutual exclusion works.
#
# Steps:
#   1. Initialize the counter key to "0" in etcd.
#
#   2. Define a worker function that:
#      a. Creates its own etcd client (each thread needs its own connection)
#      b. Creates an EtcdLock with lock_key=LOCK_KEY, holder_id=f"worker-{i}"
#      c. Acquires the lock (with a generous timeout, e.g., 60s)
#      d. Reads the counter from COUNTER_KEY, increments by 1, writes it back
#         (this is the critical section — must be protected by the lock)
#      e. Sleeps briefly (0.1s) to simulate work and increase contention window
#      f. Releases the lock
#      g. Closes its client
#      Print each step so you can see the interleaving of workers.
#
#   3. Start num_workers threads, each running the worker function.
#      Wait for all threads to complete (join).
#
#   4. Read the final counter value. It must equal num_workers.
#      Print the final value and whether the test passed.
#
# The key learning: etcd transactions provide the CAS primitive, and the lock
# built on top provides the familiar mutex interface. The lease ensures crash
# safety — a dead worker's lock automatically releases, so other workers can
# make progress.
#
# Use:
#   threading.Thread(target=worker, args=(i,)) for each worker
#   Each thread creates its own etcd3.client() (thread safety)
#   client.get(COUNTER_KEY) to read counter, client.put(COUNTER_KEY, str(new_val))
#
# Function signature:
#   def run_lock_contention(
#       client: etcd3.Etcd3Client,
#       num_workers: int,
#       lock_key: str,
#   ) -> None
def run_lock_contention(
    client: etcd3.Etcd3Client,
    num_workers: int,
    lock_key: str,
) -> None:
    raise NotImplementedError("TODO(human): Implement run_lock_contention")


def main() -> None:
    print("=" * 60)
    print("  Exercise 3: Distributed Locking")
    print("=" * 60)

    client = create_client()
    cleanup(client)

    # --- Test basic lock acquire/release ---
    print("\n--- Basic lock acquire/release ---\n")
    lock = EtcdLock(client, LOCK_KEY, "main-thread", ttl_secs=10)
    acquired = lock.acquire(timeout_secs=5.0)
    if acquired:
        print("  Lock acquired by main-thread. Holding for 2 seconds...")
        time.sleep(2)
        lock.release()
        print("  Lock released by main-thread.")
    else:
        print("  ERROR: Failed to acquire lock (should not happen).")

    # --- Test lock contention with multiple workers ---
    print("\n--- Lock contention test (5 workers) ---\n")
    cleanup(client)
    run_lock_contention(client, num_workers=5, lock_key=LOCK_KEY)

    # --- Cleanup ---
    cleanup(client)
    client.close()

    print("\n" + "=" * 60)
    print("  Exercise 3 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
