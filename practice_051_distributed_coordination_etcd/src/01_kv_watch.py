"""Exercise 1: Key-Value CRUD Operations and Watch API.

This exercise covers the foundational etcd operations:
- Putting, getting, and deleting key-value pairs
- Prefix-based queries (get all keys under a path)
- Watching for real-time changes on a key prefix

The Watch API is etcd's event-driven notification mechanism — it's
how Kubernetes controllers react to state changes without polling.
Understanding revision numbers is critical: each mutation increments
a global revision, enabling watchers to resume from where they left off.
"""

import threading
import time

import etcd3


ETCD_HOST = "localhost"
ETCD_PORT = 2379

# Prefix used for all keys in this exercise
KEY_PREFIX = "/services/"


def create_client() -> etcd3.Etcd3Client:
    """Create an etcd client connected to the first cluster node."""
    return etcd3.client(host=ETCD_HOST, port=ETCD_PORT)


def cleanup(client: etcd3.Etcd3Client) -> None:
    """Delete all keys under our prefix to start fresh."""
    client.delete_prefix(KEY_PREFIX)
    print(f"Cleaned up all keys under '{KEY_PREFIX}'")


# TODO(human): Implement kv_crud_operations
#
# Perform basic key-value CRUD operations on etcd using the /services/ prefix.
# This teaches the core KV API that everything else in etcd builds upon.
#
# Steps:
#   1. PUT 5 keys under the /services/ prefix with descriptive values:
#      - "/services/web"    -> "10.0.0.1:8080"
#      - "/services/api"    -> "10.0.0.2:3000"
#      - "/services/db"     -> "10.0.0.3:5432"
#      - "/services/cache"  -> "10.0.0.4:6379"
#      - "/services/queue"  -> "10.0.0.5:5672"
#      For each put, print the key, value, and the response header's revision.
#      The revision is a global monotonic counter — notice it increments by 1
#      with each put. This global ordering is how etcd guarantees linearizability.
#
#   2. GET a single key ("/services/web") and print its value, mod_revision,
#      create_revision, and version. The mod_revision tells you which global
#      mutation last changed this key. The version is a per-key counter.
#
#   3. GET with prefix: call client.get_prefix("/services/") and print all
#      key-value pairs found. This is how service discovery works — a load
#      balancer queries the prefix to find all registered backends.
#
#   4. DELETE a single key ("/services/cache") and print confirmation.
#      Then delete with prefix: delete all keys starting with "/services/q"
#      (this removes "/services/queue"). Print the number of keys deleted.
#
#   5. GET with prefix again to show the remaining keys after deletions.
#
# Use:
#   client.put(key, value) -> returns a PutResponse with header.revision
#   client.get(key) -> returns (value_bytes, KVMetadata)
#   client.get_prefix(prefix) -> yields (value_bytes, KVMetadata) tuples
#   client.delete(key) -> returns True if the key existed
#   client.delete_prefix(prefix) -> returns DeleteRangeResponse
#
# Note: Values returned from get() are bytes — decode with .decode("utf-8").
#       Metadata object has: .key, .mod_revision, .create_revision, .version
#
# Function signature:
#   def kv_crud_operations(client: etcd3.Etcd3Client) -> None
def kv_crud_operations(client: etcd3.Etcd3Client) -> None:
    raise NotImplementedError("TODO(human): Implement kv_crud_operations")


# TODO(human): Implement watch_with_prefix
#
# Watch all keys under a prefix for real-time change notifications.
# This is the event-driven model behind Kubernetes controllers: instead of
# polling etcd for changes, you open a watch stream and react to events
# as they arrive.
#
# Steps:
#   1. Start a watch on the KEY_PREFIX ("/services/") in a background thread.
#      Use client.watch_prefix(prefix) which returns (events_iterator, cancel).
#      In the thread, iterate over events_iterator. For each event, print:
#        - Event type: event.__class__.__name__ ("PutEvent" or "DeleteEvent")
#        - Key: event.key.decode("utf-8")
#        - Value: event.value.decode("utf-8") if it's a PutEvent (DeleteEvents
#          may have empty value)
#      The thread should stop iterating when the cancel function is called.
#
#   2. In the main thread, wait 1 second (so the watcher is ready), then
#      make several changes that the watcher should pick up:
#        - Put "/services/new-service" -> "10.0.0.10:9090"
#        - Update "/services/web" -> "10.0.0.1:8081" (changed port)
#        - Delete "/services/api"
#      Sleep briefly (0.5s) between operations so events arrive in order.
#
#   3. After all changes, wait 2 seconds for events to be processed, then
#      call cancel() to stop the watcher and join the background thread.
#
# The key learning: watches are push-based, not pull-based. The etcd server
# pushes events to the client over a persistent gRPC stream. This is far
# more efficient than polling, especially at scale (Kubernetes may watch
# thousands of keys simultaneously).
#
# Consider: what happens if the watcher disconnects and reconnects? It can
# pass start_revision to watch_prefix() to resume from the last processed
# revision, ensuring no events are missed. This is how Kubernetes controllers
# implement reliable event processing.
#
# Use:
#   events_iterator, cancel = client.watch_prefix(prefix)
#   threading.Thread(target=..., daemon=True) for background watching
#   event.key, event.value are bytes; event is PutEvent or DeleteEvent
#
# Function signature:
#   def watch_with_prefix(client: etcd3.Etcd3Client, duration_secs: float) -> None
def watch_with_prefix(client: etcd3.Etcd3Client, duration_secs: float) -> None:
    raise NotImplementedError("TODO(human): Implement watch_with_prefix")


def main() -> None:
    print("=" * 60)
    print("  Exercise 1: Key-Value CRUD and Watch API")
    print("=" * 60)

    client = create_client()

    # Start clean
    cleanup(client)

    # Part 1: CRUD operations
    print("\n--- Part 1: KV CRUD Operations ---\n")
    kv_crud_operations(client)

    # Clean up before watch demo
    cleanup(client)

    # Pre-populate some keys for the watch demo to modify
    print("\n--- Pre-populating keys for watch demo ---")
    for svc, addr in [("web", "10.0.0.1:8080"), ("api", "10.0.0.2:3000")]:
        client.put(f"{KEY_PREFIX}{svc}", addr)
        print(f"  Put {KEY_PREFIX}{svc} = {addr}")

    # Part 2: Watch demo
    print("\n--- Part 2: Watch API Demo ---\n")
    watch_with_prefix(client, duration_secs=8.0)

    # Final cleanup
    cleanup(client)
    client.close()

    print("\n" + "=" * 60)
    print("  Exercise 1 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
