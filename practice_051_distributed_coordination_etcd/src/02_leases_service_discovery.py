"""Exercise 2: Leases and Service Discovery.

This exercise implements a heartbeat-based service registration pattern:
- Services register by creating a lease (TTL) and attaching a key to it
- Services keep their registration alive by sending periodic keepalive messages
- If a service crashes (stops sending keepalives), the lease expires and the
  key is automatically deleted — the service is deregistered
- Other services discover available backends by querying the prefix

This is the same pattern used by Consul, Eureka, and every service mesh
for health-based service discovery. etcd's lease API makes it simple:
the infrastructure handles expiration and cleanup automatically.
"""

import threading
import time

import etcd3


ETCD_HOST = "localhost"
ETCD_PORT = 2379

SERVICE_PREFIX = "/services/"


def create_client() -> etcd3.Etcd3Client:
    """Create an etcd client connected to the first cluster node."""
    return etcd3.client(host=ETCD_HOST, port=ETCD_PORT)


def cleanup(client: etcd3.Etcd3Client) -> None:
    """Delete all keys under our prefix to start fresh."""
    client.delete_prefix(SERVICE_PREFIX)


# TODO(human): Implement register_service
#
# Register a service in etcd with a TTL-based lease. The service key will
# automatically be deleted if the lease is not renewed, simulating a health
# check. This is the heartbeat pattern: "I'm alive" messages keep the
# registration active.
#
# Steps:
#   1. Create a lease with the given TTL (in seconds) using client.lease(ttl).
#      The lease is a server-side object with an ID and countdown timer.
#      Print the lease ID and TTL.
#
#   2. Put a key at f"{SERVICE_PREFIX}{service_name}" with the address string
#      as the value, attached to the lease: client.put(key, value, lease=lease).
#      This binds the key's lifetime to the lease — if the lease expires,
#      the key is deleted. Print the registration details.
#
#   3. Start a background thread that sends keepalive messages every
#      (ttl_secs / 3) seconds. Use lease.refresh() to renew the lease.
#      The thread should run until a threading.Event (stop_event) is set.
#      Each keepalive should print a brief message so you can see the
#      heartbeat rhythm.
#
#   4. Return a tuple of (lease, stop_event) so the caller can stop the
#      keepalive thread and/or revoke the lease later.
#
# The key insight: the keepalive interval should be significantly less than
# the TTL. A common pattern is ttl/3 — this gives 2 chances to renew before
# expiration, tolerating transient network issues. If all keepalives fail
# for the full TTL duration, the service is presumed dead.
#
# Use:
#   lease = client.lease(ttl=ttl_secs)  -> Lease object
#   client.put(key, value, lease=lease)  -> attaches key to lease
#   lease.refresh()                      -> sends keepalive, resets TTL timer
#   threading.Event() for stop signaling
#   threading.Thread(target=..., daemon=True)
#
# Function signature:
#   def register_service(
#       client: etcd3.Etcd3Client,
#       service_name: str,
#       address: str,
#       ttl_secs: int,
#   ) -> tuple[etcd3.Lease, threading.Event]
def register_service(
    client: etcd3.Etcd3Client,
    service_name: str,
    address: str,
    ttl_secs: int,
) -> tuple[etcd3.Lease, threading.Event]:
    raise NotImplementedError("TODO(human): Implement register_service")


# TODO(human): Implement discover_services
#
# Query etcd for all currently registered services under the prefix.
# This is the "consumer side" of service discovery — a load balancer,
# API gateway, or another microservice calls this to find backends.
#
# Steps:
#   1. Use client.get_prefix(SERVICE_PREFIX) to retrieve all registered
#      services. This returns an iterator of (value_bytes, KVMetadata).
#
#   2. For each service found, extract:
#      - Service name: parse from the key (remove the prefix)
#      - Address: decode the value bytes
#      - Key metadata: mod_revision and version
#
#   3. Print a formatted table of discovered services.
#
#   4. Return a list of tuples: [(service_name, address, mod_revision), ...]
#
# In production, a service discovery consumer would also set up a watch
# on the prefix to get real-time notifications when services come and go,
# rather than polling this function periodically.
#
# Use:
#   client.get_prefix(prefix) -> yields (value_bytes, KVMetadata)
#   metadata.key -> bytes (the full key path)
#   metadata.mod_revision -> int (global revision of last modification)
#
# Function signature:
#   def discover_services(
#       client: etcd3.Etcd3Client,
#   ) -> list[tuple[str, str, int]]
def discover_services(
    client: etcd3.Etcd3Client,
) -> list[tuple[str, str, int]]:
    raise NotImplementedError("TODO(human): Implement discover_services")


# TODO(human): Implement simulate_service_crash
#
# Simulate a service crash by revoking its lease (or stopping keepalives)
# and observe the automatic deregistration via a watch.
#
# Steps:
#   1. Set up a watch on SERVICE_PREFIX to capture the DELETE event when
#      the lease expires. Use client.watch_prefix(prefix) in a background
#      thread. Record the timestamp when the watcher is started.
#
#   2. Revoke the lease using lease.revoke(). This immediately expires the
#      lease and deletes all keys attached to it. In a real crash, the
#      service would simply stop sending keepalives, and the lease would
#      expire after the TTL. Revoking simulates an instant crash.
#      Record the timestamp of revocation.
#
#   3. In the watcher thread, when a DeleteEvent is received for the
#      service's key, record the timestamp and print:
#        - Which service was deregistered (key name)
#        - Time elapsed between revocation and the watch event
#      This delta shows etcd's event propagation latency (typically <10ms
#      for a healthy local cluster).
#
#   4. After observing the event (or a timeout of 5 seconds), cancel the
#      watcher and join the thread.
#
# The key learning: lease revocation (or expiration) triggers a DELETE
# event on the watch stream. This is how the rest of the system learns
# that a service is gone — without polling, without delay beyond the
# TTL window.
#
# Use:
#   events_iterator, cancel = client.watch_prefix(prefix)
#   lease.revoke() -> immediately destroys the lease and attached keys
#   time.monotonic() for measuring elapsed time
#   threading.Thread + threading.Event for coordination
#
# Function signature:
#   def simulate_service_crash(
#       client: etcd3.Etcd3Client,
#       lease: etcd3.Lease,
#       service_name: str,
#   ) -> None
def simulate_service_crash(
    client: etcd3.Etcd3Client,
    lease: etcd3.Lease,
    service_name: str,
) -> None:
    raise NotImplementedError("TODO(human): Implement simulate_service_crash")


def main() -> None:
    print("=" * 60)
    print("  Exercise 2: Leases and Service Discovery")
    print("=" * 60)

    client = create_client()
    cleanup(client)

    # --- Register 3 services ---
    print("\n--- Registering services ---\n")
    services_info: list[tuple[str, str, int]] = [
        ("web-server",   "10.0.1.1:8080", 15),
        ("api-gateway",  "10.0.1.2:3000", 15),
        ("worker-node",  "10.0.1.3:9090", 15),
    ]

    registrations: list[tuple[str, etcd3.Lease, threading.Event]] = []
    for name, addr, ttl in services_info:
        lease, stop_event = register_service(client, name, addr, ttl)
        registrations.append((name, lease, stop_event))
        time.sleep(0.3)  # Brief pause between registrations

    # --- Discover services ---
    print("\n--- Discovering services ---\n")
    time.sleep(1)  # Let registrations settle
    discovered = discover_services(client)
    print(f"\n  Total services discovered: {len(discovered)}")

    # --- Simulate a crash ---
    print("\n--- Simulating service crash (worker-node) ---\n")
    crash_name, crash_lease, crash_stop = registrations[2]
    crash_stop.set()  # Stop keepalive thread first
    time.sleep(0.5)
    simulate_service_crash(client, crash_lease, crash_name)

    # --- Discover again to see the missing service ---
    print("\n--- Discovering services after crash ---\n")
    time.sleep(1)
    discovered = discover_services(client)
    print(f"\n  Total services discovered: {len(discovered)}")

    # --- Cleanup ---
    print("\n--- Cleaning up ---")
    for name, lease, stop_event in registrations:
        stop_event.set()
        try:
            lease.revoke()
        except Exception:
            pass  # Already revoked
    time.sleep(1)
    cleanup(client)
    client.close()

    print("\n" + "=" * 60)
    print("  Exercise 2 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
