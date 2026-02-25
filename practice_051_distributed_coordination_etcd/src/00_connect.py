"""Exercise 0: Verify etcd cluster connectivity.

Connects to the 3-node etcd cluster via the Python client, prints the
cluster version, leader information, and member list. This script has
no TODO(human) — it's fully scaffolded to let you verify your setup
before starting the exercises.
"""

import sys

import etcd3


# Connection endpoints for our 3-node cluster.
# etcd1 is on localhost:2379, etcd2 on :2381, etcd3 on :2383.
ENDPOINTS = [
    ("localhost", 2379),
    ("localhost", 2381),
    ("localhost", 2383),
]


def check_connection(host: str, port: int) -> bool:
    """Try to connect to a single etcd endpoint and print its status."""
    try:
        client = etcd3.client(host=host, port=port)
        status = client.status()
        print(f"  Endpoint {host}:{port}")
        print(f"    Version:  {status.version}")
        print(f"    DB size:  {status.db_size} bytes")
        print(f"    Leader:   {status.leader}")
        print(f"    Raft idx: {status.raft_index}")
        print(f"    Raft trm: {status.raft_term}")
        client.close()
        return True
    except Exception as e:
        print(f"  Endpoint {host}:{port} -- FAILED: {e}")
        return False


def list_members(host: str, port: int) -> None:
    """Print all cluster members from the given endpoint."""
    try:
        client = etcd3.client(host=host, port=port)
        members = list(client.members)
        print(f"\n  Cluster has {len(members)} members:")
        for m in members:
            print(f"    - Name: {m.name}")
            print(f"      ID:   {m.id}")
            print(f"      Peer URLs:   {list(m.peer_urls)}")
            print(f"      Client URLs: {list(m.client_urls)}")
        client.close()
    except Exception as e:
        print(f"  Failed to list members: {e}")


def test_kv_roundtrip(host: str, port: int) -> bool:
    """Write and read a test key to verify the cluster is writable."""
    try:
        client = etcd3.client(host=host, port=port)
        test_key = "/practice_051/connectivity_test"
        test_val = "hello_etcd"

        client.put(test_key, test_val)
        value, metadata = client.get(test_key)

        if value is None:
            print("  KV roundtrip FAILED: key not found after put")
            client.delete(test_key)
            client.close()
            return False

        decoded = value.decode("utf-8")
        if decoded != test_val:
            print(f"  KV roundtrip FAILED: expected '{test_val}', got '{decoded}'")
            client.delete(test_key)
            client.close()
            return False

        print(f"\n  KV roundtrip OK:")
        print(f"    Key:      {test_key}")
        print(f"    Value:    {decoded}")
        print(f"    Revision: {metadata.mod_revision}")
        print(f"    Version:  {metadata.version}")

        # Clean up
        client.delete(test_key)
        client.close()
        return True
    except Exception as e:
        print(f"  KV roundtrip FAILED: {e}")
        return False


def main() -> None:
    print("=" * 60)
    print("  Practice 051: etcd Cluster Connectivity Check")
    print("=" * 60)

    # Check each endpoint
    print("\n[1/3] Checking individual endpoints...")
    results = []
    for host, port in ENDPOINTS:
        ok = check_connection(host, port)
        results.append(ok)

    healthy = sum(results)
    print(f"\n  {healthy}/{len(ENDPOINTS)} endpoints healthy")

    if healthy == 0:
        print("\n  ERROR: No endpoints reachable.")
        print("  Make sure Docker containers are running:")
        print("    docker compose up -d")
        print("    docker compose ps")
        sys.exit(1)

    # List cluster members (from first healthy endpoint)
    print("\n[2/3] Listing cluster members...")
    for (host, port), ok in zip(ENDPOINTS, results):
        if ok:
            list_members(host, port)
            break

    # Test KV roundtrip
    print("\n[3/3] Testing KV write/read roundtrip...")
    for (host, port), ok in zip(ENDPOINTS, results):
        if ok:
            kv_ok = test_kv_roundtrip(host, port)
            break

    print("\n" + "=" * 60)
    if healthy >= 2 and kv_ok:
        print("  All checks passed. Cluster is ready for exercises.")
    elif healthy >= 2:
        print("  WARNING: Cluster reachable but KV roundtrip failed.")
    else:
        print(f"  WARNING: Only {healthy}/3 endpoints healthy.")
        print("  The cluster may still work (quorum needs 2/3).")
    print("=" * 60)


if __name__ == "__main__":
    main()
