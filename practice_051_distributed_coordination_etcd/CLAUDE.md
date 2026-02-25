# Practice 051 — Distributed Coordination: etcd

## Technologies

- **etcd** (v3.5) — Distributed, reliable key-value store for coordination data
- **python-etcd3** (`etcd3` pip package) — Python gRPC client for etcd v3 API
- **Docker Compose** — 3-node etcd cluster orchestration
- **gRPC / Protobuf** — Wire protocol for etcd v3 API

## Stack

Python 3.12 (uv), Docker / Docker Compose

## Theoretical Context

### What etcd Is and the Problem It Solves

etcd is a distributed, reliable key-value store designed for the most critical data in distributed systems: configuration, service discovery, coordination state, and distributed locks. The name comes from the Unix `/etc` directory (where system configuration lives) plus "d" for distributed. Created by CoreOS in 2013 (now part of the CNCF), etcd is the backbone of Kubernetes — every piece of cluster state (pods, services, deployments, secrets) is stored in etcd. When you run `kubectl apply`, the API server writes to etcd; when a controller reacts to a pod change, it's watching an etcd key prefix.

etcd is designed for **consistency over throughput**. It targets ~10,000 writes/sec and sub-10ms latency for a healthy cluster. It is NOT a general-purpose database — you would never store application data (user profiles, orders, logs) in etcd. Its purpose is coordination: storing the small amount of data that many distributed components need to agree on (who is the leader, which services are alive, what is the current configuration).

The core guarantee is **linearizable reads and writes**: every read returns the most recent write, and writes are applied in a total global order. This makes etcd suitable for building correct distributed algorithms on top of it.

**References:**
- [etcd Official Documentation](https://etcd.io/docs/v3.5/)
- [etcd: What It Is and Why It Matters (IBM)](https://www.ibm.com/think/topics/etcd)
- [Kubernetes etcd documentation](https://kubernetes.io/docs/concepts/overview/components/#etcd)

### Architecture: Raft Consensus Under the Hood

etcd uses the **Raft consensus algorithm** to replicate data across cluster members. Raft is a leader-based protocol designed for understandability (compared to Paxos). At any time, one member is the **leader** and the rest are **followers**. All writes go through the leader, which replicates entries to followers via an append-only log.

**How a write works:**
1. Client sends a write (Put) request to any cluster member.
2. If the member is not the leader, it forwards the request to the leader.
3. The leader appends the entry to its log and sends AppendEntries RPCs to all followers.
4. Once a **majority** (quorum) acknowledges, the entry is **committed** — it will never be lost.
5. The leader applies the entry to its state machine (the key-value store) and responds to the client.

**Cluster sizing and fault tolerance:** etcd requires an **odd number** of members (3, 5, or 7 are typical). A cluster of N members tolerates **(N-1)/2** failures:
- 3 members: tolerates 1 failure (quorum = 2)
- 5 members: tolerates 2 failures (quorum = 3)
- 7 members: tolerates 3 failures (quorum = 4)

**Leader election:** If the leader crashes, followers detect the missing heartbeats after an election timeout. A follower becomes a **candidate**, increments its term number, votes for itself, and requests votes from other members. The candidate that receives a majority of votes becomes the new leader. Raft guarantees that exactly one leader exists per term, and the cluster resumes normal operation within ~1 second of a leader failure.

**Read consistency:** By default, etcd provides **serializable reads** (may return stale data from a follower). For **linearizable reads** (strongly consistent), the client can request that the serving member confirms it is still the leader before responding. The python-etcd3 client connects to a single endpoint by default, which provides serializable reads. For linearizable reads in production, clients typically connect to the leader or use the `--consistency=l` flag in etcdctl.

**References:**
- [Raft Paper — In Search of an Understandable Consensus Algorithm](https://raft.github.io/raft.pdf)
- [etcd Raft Implementation (Deep Dive)](https://deepwiki.com/openshift/etcd/3-raft-consensus-implementation)
- [Inside Kubernetes: Raft Algorithm in etcd](https://ezyinfra.dev/blog/raft-algo-backup-etcd)

### Key Features of etcd v3

#### 1. Key-Value Store with Hierarchical Namespace

Keys in etcd are arbitrary byte strings, but by convention use `/`-delimited paths to simulate a hierarchy: `/services/web/config`, `/services/api/host`. This convention enables **prefix queries**: get all keys under `/services/` returns every service's configuration. Internally, keys are stored in a B-tree ordered lexicographically, making prefix scans efficient.

Values are also byte strings (up to 1.5 MB per value, but best kept small — KBs, not MBs).

Every mutation increments a global **revision** counter (a monotonically increasing 64-bit integer). Each key-value pair also tracks:
- **create_revision**: the global revision when this key was first created
- **mod_revision**: the global revision when this key was last modified
- **version**: per-key version counter (incremented on each modification, reset to 0 on delete)

Revisions enable **multi-version concurrency control (MVCC)**: etcd keeps a history of all key states within a configurable window. Old revisions can be retrieved (time-travel reads) and are cleaned up via **compaction**.

#### 2. Watch API — Event-Driven Notifications

The Watch API lets clients subscribe to changes on a key or key range (prefix). When a key is created, updated, or deleted, etcd pushes a **WatchEvent** to all watchers with: event type (PUT or DELETE), the key, the new value, and the revision.

This is the foundation of Kubernetes controllers: a Deployment controller watches for Pod changes and reacts accordingly, without polling. Watches are efficient — etcd maintains a single gRPC stream per watcher and multiplexes events.

Critical feature: **watch from a specific revision**. If a client disconnects and reconnects, it can resume watching from the last revision it processed, ensuring no events are missed (as long as the revision hasn't been compacted away). This enables exactly-once processing semantics.

#### 3. Lease API — Time-to-Live Keys

A **lease** is a time-to-live (TTL) grant from etcd. Clients create a lease with a TTL (e.g., 10 seconds), then attach keys to it. If the lease expires (the client doesn't send keepalive messages), all keys attached to that lease are automatically deleted.

This is the heartbeat mechanism for **service registration**: a service registers itself by creating a key `/services/web-01` with its address, attached to a 10-second lease. It sends keepalive messages every ~3 seconds. If the service crashes (stops sending keepalives), the lease expires after 10 seconds, the key is auto-deleted, and watchers on the `/services/` prefix receive a DELETE event — the service is automatically deregistered.

Lease operations:
- **Grant**: create a lease with a TTL, returns a lease_id
- **Keepalive**: refresh the lease TTL (must be sent before expiration)
- **Revoke**: explicitly destroy the lease (and all attached keys)
- **TTL**: query remaining time on a lease

#### 4. Transactions — Atomic Compare-and-Swap

etcd transactions provide an atomic **If/Then/Else** construct:

```
If (key1.version == 3 AND key2.value == "ready")
Then (put key3="started", delete key4)
Else (get key1)
```

All comparisons are evaluated atomically. If all comparisons succeed, the Then branch executes; otherwise, the Else branch executes. The entire transaction is a single revision increment.

Comparison targets: `value`, `version`, `create_revision`, `mod_revision`. Comparison operators: `==`, `!=`, `<`, `>`.

Transactions are the building block for all higher-level coordination primitives: locks, elections, barriers, and double-checked locking patterns.

#### 5. Distributed Locks — Built on Leases + Transactions

A distributed lock in etcd combines leases and transactions:

1. **Acquire**: Create a lease (so the lock auto-releases if holder crashes). Then execute a transaction: If(lock_key does not exist) Then(put lock_key=holder_id with lease) Else(fail). If the transaction fails (someone else holds the lock), watch the lock key and retry when it's deleted.
2. **Hold**: Keep the lease alive by sending periodic keepalives.
3. **Release**: Delete the lock key (or let the lease expire on crash).

This provides **mutual exclusion** with **automatic failure recovery** — no risk of permanent deadlock from a crashed lock holder.

#### 6. Leader Election — Built on Locks

Leader election is a specialized use of distributed locks:

1. Multiple candidates try to acquire the same key via CAS transaction.
2. The winner becomes the leader and keeps the key alive via lease keepalive.
3. Losers watch the key — when the leader crashes (lease expires, key deleted) or resigns (deletes the key), they retry.
4. Exactly one leader is elected at any time due to the CAS guarantee.

etcd's Go client includes a built-in `concurrency.Election` package. In Python, we build it from the transaction and lease primitives.

### etcd vs Alternatives

| Feature | etcd | ZooKeeper | Consul | Redis (Redlock) |
|---------|------|-----------|--------|-----------------|
| **Consensus** | Raft | ZAB (Paxos-like) | Raft (via Serf gossip + Raft for servers) | None (single-node); Redlock is advisory |
| **API** | gRPC (v3) | Custom TCP protocol | HTTP + DNS | RESP protocol |
| **Data model** | Flat KV with prefix convention | Hierarchical znodes | KV + service catalog | KV (strings, hashes, lists, etc.) |
| **Watch** | Efficient gRPC streaming, resume from revision | ZooKeeper watches (one-shot, must re-register) | Blocking queries (long-poll HTTP) | Pub/Sub (no history replay) |
| **Lease/TTL** | First-class lease API with keepalive | Ephemeral znodes (auto-deleted on session close) | TTL on KV, health-check based | `EXPIRE` command on keys |
| **Transactions** | Full If/Then/Else with multi-key comparisons | Multi-op (version-based, less expressive) | Check-and-Set (CAS) on single key | `WATCH`/`MULTI`/`EXEC` (optimistic locking) |
| **Use case** | Kubernetes, small coordination data | Hadoop/Kafka metadata, ZK-based systems | Service mesh, multi-DC discovery | Caching, general-purpose; Redlock for advisory locks |
| **Maturity** | CNCF graduated, Kubernetes backbone | Very mature (Apache, ~2010), declining adoption | HashiCorp, feature-rich but more complex | Ubiquitous, but not designed for consensus |
| **Language** | Go | Java | Go | C |

**When to use etcd:** You need a strongly consistent, simple key-value store for coordination data (config, service registry, locks, elections). Especially if you're already in the Kubernetes ecosystem. etcd's API is simpler than ZooKeeper's and more focused than Consul's.

**When NOT to use etcd:** You need high-throughput data storage, complex queries, large values (>1 MB), or multi-datacenter replication (etcd is designed for single-datacenter clusters).

**References:**
- [etcd versus other key-value stores](https://etcd.io/docs/v3.5/learning/why/)
- [Consul vs ZooKeeper vs etcd (StackShare)](https://stackshare.io/stackups/consul-vs-etcd-vs-zookeeper)
- [How To Choose Between Etcd, ZooKeeper, and Consul](https://blog.devops.dev/how-to-choose-between-etcd-zookeeper-and-consul-2c0e6a3c48da)
- [In-Depth Comparison: Consul, etcd, ZooKeeper, Nacos](https://medium.com/@karim.albakry/in-depth-comparison-of-distributed-coordination-tools-consul-etcd-zookeeper-and-nacos-a6f8e5d612a6)

### Performance and Operational Considerations

- **Write throughput**: ~10K writes/sec (sequential), ~30K with concurrent clients (varies by hardware and value size)
- **Read throughput**: Much higher — serializable reads don't go through Raft
- **Latency**: Sub-10ms for healthy clusters on fast networks
- **Data size**: Recommended max DB size is 8 GB (default 2 GB). Keys should be small, values under a few KB
- **Compaction**: etcd keeps all revisions by default. You MUST run periodic compaction to prevent unbounded disk growth. In production: auto-compaction every hour or by revision count
- **Defragmentation**: After compaction, free space is not returned to disk immediately. Run `etcdctl defrag` periodically
- **Backup**: `etcdctl snapshot save` for point-in-time backups; critical for Kubernetes clusters

### Python Client: python-etcd3

This practice uses the `etcd3` pip package ([kragniz/python-etcd3](https://github.com/kragniz/python-etcd3)), the most widely used Python client for etcd v3. It communicates via gRPC and provides a synchronous API. Key methods:

- `etcd3.client(host, port)` — create a client connection
- `client.put(key, value, lease=None)` — write a key-value pair
- `client.get(key)` — get value and metadata for a key
- `client.get_prefix(prefix)` — get all KV pairs under a prefix
- `client.delete(key)` — delete a key
- `client.delete_prefix(prefix)` — delete all keys under a prefix
- `client.watch(key)` / `client.watch_prefix(prefix)` — returns `(events_iterator, cancel_fn)`
- `client.add_watch_callback(key, callback)` — callback-based watching
- `client.lease(ttl)` — grant a new lease
- `lease.refresh()` — send a keepalive
- `lease.revoke()` — revoke the lease
- `client.transaction(compare, success, failure)` — atomic If/Then/Else
- `client.status()` — cluster member status
- `client.members` — list of cluster members (property)

**Note on maintenance:** The python-etcd3 library is not actively maintained (last release ~2020), but it remains functional with etcd v3.5 and is the most documented Python option. For production use, consider the etcd HTTP/gRPC gateway API directly or the Go client.

**References:**
- [python-etcd3 API Documentation](https://python-etcd3.readthedocs.io/en/latest/usage.html)
- [python-etcd3 GitHub](https://github.com/kragniz/python-etcd3)
- [etcd Python client discussion (etcd-io/etcd#18211)](https://github.com/etcd-io/etcd/discussions/18211)

## Description

Build four exercises that progressively explore etcd's coordination primitives using a local 3-node cluster:

1. **Key-value CRUD + Watch API** — Store and retrieve service configuration data, watch for real-time changes
2. **Leases and Service Discovery** — Implement heartbeat-based service registration with automatic deregistration on failure
3. **Distributed Locking** — Build a CAS-based distributed lock using transactions, test under contention
4. **Leader Election** — Implement a leader election protocol with automatic failover

### What you'll learn

1. **etcd fundamentals** — KV operations, prefix queries, revision semantics
2. **Event-driven coordination** — Watch API for reacting to state changes in real time
3. **Service discovery pattern** — Lease-based registration with heartbeat and auto-deregistration
4. **Distributed locking** — How to build correct mutual exclusion from CAS transactions
5. **Leader election** — How multiple candidates coordinate to elect exactly one leader with failover
6. **etcd cluster operations** — Working with a multi-node cluster, observing fault tolerance

## Instructions

### Phase 1: Setup & Connection (~10 min)

1. Start the 3-node etcd cluster: `docker compose up -d`
2. Verify cluster health: `docker exec practice_051-etcd1 etcdctl endpoint health --cluster`
3. List members: `docker exec practice_051-etcd1 etcdctl member list`
4. Install Python dependencies: `uv sync`
5. Run `uv run python src/00_connect.py` to verify the Python client can connect to the cluster. This script prints the cluster version, leader ID, and all members. If connection fails, check that Docker containers are running and ports 2379/2381/2383 are exposed.

### Phase 2: Key-Value Operations & Watch API (~20 min)

1. **Exercise (`src/01_kv_watch.py`):** Implement basic CRUD operations on etcd keys using the `/services/` prefix convention, then set up a watch to observe real-time changes. This exercise teaches the foundational KV API — how keys are stored, queried by prefix, and how revisions track the history of changes. The Watch API is the mechanism behind Kubernetes controllers: instead of polling for changes, you subscribe to a key range and react to events as they happen. Understanding watches is essential because every higher-level pattern (service discovery, locks, elections) relies on watching for key changes.

### Phase 3: Leases & Service Discovery (~25 min)

1. **Exercise (`src/02_leases_service_discovery.py`):** Implement a service registration system where services register themselves with a TTL lease and keep it alive via periodic heartbeats. Other services discover them by querying the prefix. When a service crashes (stops sending keepalives), its key auto-deletes. This exercise teaches the lease API — the mechanism that makes etcd-based service discovery self-healing. Without leases, a crashed service would leave stale entries forever. The keepalive pattern (register + periodic refresh) is the same pattern used in Consul, Eureka, and every other service mesh for health-based deregistration.

### Phase 4: Distributed Locking (~25 min)

1. **Exercise (`src/03_distributed_lock.py`):** Build a distributed lock using etcd transactions (compare-and-swap). Multiple workers compete for the same lock, perform a critical section (incrementing a shared counter in etcd), and release. The transaction ensures mutual exclusion: only one worker can hold the lock at any time. This exercise teaches how to compose etcd's atomic transaction API into a correct mutual exclusion primitive. The CAS pattern (If key not exists, Then create it) is the fundamental building block for distributed coordination — it's the same pattern used in ZooKeeper recipes, DynamoDB conditional writes, and Redis `SETNX`.

### Phase 5: Leader Election (~20 min)

1. **Exercise (`src/04_leader_election.py`):** Implement leader election where multiple candidates compete to become leader by acquiring an etcd key via CAS. The winner maintains leadership via lease keepalive. When the leader crashes or resigns, watchers detect the key deletion and a new election occurs. This exercise combines everything: leases (auto-expire on crash), transactions (CAS for election), and watches (detect leadership changes). Leader election is the coordination primitive behind Kafka controller election, Kubernetes controller manager HA, and distributed scheduler architectures.

## Motivation

- **Kubernetes foundation**: etcd is the brain of Kubernetes. Understanding it explains why K8s behaves the way it does (consistency guarantees, watch-based controllers, leader election for control plane HA). Directly relevant to practices 006a/006b.
- **Distributed coordination primitives**: Locks, elections, and service discovery are universal patterns. Learning them via etcd's clean API provides transferable knowledge for ZooKeeper, Consul, or any coordination system.
- **Raft consensus in practice**: etcd is one of the best real-world case studies for the Raft algorithm. Understanding etcd's behavior under node failures gives concrete intuition for consensus theory.
- **Production debugging**: When Kubernetes misbehaves, the root cause is often etcd (slow writes, leader elections, compaction backlog). Understanding etcd internals is a senior SRE/platform engineering differentiator.
- **Complements practices 014 (SAGA) and 049a/049b (Raft)**: SAGAs coordinate transactions across services; etcd provides the coordination infrastructure those services often rely on. The Raft practice teaches the algorithm; this practice shows its production embodiment.

## Commands

All commands run from `practice_051_distributed_coordination_etcd/`.

### Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start 3-node etcd cluster (ports 2379, 2381, 2383) |
| `docker compose ps` | Check container status for all 3 etcd nodes |
| `docker compose logs -f etcd1` | Follow logs for etcd1 (useful for watching Raft elections) |
| `docker exec practice_051-etcd1 etcdctl endpoint health --cluster` | Verify all 3 endpoints are healthy |
| `docker exec practice_051-etcd1 etcdctl member list` | List cluster members with IDs and peer URLs |
| `docker exec practice_051-etcd1 etcdctl endpoint status --cluster -w table` | Show leader, DB size, Raft term/index per member |
| `docker compose down -v` | Stop cluster and remove data volumes |

### Python Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies (etcd3, grpcio, protobuf) |

### Exercises

| Command | Description |
|---------|-------------|
| `uv run python src/00_connect.py` | Test etcd connection, print cluster version and members |
| `uv run python src/01_kv_watch.py` | Exercise 1: KV CRUD operations and Watch API |
| `uv run python src/02_leases_service_discovery.py` | Exercise 2: Leases and service registration/discovery |
| `uv run python src/03_distributed_lock.py` | Exercise 3: Distributed locking with CAS transactions |
| `uv run python src/04_leader_election.py` | Exercise 4: Leader election with failover |

### Useful etcdctl Commands (for debugging)

| Command | Description |
|---------|-------------|
| `docker exec practice_051-etcd1 etcdctl put /test/key "hello"` | Put a key via etcdctl (verify cluster works) |
| `docker exec practice_051-etcd1 etcdctl get /test/key` | Get a key via etcdctl |
| `docker exec practice_051-etcd1 etcdctl get /services/ --prefix` | Get all keys under /services/ prefix |
| `docker exec practice_051-etcd1 etcdctl watch /services/ --prefix` | Watch all changes under /services/ (interactive) |
| `docker exec practice_051-etcd1 etcdctl lease grant 10` | Grant a 10-second lease |
| `docker exec practice_051-etcd1 etcdctl lease list` | List all active leases |

## References

- [etcd Official Documentation (v3.5)](https://etcd.io/docs/v3.5/)
- [etcd API Reference](https://etcd.io/docs/v3.5/learning/api/)
- [etcd API Guarantees](https://etcd.io/docs/v3.5/learning/api_guarantees/)
- [etcd Data Model](https://etcd.io/docs/v3.5/learning/data_model/)
- [Raft Consensus Algorithm](https://raft.github.io/)
- [Raft Paper (Ongaro & Ousterhout)](https://raft.github.io/raft.pdf)
- [python-etcd3 Documentation](https://python-etcd3.readthedocs.io/en/latest/usage.html)
- [python-etcd3 GitHub](https://github.com/kragniz/python-etcd3)
- [Jepsen: etcd 3.4.3 Analysis](https://jepsen.io/analyses/etcd-3.4.3)
- [Kubernetes etcd documentation](https://kubernetes.io/docs/concepts/overview/components/#etcd)

## State

`not-started`
