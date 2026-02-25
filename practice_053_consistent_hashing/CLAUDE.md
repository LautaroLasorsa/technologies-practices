# Practice 053: Consistent Hashing & DHTs

## Technologies

- **Consistent Hashing** -- Ring-based key-to-server mapping with minimal disruption on topology changes
- **Jump Consistent Hashing** -- Google's zero-memory, O(ln N) consistent hash algorithm
- **Rendezvous Hashing (HRW)** -- Highest Random Weight hashing for distributed agreement
- **Chord DHT** -- Distributed Hash Table with O(log N) lookup via finger tables
- **hashlib / bisect** -- Python standard library for cryptographic hashing and sorted-list operations
- **matplotlib** -- Visualization of hash rings, distributions, and benchmarks

## Stack

- Python 3.12 (uv)
- No Docker needed -- pure algorithmic practice

## Theoretical Context

### The Problem: Data Placement in Distributed Systems

In any distributed system with N servers storing K keys, the fundamental question is: **which server stores a given key?** The naive approach -- `server = hash(key) % N` -- works until N changes. Adding or removing a single server causes `hash(key) % N != hash(key) % (N+1)` for most keys, reshuffling approximately `K * (N-1) / N` keys (nearly all of them). For a cluster with 10 servers and 1 million keys, adding one server moves ~909,000 keys. This is catastrophic for caches (mass invalidation, "thundering herd" to backends), databases (massive data migration), and any system where key-to-server stability matters.

The core requirement is: **when the set of servers changes, minimize the number of keys that must move**.

### Consistent Hashing (Karger et al., 1997)

Consistent hashing was introduced by David Karger, Eric Lehman, Tom Leighton, Rina Panigrahy, Matthew Levine, and Daniel Lewin in their 1997 paper "Consistent Hashing and Random Trees: Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web" (STOC '97). Notably, Daniel Lewin co-founded Akamai Technologies based on this work -- consistent hashing is the foundation of modern CDN load balancing.

**Algorithm.** Both keys and servers are hashed onto a circular space (a "ring") of size 2^m (typically m=32 or m=128). The hash function maps each server identifier (e.g., IP address, hostname) and each key to a position on this ring. To find which server stores a given key:

1. Hash the key to a position on the ring.
2. Walk clockwise from that position until you encounter a server.
3. That server is responsible for the key.

Equivalently: each server "owns" the arc of the ring from its predecessor (exclusive) to itself (inclusive). All keys falling in that arc belong to that server.

**Key migration.** When a server S is added, it takes responsibility for keys in the arc between its predecessor and itself -- keys that previously belonged to S's successor. Only those keys move. When a server is removed, its keys transfer to its successor. In both cases, only `O(K/N)` keys are affected, compared to `O(K)` for naive modulo hashing. This is the minimum possible disruption: if you add one server to N, the ideal is to move exactly `K/(N+1)` keys, and consistent hashing achieves this asymptotically.

**Lookup complexity.** With a sorted array of server positions and binary search (`bisect`), lookup is `O(log M)` where M is the total number of positions on the ring (number of servers times virtual nodes per server).

### Virtual Nodes (vnodes)

With only N physical servers on the ring, the arc lengths (and thus the number of keys per server) can be highly uneven -- especially for small N. If 5 servers are placed randomly on a ring, their arc lengths follow a Dirichlet distribution; the expected maximum load is `O(K * log(N) / N)` rather than the ideal `K/N`.

**Solution: virtual nodes.** Each physical server is mapped to V positions on the ring using distinct hash inputs: e.g., `hash("server-A:vnode0")`, `hash("server-A:vnode1")`, ..., `hash("server-A:vnode149")`. With V virtual nodes per physical server:

- The ring has `N * V` total positions, and by the law of large numbers, the arc lengths become increasingly uniform as V grows.
- The standard deviation of load per server decreases as `O(1/sqrt(V))`.
- In practice, V = 100-200 gives good balance; Amazon Dynamo uses ~150 virtual nodes per physical node.
- Adding a physical server adds V positions scattered across the ring, each stealing a small slice from a different existing server -- spreading the migration cost evenly.

**Trade-off:** More virtual nodes means more memory (storing V positions per server) and slightly slower lookup (binary search over N*V entries). Typical production systems choose V = 100-200 as the sweet spot.

### Jump Consistent Hashing (Lamping & Veach, 2014)

Google engineers John Lamping and Eric Veach introduced jump consistent hashing in their 2014 paper "A Fast, Minimal Memory, Consistent Hash Algorithm" (arXiv:1406.2294). It solves the same problem with radically different trade-offs.

**Algorithm.** Given a 64-bit key and N buckets, the algorithm returns a bucket number in `[0, N)`. It uses a linear congruential generator (LCG) seeded by the key to simulate the process of adding buckets one by one: for each new bucket j, the key "jumps" to bucket j with probability `1/j` (like the reservoir sampling argument). The elegant insight is that you can skip ahead -- instead of checking every bucket, you can compute the next bucket to jump to directly, yielding `O(ln N)` expected iterations.

The complete algorithm in pseudocode (from the paper):

```
int32_t JumpConsistentHash(uint64_t key, int32_t num_buckets) {
    int64_t b = -1, j = 0;
    while (j < num_buckets) {
        b = j;
        key = key * 2862933555777941757ULL + 1;
        j = (b + 1) * (double(1LL << 31) / double((key >> 33) + 1));
    }
    return b;
}
```

**Properties:**
- **Zero memory:** No ring, no sorted array -- just arithmetic. O(1) space.
- **O(ln N) time:** Expected number of iterations is `ln(N)`.
- **Perfectly balanced:** Each bucket gets exactly `1/N` of keys (no variance from virtual nodes).
- **Minimal disruption:** When going from N to N+1 buckets, exactly `1/(N+1)` of keys move.

**Limitation:** Jump consistent hashing only supports adding or removing buckets at the end (bucket N-1). You cannot remove an arbitrary server from the middle. This makes it suitable for systems where servers are numbered sequentially and only the total count changes (e.g., sharded storage with controlled scaling), but not for systems with arbitrary server failures.

### Rendezvous Hashing / Highest Random Weight (Thaler & Ravishankar, 1996-1998)

Rendezvous hashing, also called Highest Random Weight (HRW) hashing, was invented by David Thaler and Chinya Ravishankar at the University of Michigan. It was described in a 1996 technical report (CSE-TR-316-96) and formally published in 1998 as "Using Name-Based Mappings to Increase Hit Rates" (IEEE/ACM Transactions on Networking, Vol. 6, Issue 1).

**Algorithm.** For a given key, compute `hash(key, server_id)` for every server in the cluster. The server with the highest hash value "wins" and is assigned the key. That's it -- no ring, no sorted positions.

```python
def get_server(key: str, servers: list[str]) -> str:
    return max(servers, key=lambda s: hash(key + s))
```

**Properties:**
- **Simple:** The algorithm is trivially understood and implemented.
- **Balanced:** With a good hash function, keys are uniformly distributed.
- **Minimal disruption:** When a server is added or removed, only `O(K/N)` keys move -- the same optimal bound as consistent hashing. A key only moves if the added server has a higher hash than its current server, which happens with probability `1/(N+1)`.
- **Supports arbitrary topology changes:** Any server can be added or removed, unlike jump consistent hashing.
- **O(N) per lookup:** Must compute hash against every server. This is the main disadvantage. For clusters with thousands of servers, this becomes significant.

**Comparison with consistent hashing:** Rendezvous hashing achieves the same key migration properties without any data structure (no ring, no sorted array). However, lookup is O(N) vs O(log M) for consistent hashing with binary search. For small N (< 100 servers), rendezvous hashing is often preferred for its simplicity. For large N, consistent hashing with virtual nodes is more efficient.

Microsoft's Cache Array Routing Protocol (CARP) adopted rendezvous hashing in 1998 for distributed cache coordination.

### Distributed Hash Tables (DHTs)

A DHT is a decentralized distributed system that provides a hash-table-like interface: `put(key, value)` and `get(key)` -- but data is spread across many nodes with no central directory. Each node stores a portion of the key-value space and can route queries to the correct node in a bounded number of hops.

DHTs are the backbone of peer-to-peer systems: BitTorrent (Kademlia-based Mainline DHT for trackerless torrents), IPFS (Kademlia-based content routing), and historically Napster successors like Gnutella2. They are also conceptually foundational for understanding distributed databases like Cassandra and DynamoDB, which use DHT-like consistent hashing for data placement.

Key DHT protocols:

| Protocol | Topology | Distance Metric | Lookup Hops | Year |
|----------|----------|-----------------|-------------|------|
| **Chord** | Ring | Clockwise distance on ring | O(log N) | 2001 |
| **Kademlia** | Binary tree (XOR) | XOR of node IDs | O(log N) | 2002 |
| **Pastry** | Prefix-based | Shared prefix length | O(log N) | 2001 |
| **Tapestry** | Prefix-based | Suffix matching | O(log N) | 2001 |
| **CAN** | d-dimensional torus | Cartesian distance | O(d * N^(1/d)) | 2001 |

### Chord Protocol (Stoica et al., 2001)

Chord was introduced by Ion Stoica, Robert Morris, David Karger, M. Frans Kaashoek, and Hari Balakrishnan in "Chord: A Scalable Peer-to-peer Lookup Service for Internet Applications" (SIGCOMM '01). The paper won the ACM SIGCOMM Test of Time Award in 2011.

**Structure.** Chord organizes nodes on a ring of size `2^m` (m-bit identifier space). Each node has an m-bit ID (typically the hash of its IP address). Each key is assigned to the first node whose ID is equal to or follows the key's hash on the ring -- this node is called the key's **successor**.

**Finger table.** Each node maintains a routing table (the "finger table") with m entries. Entry i (0-indexed) of node n's finger table points to the successor of `(n + 2^i) mod 2^m`. This means:
- Entry 0 points to the immediate successor (distance 1).
- Entry 1 points to the successor at distance 2.
- Entry 2 points to the successor at distance 4.
- Entry i points to the successor at distance 2^i.

This exponentially increasing stride is the key to O(log N) lookup: each hop at least halves the remaining distance to the target.

**Lookup algorithm.** To find the node responsible for key k:
1. If k falls between the current node and its successor, return successor.
2. Otherwise, find the closest preceding finger: scan the finger table from entry m-1 down to 0, and find the largest finger that precedes k on the ring.
3. Forward the query to that finger node and repeat.

Each step at least halves the distance, so lookup completes in O(log N) hops with high probability when N nodes are in the ring.

**Node join.** When a new node n joins via an existing node n':
1. n asks n' to find n's successor: `successor = n'.find_successor(n.id)`.
2. n sets its successor and copies responsibility for keys in the range `(predecessor, n]` from its successor.
3. n notifies its successor to update its predecessor pointer.
4. Finger tables are updated via the stabilization protocol.

**Stabilization.** Chord runs periodic maintenance:
- **stabilize():** Each node n asks its successor for its predecessor p. If p is between n and successor, then p is a better successor for n (a new node joined between them). Update accordingly.
- **fix_fingers():** Periodically pick a random finger table entry i and recompute it by calling `find_successor(n + 2^i)`.
- **check_predecessor():** Verify that the predecessor is still alive; if not, set predecessor to nil.

**Complexity guarantees:**
- Lookup: O(log N) hops
- Storage per node: O(log N) finger table entries + keys in the node's range
- Join/leave: O(log^2 N) messages to update finger tables

### Replication on Consistent Hashing Rings

In production systems, keys are replicated across multiple servers for fault tolerance. The standard approach: store each key on the next R servers clockwise on the ring (R = replication factor). The list of R servers responsible for a key is called the **preference list**.

Amazon Dynamo (DeCandia et al., "Dynamo: Amazon's Highly Available Key-value Store", SOSP 2007) uses this approach with a "sloppy quorum": reads and writes go to the first N healthy nodes from the preference list, allowing the system to remain available even when some replicas are down. Dynamo pioneered the combination of consistent hashing + virtual nodes + sloppy quorum + vector clocks for eventually-consistent key-value storage.

Apache Cassandra uses a similar token ring with virtual nodes (called "vnodes") where each node owns multiple token ranges. The replication factor is configured per keyspace, and replicas are the next RF-1 distinct physical nodes clockwise on the ring.

### Real-World Applications

| System | How It Uses Consistent Hashing |
|--------|-------------------------------|
| **Amazon DynamoDB** | Consistent hashing ring with virtual nodes for partition placement, preference lists for replication |
| **Apache Cassandra** | Token ring with vnodes; each node owns multiple token ranges |
| **Redis Cluster** | 16384 hash slots distributed across nodes; keys map to slots via CRC16 |
| **Memcached** | Client-side consistent hashing (libketama) for cache distribution |
| **Akamai CDN** | Consistent hashing for request routing to edge servers (original motivation for the 1997 paper) |
| **Discord** | Consistent hashing for gateway server routing (user-to-gateway assignment) |
| **Riak** | Dynamo-inspired ring with virtual nodes |
| **Apache Kafka** | Consumer group partition assignment uses range/round-robin, but some client libraries offer sticky/consistent-hash assignors |

### References

- Karger, D. et al. (1997). "Consistent Hashing and Random Trees: Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web." *Proceedings of the 29th Annual ACM Symposium on Theory of Computing (STOC '97)*.
- Stoica, I. et al. (2001). "Chord: A Scalable Peer-to-peer Lookup Service for Internet Applications." *ACM SIGCOMM '01*.
- Lamping, J. & Veach, E. (2014). "A Fast, Minimal Memory, Consistent Hash Algorithm." *arXiv:1406.2294*.
- Thaler, D. & Ravishankar, C. (1998). "Using Name-Based Mappings to Increase Hit Rates." *IEEE/ACM Transactions on Networking, 6(1)*.
- DeCandia, G. et al. (2007). "Dynamo: Amazon's Highly Available Key-value Store." *SOSP '07*.

## Description

Implement and compare the three major consistent hashing algorithms, then build a simplified Chord DHT to understand O(log N) routing in distributed hash tables. The practice progresses from basic ring-based consistent hashing through key migration analysis to a working Chord implementation, finishing with a head-to-head comparison of all approaches.

### What you'll learn

1. **Consistent hashing fundamentals** -- Ring structure, virtual nodes, clockwise assignment, and why O(K/N) migration is optimal.
2. **Key migration analysis** -- Quantify how many keys move when servers join/leave, comparing consistent vs naive hashing empirically.
3. **Chord DHT internals** -- Finger tables, O(log N) routing, node join protocol, and stabilization.
4. **Algorithm trade-offs** -- When to use ring-based vs jump vs rendezvous hashing (memory, latency, flexibility).

## Instructions

### Exercise 1: Consistent Hash Ring with Virtual Nodes (~25 min)

**File:** `src/01_consistent_hash_ring.py`

Build a consistent hash ring from scratch. You will implement the core data structure that maps keys to servers using a sorted ring of virtual node positions with O(log N) binary-search lookup.

**What it teaches:** The ring abstraction is the foundation of consistent hashing. By implementing add/remove/lookup, you internalize why sorted positions + binary search gives efficient clockwise traversal, why virtual nodes improve balance, and how the ring structure naturally minimizes key migration. The visualization makes the abstract ring concrete.

1. First, run `src/00_hash_utils.py` to verify the hash utilities work.
2. Implement `ConsistentHashRing.__init__` and `add_node` -- the ring data structure using `bisect` for O(log N) lookup. Each physical node maps to `num_virtual_nodes` positions on the ring.
3. Implement `ConsistentHashRing.get_node` -- clockwise lookup via `bisect_right` with wrap-around.
4. Implement `ConsistentHashRing.remove_node` -- clean removal of all virtual node positions for a physical node.
5. Implement `visualize_ring` -- matplotlib visualization of the ring showing nodes, keys, and assignments.
6. Run the script and inspect the load distribution and ring visualization.

### Exercise 2: Key Migration Analysis (~20 min)

**File:** `src/02_key_migration.py`

Quantify the key migration cost of adding/removing servers under consistent hashing vs naive modulo hashing. This is the core value proposition of consistent hashing made measurable.

**What it teaches:** The theoretical claim is that consistent hashing moves O(K/N) keys vs O(K) for naive hashing. By measuring this empirically for different cluster sizes, you verify the theory and see the dramatic practical difference. The bar chart makes the comparison visually striking.

1. Implement `measure_key_migration` -- snapshot key assignments before and after a topology change, count movements.
2. Implement `compare_naive_vs_consistent` -- run both strategies, compare migration ratios, and generate a comparison plot.
3. Run the script for various cluster sizes (3, 5, 10, 20 servers) and verify the results match theoretical predictions.

### Exercise 3: Chord DHT with Finger Tables (~30 min)

**File:** `src/03_chord_dht.py`

Implement a simplified Chord DHT to understand how distributed hash tables achieve O(log N) lookup using finger tables. This is the most algorithmically rich exercise.

**What it teaches:** Chord's finger table is a brilliant data structure: entry i covers distance 2^i on the ring, so each hop halves the remaining distance. By implementing find_successor, join, stabilize, and fix_fingers, you understand the complete lifecycle of a DHT node. The hop-count measurement confirms O(log N) empirically.

1. Implement `ChordNode.__init__` -- the finger table structure with m entries.
2. Implement `ChordNode.find_successor` -- the O(log N) lookup algorithm using finger table routing.
3. Implement `ChordNode.join` -- joining an existing ring and transferring key responsibility.
4. Implement `stabilize` and `fix_fingers` -- the maintenance protocols that keep routing correct.
5. Run the script to create a 16-node ring, perform lookups, and verify O(log N) average hops.

### Exercise 4: Algorithm Comparison (~15 min)

**File:** `src/04_comparison.py`

Implement jump consistent hashing and rendezvous hashing, then benchmark all three approaches head-to-head on load balance, lookup speed, key migration, and memory usage.

**What it teaches:** Each hashing algorithm makes different trade-offs. Jump hashing is fastest and perfectly balanced but only supports sequential server changes. Rendezvous hashing is simplest but O(N) per lookup. Consistent hashing with vnodes is the most flexible but uses more memory. Seeing all three side by side crystallizes when to use each in practice.

1. Implement `JumpConsistentHash` -- the ~5-line algorithm from Lamping & Veach's paper.
2. Implement `RendezvousHash` -- highest random weight selection.
3. Implement `benchmark_comparison` -- measure all three on load balance, latency, migration, and memory.
4. Run the script and analyze the multi-panel comparison chart.

## Motivation

Consistent hashing is a fundamental building block of distributed systems -- it appears in virtually every distributed database (DynamoDB, Cassandra), cache layer (Memcached, Redis Cluster), CDN (Akamai), and peer-to-peer network (BitTorrent, IPFS). Understanding it deeply is essential for:

- **System design interviews**: Consistent hashing is a top-3 topic in distributed systems interviews. Knowing the algorithm, virtual nodes, and trade-offs vs alternatives (jump, rendezvous) sets you apart.
- **Production debugging**: When a node fails in a consistent-hashing-based cluster, understanding which keys migrate and where helps diagnose hot spots and rebalancing issues.
- **Architecture decisions**: Choosing between consistent hashing, jump hashing, and rendezvous hashing requires understanding their trade-offs in memory, latency, and flexibility.
- **Complementary to existing skills**: Builds on the distributed systems patterns from practices 014 (SAGA), 049 (Raft), and 050 (CRDTs), adding the data placement layer.

## Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies (matplotlib, numpy) |
| `uv run python src/00_hash_utils.py` | Run hash utility self-tests |
| `uv run python src/01_consistent_hash_ring.py` | Run consistent hash ring implementation |
| `uv run python src/02_key_migration.py` | Run key migration analysis and generate comparison plots |
| `uv run python src/03_chord_dht.py` | Run Chord DHT implementation and verify O(log N) lookup |
| `uv run python src/04_comparison.py` | Run full comparison of all three hashing algorithms |

## State

`not-started`
