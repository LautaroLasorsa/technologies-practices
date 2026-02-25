# Practice 050: CRDTs — Conflict-Free Replicated Data Types

## Technologies

- **CRDTs** -- Data structures that guarantee convergence across replicas without coordination
- **Pure Python** -- Built from scratch using `dataclasses`, `abc`, `typing`, `uuid`, `copy`
- **No external dependencies** -- All CRDT logic implemented by hand to understand internals

## Stack

- Python 3.12+ (uv)
- No Docker needed

## Theoretical Context

### The Problem: Consistency vs. Availability in Distributed Systems

In distributed systems, data is replicated across multiple nodes for fault tolerance and low latency. When multiple replicas accept concurrent writes, conflicts arise: if node A sets `x = 1` and node B sets `x = 2` simultaneously, what should the final value be? Traditional solutions require **coordination** — locks, leader election, or consensus protocols (Paxos, Raft) — which sacrifice availability during network partitions and add latency to every write.

The CAP theorem (Brewer, 2000) formalizes this trade-off: a distributed system can provide at most two of Consistency, Availability, and Partition tolerance. Since network partitions are inevitable in practice, the real choice is between CP (consistent but sometimes unavailable — e.g., databases using 2PC or Raft) and AP (always available but eventually consistent — e.g., Dynamo-style systems). CRDTs are the mathematical foundation that makes AP systems work correctly.

### What CRDTs Are

A **Conflict-free Replicated Data Type** (CRDT) is a data structure that can be replicated across multiple nodes, updated independently and concurrently without any coordination between replicas, and is **mathematically guaranteed** to converge to a consistent state when replicas synchronize. The word "conflict-free" means the data structure's design eliminates conflicts by construction — there is no conflict resolution logic because conflicts cannot occur.

CRDTs provide **Strong Eventual Consistency (SEC)**: any two replicas that have received the same set of updates (in any order) are guaranteed to be in the same state. This is stronger than plain eventual consistency (which only promises convergence "eventually" without specifying when) because SEC guarantees **immediate convergence** once updates are delivered — no background reconciliation or conflict resolution needed.

The concept was formalized by Marc Shapiro, Nuno Preguica, Carlos Baquero, and Marek Zawirski in their 2011 paper "Conflict-free Replicated Data Types" (INRIA RR-7687) and the companion "A Comprehensive Study of Convergent and Commutative Replicated Data Types" (INRIA RR-7506).

### Two Families of CRDTs

#### State-based CRDTs (CvRDTs — Convergent Replicated Data Types)

Replicas periodically exchange their **full state**. Each replica merges received states using a `merge` function. The merge function must form a **join-semilattice** — meaning it must be:

- **Commutative**: `merge(A, B) = merge(B, A)` — order of merging doesn't matter
- **Associative**: `merge(merge(A, B), C) = merge(A, merge(B, C))` — grouping doesn't matter
- **Idempotent**: `merge(A, A) = A` — merging with yourself changes nothing

Additionally, all update operations must be **monotonically increasing** in the semilattice — state can only move "upward" (grow), never backward. This ensures that once information is added, it cannot be lost through merging.

**Advantages**: Simple protocol — just gossip your state to neighbors. Tolerates message loss, duplication, and reordering (idempotent merge handles all of these). **Disadvantage**: Transmitting full state can be expensive for large data structures (mitigated by delta-state CRDTs, which send only the delta since last sync).

#### Operation-based CRDTs (CmRDTs — Commutative Replicated Data Types)

Instead of exchanging states, replicas broadcast **operations** (e.g., "increment by 1", "add element X"). Each replica applies received operations to its local state. Operations must be **commutative** for concurrent operations — applying them in any order must yield the same result. Operations that are causally related (one happened-before the other) must be delivered in causal order.

**Advantages**: Smaller messages (just the operation, not the full state). **Disadvantages**: Requires a **reliable causal broadcast** layer — each operation must be delivered exactly once and in causal order. This is a stronger requirement on the network layer than state-based CRDTs need.

#### Equivalence

Shapiro et al. proved that the two families are theoretically equivalent: any state-based CRDT can be expressed as an operation-based CRDT and vice versa. The choice between them is a practical engineering decision based on network characteristics and payload sizes.

### Mathematical Foundation: Join-Semilattices

A **join-semilattice** is a partially ordered set (S, <=) where every pair of elements has a **least upper bound** (LUB), also called the **join**. The merge function computes this LUB.

**Example**: For a G-Counter with 3 replicas, states are vectors like `[2, 0, 1]`. The partial order is element-wise: `[2, 0, 1] <= [3, 0, 2]`. The join (merge) is element-wise max: `join([2, 0, 1], [1, 3, 0]) = [2, 3, 1]`. This forms a semilattice because:

- Max is commutative: `max(a,b) = max(b,a)`
- Max is associative: `max(max(a,b),c) = max(a,max(b,c))`
- Max is idempotent: `max(a,a) = a`

The **monotonicity** requirement means that local updates (e.g., incrementing your own counter entry) must only move the state upward in the partial order. Incrementing `[2, 0, 1]` at replica 0 gives `[3, 0, 1]`, which is >= `[2, 0, 1]` — monotonically increasing.

### Key CRDT Types

#### G-Counter (Grow-only Counter)

The simplest CRDT. Each of N replicas maintains a vector of N integers, one per replica. Replica `i` can only increment `vector[i]`. The counter value is `sum(vector)`. Merge takes element-wise max.

Why max and not sum? Because state-based merge must be idempotent. If replica A has `[5, 0]` and sends its state twice, merging with sum would give `[10, 0]` — wrong. Merging with max gives `[5, 0]` — correct.

**Semilattice**: vectors ordered element-wise, join = element-wise max.

#### PN-Counter (Positive-Negative Counter)

Supports both increment and decrement. Implemented as two G-Counters: `P` (positive/increments) and `N` (negative/decrements). `increment(replica_id)` adds to P, `decrement(replica_id)` adds to N. Value = `P.value() - N.value()`. Merge merges P and N independently.

This works because each G-Counter is independently a valid CRDT, and composing two CRDTs yields a CRDT. The value can go negative — the name "Positive-Negative" refers to the two internal counters, not the value range.

#### LWW-Register (Last-Writer-Wins Register)

Stores a single value with a timestamp. `assign(value, timestamp)` updates only if the new timestamp is higher. `merge(other)` keeps whichever has the higher timestamp. Tie-breaking (equal timestamps) uses a deterministic rule — typically lexicographic comparison of replica IDs.

**Trade-off**: Simple and widely used, but concurrent writes result in one being silently discarded. Amazon's CTO Werner Vogels described this as "the customer who made their purchase just a millisecond after the first one wins." Suitable when losing a concurrent write is acceptable (e.g., user profile updates).

**Semilattice**: ordered by timestamp (with tie-breaking), join = pick the greater.

#### MV-Register (Multi-Value Register)

Instead of picking one winner, keeps ALL concurrent values (like Amazon Dynamo's shopping cart). Uses **vector clocks** instead of timestamps to detect concurrency. When two writes are concurrent (neither happened-before the other), both values are preserved. The application or user resolves the conflict at read time.

**Semilattice**: ordered by vector clock dominance; concurrent values are incomparable and both kept.

#### G-Set (Grow-only Set)

Elements can only be added, never removed. Merge = set union. The simplest set CRDT.

**Semilattice**: subsets ordered by inclusion, join = union.

#### 2P-Set (Two-Phase Set)

Two G-Sets: `A` (added elements) and `R` (removed/tombstone elements). An element is in the set if it's in A but not in R. Once removed, an element can **never be re-added** — this is the "two-phase" restriction (add phase, then optionally remove phase, permanently).

**Limitation**: The tombstone set R grows forever (metadata bloat) and elements cannot be re-added. This is why OR-Set was invented.

#### OR-Set (Observed-Remove Set, also called Add-Wins Set)

The most practical set CRDT. Each `add(element)` generates a globally unique tag (e.g., UUID). Internal state maps each element to a set of unique tags. `remove(element)` removes only the tags **currently observed** at that replica. If another replica concurrently adds the same element (generating a new tag), that tag survives the remove — hence "add-wins" semantics.

**Merge**: For each element, take the union of tags that are in either replica but haven't been explicitly removed by either. Formally: `merged_tags(e) = (A_tags(e) | B_tags(e)) - (A_removed(e) | B_removed(e))`, but implementations typically track only active tags and use set difference during merge.

**Why this allows re-add after remove**: Removing an element removes its current tags. Adding it again generates a fresh tag not in any removal set. The element reappears with the new tag.

**Semilattice**: defined over the tag sets; merge is a carefully constructed union that respects add-wins semantics.

#### LWW-Element-Set

Each element has an add-timestamp and a remove-timestamp. Element is present if `add_timestamp > remove_timestamp`. Merge takes element-wise max of timestamps. Simpler than OR-Set but shares LWW-Register's "last write wins" semantics — concurrent add and remove of the same element is resolved by timestamp, which may not match user intent.

### Trade-offs and Limitations

- **Metadata growth**: CRDTs grow monotonically — tombstones, vector clocks, and unique tags accumulate. Garbage collection requires coordination (ironic for a coordination-free data structure), typically done during quiescent periods.
- **Expressiveness**: Not all data structures have natural CRDT representations. Sequences (text editing) are particularly complex — solutions like RGA, LSEQ, and YATA exist but have varying performance characteristics.
- **Semantics**: CRDTs define conflict resolution mathematically, but the resolution may not match application intent. OR-Set's "add-wins" is a policy choice, not always the right one.
- **Size overhead**: State-based CRDTs transmit full state (mitigated by delta-CRDTs). OR-Set's unique tags add per-element overhead.

### Real-World Usage

| System | CRDT Usage |
|--------|-----------|
| **Redis Enterprise** | CRDT-based multi-master geo-replication for counters, sets, strings |
| **Riak** | Distributed KV store with built-in CRDT support (counters, sets, maps, registers) |
| **Automerge** | JSON CRDT for collaborative editing (based on Kleppmann's work), Rust core with JS/WASM bindings |
| **Yjs** | High-performance CRDT framework for real-time collaborative text editing (used by JupyterLab, Nimbus Note) |
| **Apple Notes** | Uses CRDTs for offline sync across iCloud devices |
| **TomTom** | CRDTs for synchronizing navigation data across user devices |
| **Bet365** | Stores hundreds of MB in Riak OR-Sets for live betting data |
| **SoundCloud Roshi** | CRDT set layer on top of Redis for activity feeds |

### References

- Shapiro, M., Preguica, N., Baquero, C., Zawirski, M. (2011). "Conflict-free Replicated Data Types." INRIA Research Report RR-7687.
- Shapiro, M., Preguica, N., Baquero, C., Zawirski, M. (2011). "A Comprehensive Study of Convergent and Commutative Replicated Data Types." INRIA Research Report RR-7506.
- Kleppmann, M., Beresford, A. R. (2017). "A Conflict-Free Replicated JSON Datatype." IEEE Transactions on Parallel and Distributed Systems.
- Almeida, P. S., Shoker, A., Baquero, C. (2016). "Delta State Replicated Data Types." Journal of Parallel and Distributed Computing.
- crdt.tech -- Community resource hub for CRDT research and implementations.

## Description

Build a complete suite of CRDT implementations from scratch in pure Python. No external libraries — the goal is to deeply understand the mathematical properties (join-semilattice, commutativity, idempotency, monotonicity) that make CRDTs work, and to verify those properties empirically through simulation.

### What you'll build

1. **State-based counters** -- G-Counter and PN-Counter with merge-based synchronization
2. **Registers and sets** -- LWW-Register (timestamp-based) and OR-Set (tag-based, add-wins)
3. **Operation-based CRDTs** -- Counter with causal broadcast delivery simulation
4. **Convergence verification** -- Multi-replica simulation with network partitions, plus property-based testing of semilattice axioms

### What you'll learn

1. **Semilattice merge** -- Why `max` (not `sum`) is the correct merge for counters, and how this generalizes
2. **Metadata design** -- How unique tags in OR-Set solve the "re-add after remove" problem that 2P-Set cannot
3. **Causal delivery** -- Why operation-based CRDTs need vector clocks and what breaks without causal ordering
4. **Convergence testing** -- How to empirically verify that your implementations satisfy CRDT axioms under adversarial conditions

## Instructions

### Exercise 1: G-Counter and PN-Counter (~25 min)

**File:** `src/01_counters.py`

**Concepts:** State-based CRDT fundamentals. The G-Counter is the "hello world" of CRDTs — a vector where each replica owns one entry, merges via element-wise max, and queries via sum. The PN-Counter extends this to support decrements by composing two G-Counters.

**TODO(human) tasks:**
- Implement `GCounter` — Internal dict mapping `replica_id -> count`. `increment()` bumps your own entry. `value()` sums all entries. `merge()` takes element-wise max. Think about why max is correct (idempotency) and why sum would break.
- Implement `PNCounter` — Two GCounters (P for increments, N for decrements). `value() = P.value() - N.value()`. Merge delegates to both internal counters. Think about why the value can go negative and why that's fine.

After implementing, the scaffolded tests create 3 replicas, run concurrent increments/decrements, sync in various orders, and verify all replicas converge to the same value.

### Exercise 2: LWW-Register and OR-Set (~30 min)

**File:** `src/02_registers_sets.py`

**Concepts:** More complex state-based CRDTs. LWW-Register resolves conflicts by timestamp (simple but lossy). OR-Set uses unique tags per add-operation to support add-wins semantics — the key insight is that `remove` only removes tags that exist at the time of removal, so a concurrent `add` (with a fresh tag) survives.

**TODO(human) tasks:**
- Implement `LWWRegister` — State is `(value, timestamp, replica_id)`. `assign()` updates only if new timestamp is strictly greater (or equal timestamp but higher replica_id for tie-breaking). `merge()` keeps the winner.
- Implement `ORSet` — State is a dict mapping `element -> set of (unique_tag, replica_id)` pairs. `add()` generates a fresh UUID tag. `remove()` removes all currently observed tags for that element. `merge()` takes union of tags from both replicas, minus tags that were explicitly removed. `value()` returns elements with at least one active tag.

The OR-Set is the most complex CRDT in this practice. Pay attention to how the merge handles the case where one replica added an element (new tag) while another removed it (different tag) — both operations should take effect.

### Exercise 3: Operation-based CRDTs (~25 min)

**File:** `src/03_op_based.py`

**Concepts:** The other CRDT family — instead of merging full state, replicas broadcast operations. This requires a **causal broadcast** layer: operations must be delivered exactly once and in causal order (if operation A happened-before operation B at the sender, all replicas must see A before B). You'll implement both the CRDT and the broadcast layer.

**TODO(human) tasks:**
- Implement `OpBasedCounter` — Each operation is `(op_type, amount, replica_id, vector_clock)`. Apply is just add/subtract. Must track applied operations to handle re-delivery.
- Implement `CausalBroadcast` — Each replica has a vector clock. When broadcasting, stamp the op with the sender's current clock and increment it. When receiving, an op is deliverable only if all causally prior ops have been delivered (check vector clock). Buffer ops that arrive too early.

The test creates 5 replicas, broadcasts operations with simulated random delays and reordering, and verifies that causal delivery ensures convergence.

### Exercise 4: Convergence Simulation (~30 min)

**File:** `src/04_convergence_test.py`

**Concepts:** Put it all together. Simulate a realistic distributed environment: multiple replicas running concurrently, network partitions that split replicas into groups, and eventual healing. Verify that all CRDT implementations converge after partition healing. Also verify the semilattice properties (commutativity, associativity, idempotency) through property-based testing.

**TODO(human) tasks:**
- Implement `simulate_concurrent_updates()` — Create N replicas, run random updates, apply partition schedules (during partitions, replicas can only sync within their group), then do full sync and check convergence.
- Implement `verify_semilattice_properties()` — Generate random CRDT states, verify merge is commutative, associative, and idempotent. Report any violations.

This exercise ties together the theory (semilattice axioms) with practice (convergence under adversarial network conditions).

## Motivation

CRDTs are foundational to modern distributed systems. Understanding them deeply is valuable because:

- **Distributed databases** (Riak, Redis Enterprise, CockroachDB) use CRDTs internally for conflict resolution
- **Real-time collaboration** (Google Docs-style) is built on sequence CRDTs (Automerge, Yjs)
- **Edge computing and offline-first** apps rely on CRDTs to sync data without a central server
- **System design interviews** frequently ask about eventual consistency, conflict resolution, and CAP theorem trade-offs — CRDTs are the canonical answer
- **Complementary to Raft (Practice 049)** — Raft provides CP (strong consistency), CRDTs provide AP (availability). Understanding both gives a complete picture of the consistency spectrum

## Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies and create virtualenv (no external deps) |
| `uv run python src/00_crdt_base.py` | Run base classes self-test (verifies abstract interface) |
| `uv run python src/01_counters.py` | Exercise 1: G-Counter and PN-Counter |
| `uv run python src/02_registers_sets.py` | Exercise 2: LWW-Register and OR-Set |
| `uv run python src/03_op_based.py` | Exercise 3: Operation-based counter with causal broadcast |
| `uv run python src/04_convergence_test.py` | Exercise 4: Multi-replica convergence simulation |

## State

`not-started`
