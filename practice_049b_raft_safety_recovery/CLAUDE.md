# Practice 049b -- Raft Consensus: Safety & Failure Recovery

## Technologies

- **Raft Consensus Algorithm** -- Safety proofs, log compaction, linearizable reads (Ongaro & Ousterhout, 2014)
- **asyncio** -- Python's async/await runtime for simulating concurrent nodes
- **dataclasses** -- Structured Raft message types and node state
- **logging** -- Structured event logging for observing consensus behavior
- Pure Python implementation -- no external dependencies

## Stack

Python 3.12+ (uv). No Docker needed -- nodes are simulated as async tasks in a single process, communicating via direct async function calls instead of network RPCs.

## Theoretical Context

### Prerequisite: Practice 049a

This practice builds on 049a (Leader Election & Log Replication). It assumes you understand terms, node states (Follower/Candidate/Leader), RequestVote and AppendEntries RPCs, log consistency checks, and commitment via majority replication. Here we focus on the **safety properties** that make Raft correct, **failure recovery** scenarios, **log compaction**, and **linearizable reads**.

### Raft's Five Safety Properties

The Raft paper (Figure 3, Section 5.4) defines five properties that together guarantee the replicated state machine behaves as a single correct machine. These are not features to implement -- they are **invariants** that hold as consequences of the protocol rules. Verifying them on a running cluster is how you gain confidence in the implementation's correctness.

#### 1. Election Safety

**Statement**: At most one leader can be elected in a given term.

**Why it holds**: Each node votes for at most one candidate per term (stored in persistent `votedFor`), and a candidate needs a strict majority `(N/2)+1` to win. Since any two majorities overlap by at least one node, and that node cannot vote for two different candidates, two candidates cannot both win in the same term. This is the most fundamental invariant -- if it breaks, the entire protocol is unsound.

**Verification**: Record `(term, leader_id)` for every leader transition across all nodes. For each term, assert at most one distinct `leader_id` appears.

Reference: Raft paper Section 5.2.

#### 2. Leader Append-Only

**Statement**: A leader never overwrites or deletes entries in its log; it only appends new entries.

**Why it holds**: The leader code path only appends entries received from clients. Overwriting or truncation only happens in followers when the consistency check reveals a mismatch -- followers discard conflicting entries and accept the leader's entries. The leader itself never modifies existing entries.

**Verification**: After each operation on a leader, confirm that no previously existing entry has changed its term or command.

Reference: Raft paper Section 5.3.

#### 3. Log Matching Property

**Statement**: If two logs contain an entry with the same index and term, then the logs are identical in all entries up through that index.

**Why it holds**: This is an inductive invariant maintained by the AppendEntries consistency check. When a leader sends entries, it includes `prevLogIndex` and `prevLogTerm`. The follower only accepts the entries if its log matches at that point. Since entries are created by a single leader per term (Election Safety) and the leader assigns strictly increasing indices within a term, if two entries have the same index and term, they must be the same entry. By induction (the consistency check verifies the preceding entry), all entries before them must also match.

**Verification**: For all pairs of nodes, find entries with matching `(index, term)` pairs. For each such pair, verify all preceding entries also match.

Reference: Raft paper Section 5.3, Proof in Section 5.4.

#### 4. Leader Completeness

**Statement**: If a log entry is committed in a given term, that entry will be present in the logs of the leaders for all higher-numbered terms.

**Why it holds**: This follows from the election restriction. A committed entry has been replicated to a majority. A candidate can only win with votes from a majority. Since any two majorities overlap, at least one voter has the committed entry. The voter only grants its vote if the candidate's log is **at least as up-to-date** (compared by last entry's term, then index). Therefore, the winning candidate must have the committed entry. This is the deepest safety property -- it connects election and replication.

**Verification**: Track which entries are committed and in which term. Then for every subsequent leader, verify its log contains all previously committed entries.

Reference: Raft paper Section 5.4.3, proof by contradiction.

#### 5. State Machine Safety

**Statement**: If a server has applied a log entry at a given index to its state machine, no other server will ever apply a different log entry for the same index.

**Why it holds**: This follows from Log Matching and Leader Completeness. If a server applied entry E at index I, then E was committed (applied entries are always committed). By Leader Completeness, all future leaders have E at index I. By Log Matching, all followers will eventually have E at index I (the leader replicates it). Therefore, no other entry can ever be applied at index I.

**Verification**: After entries are applied, check that all nodes' applied entries at each index are identical.

Reference: Raft paper Section 5.4.3.

### Network Partitions: The Ultimate Stress Test

A **network partition** splits the cluster into groups that cannot communicate. Raft handles partitions correctly because of its majority-based design:

**Majority partition** (>= `(N/2)+1` nodes): Can elect a leader, commit entries, and serve clients normally. The majority partition doesn't know (or care) about the minority -- it has enough nodes for quorum operations.

**Minority partition** (< `(N/2)+1` nodes): Cannot elect a leader (not enough votes for majority). If the old leader is in the minority, it will continue sending heartbeats to its partition peers, but cannot commit any new entries (not enough matchIndex confirmations). Eventually the minority nodes will time out and start elections, but since they can't get a majority, they'll keep incrementing their terms fruitlessly.

**Partition heals**: When connectivity is restored, the minority nodes discover the majority partition's higher term. They step down to follower, adopt the new leader, and the leader repairs their logs via the normal AppendEntries consistency check. Any uncommitted entries on minority nodes are **overwritten** -- this is safe because they were never committed (never reached a majority).

**Key insight**: The minority partition's stale leader may have accepted client requests and appended them to its log, but it could never commit them. Clients talking to the minority partition would see timeouts (no commit confirmation). Well-designed Raft clients retry with the majority partition.

Reference: Raft paper Section 5.4; [Baeldung Raft Guide](https://www.baeldung.com/cs/raft-consensus-algorithm).

### Leader Crashes and Log Reconciliation

When a leader crashes, the new leader may find followers with divergent logs -- entries that the old leader replicated to some followers but not others, or entries from even older terms. Raft handles this via the **log repair mechanism**:

1. The new leader initializes `nextIndex[i] = leader.lastLogIndex + 1` for each follower.
2. AppendEntries RPCs to each follower include a consistency check at `prevLogIndex`.
3. If a follower's log doesn't match, it rejects the RPC. The leader decrements `nextIndex[i]` and retries.
4. Eventually, the leader finds the point where the follower's log matches, and overwrites everything after that point with the leader's log.

This mechanism guarantees convergence: after at most O(log_length) rounds of backtracking, every follower's log matches the leader's. The leader never modifies its own log -- it only forces followers to match.

**The subtle Section 5.4.2 bug**: A new leader must NOT commit entries from previous terms by counting replicas alone. It can only commit entries from its **own** term. Previous-term entries get committed indirectly when a current-term entry after them is committed. This prevents a scenario where a leader could commit an entry, crash, and a new leader could overwrite it.

Reference: Raft paper Section 5.3 (log repair), Section 5.4.2 (commitment restriction).

### Log Compaction and Snapshots

In a production system, the Raft log grows indefinitely as clients submit commands. Without compaction, it would consume unbounded storage and make restart (replaying the entire log) impossibly slow. Raft uses **snapshotting** as its compaction mechanism.

#### How Snapshots Work

1. Each server independently decides when to snapshot (typically when the log exceeds a size threshold).
2. The server serializes the **current state machine state** to stable storage. This state represents the cumulative effect of all log entries up to some index.
3. The server records metadata: `last_included_index` (the index of the last entry covered by the snapshot) and `last_included_term` (the term of that entry).
4. All log entries up to and including `last_included_index` are discarded. The snapshot replaces them.
5. After snapshotting, the log starts from `last_included_index + 1`.

#### InstallSnapshot RPC

When a leader needs to bring a very stale follower up to date, the follower may need entries that the leader has already discarded (compacted into a snapshot). In this case, the leader sends an `InstallSnapshot` RPC containing:

- `term`: Leader's current term
- `leader_id`: Leader's ID
- `last_included_index`: Index of the last entry in the snapshot
- `last_included_term`: Term of that entry
- `data`: The serialized state machine snapshot

When a follower receives InstallSnapshot:
- If the snapshot is more recent than the follower's current state (`last_included_index` > follower's `last_included_index`): discard the entire log up to `last_included_index`, load the snapshot as the new state machine state, update metadata.
- If the follower has entries beyond `last_included_index`: keep those entries (they're more recent than the snapshot).
- If the snapshot is older than the follower's current state: ignore it.

**Design choice**: Each server snapshots independently. The leader doesn't coordinate snapshotting -- it only sends snapshots when a follower is too far behind for log-based repair. This keeps the common case (log-based replication) simple and efficient.

Reference: Raft paper Section 7 (Log Compaction); [MIT 6.824 Lab 2D](https://www.mo4tech.com/raft-part-d-mit-6-824-lab2d-log-compaction.html).

### Cluster Membership Changes

Real systems need to add and remove nodes (scaling, replacing hardware). Raft provides two approaches:

#### Joint Consensus (Original Paper, Section 6)

A two-phase approach:
1. Leader creates a `C_old,new` configuration entry containing both old and new configurations. Decisions require **separate majorities from both** `C_old` and `C_new`.
2. Once `C_old,new` is committed, the leader creates a `C_new` entry. Once committed, the old configuration nodes not in `C_new` can be shut down.

This guarantees safety because at no point can two disjoint majorities make independent decisions.

#### Single-Server Changes (PhD Thesis, Simpler)

Add or remove one server at a time. If configurations differ by at most one server, any majority from the old and any majority from the new configuration must overlap by at least one server. This is simpler to implement and is used by most production Raft implementations (etcd, HashiCorp Raft). However, a [safety bug was found](https://groups.google.com/g/raft-dev/c/t4xj6dJTP6E) in 2015 relating to leader switching during single-server changes -- implementations must handle this edge case carefully.

Reference: Raft paper Section 6; [Ongaro PhD thesis](https://web.stanford.edu/~ouster/cgi-bin/papers/OngaroPhD.pdf) Chapter 4; [Alibaba Cloud analysis](https://www.alibabacloud.com/blog/raft-engineering-practices-and-the-cluster-membership-change_597742).

### Linearizable Reads

Raft provides linearizable **writes** by default (committed entries are durably replicated). But **reads** are trickier. A naive implementation that reads from the leader's local state can return stale data if the leader has been deposed but doesn't know it yet (it hasn't received a message from the new leader). Two approaches solve this:

#### Quorum Reads (ReadIndex)

The leader must confirm it is still the leader before serving a read:
1. Record the current `commitIndex` as the `readIndex`.
2. Send heartbeats to all followers. Wait for acknowledgment from a majority.
3. If a majority acknowledges, the leader is still authoritative. Wait until the state machine has applied entries up to `readIndex`.
4. Serve the read from local state.

This approach is used by etcd (`ReadOnlySafe` mode). It adds one round-trip of heartbeats per read but guarantees linearizability. The [etcd/raft README](https://github.com/etcd-io/raft/blob/main/README.md) documents this as the default and recommended approach.

#### Lease-Based Reads

The leader maintains a **lease** -- a time window during which it knows no other leader can exist:
1. After receiving heartbeat acknowledgments from a majority, the leader starts a lease timer equal to the election timeout.
2. During the lease period, the leader can serve reads locally without contacting followers (since no election can complete within the election timeout).
3. If the lease expires, fall back to quorum reads.

This is faster (no extra round-trip) but relies on **clock assumptions** -- all servers' clocks must advance at roughly the same rate. If a server's clock runs fast, it might start an election before the leader's lease expires, violating linearizability. The [LeaseGuard paper](https://arxiv.org/html/2512.15659v1) (Davis, 2025) provides a rigorous analysis of lease-based reads and common pitfalls.

**Trade-off**: Quorum reads are always safe but slower. Lease-based reads are faster but assume bounded clock drift. Most production systems offer both options.

Reference: [Diving into etcd's linearizable reads](https://pierrezemb.fr/posts/diving-into-etcd-linearizable/); [etcd raft README](https://github.com/etcd-io/raft/blob/main/README.md).

### Byzantine Failures: What Raft Does NOT Handle

Raft assumes a **crash-fault model**: nodes may crash, restart, or be partitioned, but they never lie. A crashed node stops responding; when it recovers, it resumes honestly from its persisted state. Raft does NOT handle **Byzantine faults** -- nodes that send incorrect messages, forge votes, corrupt logs, or behave maliciously.

For Byzantine fault tolerance (BFT), protocols like **PBFT** (Practical Byzantine Fault Tolerance, Castro & Liskov, 1999) are needed:
- PBFT requires `3f + 1` replicas to tolerate `f` Byzantine nodes (vs. Raft's `2f + 1` for `f` crash faults)
- PBFT uses cryptographic signatures and multiple voting rounds
- PBFT has O(n^2) message complexity per operation (vs. Raft's O(n))

In practice, BFT is used in adversarial environments (blockchain, multi-party systems). In trusted datacenter environments, crash-fault tolerance (Raft, Paxos) is sufficient and far more efficient.

Reference: [CS6213 Week 3: BFT](https://ilyasergey.net/CS6213/week-03-bft.html); [Stanford Raftlet paper](https://www.scs.stanford.edu/24sp-cs244b/projects/Raftlet_A_Byzantine_Fault_Tolerant_Raft.pdf).

### References

- Ongaro, D. & Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm." USENIX ATC '14. [https://raft.github.io/raft.pdf](https://raft.github.io/raft.pdf) -- Sections 5.4 (Safety), 6 (Membership), 7 (Compaction)
- Ongaro, D. (2014). "Consensus: Bridging Theory and Practice." PhD Thesis. [https://web.stanford.edu/~ouster/cgi-bin/papers/OngaroPhD.pdf](https://web.stanford.edu/~ouster/cgi-bin/papers/OngaroPhD.pdf)
- Davis, A. J. J. (2025). "LeaseGuard: Raft Leases Done Right." [https://arxiv.org/html/2512.15659v1](https://arxiv.org/html/2512.15659v1)
- Zemb, P. "Diving into etcd's linearizable reads." [https://pierrezemb.fr/posts/diving-into-etcd-linearizable/](https://pierrezemb.fr/posts/diving-into-etcd-linearizable/)
- etcd Raft library: [https://github.com/etcd-io/raft](https://github.com/etcd-io/raft)
- MIT 6.824 Raft FAQ: [https://pdos.csail.mit.edu/6.824/papers/raft-faq.txt](https://pdos.csail.mit.edu/6.824/papers/raft-faq.txt)

## Description

Extend the Raft core implementation from 049a with safety property verification, network partition simulation and recovery, log compaction via snapshots, and linearizable read implementations. The core Raft engine (`00_raft_core.py`) is provided fully working -- each exercise builds on it to explore a different aspect of Raft's correctness and production-readiness.

### What you'll learn

1. **Safety verification** -- How to verify Election Safety, Log Matching, and Leader Completeness on a running cluster by collecting and analyzing state history
2. **Partition recovery** -- What happens during and after a network partition: majority continues, minority stalls, logs diverge, then reconcile on heal
3. **Log compaction** -- Why snapshots are needed, how to create them, how InstallSnapshot RPC brings stale nodes up to date
4. **Linearizable reads** -- The difference between quorum reads (always safe) and lease-based reads (faster but clock-dependent), and when each can fail

### Architecture

```
  Single Python Process (asyncio event loop)
  +--------------------------------------------------+
  |                                                   |
  |   Node 0        Node 1        Node 2             |
  |  (Follower)    (Leader)      (Follower)           |
  |     |             |              |                |
  |     +------+------+------+------+                |
  |            |             |                        |
  |         Node 3        Node 4                      |
  |        (Follower)    (Follower)                   |
  |                                                   |
  |  Communication: Network object routes messages    |
  |  Partitions: NetworkSimulator drops cross-group   |
  |  Snapshots: Serialize state_machine dict          |
  +--------------------------------------------------+
```

## Instructions

### Phase 1: Core Raft Engine (~5 min)

Run the provided core Raft implementation to verify it works:

```
uv sync
uv run python src/00_raft_core.py
```

This file provides a complete working Raft cluster with leader election, log replication, AppendEntries, RequestVote, and commit logic. Read through it to understand the API surface you'll use in exercises 1-4. Key classes: `RaftNode`, `Network`, `RaftCluster`.

### Phase 2: Safety Property Verification (~25 min)

**Exercise 1: Verify Raft's safety invariants on a running cluster.**

The safest Raft implementation is one where you can **prove** correctness at runtime. This exercise builds a test harness that runs the cluster through multiple rounds of elections and operations, collects state history, and verifies three of the five safety properties.

File: `src/01_safety_verification.py`

Three TODO(human) blocks:

- `verify_election_safety(cluster_history)` -- Given a history of `(term, node_id, state)` snapshots captured during cluster operation, verify that no two nodes were simultaneously leader in the same term. This is the most fundamental Raft invariant. You'll iterate through the history grouped by term and check for duplicate leaders.

- `verify_log_matching(nodes)` -- For all pairs of nodes, find entries where both nodes have an entry at the same index with the same term. For each such pair, verify that ALL preceding entries are also identical (same term and command). This verifies the Log Matching Property which is maintained by the AppendEntries consistency check. A violation here means the consistency check has a bug.

- `verify_leader_completeness(commit_history, leader_history)` -- Given a record of which entries were committed in which term and which nodes became leader in which terms, verify that every entry committed in term T is present in the log of every leader elected in terms > T. This is the deepest safety property and the hardest to verify.

**Why this matters**: Runtime invariant checking is standard practice in production Raft implementations. etcd's Raft library includes numerous assertion checks. TLA+ model checking of Raft verifies these same properties. Building verifiers teaches you exactly what "correct" means.

Run: `uv run python src/01_safety_verification.py`

### Phase 3: Network Partition Simulation (~30 min)

**Exercise 2: Simulate a network partition and verify correct recovery.**

Network partitions are the defining challenge for consensus algorithms. This exercise simulates a 5-node cluster being split into majority [0,1,2] and minority [3,4], verifying that the majority continues making progress while the minority is stuck, and that after healing, all nodes converge.

File: `src/02_partition_recovery.py`

Two TODO(human) blocks:

- `class NetworkSimulator` with `partition(group_a, group_b)` and `heal()` -- Build a network layer that can simulate partitions. When partitioned, messages between group_a and group_b are dropped. Messages within each group are delivered normally. The heal method restores full connectivity. This simulates real-world network failures (switch failure, datacenter split).

- `async def run_partition_scenario(nodes, simulator)` -- Implement the full partition recovery scenario: (1) normal operation and commits, (2) partition into majority/minority, (3) verify majority continues committing, (4) verify minority cannot commit, (5) heal, (6) verify all logs converge. Each step must be observable and verified.

**Why this matters**: Partition tolerance is the "P" in CAP theorem. Understanding how Raft maintains consistency (C) and sacrifices availability (A) in the minority partition during a split is essential for designing partition-tolerant systems.

Run: `uv run python src/02_partition_recovery.py`

### Phase 4: Log Compaction (~25 min)

**Exercise 3: Implement snapshotting and InstallSnapshot RPC.**

Without log compaction, a Raft node's log grows without bound. This exercise adds snapshot creation (serialize state machine, discard log prefix) and the InstallSnapshot RPC (leader sends snapshot to stale followers). This is how production Raft systems keep memory and disk usage bounded.

File: `src/03_log_compaction.py`

Three TODO(human) blocks:

- `RaftNode.create_snapshot()` -- Serialize the state machine at a given log index, record the snapshot metadata, and discard all log entries up to that index. This is the core compaction operation.

- `RaftNode.handle_install_snapshot()` -- Handle the InstallSnapshot RPC from the leader. Decide whether the snapshot is newer than our state, and if so, replace our log and state machine with the snapshot data.

- `should_snapshot(node, threshold)` -- A policy function that decides when to trigger snapshotting based on log size. Called periodically; returns True when the log exceeds the threshold.

**Why this matters**: Log compaction is mandatory for any production Raft deployment. Without it, nodes accumulate unbounded state and restarts require replaying the entire history. This is Lab 2D in MIT 6.824 -- one of the most commonly failed labs.

Run: `uv run python src/03_log_compaction.py`

### Phase 5: Linearizable Reads (~20 min)

**Exercise 4: Implement quorum-based and lease-based linearizable reads.**

Raft's log provides linearizable writes, but reads need special handling. A naive "read from leader" can return stale data if the leader has been deposed. This exercise implements both safe read approaches and demonstrates the stale read scenario.

File: `src/04_linearizable_reads.py`

Two TODO(human) blocks:

- `RaftNode.read_with_quorum()` -- The always-correct approach: confirm leadership via heartbeat majority, wait for state machine to catch up, then serve the read. One extra round-trip but guaranteed linearizable.

- `RaftNode.read_with_lease()` -- The optimistic approach: if the leader's lease (based on last heartbeat round) hasn't expired, serve reads locally without contacting followers. Faster but relies on bounded clock drift.

**Why this matters**: Read performance is critical in most distributed systems (reads vastly outnumber writes). Understanding the trade-off between quorum reads (always safe, slower) and lease-based reads (faster, clock-dependent) is essential for tuning production systems. etcd uses quorum reads by default but supports lease-based reads as an optimization.

Run: `uv run python src/04_linearizable_reads.py`

## Motivation

- **Completing the Raft picture**: 049a covered the "how" (leader election, log replication). 049b covers the "why it's correct" (safety properties) and "how to run it in production" (compaction, linearizable reads, partition recovery).
- **Production Raft knowledge**: Every production Raft implementation (etcd, Consul, CockroachDB) implements snapshots, linearizable reads, and handles partitions. Understanding these is what separates textbook knowledge from production expertise.
- **Distributed systems interviews**: Safety properties, partition behavior, and linearizable reads are core interview topics at companies building distributed infrastructure. Being able to reason about and verify these properties demonstrates deep understanding.
- **Complementary to SAGA pattern**: Practice 014 covered eventual consistency. Raft provides strong consistency. Understanding both -- and when to choose each -- is essential for distributed systems architecture.

## Commands

All commands run from `practice_049b_raft_safety_recovery/`.

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies (none external -- pure Python) |
| `uv run python src/00_raft_core.py` | Run core Raft cluster demo (provided, verifies base works) |
| `uv run python src/01_safety_verification.py` | Exercise 1: Verify safety properties on running cluster |
| `uv run python src/02_partition_recovery.py` | Exercise 2: Network partition simulation and recovery |
| `uv run python src/03_log_compaction.py` | Exercise 3: Snapshot creation and InstallSnapshot RPC |
| `uv run python src/04_linearizable_reads.py` | Exercise 4: Quorum reads vs. lease-based reads |

## State

`not-started`
