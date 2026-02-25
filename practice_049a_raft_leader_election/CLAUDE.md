# Practice 049a -- Raft Consensus: Leader Election & Log Replication

## Technologies

- **Raft Consensus Algorithm** -- Understandable distributed consensus (Ongaro & Ousterhout, 2014)
- **asyncio** -- Python's async/await runtime for simulating concurrent nodes
- **dataclasses** -- Structured Raft message types and node state
- **logging** -- Structured event logging for observing consensus behavior
- Pure Python implementation -- no external dependencies

## Stack

Python 3.12+ (uv). No Docker needed -- nodes are simulated as async tasks in a single process, communicating via direct async function calls instead of network RPCs.

## Theoretical Context

### The Consensus Problem: Why It's Hard

In distributed systems, multiple machines must agree on shared state -- which value was written, who is the leader, what is the current configuration. This is the **consensus problem**: getting N nodes to agree on a single value, even when some nodes crash or messages are delayed.

Consensus sounds simple but is provably difficult. The **FLP impossibility result** (Fischer, Lynch, Paterson, 1985) proves that in a purely asynchronous system -- where there is no bound on message delivery time -- no deterministic algorithm can guarantee consensus if even a single process may crash. The intuition: you can never distinguish a crashed node from a very slow one, so the algorithm can be stuck indefinitely deciding whether to wait or proceed. This doesn't mean consensus is impossible in practice; it means every practical algorithm must make additional assumptions (typically partial synchrony: messages are eventually delivered within some bound, but that bound is unknown). Both Paxos and Raft assume partial synchrony for **liveness** (the system eventually makes progress) while guaranteeing **safety** (never returning an incorrect result) even in fully asynchronous conditions.

Beyond FLP, real systems face **network partitions** (groups of nodes cannot communicate), **Byzantine failures** (nodes behave maliciously -- though Raft only handles crash failures), and **split brain** (two groups each believe they have a leader). The consensus problem is the foundation for building **replicated state machines**: if all nodes agree on the same sequence of commands, they can each execute those commands independently and arrive at the same state. This is how distributed databases (CockroachDB, TiDB), coordination services (etcd, Consul, ZooKeeper), and configuration stores achieve fault-tolerant consistency.

### Paxos vs Raft: The Understandability Problem

**Paxos** (Lamport, 1998, "The Part-Time Parliament") was the first proven-correct consensus algorithm and remains the theoretical gold standard. However, Paxos is notoriously difficult to understand and implement. Lamport's original paper used a metaphor of a Greek parliament that confused readers; his follow-up "Paxos Made Simple" (2001) was clearer but still left significant gaps between the single-decree protocol and a practical multi-decree (Multi-Paxos) system. Real implementations of Multi-Paxos (Google's Chubby, Apache ZooKeeper's ZAB) each made different design choices, and no two implementations agree on the "canonical" Multi-Paxos design.

**Raft** (Ongaro & Ousterhout, 2014, "In Search of an Understandable Consensus Algorithm") was designed explicitly for understandability. The authors' key insight: decompose consensus into three relatively independent subproblems that can be understood separately:

1. **Leader Election** -- How to choose a single leader among the nodes
2. **Log Replication** -- How the leader replicates client commands to followers
3. **Safety** -- Why the algorithm never produces inconsistent results

A user study in the Raft paper found that students scored significantly better on Raft questions than Paxos questions after learning both. The structured decomposition and strong leader model make Raft the algorithm of choice for new distributed systems implementations. Heidi Howard's 2020 analysis ("Paxos vs Raft: Have we reached consensus on distributed consensus?") showed that the two algorithms are more similar than different, but Raft's restriction that only up-to-date nodes can become leaders simplifies both understanding and implementation at the cost of slightly constrained leader election.

### Raft Core Concepts

#### Terms: Logical Clock for Distributed Time

Raft divides time into **terms**, which are monotonically increasing integers. Each term begins with an election. If a candidate wins, it serves as leader for the rest of that term. If no candidate wins (split vote), the term ends with no leader and a new term begins immediately.

Terms act as a **logical clock** that lets nodes detect stale information. Every RPC includes the sender's term. If a node receives a message with a higher term, it immediately updates its own term and reverts to follower state. If a node receives a message with a lower term, it rejects the message. This mechanism ensures that obsolete leaders are quickly deposed.

**Key invariant**: At most one leader exists per term. This is guaranteed because each node votes for at most one candidate per term, and a candidate needs a strict majority of votes.

#### Node States: Follower, Candidate, Leader

Every Raft node is in exactly one of three states:

- **Follower**: Passive. Responds to RPCs from leaders and candidates. If it receives no communication within its election timeout, it transitions to Candidate.
- **Candidate**: Active during elections. Increments its term, votes for itself, and sends RequestVote RPCs to all other nodes. Transitions to Leader if it receives votes from a majority, back to Follower if it discovers a higher term, or stays Candidate if the election times out (starts a new election with a new term).
- **Leader**: Handles all client requests. Sends AppendEntries RPCs (including heartbeats) to all followers. Transitions to Follower only if it discovers a higher term.

The normal flow is: all nodes start as Followers -> one times out and becomes a Candidate -> wins election and becomes Leader -> serves until it crashes or a higher term appears.

### Leader Election in Detail

**Election timeout**: Each follower has a randomized election timeout (typically 150-300ms). The randomization is critical -- it ensures that in most cases, a single node times out first, becomes a candidate, and wins the election before other nodes time out. Without randomization, multiple nodes would time out simultaneously, split the votes, and the system would livelock.

**RequestVote RPC**: When a candidate starts an election, it sends a `RequestVote` to every other node containing: its term, its candidate ID, and the index and term of its last log entry. The last log entry information is used for the **election restriction**: a node only grants its vote if the candidate's log is at least as up-to-date as its own. "At least as up-to-date" means: the candidate's last log entry has a higher term, OR the same term but a higher (or equal) index. This restriction ensures that only candidates with all committed entries can become leader (the **Leader Completeness** property).

**Vote granting**: A node grants its vote if: (1) the candidate's term is >= the node's current term, (2) the node has not already voted for a different candidate in this term, and (3) the candidate's log is at least as up-to-date. Each node votes for at most one candidate per term (stored in `votedFor`), and votes are durable -- persisted before responding.

**Majority rule**: A candidate becomes leader when it receives votes from a strict majority: `(N // 2) + 1` nodes out of N total (including its own vote for itself). For a 5-node cluster, this means 3 votes. The majority quorum guarantees that two leaders cannot be elected in the same term (any two majorities must overlap by at least one node, and that node cannot vote for two different candidates).

**Split votes**: If two candidates start elections simultaneously and neither gets a majority, both time out and start new elections with incremented terms. The randomized timeout makes this unlikely to repeat -- one candidate will almost certainly time out first next round. In the worst case, split votes add latency but never violate safety.

### Log Replication in Detail

**Client requests**: All client requests go to the leader. If a follower receives a client request, it redirects to the leader. The leader appends the command to its local log as a new entry with the current term number, then sends `AppendEntries` RPCs to all followers in parallel.

**AppendEntries RPC**: The leader sends each follower an `AppendEntries` containing: its term, its node ID (so followers know who the leader is), `prevLogIndex` and `prevLogTerm` (the index and term of the log entry immediately preceding the new entries -- used for the consistency check), the `entries[]` array (the new log entries to append, or empty for heartbeats), and `leaderCommit` (the leader's current commit index).

**Consistency check**: When a follower receives `AppendEntries`, it checks whether its log contains an entry at `prevLogIndex` with term `prevLogTerm`. If yes, the logs are consistent up to that point, and the follower appends the new entries (overwriting any conflicting entries). If no, the follower rejects the RPC, and the leader decrements `nextIndex` for that follower and retries with an earlier `prevLogIndex`. This "back-tracking" mechanism repairs follower logs after leader changes. The key insight: by checking consistency one entry at a time, the leader can always find the point where the follower's log diverges and repair it. This is guaranteed by the **Log Matching Property**: if two entries in different logs have the same index and term, then all preceding entries are also identical.

**Commitment**: An entry is **committed** when the leader has replicated it to a majority of nodes. The leader tracks `matchIndex[i]` for each follower (the highest log index known to be replicated on follower i). When a majority of `matchIndex` values >= some index N, and the entry at N has the current term, the leader sets `commitIndex = N`. The term check is critical -- the leader only commits entries from its own term (entries from previous terms are committed indirectly when a current-term entry after them is committed). This prevents a subtle bug described in Section 5.4.2 of the Raft paper.

**Heartbeats**: The leader sends empty `AppendEntries` RPCs periodically (typically every 50-100ms) to prevent followers from timing out and starting elections. Heartbeats also carry the leader's `commitIndex`, allowing followers to advance their own commit indices.

### Raft Safety Guarantees

The Raft paper (Figure 3) defines five safety properties:

| Property | Statement |
|----------|-----------|
| **Election Safety** | At most one leader can be elected in a given term |
| **Leader Append-Only** | A leader never overwrites or deletes entries in its log; it only appends |
| **Log Matching** | If two logs contain an entry with the same index and term, the logs are identical in all entries up through that index |
| **Leader Completeness** | If a log entry is committed in a given term, that entry will be present in the logs of the leaders for all higher-numbered terms |
| **State Machine Safety** | If a server has applied a log entry at a given index to its state machine, no other server will ever apply a different log entry for that index |

Together, these properties guarantee **linearizable reads and writes** -- the replicated state machine behaves as if it were a single machine.

### Real-World Raft Implementations

| Implementation | Language | Used By |
|----------------|----------|---------|
| **etcd/raft** | Go | etcd, Kubernetes, CockroachDB, TiDB, Docker Swarm |
| **HashiCorp Raft** | Go | Consul, Nomad, Vault |
| **openraft** | Rust | Databend, Chroma, SeaStreamer |
| **rqlite** | Go (wraps HashiCorp Raft) | Standalone distributed SQLite |
| **Apache Ratis** | Java | Apache Ozone (HDFS replacement) |

etcd -- the key-value store behind Kubernetes -- is perhaps the most critical Raft deployment: every Kubernetes cluster relies on it for storing cluster state. Consul (service discovery) and CockroachDB (distributed SQL) also use Raft at their core. Understanding Raft means understanding the consensus layer of most modern distributed infrastructure.

### References

- Ongaro, D. & Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm." USENIX ATC '14. [https://raft.github.io/raft.pdf](https://raft.github.io/raft.pdf)
- Raft website with interactive visualization: [https://raft.github.io/](https://raft.github.io/)
- Lamport, L. (1998). "The Part-Time Parliament." ACM TOCS.
- Fischer, M., Lynch, N., Paterson, M. (1985). "Impossibility of Distributed Consensus with One Faulty Process." JACM.
- Howard, H. & Mortier, R. (2020). "Paxos vs Raft: Have we reached consensus on distributed consensus?" [https://arxiv.org/abs/2004.05074](https://arxiv.org/abs/2004.05074)
- etcd Raft library: [https://github.com/etcd-io/raft](https://github.com/etcd-io/raft)
- openraft (Rust): [https://github.com/databendlabs/openraft](https://github.com/databendlabs/openraft)

## Description

Build a simulated Raft cluster (5 nodes) running as asyncio tasks in a single process. Implement leader election with randomized timeouts, RequestVote and AppendEntries RPCs as async function calls (simulating network communication), and basic log replication with consistency checks. Test the implementation by injecting network partitions and node crashes, verifying that safety properties hold.

### What you'll learn

1. **Raft state machine** -- Follower/Candidate/Leader transitions, term monotonicity, the role of persistent state
2. **Leader election** -- RequestVote RPC, vote granting rules, election restriction (log up-to-dateness), majority detection
3. **Log replication** -- AppendEntries RPC, consistency check, log repair (nextIndex backtracking), commitment via majority
4. **Safety under failures** -- Why the algorithm preserves Election Safety, Log Matching, and Leader Completeness even during partitions and crashes

### Architecture

```
  Single Python Process (asyncio event loop)
  +--------------------------------------------------+
  |                                                  |
  |   Node 0        Node 1        Node 2            |
  |  (Follower)    (Leader)      (Follower)          |
  |     |             |              |               |
  |     +------+------+------+------+               |
  |            |             |                       |
  |         Node 3        Node 4                     |
  |        (Follower)    (Follower)                  |
  |                                                  |
  |  Communication: direct async function calls      |
  |  Failure sim: drop/delay messages per-link       |
  +--------------------------------------------------+
```

Nodes communicate via a `Network` object that routes messages. Network partitions are simulated by marking links as "down" -- messages sent across a partition are silently dropped.

## Instructions

### Phase 1: Types & Data Structures (~10 min)

Run the pre-built types module to verify the data structures compile and self-test:

```
uv sync
uv run python src/00_raft_types.py
```

This file defines all Raft message types (RequestVoteArgs, RequestVoteReply, AppendEntriesArgs, AppendEntriesReply), node states (FOLLOWER, CANDIDATE, LEADER), log entries, and cluster configuration. Read through it to understand the data model before implementing behavior.

### Phase 2: Node State Machine (~25 min)

**Exercise 1: Implement the Raft node state machine.**

The `RaftNode` class manages a single node's persistent and volatile state, and implements the state transitions described in Raft paper Figure 2 (server rules). This is the foundation -- every subsequent exercise builds on these transitions.

File: `src/01_node_state_machine.py`

Four TODO(human) blocks:
- `__init__` -- Initialize all Raft state (persistent: currentTerm, votedFor, log; volatile: commitIndex, lastApplied, state; leader-specific: nextIndex, matchIndex)
- `become_candidate()` -- Transition to CANDIDATE (increment term, vote for self, prepare RequestVote arguments)
- `become_leader()` -- Transition to LEADER (initialize nextIndex and matchIndex for each peer)
- `step_down(term)` -- Revert to FOLLOWER when a higher term is discovered (the safety mechanism that prevents stale leaders)

**Why this matters**: These four transitions are the heartbeat of Raft. Getting them right ensures the term monotonicity invariant and the single-leader-per-term property. Every RPC handler in the next exercises calls `step_down()` when it sees a higher term.

Run: `uv run python src/01_node_state_machine.py`

### Phase 3: Leader Election (~25 min)

**Exercise 2: Implement leader election with RequestVote RPC.**

Build the vote-granting logic and the election simulation loop. This exercise teaches the election restriction (only up-to-date candidates win), the randomized timeout mechanism (prevents split votes), and how majority quorums guarantee at most one leader per term.

File: `src/02_leader_election.py`

Three TODO(human) blocks:
- `handle_request_vote()` -- Process a RequestVote RPC: check term, check vote availability, check log up-to-dateness. This is the core safety mechanism for leader election.
- `run_election()` -- Simulate a single election round: candidate broadcasts RequestVote, collects replies, checks majority.
- `simulate_election_with_timeout()` -- Run the full election loop with randomized timeouts, handling split votes by incrementing terms and retrying.

**Why this matters**: Leader election is Raft's first subproblem. The election restriction (candidates must have up-to-date logs) is what ensures the Leader Completeness property -- without it, a newly elected leader might be missing committed entries.

Run: `uv run python src/02_leader_election.py`

### Phase 4: Log Replication (~25 min)

**Exercise 3: Implement log replication with AppendEntries RPC.**

Build the consistency check, log repair mechanism, and commitment rule. This exercise teaches how the leader ensures all followers have identical logs, how conflicting entries are resolved, and how entries become committed via majority replication.

File: `src/03_log_replication.py`

Three TODO(human) blocks:
- `handle_append_entries()` -- Process AppendEntries RPC: term check, consistency check (prevLogIndex/prevLogTerm), append entries, update commitIndex. This is the most complex RPC handler.
- `replicate_to_follower()` -- As leader, send entries to a specific follower. Handle rejection by decrementing nextIndex and retrying (log repair).
- `advance_commit_index()` -- As leader, find the highest index N where a majority of matchIndex[i] >= N and log[N].term == currentTerm. This is the commitment rule.

**Why this matters**: Log replication is where Raft achieves its core goal: all nodes execute the same commands in the same order. The consistency check and log repair mechanism ensure the Log Matching property. The commitment rule (only commit current-term entries) prevents a subtle safety bug described in Raft paper Section 5.4.2.

Run: `uv run python src/03_log_replication.py`

### Phase 5: Integration Test (~15 min)

**Exercise 4: Full cluster simulation with failure injection.**

Bring everything together: run a 5-node Raft cluster, submit client commands, inject a network partition, verify that safety properties hold throughout.

File: `src/04_cluster_test.py`

One TODO(human) block:
- `run_cluster()` -- Create nodes, elect a leader, replicate commands, optionally inject a network partition, verify consistency after healing. This tests the full Raft lifecycle.

**Why this matters**: Individual components may work correctly in isolation but fail when composed. This integration test verifies that leader election, log replication, and failure recovery work together. The partition test specifically validates that the minority partition cannot make progress (no leader without majority) while the majority partition continues operating normally.

Run: `uv run python src/04_cluster_test.py`

## Motivation

- **Foundation of modern infrastructure**: Raft powers etcd (Kubernetes state store), Consul (service discovery), CockroachDB (distributed SQL), and TiDB. Understanding Raft means understanding the consensus layer of most production distributed systems.
- **Essential distributed systems knowledge**: Consensus is a core concept in distributed systems design. Interviewers at companies building distributed infrastructure (Google, AWS, Confluent, CockroachDB Labs) expect candidates to understand Raft-level concepts.
- **Complements existing skills**: Practice 014 (SAGA Pattern) covered eventual consistency. Raft provides the opposite end of the spectrum: strong consistency via replicated state machines. Understanding both models is essential for making architectural trade-off decisions.
- **Direct applicability**: At AutoScheduler.AI, understanding consensus helps reason about data consistency in distributed scheduling systems, leader-follower architectures, and failure recovery strategies.

## Commands

All commands run from `practice_049a_raft_leader_election/`.

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies (none external -- pure Python) |
| `uv run python src/00_raft_types.py` | Run type definitions self-test |
| `uv run python src/01_node_state_machine.py` | Test node state transitions (Exercise 1) |
| `uv run python src/02_leader_election.py` | Run leader election simulation (Exercise 2) |
| `uv run python src/03_log_replication.py` | Run log replication simulation (Exercise 3) |
| `uv run python src/04_cluster_test.py` | Full cluster integration test (Exercise 4) |

## State

`not-started`
