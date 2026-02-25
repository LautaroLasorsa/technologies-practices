"""Exercise 1: Raft Node State Machine -- Follower/Candidate/Leader transitions.

This module implements the core RaftNode class with state transitions as
described in Raft paper Figure 2 (Server Rules). The node manages:

  Persistent state (survives restarts -- in real Raft, persisted to disk):
    - currentTerm: latest term the node has seen (monotonically increasing)
    - votedFor: candidate ID that received vote in current term (or None)
    - log[]: log entries, each containing command and term when received

  Volatile state (all servers):
    - commitIndex: index of highest log entry known to be committed
    - lastApplied: index of highest log entry applied to state machine

  Volatile state (leaders only, reinitialized after election):
    - nextIndex[]: for each follower, index of the NEXT log entry to send
    - matchIndex[]: for each follower, index of highest log entry known
                    to be replicated on that follower

Reference: Raft paper Figure 2, Section 5.1, Section 5.2.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the practice root is on sys.path so `from src.raft_types import ...` works.
_PRACTICE_ROOT = Path(__file__).resolve().parent.parent
if str(_PRACTICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PRACTICE_ROOT))

from src.raft_types import (  # noqa: E402
    AppendEntriesArgs,
    AppendEntriesReply,
    LogEntry,
    NodeState,
    RaftConfig,
    RequestVoteArgs,
    RequestVoteReply,
)

logger = logging.getLogger("raft")


class RaftNode:
    """A single Raft node implementing the state machine from Figure 2.

    This class manages all persistent and volatile state for one node.
    RPC handlers (handle_request_vote, handle_append_entries) are added
    in subsequent exercises.
    """

    # TODO(human): Implement __init__(self, node_id, config, peers)
    #
    # Initialize a Raft node with all state described in Raft paper Figure 2.
    #
    # Parameters:
    #   node_id (int): Unique identifier for this node (e.g., 0, 1, 2, 3, 4).
    #   config (RaftConfig): Cluster configuration (timeouts, num_nodes).
    #   peers (list[int]): List of peer node IDs (all nodes except this one).
    #
    # What to initialize:
    #
    #   PERSISTENT STATE (Raft paper Figure 2, "Persistent state on all servers"):
    #   - self.node_id = node_id
    #   - self.config = config
    #   - self.peers = peers
    #   - self.current_term: int = 0
    #       The latest term this server has seen. Starts at 0 and only
    #       increases. In a real implementation, this would be persisted to
    #       stable storage before responding to any RPC.
    #   - self.voted_for: int | None = None
    #       The candidate ID that received our vote in the current term,
    #       or None if we haven't voted. Reset to None when term changes.
    #       This ensures the "vote for at most one candidate per term" invariant.
    #   - self.log: list[LogEntry] = []
    #       The log entries. 0-indexed in our list, but LogEntry.index is
    #       1-based (as in the Raft paper). An empty log means no entries yet.
    #
    #   VOLATILE STATE (all servers):
    #   - self.state: NodeState = NodeState.FOLLOWER
    #       All nodes start as followers (Raft paper Section 5.2).
    #   - self.leader_id: int | None = None
    #       ID of the current leader (if known). Used to redirect clients.
    #   - self.commit_index: int = 0
    #       Index of highest log entry known to be committed. Initialized
    #       to 0 and increases monotonically. An entry is committed when
    #       the leader has replicated it to a majority.
    #   - self.last_applied: int = 0
    #       Index of highest log entry applied to the state machine.
    #       Always <= commit_index. In a real system, a background loop
    #       applies entries between last_applied and commit_index.
    #
    #   VOLATILE STATE (leaders only -- Raft paper Figure 2):
    #   - self.next_index: dict[int, int] = {}
    #       For each peer, the index of the NEXT log entry to send to that
    #       peer. Initialized to leader's last log index + 1 when elected.
    #       Decremented on AppendEntries rejection (log repair).
    #   - self.match_index: dict[int, int] = {}
    #       For each peer, the index of the highest log entry KNOWN to be
    #       replicated on that peer. Initialized to 0 when elected.
    #       Updated on successful AppendEntries. Used by advance_commit_index().
    #
    #   - self.votes_received: set[int] = set()
    #       Track which peers have granted us their vote during an election.
    #       Only meaningful when state == CANDIDATE.
    #
    # Why this matters:
    #   Every RPC handler and state transition reads/writes these fields.
    #   Getting the initial state right is critical -- for example, starting
    #   with current_term=0 means the first election will be for term 1.
    #   Starting with voted_for=None means the node is free to vote in
    #   the first election. Starting as FOLLOWER means no node tries to
    #   be leader until an election timeout fires.
    #
    # Hint: The initializer is straightforward assignment. The key insight
    # is understanding WHY each field exists -- refer to the Raft paper
    # Figure 2 column headers.

    def __init__(self, node_id: int, config: RaftConfig, peers: list[int]) -> None:
        raise NotImplementedError(
            "TODO(human): Initialize all Raft node state. "
            "See Raft paper Figure 2 for the complete list of state variables."
        )

    # ------------------------------------------------------------------
    # Helpers for log access
    # ------------------------------------------------------------------

    @property
    def last_log_index(self) -> int:
        """Index of the last log entry (0 if log is empty)."""
        return self.log[-1].index if self.log else 0

    @property
    def last_log_term(self) -> int:
        """Term of the last log entry (0 if log is empty)."""
        return self.log[-1].term if self.log else 0

    def log_entry_at(self, index: int) -> LogEntry | None:
        """Get the log entry at the given 1-based index, or None if absent.

        Our internal list is 0-indexed, but Raft log indices are 1-based.
        Entry at list position [i] has LogEntry.index == i + 1.
        """
        if index < 1 or index > len(self.log):
            return None
        return self.log[index - 1]

    def log_term_at(self, index: int) -> int:
        """Get the term of the log entry at the given index (0 if absent)."""
        entry = self.log_entry_at(index)
        return entry.term if entry else 0

    # ------------------------------------------------------------------
    # State transitions (Raft paper Section 5.2)
    # ------------------------------------------------------------------

    # TODO(human): Implement become_candidate(self) -> RequestVoteArgs
    #
    # Transition this node from FOLLOWER (or CANDIDATE on re-election) to
    # CANDIDATE state and prepare the RequestVote RPC arguments to broadcast
    # to all peers.
    #
    # Steps (Raft paper Section 5.2, "Rules for Servers: Candidates"):
    #   1. Increment self.current_term by 1.
    #      WHY: Each election attempt uses a new term. If a previous election
    #      failed (split vote), incrementing ensures a fresh start. Terms are
    #      the logical clock that orders elections and detects stale leaders.
    #
    #   2. Set self.state = NodeState.CANDIDATE.
    #
    #   3. Vote for ourselves: self.voted_for = self.node_id.
    #      WHY: A candidate always votes for itself. Combined with the rule
    #      "vote for at most one candidate per term," this means we won't
    #      accidentally vote for a competing candidate in this term.
    #
    #   4. Initialize self.votes_received = {self.node_id} (we have our own vote).
    #
    #   5. Clear self.leader_id = None (we don't know who the leader is yet).
    #
    #   6. Build and return a RequestVoteArgs with:
    #        term = self.current_term
    #        candidate_id = self.node_id
    #        last_log_index = self.last_log_index  (use the property)
    #        last_log_term = self.last_log_term    (use the property)
    #
    #   7. Log the transition: logger.info(f"Node {self.node_id} became CANDIDATE for term {self.current_term}")
    #
    # Returns:
    #   RequestVoteArgs to broadcast to all peers.
    #
    # Why this matters:
    #   This is the entry point for every election. The correctness of
    #   self-voting and term increment is critical: if a candidate doesn't
    #   vote for itself, it needs one extra external vote; if it doesn't
    #   increment the term, it might receive stale votes from a previous
    #   election attempt.
    #
    # Edge case: A CANDIDATE that times out calls become_candidate() again,
    # incrementing the term and starting a fresh election. This is how
    # split votes are resolved.

    def become_candidate(self) -> RequestVoteArgs:
        raise NotImplementedError(
            "TODO(human): Transition to CANDIDATE state. "
            "Increment term, vote for self, return RequestVoteArgs."
        )

    # TODO(human): Implement become_leader(self) -> None
    #
    # Transition this node from CANDIDATE to LEADER after winning an election.
    # Initialize the leader-only volatile state and log the transition.
    #
    # Steps (Raft paper Figure 2, "Volatile state on leaders"):
    #   1. Set self.state = NodeState.LEADER.
    #
    #   2. Set self.leader_id = self.node_id.
    #
    #   3. Initialize self.next_index: for each peer, set to
    #      self.last_log_index + 1.
    #      WHY: The leader optimistically assumes each follower's log is
    #      up-to-date. If a follower's log is actually behind, the first
    #      AppendEntries will fail the consistency check, and the leader
    #      will decrement next_index for that follower until it finds the
    #      matching point. This optimistic start means no wasted RPCs when
    #      followers ARE up-to-date (the common case).
    #
    #   4. Initialize self.match_index: for each peer, set to 0.
    #      WHY: The leader doesn't know how far each follower's log extends.
    #      match_index is updated conservatively -- only after a successful
    #      AppendEntries confirms replication. It starts at 0 (we know nothing)
    #      and only increases. The leader uses match_index to determine when
    #      entries are committed (replicated to a majority).
    #
    #   5. Log: logger.info(f"Node {self.node_id} became LEADER for term {self.current_term}")
    #
    # Returns:
    #   None. The caller (election loop) will start sending heartbeats.
    #
    # Why this matters:
    #   The initialization of next_index and match_index is the bridge between
    #   leader election and log replication. Getting these values wrong means
    #   the leader either sends too many entries (inefficient), skips entries
    #   (unsafe), or fails to detect commitment (entries never applied).
    #
    # Important: After becoming leader, the leader should immediately send
    # heartbeats to all followers to establish its authority and prevent
    # new elections. This is handled by the caller, not by this method.

    def become_leader(self) -> None:
        raise NotImplementedError(
            "TODO(human): Transition to LEADER state. "
            "Initialize next_index and match_index for each peer."
        )

    # TODO(human): Implement step_down(self, term: int) -> None
    #
    # If the given term is greater than our current term, revert to FOLLOWER
    # state and update our term. This is the universal safety mechanism that
    # ensures stale leaders/candidates immediately step down when they
    # discover a more recent term.
    #
    # Steps (Raft paper Section 5.1, "Rules for Servers: All Servers"):
    #   1. Check: if term > self.current_term:
    #        a. Set self.current_term = term
    #        b. Set self.state = NodeState.FOLLOWER
    #        c. Set self.voted_for = None
    #           WHY: We're entering a new term, and we haven't voted in it
    #           yet. Clearing voted_for allows us to vote for a candidate
    #           in this new term.
    #        d. Set self.leader_id = None
    #           WHY: We don't know who the leader is in this new term yet.
    #        e. Clear self.votes_received = set()
    #        f. Log: logger.info(f"Node {self.node_id} stepping down: term {self.current_term} (was {old_term})")
    #           (Capture the old term before updating for the log message.)
    #
    #   2. If term <= self.current_term: do nothing. This is not an error --
    #      it just means we already know about this term or a later one.
    #
    # Parameters:
    #   term (int): The term discovered in an incoming RPC or reply.
    #
    # Returns:
    #   None.
    #
    # Why this matters:
    #   step_down() is called at the TOP of every RPC handler (both request
    #   and reply). It's the mechanism that enforces term monotonicity: the
    #   single most important safety invariant in Raft. Without it:
    #   - A stale leader could continue sending AppendEntries and overwrite
    #     entries from the new leader.
    #   - A stale candidate could win an election with outdated votes.
    #   - Two leaders could coexist, violating Election Safety.
    #
    # Example scenario:
    #   Node A is leader in term 3. Network partition isolates A from the
    #   majority. The majority elects Node B as leader in term 4. When the
    #   partition heals, A receives an AppendEntries from B with term=4.
    #   A calls step_down(4), reverts to follower, and accepts B's authority.
    #   Without step_down, A would reject B's AppendEntries (thinking it's
    #   still the legitimate leader) and the cluster would have two leaders.

    def step_down(self, term: int) -> None:
        raise NotImplementedError(
            "TODO(human): Step down to FOLLOWER if term > current_term. "
            "Update term, clear votedFor, clear leader_id."
        )

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RaftNode(id={self.node_id}, state={self.state.name}, "
            f"term={self.current_term}, votedFor={self.voted_for}, "
            f"log_len={len(self.log)}, commit={self.commit_index})"
        )


# ---------------------------------------------------------------------------
# Convenience: create a cluster of RaftNodes
# ---------------------------------------------------------------------------

def create_cluster(config: RaftConfig | None = None) -> list[RaftNode]:
    """Create a cluster of RaftNodes with default config.

    Returns a list of nodes where nodes[i].node_id == i.
    Each node's peers list contains the IDs of all other nodes.
    """
    if config is None:
        config = RaftConfig()

    nodes: list[RaftNode] = []
    for i in range(config.num_nodes):
        peers = [j for j in range(config.num_nodes) if j != i]
        nodes.append(RaftNode(node_id=i, config=config, peers=peers))
    return nodes


# ---------------------------------------------------------------------------
# Main: test state transitions
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("Exercise 1: Node State Machine -- Test")
    print("=" * 60)

    config = RaftConfig(num_nodes=5)
    nodes = create_cluster(config)

    # All nodes start as followers in term 0
    print("\n--- Initial State ---")
    for node in nodes:
        print(f"  {node}")
        assert node.state == NodeState.FOLLOWER, f"Node {node.node_id} should be FOLLOWER"
        assert node.current_term == 0, f"Node {node.node_id} should be in term 0"
        assert node.voted_for is None, f"Node {node.node_id} should not have voted"

    # Node 2 becomes a candidate
    print("\n--- Node 2 becomes CANDIDATE ---")
    rv_args = nodes[2].become_candidate()
    print(f"  {nodes[2]}")
    print(f"  RequestVoteArgs: {rv_args}")
    assert nodes[2].state == NodeState.CANDIDATE
    assert nodes[2].current_term == 1
    assert nodes[2].voted_for == 2
    assert rv_args.term == 1
    assert rv_args.candidate_id == 2

    # Node 2 wins election (simulate by calling become_leader directly)
    print("\n--- Node 2 becomes LEADER ---")
    nodes[2].votes_received = {0, 1, 2}  # got majority (3/5)
    nodes[2].become_leader()
    print(f"  {nodes[2]}")
    assert nodes[2].state == NodeState.LEADER
    assert nodes[2].leader_id == 2
    assert all(nodes[2].next_index[p] == 1 for p in nodes[2].peers)
    assert all(nodes[2].match_index[p] == 0 for p in nodes[2].peers)

    # Higher term discovered: leader steps down
    print("\n--- Node 2 discovers higher term (3) ---")
    nodes[2].step_down(3)
    print(f"  {nodes[2]}")
    assert nodes[2].state == NodeState.FOLLOWER
    assert nodes[2].current_term == 3
    assert nodes[2].voted_for is None

    # step_down with equal term does nothing
    print("\n--- Node 2 step_down with same term (3) -- no change ---")
    old_state = nodes[2].state
    nodes[2].step_down(3)
    assert nodes[2].state == old_state
    assert nodes[2].current_term == 3

    # Candidate that times out starts new election
    print("\n--- Node 4 runs two elections (split vote scenario) ---")
    rv1 = nodes[4].become_candidate()
    print(f"  First election:  term={rv1.term}")
    assert nodes[4].current_term == 1

    rv2 = nodes[4].become_candidate()  # timeout, try again
    print(f"  Second election: term={rv2.term}")
    assert nodes[4].current_term == 2
    assert rv2.term == 2

    print("\n--- All Tests Passed ---")
    print("=" * 60)


if __name__ == "__main__":
    main()
