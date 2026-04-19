"""Raft data types -- all message and state types used by the Raft protocol.

This module defines the data structures from Raft paper Figure 2:
- Node states (Follower, Candidate, Leader)
- Log entries (term + index + command)
- RequestVote RPC arguments and reply
- AppendEntries RPC arguments and reply
- Cluster configuration

All types use dataclasses for clarity and immutability where appropriate.
No behavior here -- just data definitions.

Reference: Ongaro & Ousterhout (2014), Figure 2 -- "State" and "RPC" boxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Node states (Raft paper Section 5.1)
# ---------------------------------------------------------------------------

class NodeState(Enum):
    """The three possible states of a Raft node.

    Transitions:
        FOLLOWER  -> CANDIDATE  (election timeout expired, no heartbeat received)
        CANDIDATE -> LEADER     (received votes from majority)
        CANDIDATE -> FOLLOWER   (discovered higher term, or another leader elected)
        CANDIDATE -> CANDIDATE  (election timeout, start new election with new term)
        LEADER    -> FOLLOWER   (discovered higher term)
    """

    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


# ---------------------------------------------------------------------------
# Log entry (Raft paper Section 5.3)
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    """A single entry in the Raft log.

    Attributes:
        term:    The term when the entry was received by the leader.
        index:   The position in the log (1-indexed, as in the Raft paper).
        command: The state machine command (a string in our simulation).
    """

    term: int
    index: int
    command: str

    def __repr__(self) -> str:
        return f"LogEntry(term={self.term}, idx={self.index}, cmd={self.command!r})"


# ---------------------------------------------------------------------------
# RequestVote RPC (Raft paper Section 5.2)
# ---------------------------------------------------------------------------

@dataclass
class RequestVoteArgs:
    """Arguments for the RequestVote RPC, sent by candidates during elections.

    The candidate includes its last log entry information so that voters
    can enforce the election restriction: only candidates with up-to-date
    logs can become leader (Section 5.4.1).

    Attributes:
        term:           Candidate's current term.
        candidate_id:   ID of the candidate requesting the vote.
        last_log_index: Index of the candidate's last log entry (0 if log is empty).
        last_log_term:  Term of the candidate's last log entry (0 if log is empty).
    """

    term: int
    candidate_id: int
    last_log_index: int
    last_log_term: int


@dataclass
class RequestVoteReply:
    """Reply to a RequestVote RPC.

    Attributes:
        term:         The responder's current term (for candidate to update itself).
        vote_granted: True if the responder granted its vote to the candidate.
        voter_id:     ID of the node that sent this reply.
    """

    term: int
    vote_granted: bool
    voter_id: int


# ---------------------------------------------------------------------------
# AppendEntries RPC (Raft paper Section 5.3)
# ---------------------------------------------------------------------------

@dataclass
class AppendEntriesArgs:
    """Arguments for the AppendEntries RPC, sent by the leader.

    Used for both log replication (entries is non-empty) and heartbeats
    (entries is empty). The prevLogIndex/prevLogTerm fields implement the
    consistency check described in Section 5.3.

    Attributes:
        term:           Leader's current term.
        leader_id:      ID of the leader, so followers can redirect clients.
        prev_log_index: Index of the log entry immediately preceding the new entries.
                        0 if sending from the beginning of the log.
        prev_log_term:  Term of the entry at prev_log_index.
                        0 if prev_log_index is 0.
        entries:        Log entries to append (empty for heartbeats).
        leader_commit:  Leader's current commit index.
    """

    term: int
    leader_id: int
    prev_log_index: int
    prev_log_term: int
    entries: list[LogEntry] = field(default_factory=list)
    leader_commit: int = 0


@dataclass
class AppendEntriesReply:
    """Reply to an AppendEntries RPC.

    When success is False, the follower's log didn't match at prevLogIndex.
    The conflict_index and conflict_term fields help the leader skip back
    more efficiently (optimization from Section 5.3).

    Attributes:
        term:           The responder's current term.
        success:        True if the follower's log matched at prevLogIndex
                        and entries were appended successfully.
        follower_id:    ID of the responding follower.
        match_index:    The highest log index on the follower after appending
                        (used by the leader to update matchIndex).
        conflict_index: If success is False, the first index of the conflicting
                        term on the follower (for faster backtracking).
        conflict_term:  If success is False, the term of the conflicting entry.
    """

    term: int
    success: bool
    follower_id: int
    match_index: int = 0
    conflict_index: int = 0
    conflict_term: int = 0


# ---------------------------------------------------------------------------
# Cluster configuration
# ---------------------------------------------------------------------------

@dataclass
class RaftConfig:
    """Configuration for a Raft cluster simulation.

    The election timeout is randomized per-node within [min, max] to
    prevent split votes. The heartbeat interval must be much less than
    the election timeout minimum (typically heartbeat << election_timeout_min / 3).

    For simulation, we use shorter timeouts (seconds instead of milliseconds)
    to keep the demo fast while still showing the protocol behavior.

    Attributes:
        election_timeout_min_ms: Minimum election timeout in milliseconds.
        election_timeout_max_ms: Maximum election timeout in milliseconds.
        heartbeat_interval_ms:   Heartbeat interval in milliseconds.
        num_nodes:               Number of nodes in the cluster.
    """

    election_timeout_min_ms: int = 300
    election_timeout_max_ms: int = 500
    heartbeat_interval_ms: int = 100
    num_nodes: int = 5

    @property
    def majority(self) -> int:
        """The number of votes needed to win an election (strict majority)."""
        return (self.num_nodes // 2) + 1

    @property
    def election_timeout_min_s(self) -> float:
        """Election timeout minimum in seconds (for asyncio.sleep)."""
        return self.election_timeout_min_ms / 1000.0

    @property
    def election_timeout_max_s(self) -> float:
        """Election timeout maximum in seconds (for asyncio.sleep)."""
        return self.election_timeout_max_ms / 1000.0

    @property
    def heartbeat_interval_s(self) -> float:
        """Heartbeat interval in seconds (for asyncio.sleep)."""
        return self.heartbeat_interval_ms / 1000.0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Verify all types can be constructed and printed."""
    print("=" * 60)
    print("Raft Types -- Self-Test")
    print("=" * 60)

    # Node states
    for state in NodeState:
        print(f"  NodeState.{state.name}")

    # Log entry
    entry = LogEntry(term=1, index=1, command="SET x=1")
    print(f"\n  {entry}")

    # RequestVote
    rv_args = RequestVoteArgs(term=2, candidate_id=3, last_log_index=5, last_log_term=1)
    rv_reply = RequestVoteReply(term=2, vote_granted=True, voter_id=0)
    print(f"\n  RequestVoteArgs:  {rv_args}")
    print(f"  RequestVoteReply: {rv_reply}")

    # AppendEntries
    ae_args = AppendEntriesArgs(
        term=2,
        leader_id=3,
        prev_log_index=4,
        prev_log_term=1,
        entries=[LogEntry(term=2, index=5, command="SET y=2")],
        leader_commit=4,
    )
    ae_reply = AppendEntriesReply(term=2, success=True, follower_id=0, match_index=5)
    print(f"\n  AppendEntriesArgs:  {ae_args}")
    print(f"  AppendEntriesReply: {ae_reply}")

    # Config
    config = RaftConfig()
    print(f"\n  Config: {config}")
    print(f"  Majority needed: {config.majority}")
    print(f"  Election timeout: [{config.election_timeout_min_s:.3f}s, {config.election_timeout_max_s:.3f}s]")
    print(f"  Heartbeat interval: {config.heartbeat_interval_s:.3f}s")

    print("\n  All types OK.")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
