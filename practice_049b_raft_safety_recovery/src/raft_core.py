"""Raft Core Implementation -- Complete working Raft cluster.

This module provides the full Raft consensus engine that exercises 1-4 build on.
It includes: leader election, log replication, AppendEntries/RequestVote RPCs,
commitment via majority, and a simulated network layer.

This is the "solution" from Practice 049a, provided as a foundation so that
Practice 049b can focus on safety, recovery, compaction, and reads.

Architecture:
  - RaftNode: A single Raft node with full state machine
  - Network: Routes messages between nodes, supports partition simulation
  - RaftCluster: Manages a group of nodes + network + event loop

Reference: Ongaro & Ousterhout (2014), "In Search of an Understandable
Consensus Algorithm", Sections 5.1-5.4.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-8s] %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class NodeState(Enum):
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


@dataclass
class LogEntry:
    term: int
    index: int
    command: str

    def __repr__(self) -> str:
        return f"LogEntry(t={self.term}, i={self.index}, cmd={self.command!r})"


@dataclass
class RequestVoteArgs:
    term: int
    candidate_id: int
    last_log_index: int
    last_log_term: int


@dataclass
class RequestVoteReply:
    term: int
    vote_granted: bool
    voter_id: int


@dataclass
class AppendEntriesArgs:
    term: int
    leader_id: int
    prev_log_index: int
    prev_log_term: int
    entries: list[LogEntry] = field(default_factory=list)
    leader_commit: int = 0


@dataclass
class AppendEntriesReply:
    term: int
    success: bool
    follower_id: int
    match_index: int = 0
    conflict_index: int = 0
    conflict_term: int = 0


@dataclass
class InstallSnapshotArgs:
    """Arguments for the InstallSnapshot RPC (Raft paper Section 7).

    Sent by the leader to followers that are too far behind for
    log-based repair (the leader has already compacted the needed entries).

    Attributes:
        term:                Leader's current term.
        leader_id:           Leader's ID.
        last_included_index: Index of the last entry in the snapshot.
        last_included_term:  Term of the last entry in the snapshot.
        data:                Serialized state machine snapshot.
    """
    term: int
    leader_id: int
    last_included_index: int
    last_included_term: int
    data: dict[str, str]


@dataclass
class InstallSnapshotReply:
    """Reply to an InstallSnapshot RPC.

    Attributes:
        term: The responder's current term.
    """
    term: int


@dataclass
class RaftConfig:
    election_timeout_min_ms: int = 300
    election_timeout_max_ms: int = 500
    heartbeat_interval_ms: int = 100
    num_nodes: int = 5

    @property
    def majority(self) -> int:
        return (self.num_nodes // 2) + 1

    @property
    def election_timeout_min_s(self) -> float:
        return self.election_timeout_min_ms / 1000.0

    @property
    def election_timeout_max_s(self) -> float:
        return self.election_timeout_max_ms / 1000.0

    @property
    def heartbeat_interval_s(self) -> float:
        return self.heartbeat_interval_ms / 1000.0


# ---------------------------------------------------------------------------
# Network layer -- routes messages, supports partition simulation
# ---------------------------------------------------------------------------

class Network:
    """Simulates network communication between Raft nodes.

    Messages are delivered via direct async function calls. Partitions are
    simulated by maintaining a set of blocked (src, dst) pairs -- messages
    on blocked links are silently dropped.
    """

    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes
        self.nodes: dict[int, RaftNode] = {}
        self.blocked: set[tuple[int, int]] = set()
        self.message_delay_ms: float = 5.0
        self.logger = logging.getLogger("Network")

    def register(self, node: RaftNode) -> None:
        self.nodes[node.node_id] = node

    def is_reachable(self, src: int, dst: int) -> bool:
        return (src, dst) not in self.blocked

    def partition(self, group_a: list[int], group_b: list[int]) -> None:
        """Block all links between group_a and group_b."""
        for a in group_a:
            for b in group_b:
                self.blocked.add((a, b))
                self.blocked.add((b, a))
        self.logger.warning(
            "PARTITION: %s <-X-> %s (%d links blocked)",
            group_a, group_b, len(group_a) * len(group_b) * 2,
        )

    def heal(self) -> None:
        """Restore all links."""
        count = len(self.blocked)
        self.blocked.clear()
        self.logger.warning("HEAL: All links restored (%d unblocked)", count)

    async def send_request_vote(
        self, src: int, dst: int, args: RequestVoteArgs,
    ) -> RequestVoteReply | None:
        if not self.is_reachable(src, dst):
            return None
        if dst not in self.nodes:
            return None
        await asyncio.sleep(self.message_delay_ms / 1000.0)
        try:
            return self.nodes[dst].handle_request_vote(args)
        except Exception:
            return None

    async def send_append_entries(
        self, src: int, dst: int, args: AppendEntriesArgs,
    ) -> AppendEntriesReply | None:
        if not self.is_reachable(src, dst):
            return None
        if dst not in self.nodes:
            return None
        await asyncio.sleep(self.message_delay_ms / 1000.0)
        try:
            return self.nodes[dst].handle_append_entries(args)
        except Exception:
            return None

    async def send_install_snapshot(
        self, src: int, dst: int, args: InstallSnapshotArgs,
    ) -> InstallSnapshotReply | None:
        if not self.is_reachable(src, dst):
            return None
        if dst not in self.nodes:
            return None
        await asyncio.sleep(self.message_delay_ms / 1000.0)
        try:
            return self.nodes[dst].handle_install_snapshot(args)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Raft Node -- full state machine with election and replication
# ---------------------------------------------------------------------------

class RaftNode:
    """A single Raft node implementing the full consensus protocol.

    Implements: RequestVote RPC handler, AppendEntries RPC handler,
    leader election loop, log replication, and commit advancement.
    """

    def __init__(
        self,
        node_id: int,
        config: RaftConfig,
        network: Network,
    ) -> None:
        self.node_id = node_id
        self.config = config
        self.network = network
        self.logger = logging.getLogger(f"Node-{node_id}")

        # -- Persistent state (Raft paper Figure 2) --
        self.current_term: int = 0
        self.voted_for: int | None = None
        self.log: list[LogEntry] = []

        # -- Volatile state --
        self.commit_index: int = 0
        self.last_applied: int = 0
        self.state: NodeState = NodeState.FOLLOWER
        self.leader_id: int | None = None

        # -- Leader-specific volatile state --
        self.next_index: dict[int, int] = {}
        self.match_index: dict[int, int] = {}

        # -- State machine (simple key-value store) --
        self.state_machine: dict[str, str] = {}

        # -- Snapshot state (Raft paper Section 7) --
        self.snapshot_data: dict[str, str] | None = None
        self.snapshot_last_included_index: int = 0
        self.snapshot_last_included_term: int = 0

        # -- Timing --
        self.last_heartbeat_time: float = time.monotonic()
        self.election_timeout: float = self._random_election_timeout()

        # -- Leader lease for linearizable reads --
        self.lease_expiry: float = 0.0
        self.last_heartbeat_ack_time: float = 0.0

        # -- History tracking (for safety verification) --
        self.state_history: list[tuple[float, int, NodeState, int | None]] = []
        self._record_state()

        # -- Task handle --
        self._task: asyncio.Task | None = None
        self._running: bool = False

    # -- Helpers --

    def _random_election_timeout(self) -> float:
        return random.uniform(
            self.config.election_timeout_min_s,
            self.config.election_timeout_max_s,
        )

    def _record_state(self) -> None:
        self.state_history.append(
            (time.monotonic(), self.current_term, self.state, self.leader_id)
        )

    @property
    def last_log_index(self) -> int:
        return self.log[-1].index if self.log else self.snapshot_last_included_index

    @property
    def last_log_term(self) -> int:
        return self.log[-1].term if self.log else self.snapshot_last_included_term

    def _get_log_entry(self, index: int) -> LogEntry | None:
        """Get log entry by 1-based absolute index.

        Accounts for snapshot offset: actual array position is
        index - snapshot_last_included_index - 1.
        """
        offset = index - self.snapshot_last_included_index
        if offset <= 0 or offset > len(self.log):
            return None
        return self.log[offset - 1]

    def _log_slice_from(self, index: int) -> list[LogEntry]:
        """Get all log entries from the given absolute index onward."""
        offset = index - self.snapshot_last_included_index
        if offset <= 0:
            return list(self.log)
        if offset > len(self.log):
            return []
        return self.log[offset - 1:]

    @property
    def _log_length_absolute(self) -> int:
        """Absolute index of the last log entry (accounting for snapshot offset)."""
        return self.snapshot_last_included_index + len(self.log)

    def _peers(self) -> list[int]:
        return [i for i in range(self.config.num_nodes) if i != self.node_id]

    # -- State transitions --

    def step_down(self, term: int) -> None:
        """Revert to follower on discovering a higher term."""
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
            self.state = NodeState.FOLLOWER
            self.leader_id = None
            self._record_state()

    def become_candidate(self) -> RequestVoteArgs:
        """Transition to candidate: increment term, vote for self."""
        self.current_term += 1
        self.state = NodeState.CANDIDATE
        self.voted_for = self.node_id
        self.leader_id = None
        self.election_timeout = self._random_election_timeout()
        self._record_state()
        self.logger.info(
            "Became CANDIDATE for term %d", self.current_term,
        )
        return RequestVoteArgs(
            term=self.current_term,
            candidate_id=self.node_id,
            last_log_index=self.last_log_index,
            last_log_term=self.last_log_term,
        )

    def become_leader(self) -> None:
        """Transition to leader: initialize nextIndex and matchIndex."""
        self.state = NodeState.LEADER
        self.leader_id = self.node_id
        next_idx = self._log_length_absolute + 1
        for peer in self._peers():
            self.next_index[peer] = next_idx
            self.match_index[peer] = 0
        self._record_state()
        self.logger.info(
            "Became LEADER for term %d (log length=%d)",
            self.current_term, self._log_length_absolute,
        )

    # -- RequestVote RPC handler --

    def handle_request_vote(self, args: RequestVoteArgs) -> RequestVoteReply:
        """Process a RequestVote RPC (Raft paper Section 5.2)."""
        if args.term > self.current_term:
            self.step_down(args.term)

        if args.term < self.current_term:
            return RequestVoteReply(
                term=self.current_term,
                vote_granted=False,
                voter_id=self.node_id,
            )

        can_vote = self.voted_for is None or self.voted_for == args.candidate_id

        candidate_up_to_date = (
            args.last_log_term > self.last_log_term
            or (
                args.last_log_term == self.last_log_term
                and args.last_log_index >= self.last_log_index
            )
        )

        if can_vote and candidate_up_to_date:
            self.voted_for = args.candidate_id
            self.last_heartbeat_time = time.monotonic()
            return RequestVoteReply(
                term=self.current_term,
                vote_granted=True,
                voter_id=self.node_id,
            )

        return RequestVoteReply(
            term=self.current_term,
            vote_granted=False,
            voter_id=self.node_id,
        )

    # -- AppendEntries RPC handler --

    def handle_append_entries(self, args: AppendEntriesArgs) -> AppendEntriesReply:
        """Process an AppendEntries RPC (Raft paper Section 5.3)."""
        if args.term > self.current_term:
            self.step_down(args.term)

        if args.term < self.current_term:
            return AppendEntriesReply(
                term=self.current_term,
                success=False,
                follower_id=self.node_id,
            )

        # Valid leader -- reset election timer
        self.last_heartbeat_time = time.monotonic()
        self.leader_id = args.leader_id
        if self.state == NodeState.CANDIDATE:
            self.state = NodeState.FOLLOWER
            self._record_state()

        # Consistency check: log must match at prevLogIndex
        if args.prev_log_index > 0:
            if args.prev_log_index > self._log_length_absolute:
                return AppendEntriesReply(
                    term=self.current_term,
                    success=False,
                    follower_id=self.node_id,
                    conflict_index=self._log_length_absolute + 1,
                    conflict_term=0,
                )

            # Check if prevLogIndex is covered by snapshot
            if args.prev_log_index <= self.snapshot_last_included_index:
                # Entry is in snapshot -- if the term matches the snapshot's
                # last included term (or is before it), we trust the snapshot.
                # Only reject if prevLogIndex == snapshot boundary and terms mismatch.
                if (args.prev_log_index == self.snapshot_last_included_index
                        and args.prev_log_term != self.snapshot_last_included_term):
                    return AppendEntriesReply(
                        term=self.current_term,
                        success=False,
                        follower_id=self.node_id,
                        conflict_index=self.snapshot_last_included_index,
                        conflict_term=self.snapshot_last_included_term,
                    )
            else:
                prev_entry = self._get_log_entry(args.prev_log_index)
                if prev_entry is None or prev_entry.term != args.prev_log_term:
                    conflict_term = prev_entry.term if prev_entry else 0
                    conflict_index = args.prev_log_index
                    if prev_entry:
                        # Find first index of the conflicting term
                        while conflict_index > self.snapshot_last_included_index + 1:
                            e = self._get_log_entry(conflict_index - 1)
                            if e is None or e.term != conflict_term:
                                break
                            conflict_index -= 1
                    return AppendEntriesReply(
                        term=self.current_term,
                        success=False,
                        follower_id=self.node_id,
                        conflict_index=conflict_index,
                        conflict_term=conflict_term,
                    )

        # Append new entries, overwriting conflicts
        for entry in args.entries:
            if entry.index <= self.snapshot_last_included_index:
                continue  # Already covered by snapshot

            existing = self._get_log_entry(entry.index)
            if existing is not None:
                if existing.term != entry.term:
                    # Conflict: truncate from here
                    offset = entry.index - self.snapshot_last_included_index
                    self.log = self.log[:offset - 1]
                    self.log.append(entry)
                # else: already have this entry, skip
            else:
                self.log.append(entry)

        # Update commit index
        if args.leader_commit > self.commit_index:
            self.commit_index = min(args.leader_commit, self._log_length_absolute)
            self._apply_committed()

        return AppendEntriesReply(
            term=self.current_term,
            success=True,
            follower_id=self.node_id,
            match_index=self._log_length_absolute,
        )

    # -- InstallSnapshot RPC handler (stub -- Exercise 3 replaces this) --

    def handle_install_snapshot(self, args: InstallSnapshotArgs) -> InstallSnapshotReply:
        """Process an InstallSnapshot RPC (Raft paper Section 7).

        Default stub: accepts the snapshot if it's newer. Exercise 3
        asks the learner to replace this with a full implementation.
        """
        if args.term > self.current_term:
            self.step_down(args.term)

        if args.term < self.current_term:
            return InstallSnapshotReply(term=self.current_term)

        self.last_heartbeat_time = time.monotonic()
        self.leader_id = args.leader_id

        if args.last_included_index <= self.snapshot_last_included_index:
            return InstallSnapshotReply(term=self.current_term)

        # Discard log entries covered by the snapshot
        remaining: list[LogEntry] = []
        for entry in self.log:
            if entry.index > args.last_included_index:
                remaining.append(entry)
        self.log = remaining

        # Install the snapshot
        self.snapshot_data = dict(args.data)
        self.snapshot_last_included_index = args.last_included_index
        self.snapshot_last_included_term = args.last_included_term
        self.state_machine = dict(args.data)

        if self.commit_index < args.last_included_index:
            self.commit_index = args.last_included_index
        if self.last_applied < args.last_included_index:
            self.last_applied = args.last_included_index

        return InstallSnapshotReply(term=self.current_term)

    # -- Leader: replicate to a single follower --

    async def replicate_to_follower(self, peer: int) -> bool:
        """Send AppendEntries to a single follower. Returns True if successful."""
        if self.state != NodeState.LEADER:
            return False

        next_idx = self.next_index.get(peer, 1)

        # If the follower needs entries we've already compacted, send snapshot
        if next_idx <= self.snapshot_last_included_index and self.snapshot_data is not None:
            args = InstallSnapshotArgs(
                term=self.current_term,
                leader_id=self.node_id,
                last_included_index=self.snapshot_last_included_index,
                last_included_term=self.snapshot_last_included_term,
                data=dict(self.snapshot_data),
            )
            reply = await self.network.send_install_snapshot(self.node_id, peer, args)
            if reply is None:
                return False
            if reply.term > self.current_term:
                self.step_down(reply.term)
                return False
            self.next_index[peer] = self.snapshot_last_included_index + 1
            self.match_index[peer] = self.snapshot_last_included_index
            return True

        prev_log_index = next_idx - 1
        prev_log_term = 0
        if prev_log_index > 0:
            if prev_log_index == self.snapshot_last_included_index:
                prev_log_term = self.snapshot_last_included_term
            else:
                entry = self._get_log_entry(prev_log_index)
                prev_log_term = entry.term if entry else 0

        entries = self._log_slice_from(next_idx)

        args = AppendEntriesArgs(
            term=self.current_term,
            leader_id=self.node_id,
            prev_log_index=prev_log_index,
            prev_log_term=prev_log_term,
            entries=entries,
            leader_commit=self.commit_index,
        )

        reply = await self.network.send_append_entries(self.node_id, peer, args)
        if reply is None:
            return False

        if reply.term > self.current_term:
            self.step_down(reply.term)
            return False

        if reply.success:
            self.next_index[peer] = reply.match_index + 1
            self.match_index[peer] = reply.match_index
            return True
        else:
            if reply.conflict_term > 0:
                self.next_index[peer] = reply.conflict_index
            else:
                self.next_index[peer] = max(1, reply.conflict_index)
            return False

    # -- Leader: advance commit index --

    def advance_commit_index(self) -> None:
        """Find the highest N where a majority of matchIndex[i] >= N
        and log[N].term == currentTerm. (Raft paper Section 5.3/5.4)."""
        if self.state != NodeState.LEADER:
            return

        for n in range(self._log_length_absolute, self.commit_index, -1):
            entry = self._get_log_entry(n)
            if entry is None or entry.term != self.current_term:
                continue
            count = 1  # self
            for peer in self._peers():
                if self.match_index.get(peer, 0) >= n:
                    count += 1
            if count >= self.config.majority:
                self.commit_index = n
                self._apply_committed()
                break

    # -- State machine application --

    def _apply_committed(self) -> None:
        """Apply committed but not-yet-applied entries to the state machine."""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self._get_log_entry(self.last_applied)
            if entry is not None:
                self._apply_command(entry.command)

    def _apply_command(self, command: str) -> None:
        """Apply a single command to the key-value state machine.

        Supported commands: "SET key=value", "DEL key", "NOP".
        """
        if command.startswith("SET "):
            parts = command[4:].split("=", 1)
            if len(parts) == 2:
                self.state_machine[parts[0].strip()] = parts[1].strip()
        elif command.startswith("DEL "):
            key = command[4:].strip()
            self.state_machine.pop(key, None)

    # -- Client command submission --

    def submit_command(self, command: str) -> LogEntry | None:
        """Submit a client command to this node (must be leader)."""
        if self.state != NodeState.LEADER:
            return None
        entry = LogEntry(
            term=self.current_term,
            index=self._log_length_absolute + 1,
            command=command,
        )
        self.log.append(entry)
        self.match_index[self.node_id] = entry.index
        return entry

    # -- Main event loop --

    async def run(self) -> None:
        """Main loop: run election timeouts and leader heartbeats."""
        self._running = True
        while self._running:
            if self.state == NodeState.LEADER:
                await self._leader_loop_tick()
                await asyncio.sleep(self.config.heartbeat_interval_s)
            else:
                await self._follower_candidate_tick()
                await asyncio.sleep(0.05)

    async def _leader_loop_tick(self) -> None:
        """Leader: send heartbeats and replicate."""
        tasks = [self.replicate_to_follower(p) for p in self._peers()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.advance_commit_index()
        # Track heartbeat acks for lease-based reads
        ack_count = sum(1 for r in results if r is True) + 1  # +1 for self
        if ack_count >= self.config.majority:
            self.last_heartbeat_ack_time = time.monotonic()
            self.lease_expiry = (
                time.monotonic() + self.config.election_timeout_min_s
            )

    async def _follower_candidate_tick(self) -> None:
        """Follower/Candidate: check election timeout."""
        elapsed = time.monotonic() - self.last_heartbeat_time
        if elapsed >= self.election_timeout:
            await self._run_election()

    async def _run_election(self) -> None:
        """Run a single election round."""
        args = self.become_candidate()
        votes = 1

        tasks = [
            self.network.send_request_vote(self.node_id, peer, args)
            for peer in self._peers()
        ]
        replies = await asyncio.gather(*tasks, return_exceptions=True)

        for reply in replies:
            if self.state != NodeState.CANDIDATE:
                return
            if isinstance(reply, RequestVoteReply):
                if reply.term > self.current_term:
                    self.step_down(reply.term)
                    return
                if reply.vote_granted:
                    votes += 1

        if votes >= self.config.majority and self.state == NodeState.CANDIDATE:
            self.become_leader()
            await self._leader_loop_tick()

    def stop(self) -> None:
        self._running = False

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> asyncio.Task:
        self._task = asyncio.ensure_future(self.run())
        return self._task


# ---------------------------------------------------------------------------
# Raft Cluster -- manages nodes and network
# ---------------------------------------------------------------------------

class RaftCluster:
    """A Raft cluster with N nodes and a simulated network."""

    def __init__(self, config: RaftConfig | None = None) -> None:
        self.config = config or RaftConfig()
        self.network = Network(self.config.num_nodes)
        self.nodes: list[RaftNode] = []
        for i in range(self.config.num_nodes):
            node = RaftNode(i, self.config, self.network)
            self.nodes.append(node)
            self.network.register(node)

    def start_all(self) -> list[asyncio.Task]:
        """Start all nodes' event loops."""
        return [node.start() for node in self.nodes]

    def stop_all(self) -> None:
        """Stop all nodes."""
        for node in self.nodes:
            node.stop()

    def get_leader(self) -> RaftNode | None:
        """Return the current leader (if any)."""
        leaders = [n for n in self.nodes if n.state == NodeState.LEADER]
        return leaders[0] if len(leaders) == 1 else None

    async def wait_for_leader(self, timeout: float = 5.0) -> RaftNode | None:
        """Wait until a leader is elected."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            leader = self.get_leader()
            if leader is not None:
                return leader
            await asyncio.sleep(0.05)
        return None

    async def submit_and_wait(
        self, command: str, timeout: float = 3.0,
    ) -> bool:
        """Submit a command to the leader and wait for it to commit."""
        leader = self.get_leader()
        if leader is None:
            return False
        entry = leader.submit_command(command)
        if entry is None:
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if leader.commit_index >= entry.index:
                return True
            await asyncio.sleep(0.05)
        return False

    def print_cluster_state(self, label: str = "") -> None:
        """Print a summary of all nodes' states."""
        print(f"\n{'=' * 60}")
        if label:
            print(f"  {label}")
            print(f"{'=' * 60}")
        for node in self.nodes:
            leader_mark = " ***LEADER***" if node.state == NodeState.LEADER else ""
            snap_info = ""
            if node.snapshot_last_included_index > 0:
                snap_info = f"  snap_idx={node.snapshot_last_included_index}"
            print(
                f"  Node {node.node_id}: term={node.current_term:2d}  "
                f"state={node.state.name:10s}  "
                f"log_len={node._log_length_absolute:2d}  "
                f"commit={node.commit_index:2d}  "
                f"applied={node.last_applied:2d}"
                f"{snap_info}{leader_mark}"
            )
        print(f"{'=' * 60}")
