"""Exercise 3: Log Replication -- AppendEntries RPC and commitment.

This module adds log replication logic to RaftNode:
  1. handle_append_entries() -- follower processes AppendEntries from leader
  2. replicate_to_follower() -- leader sends entries to a specific follower
  3. advance_commit_index() -- leader checks if entries can be committed

These three functions implement the core of Raft's replicated log mechanism
(Raft paper Section 5.3). Together with leader election from Exercise 2,
they provide the complete Raft consensus protocol for Practice 049a.

Key concepts exercised here:
  - Consistency check (prevLogIndex / prevLogTerm matching)
  - Log repair (leader decrements nextIndex on rejection, retries)
  - Commitment rule (majority of matchIndex >= N AND log[N].term == currentTerm)
  - The subtle safety rule in Section 5.4.2: leaders only commit entries
    from their own term (previous-term entries are committed indirectly)

Reference: Raft paper Section 5.3, 5.4.2; Figure 2 "AppendEntries RPC".
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

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
from src.node import RaftNode, create_cluster  # noqa: E402

# Import election module to ensure handle_request_vote is attached
import importlib.util as _ilu

_election_spec = _ilu.spec_from_file_location(
    "leader_election",
    Path(__file__).resolve().parent / "02_leader_election.py",
)
_election_mod = _ilu.module_from_spec(_election_spec)
_election_spec.loader.exec_module(_election_mod)

run_election = _election_mod.run_election
simulate_election_with_timeout = _election_mod.simulate_election_with_timeout
print_cluster_state = _election_mod.print_cluster_state

logger = logging.getLogger("raft.replication")


# ======================================================================
# AppendEntries RPC Handler -- added to RaftNode
# ======================================================================

# TODO(human): Implement handle_append_entries(self, args: AppendEntriesArgs) -> AppendEntriesReply
#
# Process an incoming AppendEntries RPC. This is the FOLLOWER's logic --
# the leader sends us entries to append (or an empty heartbeat). We must
# check consistency, append entries if valid, and update our commit index.
#
# Parameters:
#   args (AppendEntriesArgs): The request from the leader, containing:
#     - args.term: the leader's current term
#     - args.leader_id: ID of the leader (so we can track who leads)
#     - args.prev_log_index: index of entry just before the new entries
#     - args.prev_log_term: term of entry at prev_log_index
#     - args.entries: list of LogEntry to append (empty for heartbeats)
#     - args.leader_commit: the leader's current commit index
#
# Returns:
#   AppendEntriesReply with:
#     - term: our current term
#     - success: True if we accepted and appended the entries
#     - follower_id: self.node_id
#     - match_index: our last log index after appending (if success)
#     - conflict_index/conflict_term: hint for faster backtracking (if !success)
#
# Algorithm (Raft paper Figure 2, "AppendEntries RPC: Receiver implementation"):
#
#   Step 1: If args.term < self.current_term, REJECT immediately.
#     Return AppendEntriesReply(term=self.current_term, success=False,
#            follower_id=self.node_id)
#     WHY: The sender is a stale leader. Our term is more recent, so
#     we should not accept entries from a leader that's been deposed.
#     The stale leader will see our higher term in the reply and step down.
#
#   Step 2: If args.term > self.current_term, call self.step_down(args.term).
#     Also do this if args.term == self.current_term and we're a CANDIDATE
#     (we lost the election -- the sender IS the legitimate leader).
#     WHY: Discovering a valid leader means we should revert to follower.
#     For candidates: receiving AppendEntries from a leader in the same
#     term means that leader already won the election.
#
#   Step 3: Update self.leader_id = args.leader_id.
#     Set self.state = NodeState.FOLLOWER (in case we were a candidate).
#     WHY: We now know who the leader is. Even if we were a candidate,
#     the AppendEntries proves someone else won.
#
#   Step 4: CONSISTENCY CHECK (the core of log replication safety).
#     If args.prev_log_index > 0:
#       Check if our log has an entry at args.prev_log_index with
#       term == args.prev_log_term.
#
#       If our log is TOO SHORT (len(self.log) < args.prev_log_index):
#         REJECT with conflict_index = len(self.log) + 1, conflict_term = 0.
#         WHY: We're missing entries. The leader needs to back up further.
#         conflict_index tells the leader where our log ends, so it can
#         jump back efficiently instead of decrementing one-by-one.
#
#       If our log HAS an entry at prev_log_index but the TERM DOESN'T MATCH:
#         REJECT with conflict_term = self.log_term_at(args.prev_log_index).
#         Set conflict_index to the FIRST index with that conflicting term
#         (scan backward from prev_log_index to find the start of the
#         conflicting term). Delete all entries from conflict_index onward.
#         WHY: The conflicting entries came from a deposed leader. We delete
#         them to make room for the new leader's entries. The conflict_term
#         and conflict_index hints let the leader skip past all entries from
#         that stale term in one step (optimization from Section 5.3).
#
#       If the terms MATCH: consistency check passes, proceed to Step 5.
#
#     If args.prev_log_index == 0: no check needed (appending from the start).
#
#   Step 5: APPEND new entries.
#     For each entry in args.entries:
#       - If we already have an entry at that index WITH THE SAME TERM,
#         skip it (already present, no conflict).
#       - If we have an entry at that index with a DIFFERENT TERM,
#         delete it and all following entries, then append the new one.
#         WHY: Conflicting entries from a previous leader must be replaced.
#       - If we DON'T have an entry at that index, append it.
#     After appending, our log should match the leader's through the last
#     new entry.
#
#   Step 6: UPDATE COMMIT INDEX.
#     If args.leader_commit > self.commit_index:
#       Set self.commit_index = min(args.leader_commit, index of last new entry)
#       WHY: The leader tells us its commit index, so we can apply entries
#       up to that point. We take the min because the leader might have
#       committed entries we don't have yet (if this AppendEntries only
#       includes a subset).
#
#   Step 7: Return success.
#     Return AppendEntriesReply(
#         term=self.current_term,
#         success=True,
#         follower_id=self.node_id,
#         match_index=self.last_log_index,
#     )
#
# Why this matters:
#   handle_append_entries is the most complex RPC handler in Raft. It
#   enforces the Log Matching Property: if two logs contain an entry with
#   the same index and term, all preceding entries are identical. The
#   consistency check (Step 4) is the mechanism that achieves this --
#   the leader walks back through the follower's log until it finds the
#   point of agreement, then overwrites everything after that point.
#   This handles all possible log divergence scenarios (follower missing
#   entries, follower having extra entries from a deposed leader, etc.).
#
# Hint: Be careful with 1-based vs 0-based indexing. LogEntry.index is
# 1-based (as in the Raft paper), but self.log is a 0-based Python list.
# Use self.log_entry_at(index) and self.log_term_at(index) which handle
# the conversion. When deleting entries, self.log = self.log[:index - 1]
# removes everything from 1-based 'index' onward.

def handle_append_entries(self: RaftNode, args: AppendEntriesArgs) -> AppendEntriesReply:
    raise NotImplementedError(
        "TODO(human): Implement AppendEntries RPC handler. "
        "Check term, consistency, append entries, update commit index."
    )


RaftNode.handle_append_entries = handle_append_entries


# ======================================================================
# Leader-side replication logic -- added to RaftNode
# ======================================================================

# TODO(human): Implement replicate_to_follower(self, follower: RaftNode) -> bool
#
# As the LEADER, send AppendEntries RPC to a specific follower and process
# the reply. This handles both initial replication and log repair (when the
# follower's log diverges from ours).
#
# Parameters:
#   follower (RaftNode): The follower node to replicate to.
#     (In a real system, this would be an RPC call over the network.
#      Here we call follower.handle_append_entries() directly.)
#
# Returns:
#   bool: True if replication succeeded (follower's log is up-to-date),
#         False if we stepped down (discovered higher term).
#
# Algorithm (Raft paper Figure 2, "Rules for Servers: Leaders"):
#
#   1. Get the follower_id: follower.node_id
#
#   2. Loop (retry until success or step-down):
#
#     a. Compute prev_log_index = self.next_index[follower_id] - 1
#        Compute prev_log_term = self.log_term_at(prev_log_index)
#        WHY: next_index[follower_id] is the index of the NEXT entry to
#        send. The entry just before it (prev_log_index) is used for the
#        consistency check. If next_index is 1, prev_log_index is 0
#        (meaning: start from the beginning of the log).
#
#     b. Collect entries to send:
#        entries = [self.log[i] for i in range(next_index - 1, len(self.log))]
#        (Entries from next_index to the end of the leader's log.)
#        WHY: We send ALL entries the follower is missing, not just one.
#        This minimizes the number of round-trips.
#
#     c. Build AppendEntriesArgs:
#        args = AppendEntriesArgs(
#            term=self.current_term,
#            leader_id=self.node_id,
#            prev_log_index=prev_log_index,
#            prev_log_term=prev_log_term,
#            entries=entries,
#            leader_commit=self.commit_index,
#        )
#
#     d. Send the RPC: reply = follower.handle_append_entries(args)
#
#     e. If reply.term > self.current_term:
#          Call self.step_down(reply.term) and return False.
#          WHY: The follower is in a newer term. We're a stale leader.
#
#     f. If reply.success:
#          Update self.next_index[follower_id] = reply.match_index + 1
#          Update self.match_index[follower_id] = reply.match_index
#          Log: logger.info(f"Leader {self.node_id}: replicated to Node {follower_id}, match_index={reply.match_index}")
#          Return True.
#
#     g. If NOT reply.success (consistency check failed):
#          LOG REPAIR: The follower's log doesn't match at prev_log_index.
#          Decrement self.next_index[follower_id].
#
#          OPTIMIZATION: Use the conflict hints from the reply:
#            If reply.conflict_term > 0:
#              Search our log for the LAST entry with reply.conflict_term.
#              If found, set next_index to that index + 1.
#              If not found, set next_index to reply.conflict_index.
#            Else:
#              Set next_index to reply.conflict_index (follower's log is shorter).
#          Ensure next_index >= 1 (never go below 1).
#
#          Log: logger.debug(f"Leader {self.node_id}: log mismatch with Node {follower_id}, backing up to next_index={self.next_index[follower_id]}")
#          Continue the loop (retry with the updated next_index).
#
# Why this matters:
#   replicate_to_follower implements the leader's half of log replication.
#   The retry loop with next_index decrement is the "log repair" mechanism --
#   the leader keeps backing up until it finds the point where the follower's
#   log matches, then sends everything from that point forward. This handles
#   ALL types of log divergence (missing entries, extra stale entries, gaps).
#   The conflict hints optimization reduces the number of round-trips from
#   O(log_length) to O(num_conflicting_terms) in the worst case.

def replicate_to_follower(self: RaftNode, follower: RaftNode) -> bool:
    raise NotImplementedError(
        "TODO(human): As leader, send AppendEntries to a follower. "
        "Handle success (update next/match index) and failure (log repair)."
    )


RaftNode.replicate_to_follower = replicate_to_follower


# ======================================================================
# Commitment logic -- added to RaftNode
# ======================================================================

# TODO(human): Implement advance_commit_index(self) -> int
#
# As the LEADER, check if any new entries can be committed. An entry is
# committed when it has been replicated to a majority of nodes.
#
# Parameters:
#   None (operates on self.log, self.match_index, self.commit_index).
#
# Returns:
#   int: The new commit_index after advancement (may be unchanged).
#
# Algorithm (Raft paper Section 5.3/5.4.2, Figure 2 "Rules for Leaders"):
#
#   For N from self.last_log_index down to self.commit_index + 1:
#     Count how many peers have match_index[peer] >= N.
#     Add 1 for the leader itself (the leader always has its own entries).
#     If count >= self.config.majority AND self.log_term_at(N) == self.current_term:
#       Set self.commit_index = N
#       Break (we found the highest committable index).
#
# Return self.commit_index.
#
# CRITICAL SUBTLETY -- the term check (Raft paper Section 5.4.2):
#   The condition `self.log_term_at(N) == self.current_term` is essential.
#   A leader must NOT commit entries from PREVIOUS terms using only their
#   replication count. Here's why:
#
#   Consider this scenario (from the paper):
#   - Term 2: Leader S1 replicates entry at index 2 to S2 (but not S3-S5)
#   - Term 3: S5 becomes leader, writes entry at index 2 (different value)
#   - Term 4: S1 becomes leader again, replicates index 2 to S3 (now on S1,S2,S3 = majority!)
#
#   If S1 commits index 2 (term 2 entry) and then crashes, S5 could become
#   leader in term 5 and overwrite index 2 with its term 3 entry -- violating
#   the State Machine Safety property!
#
#   The fix: S1 only commits entries from its OWN term (term 4). Once a
#   term 4 entry at index 3+ is committed, it implicitly commits all
#   preceding entries (because commitment is prefix-based and the new leader
#   must have all committed entries). This is the Leader Completeness
#   property in action.
#
# Why this matters:
#   advance_commit_index is called after every successful AppendEntries
#   reply. It's the mechanism that moves entries from "replicated" to
#   "committed" -- and committed entries are safe to apply to the state
#   machine. Getting the term check wrong leads to the exact safety bug
#   described in Section 5.4.2, which is one of the most subtle issues
#   in Raft. Many early Raft implementations got this wrong.
#
# Hint: Iterate from high to low. Once you find the first N that satisfies
# both conditions (majority replicated AND current term), that's the new
# commit_index. All entries below it are implicitly committed too.

def advance_commit_index(self: RaftNode) -> int:
    raise NotImplementedError(
        "TODO(human): As leader, find the highest N where majority has "
        "replicated it AND log[N].term == currentTerm. Set commit_index = N."
    )


RaftNode.advance_commit_index = advance_commit_index


# ======================================================================
# Main: log replication demo
# ======================================================================

async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("Exercise 3: Log Replication -- Simulation")
    print("=" * 60)

    config = RaftConfig(num_nodes=5)

    # ------------------------------------------------------------------
    # Setup: elect a leader first
    # ------------------------------------------------------------------
    print("\n[Setup] Electing a leader...")
    nodes = create_cluster(config)
    leader_id = await simulate_election_with_timeout(nodes, config)
    leader = nodes[leader_id]
    print(f"  Leader elected: Node {leader_id}")

    # ------------------------------------------------------------------
    # Test 1: Basic replication -- leader appends entries, replicates
    # ------------------------------------------------------------------
    print("\n[Test 1] Basic log replication")

    # Leader receives 3 client commands
    commands = ["SET x=1", "SET y=2", "SET z=3"]
    for i, cmd in enumerate(commands):
        entry = LogEntry(term=leader.current_term, index=leader.last_log_index + 1, command=cmd)
        leader.log.append(entry)
        print(f"  Leader appended: {entry}")

    # Replicate to all followers
    print("\n  Replicating to followers...")
    for peer_id in leader.peers:
        follower = nodes[peer_id]
        success = leader.replicate_to_follower(follower)
        assert success, f"Replication to Node {peer_id} failed!"
        print(f"  Node {peer_id}: replicated, match_index={leader.match_index[peer_id]}")

    # Advance commit index
    old_commit = leader.commit_index
    leader.advance_commit_index()
    print(f"\n  Leader commit_index: {old_commit} -> {leader.commit_index}")
    assert leader.commit_index == 3, f"Expected commit_index=3, got {leader.commit_index}"

    # Verify all logs match
    print("\n  Verifying log consistency...")
    for node in nodes:
        assert len(node.log) == 3, f"Node {node.node_id} has {len(node.log)} entries, expected 3"
        for i, entry in enumerate(node.log):
            assert entry.command == commands[i], (
                f"Node {node.node_id} entry {i}: {entry.command} != {commands[i]}"
            )
    print("  All logs match!")

    # ------------------------------------------------------------------
    # Test 2: Log repair -- follower has missing entries
    # ------------------------------------------------------------------
    print("\n[Test 2] Log repair: follower with missing entries")

    # Simulate: Node peers[0] missed entries 2 and 3
    repair_node_id = leader.peers[0]
    repair_node = nodes[repair_node_id]
    repair_node.log = repair_node.log[:1]  # keep only first entry
    leader.next_index[repair_node_id] = 4  # leader thinks follower is up-to-date
    leader.match_index[repair_node_id] = 3

    print(f"  Node {repair_node_id} log truncated to 1 entry (simulating missed entries)")
    print(f"  Leader thinks next_index={leader.next_index[repair_node_id]} (stale)")

    # Replicate -- should detect mismatch and repair
    success = leader.replicate_to_follower(repair_node)
    assert success, "Log repair failed!"
    assert len(repair_node.log) == 3, f"After repair: {len(repair_node.log)} entries, expected 3"
    print(f"  After repair: Node {repair_node_id} has {len(repair_node.log)} entries")
    print(f"  Leader next_index={leader.next_index[repair_node_id]}, match_index={leader.match_index[repair_node_id]}")
    print("  Log repair successful!")

    # ------------------------------------------------------------------
    # Test 3: Log repair -- follower has conflicting entries
    # ------------------------------------------------------------------
    print("\n[Test 3] Log repair: follower with conflicting entries")

    conflict_node_id = leader.peers[1]
    conflict_node = nodes[conflict_node_id]

    # Simulate: conflict_node has entries from a different (deposed) leader
    conflict_node.log = [
        LogEntry(term=leader.current_term, index=1, command="SET x=1"),  # matches
        LogEntry(term=99, index=2, command="STALE_CMD_A"),  # conflict!
        LogEntry(term=99, index=3, command="STALE_CMD_B"),  # conflict!
    ]
    leader.next_index[conflict_node_id] = 4
    leader.match_index[conflict_node_id] = 3

    print(f"  Node {conflict_node_id} has conflicting entries at index 2,3 (term 99)")

    success = leader.replicate_to_follower(conflict_node)
    assert success, "Conflict resolution failed!"
    assert len(conflict_node.log) == 3
    assert conflict_node.log[1].command == "SET y=2", "Conflict not resolved at index 2"
    assert conflict_node.log[2].command == "SET z=3", "Conflict not resolved at index 3"
    print(f"  After repair: conflicts resolved, log matches leader")

    # ------------------------------------------------------------------
    # Test 4: Commitment requires current term
    # ------------------------------------------------------------------
    print("\n[Test 4] Commitment rule: only commit entries from current term")

    nodes2 = create_cluster(RaftConfig(num_nodes=3))
    # Manually set up a leader in term 2
    nodes2[0].current_term = 2
    nodes2[0].state = NodeState.LEADER
    nodes2[0].leader_id = 0
    nodes2[0].next_index = {1: 1, 2: 1}
    nodes2[0].match_index = {1: 0, 2: 0}

    # Leader has an entry from term 1 (previous leader's entry)
    nodes2[0].log.append(LogEntry(term=1, index=1, command="OLD_CMD"))
    # Replicate to both followers
    for peer_id in [1, 2]:
        nodes2[peer_id].current_term = 2
        nodes2[0].replicate_to_follower(nodes2[peer_id])

    # Try to commit -- should NOT commit because entry is from term 1, not term 2
    nodes2[0].advance_commit_index()
    print(f"  After replicating term-1 entry to majority: commit_index={nodes2[0].commit_index}")
    assert nodes2[0].commit_index == 0, (
        "Should NOT commit term-1 entry in term-2 leader! (Section 5.4.2)"
    )

    # Now add a term-2 entry and replicate
    nodes2[0].log.append(LogEntry(term=2, index=2, command="NEW_CMD"))
    for peer_id in [1, 2]:
        nodes2[0].replicate_to_follower(nodes2[peer_id])
    nodes2[0].advance_commit_index()
    print(f"  After replicating term-2 entry to majority: commit_index={nodes2[0].commit_index}")
    assert nodes2[0].commit_index == 2, (
        "Should commit up to index 2 (term-2 entry implicitly commits term-1 entry)"
    )
    print("  PASSED: Commitment rule correctly enforced (Section 5.4.2)")

    # ------------------------------------------------------------------
    # Test 5: Heartbeat (empty AppendEntries)
    # ------------------------------------------------------------------
    print("\n[Test 5] Heartbeat -- empty AppendEntries")

    heartbeat_args = AppendEntriesArgs(
        term=leader.current_term,
        leader_id=leader.node_id,
        prev_log_index=leader.last_log_index,
        prev_log_term=leader.last_log_term,
        entries=[],  # empty = heartbeat
        leader_commit=leader.commit_index,
    )
    follower = nodes[leader.peers[0]]
    reply = follower.handle_append_entries(heartbeat_args)
    assert reply.success, "Heartbeat should succeed on up-to-date follower"
    print(f"  Heartbeat to Node {follower.node_id}: success={reply.success}")
    print("  PASSED: Heartbeat accepted by follower")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n  Final cluster state:")
    print_cluster_state(nodes, "After all replication tests")

    print("\n" + "=" * 60)
    print("All log replication tests completed!")
    print("=" * 60)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
