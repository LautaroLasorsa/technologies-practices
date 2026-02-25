"""Exercise 2: Leader Election -- RequestVote RPC and election simulation.

This module adds vote-handling logic to RaftNode and provides functions to
simulate the full election process: a candidate broadcasts RequestVote RPCs,
collects replies, and either becomes leader (majority) or retries (split vote).

The key safety mechanisms implemented here:
  1. Vote granting rules (Raft paper Section 5.2, 5.4.1)
  2. Election restriction -- candidate's log must be up-to-date (Section 5.4.1)
  3. Randomized election timeouts to prevent split votes (Section 5.2)
  4. Term-based step-down on every incoming RPC (Section 5.1)

Reference: Raft paper Sections 5.1, 5.2, 5.4.1; Figure 2 "RequestVote RPC".
"""

from __future__ import annotations

import asyncio
import logging
import random
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

logger = logging.getLogger("raft.election")


# ======================================================================
# RequestVote RPC Handler -- added to RaftNode
# ======================================================================

# TODO(human): Implement handle_request_vote(self, args: RequestVoteArgs) -> RequestVoteReply
#
# Process an incoming RequestVote RPC. This is the VOTER's logic -- when
# another node (the candidate) asks us for our vote, we decide whether to
# grant it based on three conditions.
#
# Parameters:
#   args (RequestVoteArgs): The request from the candidate, containing:
#     - args.term: the candidate's current term
#     - args.candidate_id: who is requesting the vote
#     - args.last_log_index: index of candidate's last log entry
#     - args.last_log_term: term of candidate's last log entry
#
# Returns:
#   RequestVoteReply with:
#     - term: our current term (so candidate can update itself)
#     - vote_granted: True if we voted for this candidate
#     - voter_id: self.node_id
#
# Algorithm (Raft paper Figure 2, "RequestVote RPC: Receiver implementation"):
#
#   Step 1: If args.term > self.current_term, call self.step_down(args.term).
#     WHY: We're in a stale term. Before making any decisions, update to
#     the new term and reset our vote. This is critical -- without it, a
#     node in an old term might reject a valid candidate.
#
#   Step 2: If args.term < self.current_term, REJECT (vote_granted=False).
#     WHY: The candidate is in a stale term. Our term is more recent,
#     so this candidate cannot possibly win a valid election. Replying
#     with our term lets the candidate discover it's stale and step down.
#
#   Step 3: Check if we can vote for this candidate:
#     Condition A -- Haven't voted yet, or already voted for this candidate:
#       (self.voted_for is None) or (self.voted_for == args.candidate_id)
#     WHY: Each node votes for at most one candidate per term. If we
#     already voted for someone else, we must reject. But if we already
#     voted for THIS candidate (e.g., retry due to lost reply), we can
#     re-grant the vote safely (idempotency).
#
#     Condition B -- Candidate's log is at least as up-to-date as ours:
#       The "election restriction" (Raft paper Section 5.4.1). This is
#       what ensures the Leader Completeness property. Compare logs:
#         - If args.last_log_term > self.last_log_term: candidate is more recent -> OK
#         - If args.last_log_term == self.last_log_term AND
#           args.last_log_index >= self.last_log_index: candidate has same or longer log -> OK
#         - Otherwise: candidate's log is LESS up-to-date -> REJECT
#       WHY: If we elect a leader whose log is missing committed entries,
#       those entries could be lost. The election restriction prevents this
#       by ensuring only candidates with all committed entries can win.
#
#   Step 4: If BOTH conditions A and B are satisfied:
#     - Set self.voted_for = args.candidate_id
#     - Return RequestVoteReply(term=self.current_term, vote_granted=True, voter_id=self.node_id)
#     - Log: logger.info(f"Node {self.node_id} voted for {args.candidate_id} in term {self.current_term}")
#
#   Step 5: Otherwise, reject:
#     - Return RequestVoteReply(term=self.current_term, vote_granted=False, voter_id=self.node_id)
#
# Why this matters:
#   handle_request_vote is the SOLE mechanism that controls who can become
#   leader. The election restriction (Condition B) is what makes Raft safe --
#   it guarantees that a newly elected leader has every committed entry in
#   its log, so it never needs to "learn" old entries from followers.
#   Without it, you could elect a leader with a gap in its log, violating
#   Leader Completeness and potentially losing committed data.
#
# Edge cases to consider:
#   - Candidate sends RequestVote with same term we're already in: valid,
#     as long as we haven't voted for someone else.
#   - Our log is empty (last_log_index=0, last_log_term=0): any candidate's
#     log is at least as up-to-date, so Condition B is always satisfied.
#   - Two candidates in the same term: we vote for the first one that
#     arrives and reject the second (Condition A fails).

def handle_request_vote(self: RaftNode, args: RequestVoteArgs) -> RequestVoteReply:
    raise NotImplementedError(
        "TODO(human): Implement RequestVote RPC handler. "
        "Check term, vote availability, and log up-to-dateness."
    )


# Attach the method to RaftNode (monkey-patch for exercise modularity)
RaftNode.handle_request_vote = handle_request_vote


# ======================================================================
# Election Simulation Functions
# ======================================================================

# TODO(human): Implement run_election(nodes, candidate_id) -> int | None
#
# Simulate a single election round: one node becomes a candidate, sends
# RequestVote to all peers, collects votes, and either becomes leader or
# returns None (failed to get majority).
#
# Parameters:
#   nodes (list[RaftNode]): All nodes in the cluster (nodes[i].node_id == i).
#   candidate_id (int): The node that will become a candidate and run the election.
#
# Returns:
#   int: The candidate_id if it won the election (became leader).
#   None: If the candidate did not receive a majority of votes.
#
# Algorithm:
#   1. Get the candidate node: candidate = nodes[candidate_id]
#   2. Call candidate.become_candidate() to get the RequestVoteArgs.
#   3. For each peer in candidate.peers:
#        - Call nodes[peer].handle_request_vote(rv_args)
#        - If the reply grants a vote, add the peer to candidate.votes_received
#        - If the reply has a higher term, call candidate.step_down(reply.term)
#          and return None immediately (candidate discovered it's stale)
#   4. After collecting all votes, check:
#        - If len(candidate.votes_received) >= candidate.config.majority:
#            Call candidate.become_leader()
#            Return candidate_id
#        - Else:
#            Return None (split vote or rejected)
#   5. Log the result: logger.info(f"Election result: ... won/lost with N votes")
#
# Why this matters:
#   This function orchestrates the election from the candidate's perspective.
#   In a real Raft implementation, the candidate would send RPCs in parallel
#   over the network. Here we call handle_request_vote() directly on each
#   peer node (simulating synchronous network calls). The important thing
#   is the LOGIC, not the transport.
#
# Note: This is a simplified simulation. Real Raft sends RequestVote RPCs
# in parallel and processes replies asynchronously. We iterate sequentially
# for clarity, which doesn't affect correctness (only performance).

async def run_election(nodes: list[RaftNode], candidate_id: int) -> int | None:
    raise NotImplementedError(
        "TODO(human): Run a single election round. "
        "Candidate sends RequestVote to all peers, checks majority."
    )


# TODO(human): Implement simulate_election_with_timeout(nodes, config) -> int
#
# Simulate the full Raft election process with randomized timeouts:
# multiple nodes may time out and start elections, possibly resulting in
# split votes that require retries.
#
# Parameters:
#   nodes (list[RaftNode]): All nodes in the cluster.
#   config (RaftConfig): Cluster configuration (for timeout values).
#
# Returns:
#   int: The node_id of the elected leader.
#
# Algorithm:
#   1. Assign a random election timeout to each node:
#        timeout[i] = random.uniform(config.election_timeout_min_s, config.election_timeout_max_s)
#      WHY: Randomized timeouts are Raft's primary mechanism to prevent
#      split votes. If all nodes had the same timeout, they'd all become
#      candidates simultaneously, split the votes, and repeat forever
#      (livelock). With randomized timeouts, one node almost always times
#      out first and wins the election before others even start.
#
#   2. Sort nodes by their timeout (ascending). The node with the shortest
#      timeout becomes a candidate first.
#
#   3. For each node (in timeout order):
#        a. Run the election for this candidate: result = await run_election(nodes, node_id)
#        b. If result is not None (election succeeded), return result.
#        c. If result is None (split vote or stale term), continue to next node.
#
#   4. If no node won (all elections failed -- very rare with randomization):
#      Repeat from step 1. This simulates the "new term, retry" behavior.
#      Limit retries to prevent infinite loops (e.g., max 10 rounds).
#
#   5. After the leader is elected, update all follower nodes so they know
#      who the leader is:
#        for node in nodes:
#            if node.node_id != leader_id:
#                node.leader_id = leader_id
#                node.current_term = nodes[leader_id].current_term
#
# Why this matters:
#   This simulates the COMPLETE election lifecycle, including the critical
#   randomized timeout mechanism. In a real system, each node runs its own
#   timer independently. Here we simulate the effect by sorting timeouts
#   to determine which node "fires" first. The retry loop handles split
#   votes, demonstrating why randomization makes them extremely rare in
#   practice (typically resolved in 1-2 rounds).
#
# Hint: Use asyncio.sleep() with a small delay between rounds to make the
# simulation more realistic and readable in logs.

async def simulate_election_with_timeout(
    nodes: list[RaftNode], config: RaftConfig
) -> int:
    raise NotImplementedError(
        "TODO(human): Run full election with randomized timeouts. "
        "Handle split votes by retrying with new terms."
    )


# ======================================================================
# Verification helpers
# ======================================================================

def verify_election_safety(nodes: list[RaftNode]) -> bool:
    """Verify Election Safety: at most one leader per term.

    Checks that no two nodes are leaders in the same term. This is the
    most fundamental Raft safety property.
    """
    leaders_by_term: dict[int, list[int]] = {}
    for node in nodes:
        if node.state == NodeState.LEADER:
            leaders_by_term.setdefault(node.current_term, []).append(node.node_id)

    for term, leaders in leaders_by_term.items():
        if len(leaders) > 1:
            logger.error(
                f"ELECTION SAFETY VIOLATION: term {term} has leaders {leaders}"
            )
            return False

    return True


def print_cluster_state(nodes: list[RaftNode], label: str = "") -> None:
    """Print the state of all nodes in the cluster."""
    if label:
        print(f"\n--- {label} ---")
    for node in nodes:
        role_marker = ""
        if node.state == NodeState.LEADER:
            role_marker = " <-- LEADER"
        elif node.state == NodeState.CANDIDATE:
            role_marker = " (candidate)"
        print(f"  {node}{role_marker}")


# ======================================================================
# Main: election demo
# ======================================================================

async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("Exercise 2: Leader Election -- Simulation")
    print("=" * 60)

    config = RaftConfig(num_nodes=5)

    # ------------------------------------------------------------------
    # Test 1: Simple election -- one candidate, no competition
    # ------------------------------------------------------------------
    print("\n[Test 1] Simple election: Node 0 runs unopposed")
    nodes = create_cluster(config)
    result = await run_election(nodes, candidate_id=0)
    print_cluster_state(nodes, "After election")
    assert result == 0, f"Expected Node 0 to win, got {result}"
    assert nodes[0].state == NodeState.LEADER
    assert verify_election_safety(nodes), "Election Safety violated!"
    print("  PASSED: Node 0 elected as leader")

    # ------------------------------------------------------------------
    # Test 2: Election with pre-existing votes (split scenario)
    # ------------------------------------------------------------------
    print("\n[Test 2] Election after another candidate already took some votes")
    nodes = create_cluster(config)

    # Node 3 already voted for itself in term 1 (started an election earlier)
    nodes[3].current_term = 1
    nodes[3].voted_for = 3

    # Node 0 tries to run for term 1 -- Node 3 won't vote for it
    result = await run_election(nodes, candidate_id=0)
    print_cluster_state(nodes, "After election")
    # Node 0 should still win: it gets votes from 0 (self), 1, 2, 4 = 4 votes
    # (Node 3 already voted for itself, so it rejects Node 0)
    if result == 0:
        print("  PASSED: Node 0 still won (got 4/5 votes, Node 3 abstained)")
    else:
        print(f"  Result: {result} (depends on implementation)")

    assert verify_election_safety(nodes), "Election Safety violated!"

    # ------------------------------------------------------------------
    # Test 3: Full election with randomized timeouts
    # ------------------------------------------------------------------
    print("\n[Test 3] Full election with randomized timeouts")
    nodes = create_cluster(config)
    leader_id = await simulate_election_with_timeout(nodes, config)
    print_cluster_state(nodes, "After election with timeouts")
    assert nodes[leader_id].state == NodeState.LEADER
    assert verify_election_safety(nodes), "Election Safety violated!"
    print(f"  PASSED: Node {leader_id} elected as leader")

    # ------------------------------------------------------------------
    # Test 4: Election restriction -- candidate with stale log rejected
    # ------------------------------------------------------------------
    print("\n[Test 4] Election restriction: stale log candidate rejected")
    nodes = create_cluster(config)

    # Give nodes 1,2,3,4 a log entry in term 1 (they're "up-to-date")
    for i in [1, 2, 3, 4]:
        nodes[i].log.append(LogEntry(term=1, index=1, command="SET x=1"))
        nodes[i].current_term = 1

    # Node 0 has an EMPTY log but tries to become leader
    # The election restriction should prevent it: nodes 1-4 won't vote
    # for a candidate whose log is less up-to-date than theirs.
    result = await run_election(nodes, candidate_id=0)
    print_cluster_state(nodes, "After stale-log election attempt")
    if result is None:
        print("  PASSED: Node 0 (empty log) correctly rejected by up-to-date nodes")
    else:
        print(f"  WARNING: Node 0 won despite stale log (result={result})")
        print("  Check your election restriction implementation!")

    # ------------------------------------------------------------------
    # Test 5: Election restriction -- up-to-date candidate wins
    # ------------------------------------------------------------------
    print("\n[Test 5] Election restriction: up-to-date candidate wins")
    nodes = create_cluster(config)

    # All nodes have the same log
    for node in nodes:
        node.log.append(LogEntry(term=1, index=1, command="SET x=1"))
        node.current_term = 1

    # Node 2 has an EXTRA entry (more up-to-date)
    nodes[2].log.append(LogEntry(term=1, index=2, command="SET y=2"))

    result = await run_election(nodes, candidate_id=2)
    print_cluster_state(nodes, "After up-to-date candidate election")
    assert result == 2, f"Expected Node 2 (most up-to-date) to win, got {result}"
    print("  PASSED: Node 2 (most up-to-date log) elected as leader")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("All election tests completed!")
    print("=" * 60)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
