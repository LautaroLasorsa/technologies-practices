"""Exercise 1: Raft Safety Property Verification.

This module runs a Raft cluster through multiple rounds of operations
(elections, commands, simulated failures) and then verifies the three
most important Raft safety properties on the collected state history:

  1. Election Safety    -- at most one leader per term
  2. Log Matching       -- matching (index, term) implies identical prefix
  3. Leader Completeness -- committed entries present in all future leaders

These are not "tests" in the unit-test sense -- they are runtime invariant
checks on actual cluster behavior. Production Raft implementations (etcd,
HashiCorp Raft) include similar assertion checks. TLA+ model checking of
Raft verifies these same properties exhaustively.

Reference: Raft paper Figure 3 (Safety), Section 5.4.

Usage:
    uv run python src/01_safety_verification.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Allow importing the core module from the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from raft_core import (  # noqa: E402
    LogEntry,
    NodeState,
    RaftCluster,
    RaftConfig,
    RaftNode,
)


# ---------------------------------------------------------------------------
# Types for safety verification
# ---------------------------------------------------------------------------

@dataclass
class SafetyResult:
    """Result of a safety property check."""
    property_name: str
    is_safe: bool
    violations: list[str]

    def __str__(self) -> str:
        status = "PASS" if self.is_safe else "FAIL"
        lines = [f"[{status}] {self.property_name}"]
        for v in self.violations:
            lines.append(f"  VIOLATION: {v}")
        return "\n".join(lines)


@dataclass
class ClusterSnapshot:
    """A snapshot of the cluster state at a moment in time.

    Captured periodically during cluster operation to build a history
    that can be verified after the fact.
    """
    timestamp: float
    node_states: dict[int, NodeState]
    node_terms: dict[int, int]
    node_leaders: dict[int, int | None]
    node_logs: dict[int, list[LogEntry]]
    node_commit_indices: dict[int, int]

    @staticmethod
    def capture(cluster: RaftCluster) -> ClusterSnapshot:
        """Capture the current state of all nodes in the cluster."""
        return ClusterSnapshot(
            timestamp=time.monotonic(),
            node_states={n.node_id: n.state for n in cluster.nodes},
            node_terms={n.node_id: n.current_term for n in cluster.nodes},
            node_leaders={n.node_id: n.leader_id for n in cluster.nodes},
            node_logs={
                n.node_id: list(n.log) for n in cluster.nodes
            },
            node_commit_indices={n.node_id: n.commit_index for n in cluster.nodes},
        )


# ---------------------------------------------------------------------------
# Safety verifiers -- TODO(human)
# ---------------------------------------------------------------------------

def verify_election_safety(history: list[ClusterSnapshot]) -> SafetyResult:
    """Verify Election Safety: at most one leader per term.

    This is the most fundamental Raft invariant (Raft paper Figure 3,
    Section 5.2). If this property is violated, the entire protocol is
    broken -- two leaders in the same term means conflicting commands
    can be committed, destroying state machine consistency.

    Algorithm:
      1. Iterate through every snapshot in the history.
      2. For each snapshot, find all nodes that are in LEADER state.
      3. Group leaders by their current_term.
      4. For each term, if there are two or more DIFFERENT node IDs
         claiming to be leader, that's a violation.
      5. Collect all violations into a list of descriptive strings.

    Note: The same node may appear as leader in multiple snapshots for
    the same term -- that's fine (it's the same leader persisting).
    The violation is when TWO DIFFERENT nodes are leader in the same term.

    Consider using a dict[int, set[int]] mapping term -> set of leader IDs
    seen across all snapshots. After processing all snapshots, check each
    term: if the set has more than one element, it's a violation.

    Args:
        history: List of ClusterSnapshot captured during cluster operation.

    Returns:
        SafetyResult with is_safe=True if no term has multiple leaders,
        or is_safe=False with violation descriptions listing the
        conflicting leaders and their term.
    """
    # TODO(human): Implement Election Safety verification.
    #
    # Steps:
    #   1. Create a dict: term_leaders: dict[int, set[int]] = {}
    #   2. For each snapshot in history:
    #        For each (node_id, state) in snapshot.node_states:
    #          If state == NodeState.LEADER:
    #            Add node_id to term_leaders[snapshot.node_terms[node_id]]
    #   3. For each term in term_leaders:
    #        If len(term_leaders[term]) > 1:
    #          Add violation: f"Term {term}: leaders {term_leaders[term]}"
    #   4. Return SafetyResult(
    #        property_name="Election Safety",
    #        is_safe=(len(violations) == 0),
    #        violations=violations,
    #      )
    #
    # Reference: Raft paper Section 5.2 -- "at most one leader can be
    # elected in a given term" -- guaranteed by majority vote + single
    # vote per term per node.
    raise NotImplementedError(
        "TODO(human): Implement verify_election_safety. "
        "Scan the history for terms with multiple distinct leader node IDs."
    )


def verify_log_matching(nodes: list[RaftNode]) -> SafetyResult:
    """Verify the Log Matching Property across all node pairs.

    The Log Matching Property (Raft paper Figure 3, Section 5.3) states:
    "If two logs contain an entry with the same index and term, then the
    logs are identical in all entries up through the given index."

    This is an inductive invariant maintained by the AppendEntries
    consistency check. If this property is violated, the consistency
    check has a bug -- followers accepted entries without verifying
    that their logs matched the leader's at the preceding position.

    Algorithm:
      1. For each pair of nodes (i, j) where i < j:
      2. Find the minimum log length between node_i and node_j.
      3. For each index from 1 to min_length:
           a. Get entry_i = node_i.log[index - 1]
           b. Get entry_j = node_j.log[index - 1]
           c. If entry_i.term == entry_j.term (same term at same index):
              This means by the Log Matching Property, all PRECEDING
              entries must also match. Verify this by checking that for
              every preceding index k (from 1 to index-1), the terms
              and commands of node_i.log[k-1] and node_j.log[k-1] match.
              If any preceding entry differs, that's a violation.
      4. Collect all violations with details: which nodes, which index
         triggered the check, and which preceding index failed.

    Note: It's fine for entries at the same index to have DIFFERENT terms
    -- that just means the nodes have divergent logs (which will be
    repaired by the leader). The property only applies when same
    index AND same term.

    Args:
        nodes: List of all RaftNode instances in the cluster.

    Returns:
        SafetyResult with is_safe=True if the property holds for all
        node pairs, or is_safe=False with the first violation found.
    """
    # TODO(human): Implement Log Matching Property verification.
    #
    # Steps:
    #   1. violations = []
    #   2. For i in range(len(nodes)):
    #        For j in range(i + 1, len(nodes)):
    #          log_i = nodes[i].log
    #          log_j = nodes[j].log
    #          min_len = min(len(log_i), len(log_j))
    #          For idx in range(1, min_len + 1):  # 1-based
    #            entry_i = log_i[idx - 1]
    #            entry_j = log_j[idx - 1]
    #            if entry_i.term == entry_j.term:
    #              # Same index, same term -> check all preceding entries
    #              for k in range(1, idx):
    #                pred_i = log_i[k - 1]
    #                pred_j = log_j[k - 1]
    #                if pred_i.term != pred_j.term or pred_i.command != pred_j.command:
    #                  violations.append(
    #                    f"Nodes {nodes[i].node_id},{nodes[j].node_id}: "
    #                    f"match at idx={idx} term={entry_i.term} "
    #                    f"but differ at idx={k}: "
    #                    f"({pred_i.term},{pred_i.command!r}) vs "
    #                    f"({pred_j.term},{pred_j.command!r})"
    #                  )
    #                  break  # One violation per pair is enough
    #   3. Return SafetyResult(...)
    #
    # Reference: Raft paper Section 5.3 -- maintained by the AppendEntries
    # consistency check (prevLogIndex / prevLogTerm matching).
    raise NotImplementedError(
        "TODO(human): Implement verify_log_matching. "
        "For each node pair, if they share an (index, term), "
        "verify all preceding entries are identical."
    )


def verify_leader_completeness(
    commit_history: list[tuple[int, LogEntry]],
    leader_history: list[tuple[int, int]],
    nodes: list[RaftNode],
) -> SafetyResult:
    """Verify Leader Completeness: committed entries exist in all future leaders.

    The Leader Completeness Property (Raft paper Section 5.4.3) states:
    "If a log entry is committed in a given term, that entry will be
    present in the logs of the leaders for all higher-numbered terms."

    This is the deepest safety property and connects the election
    restriction (candidates must have up-to-date logs) with the
    commitment rule (only commit entries from the current term).
    Together they guarantee that once an entry is committed, no future
    leader can exist without it.

    Algorithm:
      1. commit_history contains (term, entry) pairs: the term in which
         each entry was committed, and the entry itself (with index, term,
         command).
      2. leader_history contains (term, node_id) pairs: which node became
         leader in which term, sorted chronologically.
      3. For each committed entry (committed_in_term, entry):
           For each leader (leader_term, leader_node_id) in leader_history:
             If leader_term > committed_in_term:
               Check that the leader's log (from `nodes`) contains an entry
               at entry.index with the same term and command.
               If not found, that's a violation.

    Note: We check the leader's log at the time of verification (end of
    simulation). In a real system, you'd check the log at the moment the
    leader was elected. For our simulation, since we don't truncate
    committed entries, end-of-run logs are sufficient.

    Args:
        commit_history: List of (term_committed_in, LogEntry) -- entries
            that were committed and the term when commitment happened.
        leader_history: List of (term, node_id) -- each leader election
            event, chronologically ordered.
        nodes: All RaftNode instances (for reading final log state).

    Returns:
        SafetyResult with is_safe=True if every committed entry is present
        in every subsequent leader's log, or is_safe=False with details
        about which entry is missing from which leader.
    """
    # TODO(human): Implement Leader Completeness verification.
    #
    # Steps:
    #   1. Build a lookup: node_id -> node for quick access.
    #      nodes_by_id = {n.node_id: n for n in nodes}
    #   2. violations = []
    #   3. For each (committed_term, entry) in commit_history:
    #        For each (leader_term, leader_id) in leader_history:
    #          if leader_term > committed_term:
    #            leader_node = nodes_by_id[leader_id]
    #            # Check if entry.index exists in leader's log
    #            if entry.index > len(leader_node.log):
    #              violations.append(
    #                f"Entry (idx={entry.index}, term={entry.term}, "
    #                f"cmd={entry.command!r}) committed in term "
    #                f"{committed_term} missing from leader Node "
    #                f"{leader_id} of term {leader_term} "
    #                f"(log length={len(leader_node.log)})"
    #              )
    #            else:
    #              leader_entry = leader_node.log[entry.index - 1]
    #              if leader_entry.term != entry.term or \
    #                 leader_entry.command != entry.command:
    #                violations.append(
    #                  f"Entry at idx={entry.index} differs: "
    #                  f"committed ({entry.term},{entry.command!r}) vs "
    #                  f"leader {leader_id} term {leader_term} "
    #                  f"({leader_entry.term},{leader_entry.command!r})"
    #                )
    #   4. Return SafetyResult(
    #        property_name="Leader Completeness",
    #        is_safe=(len(violations) == 0),
    #        violations=violations,
    #      )
    #
    # Reference: Raft paper Section 5.4.3 -- proof by contradiction
    # using the election restriction and majority overlap.
    raise NotImplementedError(
        "TODO(human): Implement verify_leader_completeness. "
        "Check that every committed entry exists in every leader "
        "elected in subsequent terms."
    )


# ---------------------------------------------------------------------------
# Cluster simulation with history collection
# ---------------------------------------------------------------------------

async def run_safety_simulation() -> None:
    """Run a Raft cluster through multiple phases and verify safety.

    Phases:
      1. Normal operation: elect leader, commit commands
      2. Leader crash: kill leader, wait for new election
      3. More commands under new leader
      4. Network partition and heal
      5. Final commands

    After all phases, run all three safety verifiers on the collected
    history and print results.
    """
    print("=" * 60)
    print("  Exercise 1: Raft Safety Property Verification")
    print("=" * 60)

    config = RaftConfig(
        election_timeout_min_ms=300,
        election_timeout_max_ms=500,
        heartbeat_interval_ms=100,
        num_nodes=5,
    )
    cluster = RaftCluster(config)

    # History collectors
    snapshots: list[ClusterSnapshot] = []
    commit_history: list[tuple[int, LogEntry]] = []
    leader_history: list[tuple[int, int]] = []

    def capture() -> None:
        snapshots.append(ClusterSnapshot.capture(cluster))

    def record_leaders() -> None:
        for node in cluster.nodes:
            if node.state == NodeState.LEADER:
                event = (node.current_term, node.node_id)
                if not leader_history or leader_history[-1] != event:
                    leader_history.append(event)

    tasks = cluster.start_all()

    try:
        # -- Phase 1: Normal operation --
        print("\n[Phase 1] Normal operation -- waiting for leader...")
        leader = await cluster.wait_for_leader(timeout=5.0)
        if leader is None:
            print("ERROR: No leader elected!")
            return

        print(f"  Leader: Node {leader.node_id} (term {leader.current_term})")
        record_leaders()
        capture()

        # Submit commands
        commands_phase1 = ["SET a=1", "SET b=2", "SET c=3"]
        for cmd in commands_phase1:
            ok = await cluster.submit_and_wait(cmd, timeout=3.0)
            if ok:
                # Find the committed entry
                for entry in reversed(leader.log):
                    if entry.command == cmd:
                        commit_history.append((leader.current_term, entry))
                        break
            print(f"  {cmd} -> {'committed' if ok else 'FAILED'}")

        await asyncio.sleep(0.5)
        capture()
        record_leaders()

        # -- Phase 2: Simulate leader crash (stop the leader) --
        old_leader_id = leader.node_id
        print(f"\n[Phase 2] Crashing leader Node {old_leader_id}...")
        leader.stop()
        leader.state = NodeState.FOLLOWER
        leader._record_state()

        # Wait for new leader
        await asyncio.sleep(2.0)
        capture()

        new_leader = await cluster.wait_for_leader(timeout=5.0)
        if new_leader is None:
            print("  ERROR: No new leader elected!")
            return

        print(f"  New leader: Node {new_leader.node_id} (term {new_leader.current_term})")
        record_leaders()
        capture()

        # -- Phase 3: Commands under new leader --
        commands_phase3 = ["SET d=4", "SET e=5"]
        for cmd in commands_phase3:
            ok = await cluster.submit_and_wait(cmd, timeout=3.0)
            if ok:
                for entry in reversed(new_leader.log):
                    if entry.command == cmd:
                        commit_history.append((new_leader.current_term, entry))
                        break
            print(f"  {cmd} -> {'committed' if ok else 'FAILED'}")

        await asyncio.sleep(0.5)
        capture()

        # -- Phase 4: Network partition --
        print("\n[Phase 4] Partitioning [0,1,2] <-> [3,4]...")
        cluster.network.partition([0, 1, 2], [3, 4])
        await asyncio.sleep(1.5)
        capture()
        record_leaders()

        # Try to commit on majority side
        majority_leader = None
        for nid in [0, 1, 2]:
            if cluster.nodes[nid].state == NodeState.LEADER:
                majority_leader = cluster.nodes[nid]
                break

        if majority_leader:
            cmd = "SET f=6"
            entry_obj = majority_leader.submit_command(cmd)
            await asyncio.sleep(1.0)
            if entry_obj and majority_leader.commit_index >= entry_obj.index:
                print(f"  Majority: {cmd} -> committed")
                commit_history.append((majority_leader.current_term, entry_obj))
            else:
                print(f"  Majority: {cmd} -> pending")

        capture()

        # Heal partition
        print("\n[Phase 5] Healing partition...")
        cluster.network.heal()
        await asyncio.sleep(2.0)
        capture()
        record_leaders()

        # Final commands
        final_leader = await cluster.wait_for_leader(timeout=5.0)
        if final_leader:
            cmd = "SET g=7"
            ok = await cluster.submit_and_wait(cmd, timeout=3.0)
            if ok:
                for entry in reversed(final_leader.log):
                    if entry.command == cmd:
                        commit_history.append((final_leader.current_term, entry))
                        break
            print(f"  {cmd} -> {'committed' if ok else 'FAILED'}")

        await asyncio.sleep(0.5)
        capture()

    finally:
        cluster.stop_all()
        for t in tasks:
            t.cancel()
        await asyncio.sleep(0.1)

    # -- Run safety verifiers --
    print("\n" + "=" * 60)
    print("  Safety Verification Results")
    print("=" * 60)

    result1 = verify_election_safety(snapshots)
    print(f"\n{result1}")

    result2 = verify_log_matching(cluster.nodes)
    print(f"\n{result2}")

    result3 = verify_leader_completeness(
        commit_history, leader_history, cluster.nodes
    )
    print(f"\n{result3}")

    all_pass = result1.is_safe and result2.is_safe and result3.is_safe
    print(f"\n{'=' * 60}")
    print(f"  Overall: {'ALL PROPERTIES HOLD' if all_pass else 'VIOLATIONS DETECTED'}")
    print(f"{'=' * 60}")

    # Print final cluster state for reference
    cluster.print_cluster_state("Final Cluster State")


if __name__ == "__main__":
    asyncio.run(run_safety_simulation())
