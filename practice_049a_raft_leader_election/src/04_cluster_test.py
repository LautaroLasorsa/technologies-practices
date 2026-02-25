"""Exercise 4: Full Cluster Integration Test -- run Raft with failure injection.

This module brings together all previous exercises into a complete Raft cluster
simulation. It tests three scenarios:

  Scenario 1: Normal operation -- elect leader, replicate commands, verify consistency.
  Scenario 2: Leader crash -- stop the leader, verify re-election and continued operation.
  Scenario 3: Network partition -- split the cluster, verify safety (minority can't commit),
              heal the partition, verify all nodes converge to the same log.

The cluster runs as asyncio tasks in a single process. Communication between
nodes is via direct function calls (simulating RPCs). Failures are simulated
by marking nodes as "crashed" (skip their RPCs) or links as "partitioned"
(silently drop messages between partitioned groups).

Reference: Raft paper Section 5 (complete algorithm), Section 8 (client interaction).
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

# Import replication module (which also imports election module, attaching all methods)
import importlib.util as _ilu

_repl_spec = _ilu.spec_from_file_location(
    "log_replication",
    Path(__file__).resolve().parent / "03_log_replication.py",
)
_repl_mod = _ilu.module_from_spec(_repl_spec)
_repl_spec.loader.exec_module(_repl_mod)

run_election = _repl_mod.run_election
simulate_election_with_timeout = _repl_mod.simulate_election_with_timeout
print_cluster_state = _repl_mod.print_cluster_state

logger = logging.getLogger("raft.cluster")


# ======================================================================
# Cluster simulation helper
# ======================================================================

def submit_command(leader: RaftNode, command: str) -> LogEntry:
    """Submit a client command to the leader. Returns the new log entry.

    In a real system, the client sends a request to the leader, the leader
    appends it to its log, and returns after the entry is committed.
    Here we just append to the leader's log directly.
    """
    entry = LogEntry(
        term=leader.current_term,
        index=leader.last_log_index + 1,
        command=command,
    )
    leader.log.append(entry)
    return entry


def replicate_all(leader: RaftNode, nodes: list[RaftNode], crashed: set[int] | None = None) -> bool:
    """Leader replicates to all non-crashed followers. Returns True if leader is still valid."""
    if crashed is None:
        crashed = set()

    for peer_id in leader.peers:
        if peer_id in crashed:
            continue
        success = leader.replicate_to_follower(nodes[peer_id])
        if not success:
            return False  # leader stepped down
    return True


def verify_logs_consistent(nodes: list[RaftNode], up_to_index: int, skip: set[int] | None = None) -> bool:
    """Verify that all non-skipped nodes have identical logs up to the given index."""
    if skip is None:
        skip = set()

    reference = None
    reference_id = -1
    for node in nodes:
        if node.node_id in skip:
            continue
        if reference is None:
            reference = node.log[:up_to_index]
            reference_id = node.node_id
            continue

        node_log = node.log[:up_to_index]
        if len(node_log) != len(reference):
            logger.error(
                f"Log length mismatch: Node {reference_id} has {len(reference)} entries, "
                f"Node {node.node_id} has {len(node_log)}"
            )
            return False
        for i, (ref_entry, node_entry) in enumerate(zip(reference, node_log)):
            if ref_entry.command != node_entry.command or ref_entry.term != node_entry.term:
                logger.error(
                    f"Log mismatch at index {i+1}: "
                    f"Node {reference_id}={ref_entry}, Node {node.node_id}={node_entry}"
                )
                return False
    return True


# ======================================================================
# Cluster test -- the main exercise
# ======================================================================

# TODO(human): Implement run_cluster(num_nodes, num_commands, partition_at=None)
#
# Run a complete Raft cluster simulation: create nodes, elect a leader,
# submit client commands, replicate to followers, and optionally inject
# a network partition to test safety and recovery.
#
# Parameters:
#   num_nodes (int): Number of nodes in the cluster (e.g., 5).
#   num_commands (int): Number of client commands to submit (e.g., 10).
#   partition_at (int | None): If set, simulate a network partition after
#     this many commands. The partition splits the cluster into a majority
#     group (containing the leader) and a minority group. After remaining
#     commands are processed by the majority, the partition is healed and
#     the minority catches up.
#
# Returns:
#   dict with keys:
#     "leader_id": int -- the final leader's node_id
#     "commit_index": int -- the leader's final commit_index
#     "logs": dict[int, list[str]] -- each node's log as a list of command strings
#     "all_consistent": bool -- whether all nodes have identical committed logs
#
# Algorithm:
#
#   Phase 1: SETUP
#     1. Create a cluster with create_cluster(RaftConfig(num_nodes=num_nodes)).
#     2. Elect a leader using simulate_election_with_timeout().
#     3. Print the leader and initial cluster state.
#
#   Phase 2: COMMAND SUBMISSION & REPLICATION
#     4. For i in range(num_commands):
#          a. Submit a command: submit_command(leader, f"CMD_{i}")
#          b. Replicate to all followers: replicate_all(leader, nodes)
#          c. Advance commit index: leader.advance_commit_index()
#          d. If partition_at is not None and i == partition_at - 1:
#               Simulate a network partition (see Phase 3 below).
#
#     After processing commands (and after partition heals, if applicable):
#     5. Propagate the leader's commit_index to followers via one final
#        round of AppendEntries (heartbeats carry leader_commit).
#        For each non-crashed follower, call leader.replicate_to_follower().
#
#   Phase 3: NETWORK PARTITION (only if partition_at is not None)
#     When the partition triggers:
#       a. Divide nodes into two groups:
#          - majority: the leader + enough followers to form a majority
#            (e.g., for 5 nodes: leader + 2 followers = 3 nodes)
#          - minority: the remaining followers
#          Example for 5 nodes, leader=0:
#            majority = {0, 1, 2}, minority = {3, 4}
#
#       b. The minority nodes are "unreachable" -- skip them in replicate_all
#          by passing their IDs in the `crashed` set.
#          Print: f"PARTITION: majority={majority}, minority={minority}"
#
#       c. Continue submitting remaining commands to the leader, replicating
#          ONLY to the majority group.
#
#       d. After all commands are submitted, HEAL the partition:
#          Print: "PARTITION HEALED"
#          Replicate from the leader to the minority nodes. The leader's
#          log repair mechanism (replicate_to_follower's retry loop) will
#          bring the minority nodes up-to-date.
#
#       e. Do a final advance_commit_index() to update commitment.
#
#   Phase 4: VERIFICATION
#     6. Verify log consistency: all nodes should have identical logs up to
#        the leader's commit_index.
#        Use verify_logs_consistent(nodes, leader.commit_index).
#
#     7. Build and return the result dict.
#
# Why this matters:
#   This integration test validates the COMPLETE Raft lifecycle. Individual
#   components may work in isolation but fail when composed. Specifically:
#
#   - Normal operation tests the happy path: election -> replication -> commitment.
#     If this fails, there's a bug in the basic protocol.
#
#   - Network partition tests the core safety property: the minority partition
#     CANNOT make progress (no leader without majority), while the majority
#     partition continues normally. After the partition heals, the minority
#     nodes catch up via log repair. If the minority had somehow committed
#     different entries, the logs would diverge -- which would be a safety
#     violation.
#
#   - The test also implicitly validates: term monotonicity (step_down),
#     election restriction (leader has all committed entries), log matching
#     (consistency check prevents divergence), and commitment (majority rule
#     + current-term check).
#
# Hint: For the partition, you don't need actual async networking. Just
# pass the set of minority node IDs as the `crashed` parameter to
# replicate_all(). This causes the leader to skip those nodes during
# replication, simulating network unreachability.

async def run_cluster(
    num_nodes: int = 5,
    num_commands: int = 10,
    partition_at: int | None = None,
) -> dict:
    raise NotImplementedError(
        "TODO(human): Implement full cluster simulation with optional partition."
    )


# ======================================================================
# Main: run all scenarios
# ======================================================================

async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 70)
    print("Exercise 4: Full Cluster Integration Test")
    print("=" * 70)

    results = {}

    # ------------------------------------------------------------------
    # Scenario 1: Normal operation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCENARIO 1: Normal Operation (5 nodes, 10 commands)")
    print("=" * 70)

    result = await run_cluster(num_nodes=5, num_commands=10)
    results["normal"] = result
    print(f"\n  Leader: Node {result['leader_id']}")
    print(f"  Commit index: {result['commit_index']}")
    print(f"  All logs consistent: {result['all_consistent']}")
    if result["all_consistent"]:
        print("  >> PASSED: Normal operation works correctly")
    else:
        print("  >> FAILED: Logs are inconsistent!")

    # ------------------------------------------------------------------
    # Scenario 2: Leader crash and re-election
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCENARIO 2: Leader Crash & Re-election")
    print("=" * 70)

    config = RaftConfig(num_nodes=5)
    nodes = create_cluster(config)

    # Elect initial leader
    print("\n  Electing initial leader...")
    leader_id = await simulate_election_with_timeout(nodes, config)
    leader = nodes[leader_id]
    print(f"  Initial leader: Node {leader_id}")

    # Submit some commands
    for i in range(5):
        submit_command(leader, f"BEFORE_CRASH_{i}")
    replicate_all(leader, nodes)
    leader.advance_commit_index()
    print(f"  Submitted 5 commands, commit_index={leader.commit_index}")

    # "Crash" the leader
    crashed_leader_id = leader_id
    print(f"\n  CRASH: Node {crashed_leader_id} is down!")
    nodes[crashed_leader_id].state = NodeState.FOLLOWER  # simulate crash

    # Remaining nodes elect a new leader
    # Reset terms so election can proceed
    surviving_nodes = [n for n in nodes if n.node_id != crashed_leader_id]
    surviving_ids = [n.node_id for n in surviving_nodes]

    # Find the surviving node with the most up-to-date log
    best_candidate = max(surviving_ids, key=lambda nid: (nodes[nid].last_log_term, nodes[nid].last_log_index))
    print(f"  Best candidate among survivors: Node {best_candidate}")

    # Run election for the best candidate
    new_leader_id = await run_election(nodes, best_candidate)
    if new_leader_id is not None:
        new_leader = nodes[new_leader_id]
        print(f"  New leader elected: Node {new_leader_id}")

        # Submit more commands to the new leader
        for i in range(3):
            submit_command(new_leader, f"AFTER_CRASH_{i}")
        replicate_all(new_leader, nodes, crashed={crashed_leader_id})
        new_leader.advance_commit_index()
        print(f"  Submitted 3 more commands, commit_index={new_leader.commit_index}")

        # Verify consistency among survivors
        consistent = verify_logs_consistent(nodes, new_leader.commit_index, skip={crashed_leader_id})
        print(f"  Surviving nodes consistent: {consistent}")
        if consistent:
            print("  >> PASSED: Cluster recovered from leader crash")
        else:
            print("  >> FAILED: Logs diverged after leader crash!")
        results["leader_crash"] = {"new_leader": new_leader_id, "consistent": consistent}
    else:
        print("  >> FAILED: Could not elect new leader")
        results["leader_crash"] = {"new_leader": None, "consistent": False}

    # ------------------------------------------------------------------
    # Scenario 3: Network partition
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCENARIO 3: Network Partition (5 nodes, partition after 5 of 10 commands)")
    print("=" * 70)

    result = await run_cluster(num_nodes=5, num_commands=10, partition_at=5)
    results["partition"] = result
    print(f"\n  Leader: Node {result['leader_id']}")
    print(f"  Commit index: {result['commit_index']}")
    print(f"  All logs consistent: {result['all_consistent']}")
    if result["all_consistent"]:
        print("  >> PASSED: Cluster consistent after partition heal")
    else:
        print("  >> FAILED: Logs diverged after partition!")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, result in results.items():
        if isinstance(result, dict) and "all_consistent" in result:
            passed = result["all_consistent"]
        elif isinstance(result, dict) and "consistent" in result:
            passed = result["consistent"]
        else:
            passed = False
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  ALL SCENARIOS PASSED -- Raft implementation is correct!")
    else:
        print("  SOME SCENARIOS FAILED -- review the implementation")
    print("=" * 70)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
