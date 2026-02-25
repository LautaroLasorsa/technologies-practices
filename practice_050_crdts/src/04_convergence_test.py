#!/usr/bin/env python3
"""
Exercise 4: Convergence testing across all CRDT types.

This exercise ties together the theory (join-semilattice axioms) with practice
(convergence under adversarial network conditions). You'll:

  1. Simulate multi-replica environments with network partitions
  2. Verify semilattice properties (commutativity, associativity, idempotency)
     through property-based testing with random CRDT states
  3. Test convergence across ALL CRDT types from exercises 1-3
"""
from __future__ import annotations

import random
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

# Ensure sibling modules are importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crdt_base import StateCRDT, ReplicaNetwork, all_converged  # noqa: E402
from counters import GCounter, PNCounter  # noqa: E402
from registers_sets import LWWRegister, ORSet  # noqa: E402


# ---------------------------------------------------------------------------
# Random CRDT state generators
# ---------------------------------------------------------------------------

# These generators create CRDT replicas with random state. They're used by
# the property-based tests to verify semilattice axioms hold for arbitrary
# states, not just the specific states in hand-crafted unit tests.

def random_gcounter(num_replicas: int = 3, max_count: int = 20) -> GCounter:
    """Generate a G-Counter with random counts across replicas."""
    counts = {}
    for i in range(num_replicas):
        rid = f"R{i}"
        counts[rid] = random.randint(0, max_count)
    return GCounter(counts)


def random_pncounter(num_replicas: int = 3, max_count: int = 20) -> PNCounter:
    """Generate a PN-Counter with random P and N counts."""
    p = random_gcounter(num_replicas, max_count)
    n = random_gcounter(num_replicas, max_count)
    return PNCounter(p, n)


def random_lww_register(max_timestamp: float = 100.0) -> LWWRegister:
    """Generate a LWW-Register with a random value and timestamp."""
    values = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    replicas = ["A", "B", "C", "D"]
    return LWWRegister(
        value=random.choice(values),
        timestamp=random.uniform(0.0, max_timestamp),
        replica_id=random.choice(replicas),
    )


def random_orset(max_elements: int = 5, max_tags_per: int = 3) -> ORSet:
    """Generate an OR-Set with random elements and tags."""
    possible_elements = ["apple", "banana", "cherry", "date", "elderberry",
                         "fig", "grape"]
    elements: dict[Any, set[str]] = {}
    num_elements = random.randint(0, max_elements)
    for elem in random.sample(possible_elements, min(num_elements, len(possible_elements))):
        num_tags = random.randint(1, max_tags_per)
        tags = {str(uuid.uuid4()) for _ in range(num_tags)}
        elements[elem] = tags
    return ORSet(elements)


# ---------------------------------------------------------------------------
# Semilattice property verification
# ---------------------------------------------------------------------------

# TODO(human): Implement verify_semilattice_properties().
#
# This function tests the three semilattice axioms that ALL state-based CRDTs
# must satisfy. If ANY axiom fails for ANY randomly generated state, the CRDT
# implementation is INCORRECT and will not guarantee convergence.
#
# The three axioms:
#
#   1. COMMUTATIVITY: merge(A, B) must produce the same result as merge(B, A).
#      Meaning: the order in which replicas sync doesn't matter.
#      Test: Create copies of A and B. Merge B into A-copy. Merge A into B-copy.
#      Verify A-copy.value() == B-copy.value().
#
#   2. ASSOCIATIVITY: merge(merge(A, B), C) == merge(A, merge(B, C)).
#      Meaning: grouping of multi-way merges doesn't matter.
#      Test: Create copies. Path 1: merge B into A, then merge C into A.
#      Path 2: merge C into B, then merge B into A. Verify same value.
#
#   3. IDEMPOTENCY: merge(A, A) produces no change.
#      Meaning: re-merging the same state is a no-op.
#      Test: Record A's value. Merge a copy of A into A. Verify value unchanged.
#
# For each CRDT type, generate N random states and verify all three axioms.
# Report the total number of checks and any failures.
#
# IMPORTANT: Always work with COPIES when testing. The merge operation is
# destructive (mutates self), so you need fresh copies for each test path.
#
# This is a form of PROPERTY-BASED TESTING (like Hypothesis/QuickCheck):
# instead of testing specific examples, you test that mathematical properties
# hold for randomly generated inputs. This catches edge cases that hand-written
# tests miss.


def verify_semilattice_properties(
    crdt_name: str,
    generator: Callable[[], StateCRDT],
    num_trials: int = 100,
) -> tuple[int, int]:
    """Verify semilattice axioms for a CRDT type using random states.

    Args:
        crdt_name:   Name of the CRDT (for reporting)
        generator:   Callable that returns a random CRDT instance
        num_trials:  Number of random trials per axiom

    Returns:
        (total_checks, failures): count of checks performed and failures found
    """
    # TODO(human): Implement the property-based verification.
    #
    # For each trial (num_trials iterations):
    #
    #   Generate three random CRDT states: a, b, c using generator()
    #
    #   --- Test 1: Commutativity ---
    #   a1 = a.copy(); a1.merge(b.copy())       -> result1 = a1.value()
    #   b1 = b.copy(); b1.merge(a.copy())       -> result2 = b1.value()
    #   Assert result1 == result2
    #   If not, increment failures, print the violation
    #
    #   --- Test 2: Associativity ---
    #   Path 1: ab = a.copy(); ab.merge(b.copy()); ab.merge(c.copy()) -> v1
    #   Path 2: bc = b.copy(); bc.merge(c.copy()); a2 = a.copy(); a2.merge(bc) -> v2
    #   Assert v1 == v2
    #   If not, increment failures, print the violation
    #
    #   --- Test 3: Idempotency ---
    #   a3 = a.copy()
    #   val_before = a3.value()
    #   a3.merge(a3.copy())
    #   val_after = a3.value()
    #   Assert val_before == val_after
    #   If not, increment failures, print the violation
    #
    # Print a summary: "{crdt_name}: {total_checks} checks, {failures} failures"
    # Return (total_checks, failures)
    #
    # total_checks = num_trials * 3 (three axioms per trial)
    raise NotImplementedError("TODO(human): Implement verify_semilattice_properties")


# ---------------------------------------------------------------------------
# Convergence simulation with partitions
# ---------------------------------------------------------------------------

# TODO(human): Implement simulate_concurrent_updates().
#
# This function creates a realistic distributed simulation:
#   1. N replicas start with the same initial state
#   2. A network partition splits them into groups
#   3. Each group performs local updates independently (concurrent with the other group)
#   4. The partition heals and replicas sync
#   5. Verify ALL replicas converge to the same state
#
# This tests the FUNDAMENTAL GUARANTEE of CRDTs: replicas that have received
# the same set of updates (in any order) will have the same state.
#
# The simulation should test both:
#   - WITHIN-PARTITION convergence: replicas in the same group agree during partition
#   - POST-HEALING convergence: ALL replicas agree after the partition heals
#
# Use the ReplicaNetwork class from 00_crdt_base.py for the simulation.


def simulate_concurrent_updates(
    crdt_name: str,
    replica_factory: Callable[[str], StateCRDT],
    updater: Callable[[StateCRDT, str], None],
    replica_ids: list[str],
    partition_groups: list[set[str]],
    updates_per_replica: int = 5,
) -> bool:
    """Simulate concurrent updates with a network partition, then verify convergence.

    Args:
        crdt_name:          Name of the CRDT being tested (for reporting)
        replica_factory:    Callable(replica_id) -> new CRDT instance
        updater:            Callable(crdt, replica_id) -> performs a random update
        replica_ids:        List of replica IDs to create
        partition_groups:   List of sets defining the partition (e.g., [{"A","B"}, {"C","D"}])
        updates_per_replica: Number of random updates each replica performs

    Returns:
        True if convergence was achieved after healing, False otherwise
    """
    # TODO(human): Implement the partition simulation.
    #
    # Steps:
    #
    #   1. CREATE REPLICAS
    #      replicas = {rid: replica_factory(rid) for rid in replica_ids}
    #      net = ReplicaNetwork(replicas=replicas, min_delay=0, max_delay=2)
    #
    #   2. INITIAL SYNC (all replicas start from the same base state)
    #      net.full_sync()
    #      Print initial values.
    #
    #   3. SET PARTITION
    #      net.set_partition(partition_groups)
    #      Print which groups are partitioned.
    #
    #   4. CONCURRENT UPDATES DURING PARTITION
    #      For each replica, call updater(replicas[rid], rid) updates_per_replica times.
    #      Then do net.full_sync() -- only replicas in the same group can communicate.
    #      Print values during partition.
    #
    #   5. VERIFY WITHIN-PARTITION CONVERGENCE
    #      For each partition group, check that all replicas in the group have the
    #      same value (they could sync within the group).
    #      Print whether each group converged internally.
    #
    #   6. HEAL PARTITION
    #      net.heal_partition()
    #      net.full_sync()
    #      Print post-healing values.
    #
    #   7. VERIFY FULL CONVERGENCE
    #      Check all_converged(net.replicas).
    #      Print the final converged value.
    #      Return True if converged, False otherwise.
    #
    # This tests the end-to-end CRDT guarantee: even though replicas received
    # different updates in different orders, they all reach the same final state.
    raise NotImplementedError("TODO(human): Implement simulate_concurrent_updates")


# ---------------------------------------------------------------------------
# Update functions for each CRDT type (used by simulate_concurrent_updates)
# ---------------------------------------------------------------------------

def update_gcounter(crdt: StateCRDT, replica_id: str) -> None:
    """Perform a random G-Counter update."""
    assert isinstance(crdt, GCounter)
    crdt.increment(replica_id, random.randint(1, 10))


def update_pncounter(crdt: StateCRDT, replica_id: str) -> None:
    """Perform a random PN-Counter update."""
    assert isinstance(crdt, PNCounter)
    if random.random() < 0.6:
        crdt.increment(replica_id, random.randint(1, 10))
    else:
        crdt.decrement(replica_id, random.randint(1, 5))


def update_lww_register(crdt: StateCRDT, replica_id: str) -> None:
    """Perform a random LWW-Register update."""
    assert isinstance(crdt, LWWRegister)
    values = ["alice", "bob", "carol", "dave", "eve"]
    # Use a monotonically increasing timestamp (current "time" + random jitter)
    # In real systems this would be a Lamport clock or HLC
    ts = random.uniform(0.0, 100.0)
    crdt.assign(random.choice(values), timestamp=ts, replica_id=replica_id)


def update_orset(crdt: StateCRDT, replica_id: str) -> None:
    """Perform a random OR-Set update (add or remove)."""
    assert isinstance(crdt, ORSet)
    elements = ["apple", "banana", "cherry", "date", "elderberry"]
    elem = random.choice(elements)
    if random.random() < 0.7:
        crdt.add(elem)
    else:
        crdt.remove(elem)


# ---------------------------------------------------------------------------
# Tests (scaffolded)
# ---------------------------------------------------------------------------

def test_semilattice_gcounter() -> None:
    """Verify G-Counter satisfies semilattice axioms."""
    print("\n[Test] G-Counter semilattice properties")
    print("-" * 40)
    checks, failures = verify_semilattice_properties("GCounter", random_gcounter, num_trials=200)
    assert failures == 0, f"GCounter: {failures} semilattice violations!"
    print(f"  {checks} checks, {failures} failures")
    print("  PASSED")


def test_semilattice_pncounter() -> None:
    """Verify PN-Counter satisfies semilattice axioms."""
    print("\n[Test] PN-Counter semilattice properties")
    print("-" * 40)
    checks, failures = verify_semilattice_properties("PNCounter", random_pncounter, num_trials=200)
    assert failures == 0, f"PNCounter: {failures} semilattice violations!"
    print(f"  {checks} checks, {failures} failures")
    print("  PASSED")


def test_semilattice_lww_register() -> None:
    """Verify LWW-Register satisfies semilattice axioms."""
    print("\n[Test] LWW-Register semilattice properties")
    print("-" * 40)
    checks, failures = verify_semilattice_properties(
        "LWWRegister", random_lww_register, num_trials=200,
    )
    assert failures == 0, f"LWWRegister: {failures} semilattice violations!"
    print(f"  {checks} checks, {failures} failures")
    print("  PASSED")


def test_semilattice_orset() -> None:
    """Verify OR-Set satisfies semilattice axioms."""
    print("\n[Test] OR-Set semilattice properties")
    print("-" * 40)
    checks, failures = verify_semilattice_properties("ORSet", random_orset, num_trials=200)
    assert failures == 0, f"ORSet: {failures} semilattice violations!"
    print(f"  {checks} checks, {failures} failures")
    print("  PASSED")


def test_convergence_gcounter() -> None:
    """Test G-Counter convergence with partition."""
    print("\n[Test] G-Counter convergence with partition")
    print("-" * 40)
    ok = simulate_concurrent_updates(
        crdt_name="GCounter",
        replica_factory=lambda rid: GCounter(),
        updater=update_gcounter,
        replica_ids=["A", "B", "C", "D"],
        partition_groups=[{"A", "B"}, {"C", "D"}],
        updates_per_replica=10,
    )
    assert ok, "G-Counter should converge after partition healing"
    print("  PASSED")


def test_convergence_pncounter() -> None:
    """Test PN-Counter convergence with partition."""
    print("\n[Test] PN-Counter convergence with partition")
    print("-" * 40)
    ok = simulate_concurrent_updates(
        crdt_name="PNCounter",
        replica_factory=lambda rid: PNCounter(),
        updater=update_pncounter,
        replica_ids=["A", "B", "C", "D"],
        partition_groups=[{"A", "B"}, {"C", "D"}],
        updates_per_replica=10,
    )
    assert ok, "PN-Counter should converge after partition healing"
    print("  PASSED")


def test_convergence_lww_register() -> None:
    """Test LWW-Register convergence with partition."""
    print("\n[Test] LWW-Register convergence with partition")
    print("-" * 40)
    ok = simulate_concurrent_updates(
        crdt_name="LWWRegister",
        replica_factory=lambda rid: LWWRegister(),
        updater=update_lww_register,
        replica_ids=["A", "B", "C", "D", "E"],
        partition_groups=[{"A", "B"}, {"C", "D", "E"}],
        updates_per_replica=8,
    )
    assert ok, "LWW-Register should converge after partition healing"
    print("  PASSED")


def test_convergence_orset() -> None:
    """Test OR-Set convergence with partition."""
    print("\n[Test] OR-Set convergence with partition")
    print("-" * 40)
    ok = simulate_concurrent_updates(
        crdt_name="ORSet",
        replica_factory=lambda rid: ORSet(),
        updater=update_orset,
        replica_ids=["A", "B", "C", "D"],
        partition_groups=[{"A", "B"}, {"C", "D"}],
        updates_per_replica=10,
    )
    assert ok, "OR-Set should converge after partition healing"
    print("  PASSED")


def test_convergence_three_way_partition() -> None:
    """Test convergence with a 3-way partition (more adversarial)."""
    print("\n[Test] 3-way partition convergence (PN-Counter)")
    print("-" * 40)
    ok = simulate_concurrent_updates(
        crdt_name="PNCounter (3-way partition)",
        replica_factory=lambda rid: PNCounter(),
        updater=update_pncounter,
        replica_ids=["A", "B", "C", "D", "E", "F"],
        partition_groups=[{"A", "B"}, {"C", "D"}, {"E", "F"}],
        updates_per_replica=15,
    )
    assert ok, "Should converge even with 3-way partition"
    print("  PASSED")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: dict[str, bool]) -> None:
    """Print a summary table of all convergence tests."""
    print("\n" + "=" * 60)
    print("CONVERGENCE TEST SUMMARY")
    print("=" * 60)
    print(f"  {'Test':<45} {'Result':<10}")
    print(f"  {'-' * 45} {'-' * 10}")
    all_ok = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<45} {status}")
        if not passed:
            all_ok = False
    print(f"\n  Overall: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Exercise 4: Convergence Testing")
    print("=" * 60)

    random.seed(2024)  # reproducible

    # Part 1: Semilattice property verification
    print("\n>>> Part 1: Semilattice Property Verification")
    test_semilattice_gcounter()
    test_semilattice_pncounter()
    test_semilattice_lww_register()
    test_semilattice_orset()

    # Part 2: Convergence with partitions
    print("\n>>> Part 2: Convergence with Network Partitions")
    results: dict[str, bool] = {}

    test_convergence_gcounter()
    results["GCounter (2-way partition)"] = True

    test_convergence_pncounter()
    results["PNCounter (2-way partition)"] = True

    test_convergence_lww_register()
    results["LWWRegister (2-way partition)"] = True

    test_convergence_orset()
    results["ORSet (2-way partition)"] = True

    test_convergence_three_way_partition()
    results["PNCounter (3-way partition)"] = True

    print_summary(results)

    print("\n" + "=" * 60)
    print("All Exercise 4 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
