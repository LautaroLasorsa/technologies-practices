#!/usr/bin/env python3
"""
Exercise 1: G-Counter and PN-Counter (state-based CRDTs).

G-Counter = grow-only counter, the simplest CRDT.
PN-Counter = positive-negative counter, composes two G-Counters for increment + decrement.

Both are state-based: replicas exchange full state and merge via join-semilattice.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure sibling modules are importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crdt_base import StateCRDT, ReplicaNetwork, all_converged  # noqa: E402


# ---------------------------------------------------------------------------
# G-Counter
# ---------------------------------------------------------------------------

# TODO(human): Implement the G-Counter CRDT.
#
# The G-Counter (Grow-only Counter) is the simplest CRDT. It demonstrates the
# core insight of state-based CRDTs: each replica owns a piece of the state that
# only it can modify, and merge takes the maximum of each piece.
#
# Internal state:
#   self._counts: dict[str, int]  -- maps replica_id -> that replica's local count
#
# Semilattice structure:
#   - Partial order: A <= B iff A[i] <= B[i] for all replica_id i
#   - Join (merge):  result[i] = max(A[i], B[i]) for all i
#   - Query (value): sum of all entries in the dict
#
# CRITICAL INSIGHT -- Why merge uses max, not sum:
#   State-based merge must be IDEMPOTENT: merge(A, A) must equal A.
#   If merge used sum, merging a state with itself would double all counts.
#   max(x, x) == x, so max is idempotent. It's also commutative (max(a,b)
#   == max(b,a)) and associative (max(max(a,b),c) == max(a,max(b,c))),
#   forming a valid join-semilattice.
#
# Reference: Shapiro et al. 2011, Section 3.1 "G-Counter"


class GCounter(StateCRDT[int]):
    """Grow-only counter (state-based CRDT)."""

    def __init__(self, counts: dict[str, int] | None = None) -> None:
        # TODO(human): Store a COPY of counts (or empty dict if None).
        # Copying prevents aliasing bugs where two replicas share the same dict.
        raise NotImplementedError("TODO(human): Implement GCounter.__init__")

    def increment(self, replica_id: str, amount: int = 1) -> None:
        # TODO(human): Add `amount` to self._counts[replica_id].
        # If replica_id not in dict, initialize to 0 first.
        # amount must be >= 0 (grow-only!). Raise ValueError if negative.
        # This enforces the monotonicity requirement: state can only grow.
        raise NotImplementedError("TODO(human): Implement GCounter.increment")

    def value(self) -> int:
        # TODO(human): Return sum(self._counts.values()).
        # This is the "query" operation -- the externally visible counter value.
        # The sum aggregates all replicas' contributions into a single int.
        raise NotImplementedError("TODO(human): Implement GCounter.value")

    def merge(self, other: StateCRDT[int]) -> None:
        # TODO(human): Element-wise max merge.
        # For each replica_id in other._counts:
        #   self._counts[replica_id] = max(self._counts.get(replica_id, 0),
        #                                  other._counts[replica_id])
        # Also include any replica_ids only in self (they stay as-is via max with 0).
        # This is the join operation of the semilattice.
        raise NotImplementedError("TODO(human): Implement GCounter.merge")

    def copy(self) -> GCounter:
        # TODO(human): Return GCounter(dict(self._counts)) -- deep copy.
        # This simulates serializing state to send to another replica.
        raise NotImplementedError("TODO(human): Implement GCounter.copy")

    def __repr__(self) -> str:
        # TODO(human): Return f"GCounter({self._counts})"
        raise NotImplementedError("TODO(human): Implement GCounter.__repr__")


# ---------------------------------------------------------------------------
# PN-Counter
# ---------------------------------------------------------------------------

# TODO(human): Implement the PN-Counter CRDT.
#
# The PN-Counter supports both increment and decrement by composing two
# G-Counters: P (positive/increments) and N (negative/decrements).
#
# The value is P.value() - N.value(). The key insight is that we never
# actually subtract from a G-Counter (it's grow-only). Instead, we INCREMENT
# the N counter when we want to decrement the overall value. This preserves
# the monotonicity property of each internal G-Counter.
#
# This demonstrates CRDT composition: composing two valid CRDTs yields
# a valid CRDT, because merge distributes over the composition.
#
# The value CAN go negative -- "Positive-Negative" refers to the two internal
# counters (P and N), not the range of the output value.
#
# Reference: Shapiro et al. 2011, Section 3.2 "PN-Counter"


class PNCounter(StateCRDT[int]):
    """Positive-Negative counter (state-based CRDT)."""

    def __init__(self, p: GCounter | None = None, n: GCounter | None = None) -> None:
        # TODO(human): Initialize self._p and self._n.
        # Use provided GCounters or create fresh empty ones.
        raise NotImplementedError("TODO(human): Implement PNCounter.__init__")

    def increment(self, replica_id: str, amount: int = 1) -> None:
        # TODO(human): Delegate to self._p.increment(replica_id, amount).
        # Incrementing the positive counter increases the overall value.
        raise NotImplementedError("TODO(human): Implement PNCounter.increment")

    def decrement(self, replica_id: str, amount: int = 1) -> None:
        # TODO(human): Delegate to self._n.increment(replica_id, amount).
        # Note: we INCREMENT the N counter (it's a G-Counter, grow-only).
        # This decreases the overall value because value = P - N.
        raise NotImplementedError("TODO(human): Implement PNCounter.decrement")

    def value(self) -> int:
        # TODO(human): Return self._p.value() - self._n.value().
        # Can be negative! That's fine -- if decrements exceed increments.
        raise NotImplementedError("TODO(human): Implement PNCounter.value")

    def merge(self, other: StateCRDT[int]) -> None:
        # TODO(human): Merge both P and N independently.
        # self._p.merge(other._p)
        # self._n.merge(other._n)
        # This works because merge distributes over composition --
        # each G-Counter merges independently and the combined result is correct.
        raise NotImplementedError("TODO(human): Implement PNCounter.merge")

    def copy(self) -> PNCounter:
        # TODO(human): Return PNCounter(self._p.copy(), self._n.copy()).
        raise NotImplementedError("TODO(human): Implement PNCounter.copy")

    def __repr__(self) -> str:
        # TODO(human): Return f"PNCounter(value={self.value()}, P={self._p}, N={self._n})"
        raise NotImplementedError("TODO(human): Implement PNCounter.__repr__")


# ---------------------------------------------------------------------------
# Tests (scaffolded)
# ---------------------------------------------------------------------------

def test_gcounter_basic() -> None:
    """Test G-Counter with 3 replicas: independent increments then merge."""
    print("\n[Test] G-Counter basic operations")
    print("-" * 40)

    # Create 3 independent replicas
    a = GCounter()
    b = GCounter()
    c = GCounter()

    # Each replica increments its own entry
    a.increment("A", 5)
    b.increment("B", 3)
    c.increment("C", 7)

    print(f"  A after local ops: {a} -> value={a.value()}")
    print(f"  B after local ops: {b} -> value={b.value()}")
    print(f"  C after local ops: {c} -> value={c.value()}")

    assert a.value() == 5, f"Expected 5, got {a.value()}"
    assert b.value() == 3, f"Expected 3, got {b.value()}"
    assert c.value() == 7, f"Expected 7, got {c.value()}"

    # Merge A <- B, A <- C
    a.merge(b.copy())
    a.merge(c.copy())
    print(f"  A after merging B and C: {a} -> value={a.value()}")
    assert a.value() == 15, f"Expected 15, got {a.value()}"

    # Merge B <- A (B now gets A's and C's data via A)
    b.merge(a.copy())
    print(f"  B after merging A: {b} -> value={b.value()}")
    assert b.value() == 15

    # Merge C <- B
    c.merge(b.copy())
    print(f"  C after merging B: {c} -> value={c.value()}")
    assert c.value() == 15

    print("  PASSED")


def test_gcounter_idempotent_merge() -> None:
    """Verify that merging the same state multiple times is a no-op (idempotent)."""
    print("\n[Test] G-Counter merge idempotency")
    print("-" * 40)

    a = GCounter()
    a.increment("A", 10)

    b = GCounter()
    b.increment("B", 5)

    # Merge b into a three times -- should be same as merging once
    a.merge(b.copy())
    val_after_one = a.value()
    a.merge(b.copy())
    val_after_two = a.value()
    a.merge(b.copy())
    val_after_three = a.value()

    print(f"  After 1 merge: {val_after_one}")
    print(f"  After 2 merges: {val_after_two}")
    print(f"  After 3 merges: {val_after_three}")

    assert val_after_one == val_after_two == val_after_three == 15
    print("  PASSED (merge is idempotent)")


def test_gcounter_commutativity() -> None:
    """Verify merge is commutative: merge(A,B) == merge(B,A)."""
    print("\n[Test] G-Counter merge commutativity")
    print("-" * 40)

    a = GCounter()
    a.increment("A", 7)
    b = GCounter()
    b.increment("B", 3)

    # Path 1: A merges B
    a1 = a.copy()
    a1.merge(b.copy())

    # Path 2: B merges A
    b1 = b.copy()
    b1.merge(a.copy())

    print(f"  A.merge(B) -> value={a1.value()}")
    print(f"  B.merge(A) -> value={b1.value()}")

    assert a1.value() == b1.value(), "Merge should be commutative"
    print("  PASSED (merge is commutative)")


def test_gcounter_network() -> None:
    """Test G-Counter convergence through ReplicaNetwork with delays."""
    print("\n[Test] G-Counter convergence via network")
    print("-" * 40)

    replicas = {
        "A": GCounter(),
        "B": GCounter(),
        "C": GCounter(),
    }

    # Each replica does some local work
    replicas["A"].increment("A", 10)
    replicas["A"].increment("A", 5)  # A increments twice
    replicas["B"].increment("B", 7)
    replicas["C"].increment("C", 3)
    replicas["C"].increment("C", 2)  # C increments twice

    net = ReplicaNetwork(replicas=replicas, min_delay=0, max_delay=2)

    print(f"  Before sync: {net.all_values()}")
    assert not all_converged(net.replicas), "Should not be converged yet"

    syncs = net.full_sync()
    print(f"  After full sync ({syncs} syncs): {net.all_values()}")
    assert all_converged(net.replicas), "Should have converged"
    assert replicas["A"].value() == 27, f"Expected 27, got {replicas['A'].value()}"
    print("  PASSED")


def test_gcounter_reject_negative() -> None:
    """G-Counter must reject negative increments (grow-only!)."""
    print("\n[Test] G-Counter rejects negative increment")
    print("-" * 40)

    g = GCounter()
    try:
        g.increment("A", -1)
        print("  FAIL: should have raised ValueError")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  OK: correctly raised ValueError: {e}")
        print("  PASSED")


def test_pncounter_basic() -> None:
    """Test PN-Counter with increments and decrements."""
    print("\n[Test] PN-Counter basic operations")
    print("-" * 40)

    a = PNCounter()
    b = PNCounter()

    a.increment("A", 10)
    a.decrement("A", 3)   # A: +10 -3 = 7
    b.increment("B", 5)
    b.decrement("B", 8)   # B: +5 -8 = -3

    print(f"  A: {a} -> value={a.value()}")
    print(f"  B: {b} -> value={b.value()}")

    assert a.value() == 7, f"Expected 7, got {a.value()}"
    assert b.value() == -3, f"Expected -3, got {b.value()}"

    # Merge
    a.merge(b.copy())
    b.merge(a.copy())

    print(f"  A after merge: value={a.value()}")
    print(f"  B after merge: value={b.value()}")

    assert a.value() == b.value() == 4, f"Expected 4, got A={a.value()}, B={b.value()}"
    print("  PASSED")


def test_pncounter_network_with_partition() -> None:
    """Test PN-Counter convergence with a network partition."""
    print("\n[Test] PN-Counter convergence with partition")
    print("-" * 40)

    replicas = {
        "A": PNCounter(),
        "B": PNCounter(),
        "C": PNCounter(),
    }

    net = ReplicaNetwork(replicas=replicas, min_delay=0, max_delay=1)

    # Phase 1: Partition {A, B} vs {C}
    net.set_partition([{"A", "B"}, {"C"}])

    replicas["A"].increment("A", 10)
    replicas["B"].decrement("B", 3)
    replicas["C"].increment("C", 100)

    net.full_sync()
    print(f"  During partition: {net.all_values()}")

    # A and B should agree, C is separate
    assert replicas["A"].value() == replicas["B"].value() == 7
    assert replicas["C"].value() == 100

    # Phase 2: Heal and sync
    net.heal_partition()
    replicas["A"].increment("A", 5)  # More updates after healing
    net.full_sync()

    print(f"  After healing: {net.all_values()}")
    assert all_converged(net.replicas), "Should have converged after healing"
    # Expected: A=15, B_dec=3, C=100 -> 15 - 3 + 100 = 112
    assert replicas["A"].value() == 112, f"Expected 112, got {replicas['A'].value()}"
    print("  PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Exercise 1: G-Counter and PN-Counter")
    print("=" * 60)

    test_gcounter_basic()
    test_gcounter_idempotent_merge()
    test_gcounter_commutativity()
    test_gcounter_network()
    test_gcounter_reject_negative()

    test_pncounter_basic()
    test_pncounter_network_with_partition()

    print("\n" + "=" * 60)
    print("All Exercise 1 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
