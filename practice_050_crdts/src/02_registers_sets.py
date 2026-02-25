#!/usr/bin/env python3
"""
Exercise 2: LWW-Register and OR-Set (state-based CRDTs).

LWW-Register = Last-Writer-Wins Register, resolves conflicts by timestamp.
OR-Set = Observed-Remove Set, uses unique tags per add for add-wins semantics.

These are more complex state-based CRDTs that handle richer data types
(single values and sets) while maintaining the join-semilattice guarantee.
"""
from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure sibling modules are importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crdt_base import StateCRDT, ReplicaNetwork, all_converged  # noqa: E402


# ---------------------------------------------------------------------------
# LWW-Register
# ---------------------------------------------------------------------------

# TODO(human): Implement the LWW-Register CRDT.
#
# The LWW-Register (Last-Writer-Wins Register) stores a single value and
# resolves concurrent writes by keeping the one with the highest timestamp.
# It's the simplest conflict resolution strategy and is widely used in practice
# (e.g., Cassandra columns, DynamoDB attributes).
#
# Internal state:
#   self._value: Any           -- the current stored value (can be any type)
#   self._timestamp: float     -- wall-clock or logical timestamp of the last write
#   self._replica_id: str      -- ID of the replica that performed the last write
#
# Semilattice structure:
#   - Partial order: A <= B iff (A.timestamp < B.timestamp) or
#                                (A.timestamp == B.timestamp and A.replica_id <= B.replica_id)
#   - Join (merge): keep the register with the greater (timestamp, replica_id) pair
#   - This is a total order (every pair is comparable), which is a special case of a
#     join-semilattice. Total orders are trivially valid semilattices.
#
# Tie-breaking rule:
#   When timestamps are exactly equal, we need a deterministic tie-breaker.
#   Using replica_id (lexicographic comparison) ensures all replicas agree on
#   the winner without coordination. The choice of tie-breaking rule is arbitrary
#   but must be CONSISTENT across all replicas.
#
# TRADE-OFF -- What LWW sacrifices:
#   Concurrent writes result in one being SILENTLY DISCARDED. If replica A writes
#   "Alice" and replica B writes "Bob" at nearly the same time, one write vanishes.
#   The user who wrote the lost value gets no notification. This is acceptable for
#   some use cases (user profile updates, sensor readings) but dangerous for others
#   (financial transactions, inventory counts).
#
# Reference: Shapiro et al. 2011, Section 3.2.2 "LWW-Register"


class LWWRegister(StateCRDT[Any]):
    """Last-Writer-Wins Register (state-based CRDT)."""

    def __init__(
        self,
        value: Any = None,
        timestamp: float = 0.0,
        replica_id: str = "",
    ) -> None:
        # TODO(human): Store value, timestamp, and replica_id.
        # These three fields fully define the register's state.
        # The initial state (None, 0.0, "") represents an empty register
        # that will lose to any real write.
        raise NotImplementedError("TODO(human): Implement LWWRegister.__init__")

    def assign(self, value: Any, timestamp: float, replica_id: str) -> None:
        # TODO(human): Update the register if the new write "wins".
        #
        # A new write wins if:
        #   1. new timestamp > current timestamp, OR
        #   2. new timestamp == current timestamp AND new replica_id > current replica_id
        #
        # If the new write does NOT win (older or same timestamp with lower replica_id),
        # do nothing -- the current value is already the winner.
        #
        # This is the UPDATE operation. It must be MONOTONICALLY INCREASING in the
        # semilattice: after assign(), the state must be >= the old state. Since we
        # only keep the value with the greatest (timestamp, replica_id), and we only
        # update when the new one is strictly greater, monotonicity is guaranteed.
        raise NotImplementedError("TODO(human): Implement LWWRegister.assign")

    def value(self) -> Any:
        # TODO(human): Return self._value.
        # The register's externally visible value is simply the stored value.
        raise NotImplementedError("TODO(human): Implement LWWRegister.value")

    def merge(self, other: StateCRDT[Any]) -> None:
        # TODO(human): Keep whichever register has the greater (timestamp, replica_id).
        #
        # This is equivalent to calling self.assign(other._value, other._timestamp,
        # other._replica_id) -- the assign method already implements the comparison.
        #
        # Semilattice properties hold because:
        #   - Commutative: max(A, B) == max(B, A) for total orders
        #   - Associative: max(max(A, B), C) == max(A, max(B, C))
        #   - Idempotent: max(A, A) == A
        raise NotImplementedError("TODO(human): Implement LWWRegister.merge")

    def copy(self) -> LWWRegister:
        # TODO(human): Return LWWRegister(self._value, self._timestamp, self._replica_id).
        # For simple immutable values (str, int, etc.) this is sufficient.
        # For mutable values you'd need a deep copy of self._value.
        raise NotImplementedError("TODO(human): Implement LWWRegister.copy")

    def __repr__(self) -> str:
        # TODO(human): Return a readable representation showing value, timestamp,
        # and replica_id. Example: "LWWRegister('Alice', ts=3.0, replica='B')"
        raise NotImplementedError("TODO(human): Implement LWWRegister.__repr__")


# ---------------------------------------------------------------------------
# OR-Set (Observed-Remove Set, also called Add-Wins Set)
# ---------------------------------------------------------------------------

# TODO(human): Implement the OR-Set CRDT.
#
# The OR-Set is the most practical set CRDT and the most complex one in this
# practice. It solves two problems that simpler set CRDTs cannot:
#   1. Unlike G-Set, elements CAN be removed
#   2. Unlike 2P-Set, elements CAN be re-added after removal
#
# The key insight is that each add() generates a GLOBALLY UNIQUE TAG (UUID).
# The internal state maps each element to the set of tags that are currently
# "active" for it. An element is considered present if it has at least one
# active tag.
#
# Internal state:
#   self._elements: dict[Any, set[str]]
#     Maps each element to a set of unique tag strings.
#     An element with an empty set (or not in the dict) is not in the set.
#     An element with one or more tags IS in the set.
#
# Operations:
#   add(element):
#     Generate a fresh UUID tag. Add it to self._elements[element].
#     Now the element has a new tag that no other replica has seen.
#
#   remove(element):
#     Remove ALL tags currently associated with element at THIS replica.
#     Set self._elements[element] = empty set (or delete the key).
#     CRITICAL: this only removes tags that THIS replica has observed.
#     If another replica concurrently added the same element (generating a
#     NEW tag), that tag is NOT in our removal set and will survive merge.
#     This is why it's called "Observed-Remove" -- you can only remove
#     what you've observed.
#
#   merge(other):
#     For each element, the merged tag set should contain tags that are
#     "alive" in at least one replica:
#       - Tags in BOTH replicas -> keep (both replicas have it)
#       - Tags in self ONLY -> keep (other never saw it, so other didn't remove it)
#       - Tags in other ONLY -> keep (self never saw it, so self didn't remove it)
#       - Tags that WERE in one replica but got removed -> gone (the remove was observed)
#
#     The simplest correct implementation: for each element, take the UNION
#     of tags from both replicas. This works because:
#       - When a replica removes an element, it clears the element's tags locally.
#       - So after remove, the element's tags in that replica are empty.
#       - Union of empty + other's tags = other's tags (the concurrent add survives).
#       - Union of tags + same tags = same tags (idempotent).
#
#     More precisely: merged_tags(e) = self_tags(e) | other_tags(e)
#     This is correct because remove() clears the tags at the removing replica,
#     so those tags won't appear in that replica's state anymore.
#
#   value():
#     Return the set of elements that have at least one active tag.
#     frozenset({e for e, tags in self._elements.items() if tags})
#
# WHY ADD-WINS:
#   If replica A adds element X (tag=T1) and replica B concurrently removes
#   element X (removing tag=T0, which is the only tag B has seen):
#     - A has: X -> {T1}    (fresh tag from add)
#     - B has: X -> {}      (removed T0)
#     - merge: X -> {T1} | {} = {T1}  -- X is present! Add wins.
#
#   The add wins because B's remove could only remove tags B had observed (T0).
#   The new tag T1 was created concurrently and is invisible to B's remove.
#
# Semilattice: defined over the tag sets per element; union is the join.
# Union is commutative, associative, and idempotent -- valid semilattice.
#
# Reference: Shapiro et al. 2011, Section 3.3.5 "OR-Set"
# Also see: Bieniusa et al. 2012, "An Optimized Conflict-free Replicated Set"


class ORSet(StateCRDT[frozenset]):
    """Observed-Remove Set / Add-Wins Set (state-based CRDT)."""

    def __init__(self, elements: dict[Any, set[str]] | None = None) -> None:
        # TODO(human): Store a DEEP COPY of elements (or empty dict if None).
        # Must deep-copy both the outer dict AND each inner set to prevent
        # aliasing bugs where two replicas share the same mutable sets.
        #
        # Hint: {k: set(v) for k, v in elements.items()} copies each inner set.
        raise NotImplementedError("TODO(human): Implement ORSet.__init__")

    def add(self, element: Any) -> str:
        # TODO(human): Add element to the set with a fresh unique tag.
        #
        # Steps:
        #   1. Generate a fresh tag: tag = str(uuid.uuid4())
        #   2. If element not in self._elements, create an empty set for it
        #   3. Add the tag to self._elements[element]
        #   4. Return the tag (useful for testing/debugging)
        #
        # Why a UUID? It must be globally unique across all replicas and all
        # time. If two replicas use the same tag, a remove on one replica
        # would incorrectly affect the other's add. UUIDs guarantee uniqueness
        # without coordination (no need to talk to other replicas).
        raise NotImplementedError("TODO(human): Implement ORSet.add")

    def remove(self, element: Any) -> set[str]:
        # TODO(human): Remove all currently observed tags for element.
        #
        # Steps:
        #   1. Get the current tags for element (empty set if element not present)
        #   2. Save them (we'll return them for debugging)
        #   3. Clear the element's tag set: self._elements[element] = set()
        #      (or delete the key entirely)
        #   4. Return the set of removed tags
        #
        # IMPORTANT: This only removes tags that THIS replica currently has.
        # If another replica concurrently added the same element with a NEW tag
        # that we haven't seen yet, that tag is safe -- it's not in our local
        # state, so we can't remove it. This is the "observed-remove" semantics.
        #
        # If element is not in the set, this is a no-op (returns empty set).
        raise NotImplementedError("TODO(human): Implement ORSet.remove")

    def contains(self, element: Any) -> bool:
        # TODO(human): Return True if element has at least one active tag.
        # Check: element in self._elements and len(self._elements[element]) > 0
        raise NotImplementedError("TODO(human): Implement ORSet.contains")

    def value(self) -> frozenset:
        # TODO(human): Return the set of all elements with at least one active tag.
        # frozenset({e for e, tags in self._elements.items() if tags})
        # Using frozenset makes the return value hashable and immutable,
        # which is important for the convergence test (comparing values).
        raise NotImplementedError("TODO(human): Implement ORSet.value")

    def merge(self, other: StateCRDT[frozenset]) -> None:
        # TODO(human): Merge another OR-Set's state into this one.
        #
        # For each element present in EITHER self or other:
        #   merged_tags = self_tags | other_tags   (set union)
        #
        # Steps:
        #   1. Collect all element keys from both self and other
        #   2. For each element, compute the union of tags from both replicas
        #   3. Store the merged tag set
        #   4. Optionally clean up elements with empty tag sets
        #
        # Why union is correct:
        #   - Tags from self that other doesn't have: keep (other never saw them,
        #     so other didn't explicitly remove them)
        #   - Tags from other that self doesn't have: keep (same logic)
        #   - Tags in both: keep (both agree they're active)
        #   - Tags that were removed: they're already gone from the removing
        #     replica's state, so they don't appear in the union
        #
        # This is a valid semilattice join because set union is:
        #   - Commutative: A | B == B | A
        #   - Associative: (A | B) | C == A | (B | C)
        #   - Idempotent:  A | A == A
        raise NotImplementedError("TODO(human): Implement ORSet.merge")

    def copy(self) -> ORSet:
        # TODO(human): Return a deep copy of this OR-Set.
        # Must deep-copy the elements dict AND each inner tag set.
        # Hint: use the same pattern as __init__: {k: set(v) for k, v in ...}
        raise NotImplementedError("TODO(human): Implement ORSet.copy")

    def __repr__(self) -> str:
        # TODO(human): Return a readable representation showing present elements
        # and their tag counts. Don't print full UUIDs -- just the count per element.
        # Example: "ORSet({apple: 2 tags, banana: 1 tag})"
        # Hint: only show elements with non-empty tag sets.
        raise NotImplementedError("TODO(human): Implement ORSet.__repr__")


# ---------------------------------------------------------------------------
# Tests (scaffolded)
# ---------------------------------------------------------------------------

def test_lww_register_basic() -> None:
    """Test LWW-Register with concurrent writes and tie-breaking."""
    print("\n[Test] LWW-Register basic operations")
    print("-" * 40)

    a = LWWRegister()
    b = LWWRegister()

    # Replica A writes "Alice" at time 1.0
    a.assign("Alice", timestamp=1.0, replica_id="A")
    # Replica B writes "Bob" at time 2.0 (later, so B wins)
    b.assign("Bob", timestamp=2.0, replica_id="B")

    print(f"  A: {a}")
    print(f"  B: {b}")

    assert a.value() == "Alice"
    assert b.value() == "Bob"

    # Merge: B's timestamp is higher, so "Bob" wins
    a.merge(b.copy())
    print(f"  A after merging B: {a}")
    assert a.value() == "Bob", f"Expected 'Bob', got {a.value()!r}"

    b.merge(a.copy())
    assert b.value() == "Bob"
    print("  PASSED")


def test_lww_register_tiebreak() -> None:
    """Test LWW-Register tie-breaking when timestamps are equal."""
    print("\n[Test] LWW-Register tie-breaking")
    print("-" * 40)

    a = LWWRegister()
    b = LWWRegister()

    # Same timestamp, different replica IDs
    a.assign("from_A", timestamp=5.0, replica_id="A")
    b.assign("from_B", timestamp=5.0, replica_id="B")

    print(f"  A: {a}")
    print(f"  B: {b}")

    # "B" > "A" lexicographically, so replica B's write wins the tie
    a.merge(b.copy())
    b.merge(a.copy())

    print(f"  A after merge: {a}")
    print(f"  B after merge: {b}")

    assert a.value() == b.value() == "from_B", (
        f"Tie-break should pick replica B: A={a.value()!r}, B={b.value()!r}"
    )
    print("  PASSED (tie broken by replica_id)")


def test_lww_register_idempotent() -> None:
    """Verify LWW-Register merge is idempotent."""
    print("\n[Test] LWW-Register merge idempotency")
    print("-" * 40)

    a = LWWRegister()
    a.assign("hello", timestamp=1.0, replica_id="A")

    b = LWWRegister()
    b.assign("world", timestamp=2.0, replica_id="B")

    a.merge(b.copy())
    val1 = a.value()
    a.merge(b.copy())
    val2 = a.value()
    a.merge(b.copy())
    val3 = a.value()

    print(f"  After 1 merge: {val1!r}")
    print(f"  After 2 merges: {val2!r}")
    print(f"  After 3 merges: {val3!r}")

    assert val1 == val2 == val3 == "world"
    print("  PASSED (merge is idempotent)")


def test_lww_register_network() -> None:
    """Test LWW-Register convergence via ReplicaNetwork."""
    print("\n[Test] LWW-Register convergence via network")
    print("-" * 40)

    replicas: dict[str, LWWRegister] = {
        "A": LWWRegister(),
        "B": LWWRegister(),
        "C": LWWRegister(),
    }

    # Concurrent writes with different timestamps
    replicas["A"].assign("alpha", timestamp=1.0, replica_id="A")
    replicas["B"].assign("beta", timestamp=3.0, replica_id="B")   # highest ts
    replicas["C"].assign("gamma", timestamp=2.0, replica_id="C")

    net = ReplicaNetwork(replicas=replicas, min_delay=0, max_delay=2)  # type: ignore[arg-type]

    print(f"  Before sync: {net.all_values()}")
    syncs = net.full_sync()
    print(f"  After full sync ({syncs} syncs): {net.all_values()}")

    assert all_converged(net.replicas), "Should have converged"
    assert replicas["A"].value() == "beta", f"Expected 'beta', got {replicas['A'].value()!r}"
    print("  PASSED")


def test_orset_basic_add_remove() -> None:
    """Test OR-Set basic add/remove operations."""
    print("\n[Test] OR-Set basic add/remove")
    print("-" * 40)

    s = ORSet()

    # Add some elements
    s.add("apple")
    s.add("banana")
    s.add("cherry")

    print(f"  After adds: {s}")
    print(f"  Value: {s.value()}")

    assert s.contains("apple")
    assert s.contains("banana")
    assert s.contains("cherry")
    assert s.value() == frozenset({"apple", "banana", "cherry"})

    # Remove banana
    removed = s.remove("banana")
    print(f"  Removed banana (tags={len(removed)}): {s}")
    assert not s.contains("banana"), "banana should be removed"
    assert s.value() == frozenset({"apple", "cherry"})

    # Re-add banana (fresh tag)
    s.add("banana")
    print(f"  Re-added banana: {s}")
    assert s.contains("banana"), "banana should be back (new tag)"
    assert s.value() == frozenset({"apple", "banana", "cherry"})

    print("  PASSED")


def test_orset_add_wins_semantics() -> None:
    """Test OR-Set add-wins: concurrent add and remove of the same element.

    This is THE key test for OR-Set. Scenario:
      1. Both replicas have element X (same tag T0)
      2. Replica A removes X (clears tag T0)
      3. Replica B concurrently adds X again (generates new tag T1)
      4. After merge, X should be present (B's add wins over A's remove)
    """
    print("\n[Test] OR-Set add-wins semantics")
    print("-" * 40)

    # Setup: both replicas start with the same state (element "X" with tag "T0")
    initial_tag = "T0-shared"
    a = ORSet({"X": {initial_tag}})
    b = ORSet({"X": {initial_tag}})

    print(f"  Initial A: {a}")
    print(f"  Initial B: {b}")

    # Concurrent operations (no sync between these):
    # A removes X (removes tag T0-shared)
    a.remove("X")
    # B adds X again (generates a NEW tag, call it T1)
    new_tag = b.add("X")

    print(f"  A after remove('X'): {a}  (X present: {a.contains('X')})")
    print(f"  B after add('X'):    {b}  (X present: {b.contains('X')})")

    assert not a.contains("X"), "A removed X"
    assert b.contains("X"), "B still has X (new tag)"

    # Now merge: A gets B's state
    a.merge(b.copy())
    # And B gets A's state
    b.merge(a.copy())

    print(f"  A after merge: {a}  (X present: {a.contains('X')})")
    print(f"  B after merge: {b}  (X present: {b.contains('X')})")

    # ADD WINS: X should be present in both (B's new tag T1 survived)
    assert a.contains("X"), "Add-wins: X should be in A after merge"
    assert b.contains("X"), "Add-wins: X should be in B after merge"
    assert a.value() == b.value(), "Both should have converged"

    print("  PASSED (add wins over concurrent remove)")


def test_orset_merge_idempotent() -> None:
    """Verify OR-Set merge is idempotent."""
    print("\n[Test] OR-Set merge idempotency")
    print("-" * 40)

    a = ORSet()
    a.add("x")
    a.add("y")

    b = ORSet()
    b.add("y")
    b.add("z")

    a.merge(b.copy())
    val1 = a.value()
    a.merge(b.copy())
    val2 = a.value()
    a.merge(b.copy())
    val3 = a.value()

    print(f"  After 1 merge: {val1}")
    print(f"  After 2 merges: {val2}")
    print(f"  After 3 merges: {val3}")

    assert val1 == val2 == val3
    assert val1 == frozenset({"x", "y", "z"})
    print("  PASSED (merge is idempotent)")


def test_orset_commutativity() -> None:
    """Verify OR-Set merge is commutative."""
    print("\n[Test] OR-Set merge commutativity")
    print("-" * 40)

    # Create two OR-Sets with distinct tags
    a = ORSet()
    a.add("apple")
    a.add("banana")

    b = ORSet()
    b.add("banana")
    b.add("cherry")

    # Path 1: A merges B
    a1 = a.copy()
    a1.merge(b.copy())

    # Path 2: B merges A
    b1 = b.copy()
    b1.merge(a.copy())

    print(f"  A.merge(B) -> {a1.value()}")
    print(f"  B.merge(A) -> {b1.value()}")

    assert a1.value() == b1.value(), "Merge should be commutative"
    print("  PASSED (merge is commutative)")


def test_orset_network_with_partition() -> None:
    """Test OR-Set convergence through network with partitions."""
    print("\n[Test] OR-Set convergence with partition")
    print("-" * 40)

    replicas: dict[str, ORSet] = {
        "A": ORSet(),
        "B": ORSet(),
        "C": ORSet(),
    }

    net = ReplicaNetwork(replicas=replicas, min_delay=0, max_delay=1)  # type: ignore[arg-type]

    # Phase 1: All connected -- add some elements
    replicas["A"].add("apple")
    replicas["B"].add("banana")
    replicas["C"].add("cherry")
    net.full_sync()

    print(f"  After initial sync: {net.all_values()}")
    assert all_converged(net.replicas)

    # Phase 2: Partition {A, B} vs {C}
    net.set_partition([{"A", "B"}, {"C"}])

    # A removes cherry (has the tag from C's add)
    replicas["A"].remove("cherry")
    # C concurrently re-adds cherry (new tag!)
    replicas["C"].add("cherry")
    # B adds a new element
    replicas["B"].add("date")

    net.full_sync()
    print(f"  During partition: A={replicas['A'].value()}, "
          f"B={replicas['B'].value()}, C={replicas['C'].value()}")

    # A and B should agree; C is separate
    assert replicas["A"].value() == replicas["B"].value()
    assert "cherry" not in replicas["A"].value()  # A removed it, B agreed
    assert "cherry" in replicas["C"].value()  # C re-added it

    # Phase 3: Heal and sync
    net.heal_partition()
    net.full_sync()

    print(f"  After healing: {net.all_values()}")
    assert all_converged(net.replicas), "Should have converged"

    # cherry should be back (C's add-wins over A's remove)
    assert "cherry" in replicas["A"].value(), (
        "Add-wins: C's concurrent add of cherry should survive A's remove"
    )
    # All elements should be present
    for elem in ["apple", "banana", "cherry", "date"]:
        assert elem in replicas["A"].value(), f"{elem} should be in the merged set"

    print("  PASSED")


def test_orset_remove_nonexistent() -> None:
    """Removing an element not in the set should be a no-op."""
    print("\n[Test] OR-Set remove non-existent element")
    print("-" * 40)

    s = ORSet()
    s.add("apple")

    removed = s.remove("banana")  # not in set
    print(f"  Removed 'banana' (not in set): got {len(removed)} tags")
    assert len(removed) == 0
    assert s.value() == frozenset({"apple"})
    print("  PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Exercise 2: LWW-Register and OR-Set")
    print("=" * 60)

    test_lww_register_basic()
    test_lww_register_tiebreak()
    test_lww_register_idempotent()
    test_lww_register_network()

    test_orset_basic_add_remove()
    test_orset_add_wins_semantics()
    test_orset_merge_idempotent()
    test_orset_commutativity()
    test_orset_network_with_partition()
    test_orset_remove_nonexistent()

    print("\n" + "=" * 60)
    print("All Exercise 2 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
