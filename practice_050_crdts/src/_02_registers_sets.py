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

from _00_crdt_base import StateCRDT, ReplicaNetwork, all_converged  # noqa: E402


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
#   self._tombstones: set[str]
#     Tags that have been explicitly removed. This is necessary to prevent
#     the "resurrection bug": without tombstones, a removed tag can reappear
#     when a stale replica (that still has the tag) merges back.
#     Example: R1 adds X(tag=T1) -> merges to R2 -> R1 removes X ->
#     R2 merges back to R1 -> naive union resurrects T1. Tombstones prevent this.
#
# Operations:
#   add(element):
#     Generate a fresh UUID tag. Add it to self._elements[element].
#     Now the element has a new tag that no other replica has seen.
#
#   remove(element):
#     Move ALL tags currently associated with element into self._tombstones.
#     Clear self._elements[element] (empty set or delete the key).
#     Tombstoned tags are subtracted during merge, preventing resurrection.
#     Only tags present at THIS replica are tombstoned — a concurrent add
#     on another replica generates a fresh tag that is NOT in our tombstones,
#     so it survives merge (add-wins semantics).
#
#   merge(other):
#     For each element, the merged tag set should contain tags that are
#     "alive" — present in at least one replica AND not tombstoned by either:
#
#       merged_tags(e) = (self_tags(e) | other_tags(e)) - (self._tombstones | other._tombstones)
#       merged_tombstones = self._tombstones | other._tombstones
#
#     WHY NAIVE UNION IS WRONG (the "resurrection bug"):
#       R1 adds X (tag=T1) -> merges to R2 -> R1 removes X (clears T1) ->
#       R2 merges stale state back to R1 -> union gives {T1} -> X is back!
#       Without tombstones, we can't distinguish "never seen" from "seen and removed."
#
#     With tombstones: R1's tombstones contain T1 after remove. When R2 merges
#     back, (tags_union - tombstones_union) = ({T1} - {T1}) = {} -> X stays dead.
#
#     Concurrent add-wins still works: R1 removes X (tombstones {T0}), R2
#     concurrently adds X (new tag T1, not in any tombstone set).
#     Merge: ({T1} | {}) - ({T0} | {}) = {T1} -> X is present. Add wins.
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

    def __init__(
        self,
        elements: dict[Any, set[str]] | None = None,
        tombstones: set[str] | None = None,
    ) -> None:
        # TODO(human): Store a DEEP COPY of elements (or empty dict if None)
        # and a COPY of tombstones (or empty set if None).
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
        #   3. Add them to self._tombstones (prevents resurrection on merge)
        #   4. Clear the element's tag set: self._elements[element] = set()
        #      (or delete the key entirely)
        #   5. Return the set of removed tags
        #
        # IMPORTANT: This only removes tags that THIS replica currently has.
        # If another replica concurrently added the same element with a NEW tag
        # that we haven't seen yet, that tag is safe -- it's not in our local
        # state or tombstones, so it survives merge. This is "observed-remove" semantics.
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
        # Steps:
        #   1. Merge tombstones: self._tombstones |= other._tombstones
        #   2. Collect all element keys from both self and other
        #   3. For each element, compute:
        #        alive_tags = (self_tags | other_tags) - self._tombstones
        #      (tombstones were already merged in step 1)
        #   4. Store the alive tags. Clean up elements with empty tag sets.
        #
        # Why this is correct:
        #   - Tags from self that other doesn't have AND not tombstoned: keep
        #   - Tags from other that self doesn't have AND not tombstoned: keep
        #   - Tags in both AND not tombstoned: keep
        #   - Tags that were removed: in tombstones, so subtracted out
        #
        # Semilattice: (elements=union, tombstones=union) with subtraction.
        # The combined state (active_tags, tombstones) grows monotonically:
        #   - Tombstone set only grows (union)
        #   - Active tags = raw_tags - tombstones; new adds create fresh tags
        #     not in any tombstone set, so they survive
        #
        # Trade-off: tombstones grow forever. See OptimizedORSet for the
        # version-vector approach that avoids this (Bieniusa et al. 2012).
        raise NotImplementedError("TODO(human): Implement ORSet.merge")

    def copy(self) -> ORSet:
        # TODO(human): Return a deep copy of this OR-Set.
        # Must deep-copy the elements dict, each inner tag set, AND tombstones.
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


def test_orset_no_resurrection() -> None:
    """Verify that a removed tag doesn't reappear when a stale replica merges back.

    This is the "resurrection bug" that naive union-based merge suffers from.
    Scenario:
      1. R1 adds X (tag=T1)
      2. R1 merges into R2 (R2 now has X with tag T1)
      3. R1 removes X (T1 is tombstoned at R1)
      4. R2 merges its stale state back into R1
      -> X must NOT reappear at R1 (T1 is in R1's tombstones)
    """
    print("\n[Test] OR-Set no resurrection from stale replica")
    print("-" * 40)

    r1 = ORSet()
    tag = r1.add("X")
    print(f"  R1 adds X (tag={tag[:8]}...)")

    # Simulate merge R1 -> R2 (R2 gets a copy of R1's state)
    r2 = r1.copy()
    print(f"  R2 = copy of R1: {r2.value()}")

    # R1 removes X
    r1.remove("X")
    print(f"  R1 removes X: {r1.value()} (X present: {r1.contains('X')})")
    assert not r1.contains("X"), "R1 removed X"

    # R2's stale state merges back into R1
    r1.merge(r2.copy())
    print(f"  R1 after merging stale R2: {r1.value()} (X present: {r1.contains('X')})")
    assert not r1.contains("X"), (
        "RESURRECTION BUG: X should stay removed! "
        "Stale tag from R2 should be blocked by R1's tombstone."
    )

    # Also verify R2 converges after getting R1's state
    r2.merge(r1.copy())
    assert not r2.contains("X"), "R2 should also agree X is removed"
    assert r1.value() == r2.value(), "Both should converge"

    print("  PASSED (no resurrection)")


# ---------------------------------------------------------------------------
# Optimized OR-Set (version vectors, no tombstones)
# ---------------------------------------------------------------------------

# TODO(human): Implement the Optimized OR-Set CRDT.
#
# The tombstone-based OR-Set above has a problem: tombstones grow forever.
# Every removed tag stays in memory for the lifetime of the system. In a
# busy system with millions of add/remove cycles, this is a memory leak.
#
# The Optimized OR-Set (Bieniusa et al. 2012, Almeida et al. 2018) replaces
# tombstones with a CAUSAL CONTEXT (version vector) that summarizes which
# operations each replica has seen. This allows detecting whether a tag was
# "removed" without storing individual tombstones.
#
# KEY IDEA -- "Dots" instead of UUIDs:
#   Instead of UUID tags, each add generates a "dot": (replica_id, seq_number).
#   Replica R's sequence number increments with each add. The causal context
#   tracks the highest sequence number seen from each replica.
#
#   A dot (R, N) is "known" to a replica if N <= causal_context[R].
#   If a dot is known but NOT in the active set, it was explicitly removed.
#
# Internal state:
#   self._entries: dict[Any, set[tuple[str, int]]]
#     Maps each element to a set of "dots" (replica_id, seq_num).
#     An element with dots is present; empty means absent.
#
#   self._causal_context: dict[str, int]
#     Version vector: maps replica_id -> highest seq_num seen from that replica.
#     Summarizes ALL operations this replica has ever witnessed.
#
# Operations:
#   add(element, replica_id):
#     1. seq = causal_context.get(replica_id, 0) + 1
#     2. causal_context[replica_id] = seq
#     3. Add dot (replica_id, seq) to entries[element]
#     4. Return the dot
#
#   remove(element):
#     1. Clear entries[element] (all dots removed)
#     2. Causal context stays unchanged (it already covers those dots)
#     3. Return the removed dots
#     NOTE: No tombstones needed! The causal context remembers we've "seen"
#     those dots. If they're missing from entries, we know they were removed.
#
#   merge(other):
#     For each element, a dot survives in the merged state if:
#       - It's in self's entries AND (other hasn't seen it OR other also has it active)
#       - OR it's in other's entries AND (self hasn't seen it OR self also has it active)
#
#     Formally, for dot (r, n) and element e:
#       keep from self:  (r,n) in self.entries[e]  AND
#                        (n > other.cc[r]  OR  (r,n) in other.entries[e])
#       keep from other: (r,n) in other.entries[e] AND
#                        (n > self.cc[r]   OR  (r,n) in self.entries[e])
#
#     In words: a dot from self is REMOVED only if other has "seen" it
#     (n <= other.cc[r]) but doesn't have it active (it's not in other.entries[e]).
#     That means other explicitly removed it. Same logic symmetrically for other's dots.
#
#     Merged causal context: element-wise max of both contexts.
#
# WHY THIS WORKS for the resurrection scenario:
#   1. R1: add(X) -> dot=(R1,1), entries={X:{(R1,1)}}, cc={R1:1}
#   2. Merge R1->R2: R2 entries={X:{(R1,1)}}, cc={R1:1}
#   3. R1: remove(X) -> entries={X:{}}, cc={R1:1} (cc unchanged)
#   4. Merge R2->R1:
#      - Dot (R1,1) is in R2's entries. Is it new to R1? n=1 <= R1.cc[R1]=1, no.
#        Does R1 also have it active? No (removed). -> DON'T KEEP. Removed by R1.
#      - Result: X stays removed, no tombstones needed!
#
# WHY ADD-WINS still works:
#   1. Both: entries={X:{(orig,1)}}, cc={orig:1}
#   2. R1: remove(X) -> entries={X:{}}, cc={orig:1}
#   3. R2: add(X) -> dot=(R2,1), entries={X:{(orig,1),(R2,1)}}, cc={orig:1,R2:1}
#   4. Merge:
#      - Dot (orig,1): R1 has seen it (cc[orig]=1) but removed it. R2 has it.
#        R1 saw it and removed -> don't keep.
#      - Dot (R2,1): R1 hasn't seen it (cc[R2]=0 < 1) -> new! KEEP.
#      - Result: X -> {(R2,1)} -> X is present. Add wins!
#
# ADVANTAGE: No tombstones, no unbounded memory growth. The causal context
# is O(num_replicas), not O(num_operations).
#
# Reference: Bieniusa et al. 2012, "An Optimized Conflict-free Replicated Set"
# Also: Almeida et al. 2018, "Delta State Replicated Data Types"


Dot = tuple[str, int]  # (replica_id, sequence_number)


class OptimizedORSet(StateCRDT[frozenset]):
    """Optimized OR-Set using version vectors instead of tombstones."""

    def __init__(
        self,
        entries: dict[Any, set[Dot]] | None = None,
        causal_context: dict[str, int] | None = None,
    ) -> None:
        # TODO(human): Store deep copies of entries and causal_context.
        #
        # self._entries: dict[Any, set[tuple[str, int]]]
        #   Deep copy: {k: set(v) for k, v in entries.items()} if entries else {}
        # self._causal_context: dict[str, int]
        #   Copy: dict(causal_context) if causal_context else {}
        raise NotImplementedError("TODO(human): Implement OptimizedORSet.__init__")

    def add(self, element: Any, replica_id: str) -> Dot:
        # TODO(human): Add element with a new dot from this replica.
        #
        # Steps:
        #   1. seq = self._causal_context.get(replica_id, 0) + 1
        #   2. self._causal_context[replica_id] = seq
        #   3. dot = (replica_id, seq)
        #   4. If element not in self._entries, create empty set
        #   5. Add dot to self._entries[element]
        #   6. Return the dot
        #
        # Unlike the tombstone OR-Set, tags are (replica_id, seq_num) pairs,
        # not UUIDs. This ties each tag to the causal context, enabling
        # tombstone-free removal detection during merge.
        raise NotImplementedError("TODO(human): Implement OptimizedORSet.add")

    def remove(self, element: Any) -> set[Dot]:
        # TODO(human): Remove all dots for element.
        #
        # Steps:
        #   1. Get current dots for element (empty set if absent)
        #   2. Save them for return
        #   3. Clear self._entries[element] = set()
        #   4. Return removed dots
        #
        # NOTE: Do NOT modify causal_context here. The cc already covers
        # these dots (they were added in prior add() calls). The fact that
        # the dots are in cc but NOT in entries is what tells merge() they
        # were removed.
        raise NotImplementedError("TODO(human): Implement OptimizedORSet.remove")

    def contains(self, element: Any) -> bool:
        # TODO(human): Return True if element has at least one active dot.
        raise NotImplementedError("TODO(human): Implement OptimizedORSet.contains")

    def value(self) -> frozenset:
        # TODO(human): Return frozenset of elements with at least one active dot.
        raise NotImplementedError("TODO(human): Implement OptimizedORSet.value")

    def merge(self, other: StateCRDT[frozenset]) -> None:
        # TODO(human): Merge using causal context (no tombstones!).
        #
        # Steps:
        #   1. Collect all element keys from both self and other
        #   2. For each element, compute surviving dots:
        #
        #      surviving = set()
        #      for dot in self._entries.get(elem, set()):
        #          r, n = dot
        #          # Keep if other hasn't seen it, OR other also has it active
        #          if n > other._causal_context.get(r, 0) or dot in other._entries.get(elem, set()):
        #              surviving.add(dot)
        #      for dot in other._entries.get(elem, set()):
        #          r, n = dot
        #          # Keep if self hasn't seen it, OR self also has it active
        #          if n > self._causal_context.get(r, 0) or dot in self._entries.get(elem, set()):
        #              surviving.add(dot)
        #
        #   3. Store surviving as self._entries[elem] (clean up if empty)
        #   4. Merge causal contexts: element-wise max
        #      for r in union of keys:
        #          self._causal_context[r] = max(self._causal_context.get(r,0),
        #                                       other._causal_context.get(r,0))
        #
        # IMPORTANT: Compute ALL surviving dots BEFORE updating the causal context.
        # If you update cc first, the "n > self.cc[r]" checks use wrong values.
        raise NotImplementedError("TODO(human): Implement OptimizedORSet.merge")

    def copy(self) -> OptimizedORSet:
        # TODO(human): Deep copy entries and causal_context.
        raise NotImplementedError("TODO(human): Implement OptimizedORSet.copy")

    def __repr__(self) -> str:
        # TODO(human): Show present elements with dot counts.
        # Example: "OptORSet({apple: 2 dots, banana: 1 dot}, cc={A:3, B:2})"
        raise NotImplementedError("TODO(human): Implement OptimizedORSet.__repr__")


# ---------------------------------------------------------------------------
# Optimized OR-Set tests
# ---------------------------------------------------------------------------

def test_opt_orset_basic() -> None:
    """Test OptimizedORSet basic add/remove."""
    print("\n[Test] OptimizedORSet basic add/remove")
    print("-" * 40)

    s = OptimizedORSet()
    s.add("apple", "A")
    s.add("banana", "A")
    s.add("cherry", "B")

    print(f"  After adds: {s}")
    assert s.value() == frozenset({"apple", "banana", "cherry"})

    s.remove("banana")
    assert not s.contains("banana")
    assert s.value() == frozenset({"apple", "cherry"})

    # Re-add banana (new dot)
    s.add("banana", "A")
    assert s.contains("banana")

    print("  PASSED")


def test_opt_orset_no_resurrection() -> None:
    """Verify OptimizedORSet prevents resurrection WITHOUT tombstones."""
    print("\n[Test] OptimizedORSet no resurrection (tombstone-free)")
    print("-" * 40)

    r1 = OptimizedORSet()
    dot = r1.add("X", "R1")
    print(f"  R1 adds X: dot={dot}")

    r2 = r1.copy()

    r1.remove("X")
    print(f"  R1 removes X: value={r1.value()}")
    assert not r1.contains("X")

    # Stale merge from R2
    r1.merge(r2.copy())
    print(f"  R1 after stale merge from R2: value={r1.value()}")
    assert not r1.contains("X"), (
        "RESURRECTION BUG: X should stay removed! "
        "R1's causal context knows about dot (R1,1), and it's not in entries -> removed."
    )

    r2.merge(r1.copy())
    assert r1.value() == r2.value()
    print("  PASSED (no tombstones needed)")


def test_opt_orset_add_wins() -> None:
    """Verify add-wins semantics in OptimizedORSet."""
    print("\n[Test] OptimizedORSet add-wins")
    print("-" * 40)

    # Both start with same state
    r1 = OptimizedORSet()
    r1.add("X", "orig")
    r2 = r1.copy()

    # Concurrent: R1 removes, R2 adds
    r1.remove("X")
    r2.add("X", "R2")

    print(f"  R1 after remove: {r1.value()}")
    print(f"  R2 after add:    {r2.value()}")

    r1.merge(r2.copy())
    r2.merge(r1.copy())

    print(f"  R1 after merge: {r1.value()}")
    print(f"  R2 after merge: {r2.value()}")

    assert r1.contains("X"), "Add should win over concurrent remove"
    assert r1.value() == r2.value()
    print("  PASSED (add wins)")


def test_opt_orset_merge_idempotent() -> None:
    """Verify OptimizedORSet merge is idempotent."""
    print("\n[Test] OptimizedORSet merge idempotency")
    print("-" * 40)

    a = OptimizedORSet()
    a.add("x", "A")
    a.add("y", "A")

    b = OptimizedORSet()
    b.add("y", "B")
    b.add("z", "B")

    a.merge(b.copy())
    val1 = a.value()
    a.merge(b.copy())
    val2 = a.value()

    assert val1 == val2 == frozenset({"x", "y", "z"})
    print("  PASSED")


def test_opt_orset_commutativity() -> None:
    """Verify OptimizedORSet merge is commutative."""
    print("\n[Test] OptimizedORSet merge commutativity")
    print("-" * 40)

    a = OptimizedORSet()
    a.add("apple", "A")
    a.add("banana", "A")

    b = OptimizedORSet()
    b.add("banana", "B")
    b.add("cherry", "B")

    a1 = a.copy()
    a1.merge(b.copy())

    b1 = b.copy()
    b1.merge(a.copy())

    assert a1.value() == b1.value(), "Merge should be commutative"
    print("  PASSED")


def test_opt_orset_network() -> None:
    """Test OptimizedORSet convergence through network with partition."""
    print("\n[Test] OptimizedORSet convergence with partition")
    print("-" * 40)

    replicas: dict[str, OptimizedORSet] = {
        "A": OptimizedORSet(),
        "B": OptimizedORSet(),
        "C": OptimizedORSet(),
    }

    net = ReplicaNetwork(replicas=replicas, min_delay=0, max_delay=1)  # type: ignore[arg-type]

    # Phase 1: All connected
    replicas["A"].add("apple", "A")
    replicas["B"].add("banana", "B")
    replicas["C"].add("cherry", "C")
    net.full_sync()
    assert all_converged(net.replicas)

    # Phase 2: Partition
    net.set_partition([{"A", "B"}, {"C"}])
    replicas["A"].remove("cherry")
    replicas["C"].add("cherry", "C")  # concurrent re-add
    replicas["B"].add("date", "B")
    net.full_sync()

    assert "cherry" not in replicas["A"].value()
    assert "cherry" in replicas["C"].value()

    # Phase 3: Heal
    net.heal_partition()
    net.full_sync()
    assert all_converged(net.replicas)
    assert "cherry" in replicas["A"].value(), "Add-wins across partition"

    print(f"  Final: {replicas['A'].value()}")
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

    print("\n--- OR-Set (with tombstones) ---")
    test_orset_basic_add_remove()
    test_orset_add_wins_semantics()
    test_orset_merge_idempotent()
    test_orset_commutativity()
    test_orset_network_with_partition()
    test_orset_remove_nonexistent()
    test_orset_no_resurrection()

    print("\n--- Optimized OR-Set (version vectors, no tombstones) ---")
    test_opt_orset_basic()
    test_opt_orset_no_resurrection()
    test_opt_orset_add_wins()
    test_opt_orset_merge_idempotent()
    test_opt_orset_commutativity()
    test_opt_orset_network()

    print("\n" + "=" * 60)
    print("All Exercise 2 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
