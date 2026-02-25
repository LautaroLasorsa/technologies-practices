#!/usr/bin/env python3
"""
Base classes for CRDT implementations and a replica network simulator.

This module is FULLY SCAFFOLDED — no TODO(human) here.
It provides:
  - StateCRDT: Abstract base class for state-based (convergent) CRDTs
  - ReplicaNetwork: Simulates N replicas with message passing, random delays, and partitions
  - all_converged(): Helper to verify convergence across replicas
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Abstract base for state-based CRDTs
# ---------------------------------------------------------------------------

class StateCRDT(ABC, Generic[T]):
    """Base class for state-based (convergent) CRDTs.

    Subclasses must implement:
      - value()  -> query the current CRDT value
      - merge()  -> merge another replica's state (must be commutative, associative, idempotent)
      - copy()   -> deep-copy for simulating state transmission to another replica

    The merge function must form a join-semilattice:
      1. Commutative:  merge(A, B)  == merge(B, A)
      2. Associative:  merge(merge(A, B), C) == merge(A, merge(B, C))
      3. Idempotent:   merge(A, A)  == A   (no change)

    Additionally, all update operations must be monotonically increasing —
    after any update, the new state must be >= the old state in the semilattice
    partial order.
    """

    @abstractmethod
    def value(self) -> T:
        """Query the current value of this CRDT."""
        ...

    @abstractmethod
    def merge(self, other: StateCRDT[T]) -> None:
        """Merge another replica's state into this one.

        After merge, this replica's state is the least upper bound (join)
        of the two states.  Must satisfy:
          - commutative: self.merge(other) produces same result as other.merge(self)
          - associative: order of multi-way merges doesn't matter
          - idempotent:  self.merge(self_copy) is a no-op
        """
        ...

    @abstractmethod
    def copy(self) -> StateCRDT[T]:
        """Deep-copy this CRDT (simulates serializing state to send to another replica)."""
        ...


# ---------------------------------------------------------------------------
# Replica network simulator
# ---------------------------------------------------------------------------

@dataclass
class PendingSync:
    """A state snapshot in transit between replicas."""

    source_id: str
    target_id: str
    state_snapshot: StateCRDT
    delay: int  # ticks remaining before delivery


@dataclass
class ReplicaNetwork:
    """Simulates a network of CRDT replicas with:
      - Configurable random delays on state transfer
      - Network partitions (replicas in different groups cannot communicate)
      - Gossip-style synchronization (replicas periodically send their state to peers)

    Usage:
        net = ReplicaNetwork(replicas={"A": gcA, "B": gcB, "C": gcC})
        net.set_partition(group_a={"A", "B"}, group_b={"C"})
        net.gossip("A", "B")   # A sends state to B (succeeds, same group)
        net.gossip("A", "C")   # A sends state to C (blocked by partition)
        net.tick()              # advance time, deliver pending syncs
        net.heal_partition()    # remove partition
        net.full_sync()         # everyone syncs with everyone until stable
    """

    replicas: dict[str, StateCRDT] = field(default_factory=dict)
    pending: deque[PendingSync] = field(default_factory=deque)
    partitions: list[set[str]] = field(default_factory=list)
    min_delay: int = 0
    max_delay: int = 3
    total_syncs: int = 0

    def can_communicate(self, a: str, b: str) -> bool:
        """Check if replicas a and b can communicate (not separated by a partition)."""
        if not self.partitions:
            return True
        for group in self.partitions:
            if a in group and b in group:
                return True
        # If partitions are defined but a and b are not in the same group
        return False

    def set_partition(self, groups: list[set[str]]) -> None:
        """Set network partition. Each group is a set of replica IDs that can
        communicate within the group but not across groups.

        Example: set_partition([{"A", "B"}, {"C"}])
          -> A and B can sync, C is isolated.
        """
        self.partitions = groups

    def heal_partition(self) -> None:
        """Remove all partitions — everyone can communicate."""
        self.partitions = []

    def gossip(self, source: str, target: str) -> bool:
        """Replica `source` sends its state to `target`.

        Returns True if the message was queued (replicas can communicate),
        False if blocked by partition.
        """
        if not self.can_communicate(source, target):
            return False

        delay = random.randint(self.min_delay, self.max_delay)
        snapshot = self.replicas[source].copy()
        self.pending.append(PendingSync(
            source_id=source,
            target_id=target,
            state_snapshot=snapshot,
            delay=delay,
        ))
        return True

    def tick(self) -> int:
        """Advance one time unit. Deliver any pending syncs whose delay has expired.

        Returns the number of syncs delivered this tick.
        """
        delivered = 0
        still_pending: deque[PendingSync] = deque()

        while self.pending:
            sync = self.pending.popleft()
            sync.delay -= 1
            if sync.delay <= 0:
                # Deliver: target merges the source's state
                if self.can_communicate(sync.source_id, sync.target_id):
                    self.replicas[sync.target_id].merge(sync.state_snapshot)
                    delivered += 1
                    self.total_syncs += 1
                # If partition now blocks delivery, message is lost (realistic)
            else:
                still_pending.append(sync)

        self.pending = still_pending
        return delivered

    def drain(self) -> int:
        """Tick until all pending syncs are delivered. Returns total delivered."""
        total = 0
        while self.pending:
            total += self.tick()
        return total

    def full_sync(self, rounds: int = 3) -> int:
        """Perform full gossip: every replica sends to every other replica it can
        reach, repeated for `rounds` rounds. Returns total syncs delivered.

        This simulates the convergence behavior of an anti-entropy protocol.
        """
        total = 0
        ids = list(self.replicas.keys())
        for _ in range(rounds):
            for src in ids:
                for tgt in ids:
                    if src != tgt:
                        self.gossip(src, tgt)
            total += self.drain()
        return total

    def all_values(self) -> dict[str, object]:
        """Return {replica_id: value} for all replicas."""
        return {rid: r.value() for rid, r in self.replicas.items()}


def all_converged(replicas: dict[str, StateCRDT]) -> bool:
    """Check if all replicas have converged to the same value.

    Returns True if every replica reports the same value().
    """
    values = [r.value() for r in replicas.values()]
    if not values:
        return True
    first = values[0]
    return all(v == first for v in values)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Verify the base classes and network simulator work correctly."""
    print("=" * 60)
    print("CRDT Base Classes — Self-Test")
    print("=" * 60)

    # Test 1: StateCRDT cannot be instantiated directly
    print("\n[Test 1] StateCRDT is abstract...")
    try:
        StateCRDT()  # type: ignore[abstract]
        print("  FAIL: should have raised TypeError")
    except TypeError as e:
        print(f"  OK: {e}")

    # Test 2: ReplicaNetwork basic operations
    print("\n[Test 2] ReplicaNetwork basics...")

    # Create a trivial concrete CRDT for testing the network
    class TrivialMax(StateCRDT[int]):
        """A CRDT that just holds the max of all values seen."""
        def __init__(self, val: int = 0):
            self._val = val

        def value(self) -> int:
            return self._val

        def merge(self, other: StateCRDT[int]) -> None:
            if isinstance(other, TrivialMax):
                self._val = max(self._val, other._val)

        def copy(self) -> TrivialMax:
            return TrivialMax(self._val)

        def update(self, val: int) -> None:
            self._val = max(self._val, val)

    a, b, c = TrivialMax(0), TrivialMax(0), TrivialMax(0)
    net = ReplicaNetwork(
        replicas={"A": a, "B": b, "C": c},
        min_delay=0,
        max_delay=0,
    )

    a.update(10)
    b.update(20)
    c.update(5)

    print(f"  Before sync: {net.all_values()}")
    assert net.all_values() == {"A": 10, "B": 20, "C": 5}

    net.full_sync()
    print(f"  After full sync: {net.all_values()}")
    assert all_converged(net.replicas), "Should have converged"
    assert a.value() == 20, f"Expected 20, got {a.value()}"

    # Test 3: Partitions block communication
    print("\n[Test 3] Network partitions...")
    a2, b2, c2 = TrivialMax(0), TrivialMax(0), TrivialMax(0)
    net2 = ReplicaNetwork(
        replicas={"A": a2, "B": b2, "C": c2},
        min_delay=0,
        max_delay=0,
    )
    net2.set_partition([{"A", "B"}, {"C"}])

    a2.update(100)
    c2.update(200)

    net2.full_sync()
    print(f"  During partition: {net2.all_values()}")
    # A and B should converge (same group), C is isolated
    assert a2.value() == b2.value() == 100
    assert c2.value() == 200

    # Heal and sync
    net2.heal_partition()
    net2.full_sync()
    print(f"  After healing: {net2.all_values()}")
    assert all_converged(net2.replicas)
    assert a2.value() == 200

    print("\n" + "-" * 60)
    print("All base class self-tests passed.")
    print("-" * 60)


if __name__ == "__main__":
    _self_test()
