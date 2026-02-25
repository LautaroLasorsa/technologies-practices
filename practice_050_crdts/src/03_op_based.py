#!/usr/bin/env python3
"""
Exercise 3: Operation-based CRDTs (CmRDTs — Commutative Replicated Data Types).

Instead of merging full state, replicas broadcast operations and apply them locally.
This requires a causal broadcast layer that delivers operations exactly once and
in causal order.

You'll implement:
  1. OpBasedCounter -- an operation-based counter (CmRDT)
  2. CausalBroadcast -- a causal delivery layer using vector clocks
  3. OpBasedNetwork -- a simulation harness that ties them together
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Ensure sibling modules are importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Operation types
# ---------------------------------------------------------------------------

class OpType(Enum):
    """Types of counter operations."""
    INCREMENT = "increment"
    DECREMENT = "decrement"


@dataclass(frozen=True)
class Operation:
    """A single operation to be broadcast and applied.

    Attributes:
        op_type:    INCREMENT or DECREMENT
        amount:     positive integer amount
        origin:     replica_id that created this operation
        op_id:      unique identifier for this operation (for deduplication)
        vclock:     vector clock at the time this operation was created
                    (maps replica_id -> logical time)
    """
    op_type: OpType
    amount: int
    origin: str
    op_id: str
    vclock: dict[str, int] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.op_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Operation):
            return NotImplemented
        return self.op_id == other.op_id


# ---------------------------------------------------------------------------
# OpBasedCounter
# ---------------------------------------------------------------------------

# TODO(human): Implement the Operation-based Counter CRDT.
#
# Unlike state-based CRDTs (which merge full state), operation-based CRDTs
# (CmRDTs) work by broadcasting operations. Each replica applies received
# operations to its local state. The key requirements are:
#
#   1. COMMUTATIVITY: Concurrent operations must commute (applying them in
#      any order yields the same state). For a counter, increment(5) and
#      increment(3) commute because 0+5+3 == 0+3+5.
#
#   2. EXACTLY-ONCE DELIVERY: Each operation must be applied exactly once.
#      Duplicate delivery would double-count. The op_id field enables
#      deduplication.
#
#   3. CAUSAL ORDER: If operation A happened-before operation B at the sender,
#      all replicas must apply A before B. The CausalBroadcast layer ensures
#      this using vector clocks.
#
# Internal state:
#   self._value: int            -- the current counter value
#   self._replica_id: str       -- this replica's ID
#   self._applied: set[str]     -- set of op_ids already applied (for dedup)
#
# CONTRAST WITH STATE-BASED:
#   - State-based (CvRDT): merge is idempotent, so duplicate messages are OK.
#     Network can be unreliable.
#   - Op-based (CmRDT): apply is NOT idempotent (applying increment(5) twice
#     gives +10, not +5), so the delivery layer must guarantee exactly-once.
#     BUT messages are smaller (just the operation, not the full state).
#
# Reference: Shapiro et al. 2011, Section 2.2 "Op-based CRDTs"


class OpBasedCounter:
    """Operation-based counter (CmRDT)."""

    def __init__(self, replica_id: str) -> None:
        # TODO(human): Initialize the counter.
        #
        # Store:
        #   self._replica_id = replica_id
        #   self._value = 0
        #   self._applied: set[str] = set()  -- tracks op_ids we've already applied
        #
        # The _applied set is critical for exactly-once semantics. Without it,
        # if the network layer retransmits an operation (e.g., during retry),
        # we'd apply it twice, corrupting the counter.
        raise NotImplementedError("TODO(human): Implement OpBasedCounter.__init__")

    @property
    def value(self) -> int:
        # TODO(human): Return self._value.
        raise NotImplementedError("TODO(human): Implement OpBasedCounter.value")

    @property
    def replica_id(self) -> str:
        # TODO(human): Return self._replica_id.
        raise NotImplementedError("TODO(human): Implement OpBasedCounter.replica_id")

    def apply(self, op: Operation) -> bool:
        # TODO(human): Apply an operation to this replica's local state.
        #
        # Steps:
        #   1. CHECK DEDUP: if op.op_id is already in self._applied, return False
        #      (already applied -- skip to maintain exactly-once semantics)
        #   2. APPLY: if op.op_type is INCREMENT, add op.amount to self._value.
        #             if op.op_type is DECREMENT, subtract op.amount from self._value.
        #   3. RECORD: add op.op_id to self._applied
        #   4. Return True (operation was applied)
        #
        # This is intentionally simple -- the complexity is in the delivery layer,
        # not the CRDT itself. The counter just needs to handle add/subtract.
        # Commutativity holds because integer addition is commutative.
        raise NotImplementedError("TODO(human): Implement OpBasedCounter.apply")

    def __repr__(self) -> str:
        return f"OpCounter({self._replica_id}, value={self._value}, applied={len(self._applied)})"


# ---------------------------------------------------------------------------
# CausalBroadcast
# ---------------------------------------------------------------------------

# TODO(human): Implement the Causal Broadcast delivery layer.
#
# This is the infrastructure that makes operation-based CRDTs work. It ensures:
#   1. CAUSAL ORDER: If operation A happened-before B at the sender, all
#      replicas deliver A before B.
#   2. EXACTLY-ONCE: Each operation is delivered exactly once per replica.
#
# It uses VECTOR CLOCKS to track causal dependencies:
#   - Each replica maintains a vector clock: dict[str, int] mapping
#     replica_id -> the number of operations from that replica that have
#     been delivered locally.
#   - When replica R creates an operation, it stamps the operation with
#     R's current vector clock, then increments vclock[R].
#   - When replica X receives an operation from replica R with vclock V:
#     - The op is DELIVERABLE if:
#       (a) V[R] == X.vclock[R] + 1  (this is the NEXT expected op from R)
#       (b) For all other replicas S != R: V[S] <= X.vclock[S]
#           (all ops that causally precede this one have been delivered)
#     - If deliverable: deliver it (apply to the CRDT), then increment X.vclock[R]
#     - If NOT deliverable: buffer it for later (some causal dependency hasn't
#       arrived yet)
#
# WHY THIS MATTERS:
#   Without causal delivery, an operation that depends on a prior operation might
#   be applied first. Example:
#     - Replica A: increment(10), then decrement(5) (intending to set to 5)
#     - If another replica receives decrement(5) before increment(10), it
#       temporarily goes to -5 -- which might violate application invariants.
#   With causal delivery, the increment always arrives first because decrement
#   happened-after increment at replica A.
#
# DELIVERABILITY CONDITION EXPLAINED:
#   Condition (a) V[R] == X.vclock[R] + 1:
#     "This op is the NEXT one I expect from replica R."
#     If V[R] is higher, I'm missing some ops from R (buffer this).
#     If V[R] is lower or equal, I've already seen this op (duplicate).
#
#   Condition (b) V[S] <= X.vclock[S] for S != R:
#     "The sender had already seen all the ops from other replicas that
#     I've also seen." This ensures causal dependencies are satisfied.
#
# Reference: Shapiro et al. 2011, Section 2.3 "Causal delivery"
# Also see: Schwarz, Mattern (1994). "Detecting Causal Relationships in
# Distributed Computations"


class CausalBroadcast:
    """Causal broadcast layer for a single replica.

    Manages vector clock, buffering, and delivery for one replica.
    """

    def __init__(self, replica_id: str) -> None:
        # TODO(human): Initialize the causal broadcast state.
        #
        # Store:
        #   self._replica_id: str = replica_id
        #   self._vclock: dict[str, int] = defaultdict(int)
        #     -- vector clock: maps each known replica to the count of
        #        operations delivered from that replica
        #   self._buffer: list[Operation] = []
        #     -- operations received but not yet deliverable (waiting for
        #        causal dependencies to be satisfied)
        #   self._op_counter: int = 0
        #     -- number of operations THIS replica has created (used to
        #        generate unique op_ids and stamp vector clocks)
        #   self._delivered: set[str] = set()
        #     -- op_ids already delivered (for dedup on re-receive)
        raise NotImplementedError("TODO(human): Implement CausalBroadcast.__init__")

    @property
    def vclock(self) -> dict[str, int]:
        # TODO(human): Return dict(self._vclock) -- a copy to prevent external mutation.
        raise NotImplementedError("TODO(human): Implement CausalBroadcast.vclock")

    def create_operation(self, op_type: OpType, amount: int) -> Operation:
        # TODO(human): Create a new operation stamped with this replica's vector clock.
        #
        # Steps:
        #   1. Increment self._op_counter
        #   2. Generate a unique op_id: f"{self._replica_id}:{self._op_counter}"
        #   3. Create a SNAPSHOT of the current vector clock (copy it!)
        #   4. Increment self._vclock[self._replica_id] (this op is now "created")
        #   5. Mark op_id as delivered in self._delivered (creator auto-delivers to self)
        #   6. Return the Operation with the SNAPSHOT vclock (from step 3, BEFORE step 4)
        #
        # IMPORTANT: The vclock in the Operation is the sender's clock BEFORE
        # incrementing. This is because the deliverability check at the receiver
        # expects V[R] == receiver.vclock[R] + 1. If we put the AFTER clock,
        # the receiver would need V[R] == receiver.vclock[R] + 1 but V[R] would
        # already be incremented, making the condition off-by-one.
        #
        # Wait -- actually the convention here is: the operation's vclock[R] should
        # equal the sequence number of this operation (starting from 1). So:
        #   - Before creating: self._vclock[R] = N (we've created N ops so far)
        #   - Create op: stamp with vclock where vclock[R] = N + 1
        #   - After: self._vclock[R] = N + 1
        #
        # Simpler approach: increment vclock FIRST, then snapshot.
        # The receiver checks: op.vclock[R] == receiver.vclock[R] + 1, meaning
        # "this is operation number (my_count + 1) from R".
        #
        # Let's use this convention:
        #   1. self._vclock[self._replica_id] += 1
        #   2. snapshot = dict(self._vclock)
        #   3. Create Operation with vclock=snapshot
        raise NotImplementedError("TODO(human): Implement CausalBroadcast.create_operation")

    def is_deliverable(self, op: Operation) -> bool:
        # TODO(human): Check if an operation can be delivered NOW.
        #
        # Deliverability conditions:
        #   (a) op.vclock[op.origin] == self._vclock[op.origin] + 1
        #       This is the NEXT expected operation from the origin replica.
        #
        #   (b) For ALL other replicas S != op.origin:
        #       op.vclock.get(S, 0) <= self._vclock[S]
        #       The sender hasn't seen any ops from S that we haven't also seen.
        #
        # If BOTH conditions hold, the operation's causal dependencies are satisfied.
        # If not, some predecessor operation hasn't been delivered yet -- buffer this op.
        #
        # Also check: if op.op_id is already in self._delivered, it's a duplicate.
        # Return False for duplicates (though the caller may also check this).
        raise NotImplementedError("TODO(human): Implement CausalBroadcast.is_deliverable")

    def receive(self, op: Operation) -> list[Operation]:
        # TODO(human): Receive an operation and return all operations that can be
        # delivered (in causal order).
        #
        # Steps:
        #   1. If op.op_id is in self._delivered, return [] (duplicate, already applied)
        #   2. Add op to self._buffer
        #   3. Try to deliver: repeatedly scan the buffer for deliverable operations.
        #      For each deliverable op:
        #        a. Remove it from the buffer
        #        b. Mark it as delivered (add op_id to self._delivered)
        #        c. Update self._vclock[op.origin] += 1
        #        d. Add it to the list of delivered ops
        #      Repeat the scan until no more ops are deliverable (a newly delivered op
        #      might unblock a buffered op that was waiting for it).
        #   4. Return the list of all ops delivered in this call (in delivery order)
        #
        # WHY THE LOOP: Delivering operation A might satisfy the causal dependency
        # of buffered operation B. So after each delivery, we must re-scan the buffer.
        # This is O(n^2) in the worst case but correct. Production implementations
        # use more efficient data structures.
        raise NotImplementedError("TODO(human): Implement CausalBroadcast.receive")

    @property
    def buffer_size(self) -> int:
        """Number of operations waiting in the buffer."""
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"CausalBroadcast({self._replica_id}, "
            f"vclock={dict(self._vclock)}, buffered={len(self._buffer)})"
        )


# ---------------------------------------------------------------------------
# OpBasedNetwork -- simulation harness
# ---------------------------------------------------------------------------

@dataclass
class OpBasedNetwork:
    """Simulates a network of operation-based CRDT replicas.

    Each replica has a CausalBroadcast layer and an OpBasedCounter.
    Operations are broadcast to all other replicas with random delays.
    """

    counters: dict[str, OpBasedCounter] = field(default_factory=dict)
    broadcasts: dict[str, CausalBroadcast] = field(default_factory=dict)
    pending: deque[tuple[str, Operation, int]] = field(default_factory=deque)
    min_delay: int = 0
    max_delay: int = 5

    def add_replica(self, replica_id: str) -> None:
        """Add a new replica to the network."""
        self.counters[replica_id] = OpBasedCounter(replica_id)
        self.broadcasts[replica_id] = CausalBroadcast(replica_id)

    def create_and_broadcast(self, replica_id: str, op_type: OpType, amount: int) -> Operation:
        """Create an operation at a replica and broadcast to all others.

        The creating replica applies the operation immediately.
        Other replicas receive it with a random delay.
        """
        bc = self.broadcasts[replica_id]
        op = bc.create_operation(op_type, amount)

        # Creator applies immediately
        self.counters[replica_id].apply(op)

        # Queue for delivery to all other replicas
        for target_id in self.counters:
            if target_id != replica_id:
                delay = random.randint(self.min_delay, self.max_delay)
                self.pending.append((target_id, op, delay))

        return op

    def tick(self) -> int:
        """Advance one time unit. Deliver pending operations whose delay has expired."""
        delivered_total = 0
        still_pending: deque[tuple[str, Operation, int]] = deque()

        while self.pending:
            target_id, op, delay = self.pending.popleft()
            delay -= 1
            if delay <= 0:
                # Deliver through causal broadcast layer
                bc = self.broadcasts[target_id]
                delivered_ops = bc.receive(op)
                for delivered_op in delivered_ops:
                    self.counters[target_id].apply(delivered_op)
                    delivered_total += 1
            else:
                still_pending.append((target_id, op, delay))

        self.pending = still_pending
        return delivered_total

    def drain(self) -> int:
        """Tick until all pending operations are delivered."""
        total = 0
        max_ticks = 1000  # safety limit
        for _ in range(max_ticks):
            if not self.pending:
                break
            total += self.tick()
        return total

    def all_values(self) -> dict[str, int]:
        """Return {replica_id: counter_value} for all replicas."""
        return {rid: c.value for rid, c in self.counters.items()}

    def all_converged(self) -> bool:
        """Check if all replicas have the same counter value."""
        values = list(self.all_values().values())
        return len(set(values)) <= 1

    def buffer_status(self) -> dict[str, int]:
        """Return {replica_id: buffer_size} for all replicas."""
        return {rid: bc.buffer_size for rid, bc in self.broadcasts.items()}


# ---------------------------------------------------------------------------
# Tests (scaffolded)
# ---------------------------------------------------------------------------

def test_op_counter_basic() -> None:
    """Test OpBasedCounter with manual operation application."""
    print("\n[Test] OpBasedCounter basic operations")
    print("-" * 40)

    c = OpBasedCounter("A")
    assert c.value == 0

    op1 = Operation(OpType.INCREMENT, 5, "A", "A:1")
    op2 = Operation(OpType.INCREMENT, 3, "B", "B:1")
    op3 = Operation(OpType.DECREMENT, 2, "A", "A:2")

    assert c.apply(op1) is True
    assert c.value == 5
    assert c.apply(op2) is True
    assert c.value == 8
    assert c.apply(op3) is True
    assert c.value == 6

    print(f"  After 3 ops: {c}")

    # Duplicate detection: re-applying op1 should be a no-op
    assert c.apply(op1) is False, "Duplicate op should return False"
    assert c.value == 6, "Value should be unchanged after duplicate"
    print("  Duplicate correctly rejected")
    print("  PASSED")


def test_op_counter_commutative() -> None:
    """Verify that applying ops in different orders gives the same result."""
    print("\n[Test] OpBasedCounter commutativity")
    print("-" * 40)

    ops = [
        Operation(OpType.INCREMENT, 10, "A", "A:1"),
        Operation(OpType.DECREMENT, 3, "B", "B:1"),
        Operation(OpType.INCREMENT, 7, "C", "C:1"),
        Operation(OpType.DECREMENT, 2, "A", "A:2"),
    ]

    # Apply in original order
    c1 = OpBasedCounter("test1")
    for op in ops:
        c1.apply(op)

    # Apply in reversed order
    c2 = OpBasedCounter("test2")
    for op in reversed(ops):
        c2.apply(op)

    # Apply in random order
    c3 = OpBasedCounter("test3")
    shuffled = list(ops)
    random.shuffle(shuffled)
    for op in shuffled:
        c3.apply(op)

    print(f"  Original order: {c1.value}")
    print(f"  Reversed order: {c2.value}")
    print(f"  Random order:   {c3.value}")

    assert c1.value == c2.value == c3.value == 12
    print("  PASSED (operations commute)")


def test_causal_broadcast_basic() -> None:
    """Test CausalBroadcast with in-order delivery."""
    print("\n[Test] CausalBroadcast basic delivery")
    print("-" * 40)

    bc_a = CausalBroadcast("A")
    bc_b = CausalBroadcast("B")

    # A creates two operations
    op1 = bc_a.create_operation(OpType.INCREMENT, 5)
    op2 = bc_a.create_operation(OpType.INCREMENT, 3)

    print(f"  A created: {op1.op_id} (vclock={op1.vclock})")
    print(f"  A created: {op2.op_id} (vclock={op2.vclock})")

    # B receives op1 first (in order) -- should be deliverable
    delivered = bc_b.receive(op1)
    print(f"  B received op1: delivered {len(delivered)} ops")
    assert len(delivered) == 1
    assert delivered[0].op_id == op1.op_id

    # B receives op2 (in order) -- should be deliverable
    delivered = bc_b.receive(op2)
    print(f"  B received op2: delivered {len(delivered)} ops")
    assert len(delivered) == 1
    assert delivered[0].op_id == op2.op_id

    print(f"  B's vclock: {bc_b.vclock}")
    assert bc_b.vclock.get("A", 0) == 2, f"B should have seen 2 ops from A"
    print("  PASSED")


def test_causal_broadcast_reorder() -> None:
    """Test CausalBroadcast when operations arrive out of order.

    Key scenario: op2 from A arrives at B BEFORE op1 from A.
    op2 must be buffered until op1 is delivered (causal order).
    """
    print("\n[Test] CausalBroadcast reordering")
    print("-" * 40)

    bc_a = CausalBroadcast("A")
    bc_b = CausalBroadcast("B")

    op1 = bc_a.create_operation(OpType.INCREMENT, 5)
    op2 = bc_a.create_operation(OpType.INCREMENT, 3)

    print(f"  A's op1: {op1.op_id} vclock={op1.vclock}")
    print(f"  A's op2: {op2.op_id} vclock={op2.vclock}")

    # B receives op2 FIRST (out of order)
    delivered = bc_b.receive(op2)
    print(f"  B received op2 first: delivered {len(delivered)} ops, buffered {bc_b.buffer_size}")
    assert len(delivered) == 0, "op2 should be buffered (waiting for op1)"
    assert bc_b.buffer_size == 1

    # B receives op1 -- should deliver both op1 AND the buffered op2
    delivered = bc_b.receive(op1)
    print(f"  B received op1: delivered {len(delivered)} ops, buffered {bc_b.buffer_size}")
    assert len(delivered) == 2, f"Should deliver op1 + buffered op2, got {len(delivered)}"
    assert delivered[0].op_id == op1.op_id, "op1 should be delivered first (causal order)"
    assert delivered[1].op_id == op2.op_id, "op2 should be delivered second"
    assert bc_b.buffer_size == 0

    print("  PASSED (buffered out-of-order op, delivered in causal order)")


def test_causal_broadcast_cross_replica() -> None:
    """Test causal delivery with operations from multiple replicas.

    Scenario:
      1. A creates op_A1
      2. B receives op_A1, then creates op_B1 (which causally depends on op_A1)
      3. C receives op_B1 BEFORE op_A1
      4. C must buffer op_B1 until op_A1 arrives (causal dependency)
    """
    print("\n[Test] CausalBroadcast cross-replica causality")
    print("-" * 40)

    bc_a = CausalBroadcast("A")
    bc_b = CausalBroadcast("B")
    bc_c = CausalBroadcast("C")

    # Step 1: A creates op_A1
    op_a1 = bc_a.create_operation(OpType.INCREMENT, 10)
    print(f"  A creates op_A1: vclock={op_a1.vclock}")

    # Step 2: B receives op_A1, then creates op_B1
    delivered = bc_b.receive(op_a1)
    assert len(delivered) == 1
    op_b1 = bc_b.create_operation(OpType.DECREMENT, 3)
    print(f"  B receives op_A1, creates op_B1: vclock={op_b1.vclock}")

    # op_B1's vclock should include A:1 (because B saw op_A1 before creating op_B1)
    assert op_b1.vclock.get("A", 0) >= 1, (
        f"op_B1 should reflect causal dependency on op_A1: {op_b1.vclock}"
    )

    # Step 3: C receives op_B1 BEFORE op_A1
    delivered = bc_c.receive(op_b1)
    print(f"  C receives op_B1 first: delivered {len(delivered)}, buffered {bc_c.buffer_size}")
    assert len(delivered) == 0, "op_B1 should be buffered (C hasn't seen op_A1)"

    # Step 4: C receives op_A1 -- should deliver both
    delivered = bc_c.receive(op_a1)
    print(f"  C receives op_A1: delivered {len(delivered)}, buffered {bc_c.buffer_size}")
    assert len(delivered) == 2, f"Should deliver op_A1 + buffered op_B1, got {len(delivered)}"
    assert delivered[0].op_id == op_a1.op_id, "op_A1 delivered first"
    assert delivered[1].op_id == op_b1.op_id, "op_B1 delivered second (dependency satisfied)"

    print("  PASSED (cross-replica causal dependency respected)")


def test_causal_broadcast_duplicate() -> None:
    """Test that duplicate operations are rejected."""
    print("\n[Test] CausalBroadcast duplicate rejection")
    print("-" * 40)

    bc_a = CausalBroadcast("A")
    bc_b = CausalBroadcast("B")

    op = bc_a.create_operation(OpType.INCREMENT, 5)

    # First delivery
    delivered = bc_b.receive(op)
    assert len(delivered) == 1

    # Duplicate delivery
    delivered = bc_b.receive(op)
    print(f"  Duplicate delivery: {len(delivered)} ops delivered")
    assert len(delivered) == 0, "Duplicate should be rejected"

    print("  PASSED")


def test_network_convergence() -> None:
    """Test full network convergence with random delays and ordering."""
    print("\n[Test] OpBased network convergence (5 replicas)")
    print("-" * 40)

    random.seed(42)  # reproducible

    net = OpBasedNetwork(min_delay=0, max_delay=5)
    for rid in ["A", "B", "C", "D", "E"]:
        net.add_replica(rid)

    # Generate random operations across all replicas
    operations: list[tuple[str, OpType, int]] = []
    for rid in ["A", "B", "C", "D", "E"]:
        for _ in range(5):
            op_type = random.choice([OpType.INCREMENT, OpType.DECREMENT])
            amount = random.randint(1, 10)
            operations.append((rid, op_type, amount))

    # Shuffle to simulate concurrent creation
    random.shuffle(operations)

    print(f"  Broadcasting {len(operations)} operations...")
    for rid, op_type, amount in operations:
        op = net.create_and_broadcast(rid, op_type, amount)

    print(f"  Before drain: values={net.all_values()}")
    print(f"  Buffer status: {net.buffer_status()}")

    total_delivered = net.drain()
    print(f"  After drain ({total_delivered} deliveries): values={net.all_values()}")
    print(f"  Buffer status: {net.buffer_status()}")

    assert net.all_converged(), f"All replicas should converge: {net.all_values()}"

    # Verify the value is correct by summing all operations
    expected = 0
    for _, op_type, amount in operations:
        if op_type == OpType.INCREMENT:
            expected += amount
        else:
            expected -= amount

    actual_values = set(net.all_values().values())
    assert len(actual_values) == 1
    actual = actual_values.pop()
    assert actual == expected, f"Expected {expected}, got {actual}"

    print(f"  Final value: {actual} (correct)")
    print("  PASSED")


def test_network_heavy_reorder() -> None:
    """Stress test with many operations and high delay to maximize reordering."""
    print("\n[Test] OpBased network heavy reordering")
    print("-" * 40)

    random.seed(123)

    net = OpBasedNetwork(min_delay=1, max_delay=10)
    replica_ids = [f"R{i}" for i in range(5)]
    for rid in replica_ids:
        net.add_replica(rid)

    expected = 0
    for _ in range(20):
        rid = random.choice(replica_ids)
        op_type = random.choice([OpType.INCREMENT, OpType.DECREMENT])
        amount = random.randint(1, 100)
        net.create_and_broadcast(rid, op_type, amount)
        if op_type == OpType.INCREMENT:
            expected += amount
        else:
            expected -= amount

        # Advance time partially (not draining fully) to increase in-flight reordering
        net.tick()

    print(f"  Mid-test values: {net.all_values()}")
    print(f"  Mid-test buffers: {net.buffer_status()}")

    total = net.drain()
    print(f"  After drain ({total} deliveries): {net.all_values()}")

    assert net.all_converged(), f"Should converge: {net.all_values()}"

    actual = list(net.all_values().values())[0]
    assert actual == expected, f"Expected {expected}, got {actual}"
    print(f"  Final value: {actual} (correct)")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Exercise 3: Operation-based CRDTs")
    print("=" * 60)

    test_op_counter_basic()
    test_op_counter_commutative()
    test_causal_broadcast_basic()
    test_causal_broadcast_reorder()
    test_causal_broadcast_cross_replica()
    test_causal_broadcast_duplicate()
    test_network_convergence()
    test_network_heavy_reorder()

    print("\n" + "=" * 60)
    print("All Exercise 3 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
