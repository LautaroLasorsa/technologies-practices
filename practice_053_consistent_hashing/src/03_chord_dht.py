#!/usr/bin/env python3
"""
Exercise 3: Simplified Chord DHT with Finger Tables (~30 min).

Implement a simplified version of the Chord protocol (Stoica et al., SIGCOMM 2001)
to understand how distributed hash tables achieve O(log N) lookup using finger tables.

In real Chord, each node runs on a separate machine and communicates via RPC.
This implementation simulates the distributed system in a single process: all
nodes are Python objects, and "RPC" is a direct method call. The algorithms
are identical to the paper.

Key concepts:
  - m-bit identifier space: Nodes and keys live on a ring of size 2^m.
  - Finger table: Each node stores m entries. Entry i points to successor(n + 2^i).
    This exponential stride is the key to O(log N) lookup -- each hop at least
    halves the remaining distance to the target.
  - find_successor(id): The core lookup. Walks the finger table to find the
    node responsible for a given ID in O(log N) hops.
  - stabilize/fix_fingers: Maintenance protocols that keep routing tables
    correct as nodes join and leave.
"""
from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure sibling modules are importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hash_utils import save_plot, ensure_dirs, DATA_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# Chord constants
# ---------------------------------------------------------------------------

# Use a small m for visualization/debugging. Real Chord uses m=160 (SHA-1).
# With m=8, the ring has 256 positions -- enough for educational purposes
# with up to ~20 nodes, while keeping finger tables small and debuggable.
M = 8
RING_MOD = 2**M  # 256


def chord_hash(name: str) -> int:
    """Hash a node or key name to a position on the m-bit Chord ring.

    Uses SHA-256 truncated to M bits. In real Chord, this would be SHA-1
    applied to IP:port to get a 160-bit node ID.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:2], "big") % RING_MOD


def in_interval(x: int, a: int, b: int, inclusive_left: bool = False, inclusive_right: bool = True) -> bool:
    """Check if x is in the interval (a, b] on the Chord ring (mod RING_MOD).

    By default, the interval is half-open: (a, b] (exclusive left, inclusive right).
    This matches Chord's convention for key responsibility: node n is responsible
    for keys in (predecessor, n].

    Handles wrap-around: if a > b, the interval wraps around 0.

    Args:
        x: Position to check.
        a: Start of interval (exclusive by default).
        b: End of interval (inclusive by default).
        inclusive_left: If True, interval is [a, b] or [a, b) depending on inclusive_right.
        inclusive_right: If True, include b.
    """
    if a == b:
        # Full ring (only when a == b and we want to cover everything)
        return True

    if inclusive_left and x == a:
        return True
    if inclusive_right and x == b:
        return True

    if a < b:
        return a < x < b
    else:
        # Wrap-around: (a, RING_MOD) union [0, b)
        return x > a or x < b


# ---------------------------------------------------------------------------
# ChordNode
# ---------------------------------------------------------------------------

# TODO(human): Implement the ChordNode class.
#
# A ChordNode represents a single node in the Chord DHT. In a real system,
# each ChordNode would run on a separate machine. Here we simulate the
# distributed system in a single process.
#
# Each node maintains:
#   - self.id: int -- The node's position on the m-bit ring (0 to RING_MOD-1).
#   - self.finger: list[ChordNode | None] -- The finger table with M entries.
#       finger[i] = successor of (self.id + 2^i) mod RING_MOD.
#       This is the routing table that enables O(log N) lookup.
#       Entry 0 is the immediate successor (distance 1).
#       Entry 1 covers distance 2.
#       Entry i covers distance 2^i.
#       The exponential stride means each hop at least halves the
#       remaining distance to the target.
#   - self.predecessor: ChordNode | None -- The node immediately before this
#       one on the ring. Used by the stabilization protocol.
#   - self.data: dict[int, str] -- Key-value store for keys this node is
#       responsible for. Maps key_id -> value.
#
# The finger table is the brilliant insight of Chord. With N nodes on the
# ring, each node stores O(log N) = M entries, and lookup takes O(log N)
# hops. Compare this to a naive approach where each node only knows its
# successor: lookup would take O(N) hops (walking the entire ring).
#
# Key invariant: finger[0] is ALWAYS the immediate successor. Chord's
# stabilization protocol ensures this is always correct, even during
# concurrent joins. Other finger entries may be temporarily stale
# (they're fixed by fix_fingers), but finger[0] must be accurate
# for correctness (not just efficiency) of lookups.


class ChordNode:
    """A node in the Chord DHT."""

    def __init__(self, node_id: int) -> None:
        # TODO(human): Initialize the Chord node.
        #
        # Set:
        #   self.id = node_id
        #   self.finger = [None] * M   (M entries, all initially None)
        #   self.predecessor = None
        #   self.data = {}             (empty key-value store)
        #
        # In a real Chord implementation, the finger table starts as all-None
        # and is populated during the join protocol and stabilization.
        raise NotImplementedError("TODO(human): Implement ChordNode.__init__")

    @property
    def successor(self) -> ChordNode | None:
        """Shorthand for finger[0], the immediate successor.

        Provided -- no TODO. finger[0] = successor(self.id + 2^0) = successor(self.id + 1),
        which is the next node clockwise on the ring.
        """
        return self.finger[0]

    @successor.setter
    def successor(self, node: ChordNode | None) -> None:
        """Set the immediate successor (finger[0])."""
        self.finger[0] = node

    def find_successor(self, key_id: int) -> ChordNode:
        # TODO(human): Find the node responsible for the given key_id.
        #
        # This is the CORE ALGORITHM of Chord -- the O(log N) lookup.
        #
        # Algorithm (from Stoica et al. 2001, Figure 4):
        #
        # 1. If key_id falls in the interval (self.id, self.successor.id]
        #    (i.e., between this node and its successor, inclusive of successor),
        #    then self.successor is responsible for key_id. Return self.successor.
        #
        #    Use in_interval(key_id, self.id, self.successor.id) for this check.
        #
        # 2. Otherwise, find the closest preceding finger:
        #    Scan the finger table from entry M-1 down to 0. For each entry i,
        #    if finger[i] is not None and finger[i].id is in the interval
        #    (self.id, key_id) (exclusive both sides), then finger[i] is a
        #    good next hop -- it's the farthest finger that still precedes
        #    the target. Delegate the query to that node:
        #      return finger[i].find_successor(key_id)
        #
        #    Use in_interval(self.finger[i].id, self.id, key_id,
        #                    inclusive_left=False, inclusive_right=False)
        #
        # 3. If no preceding finger is found (shouldn't happen in a stable ring),
        #    return self.successor as a fallback.
        #
        # Why O(log N)?
        #   Each hop at least halves the distance to the target. Finger[i]
        #   covers distance 2^i, so the highest finger that precedes the
        #   target jumps at least half the remaining distance. After at most
        #   M = O(log RING_MOD) hops, we arrive at the predecessor of the
        #   target, and one more hop reaches the successor.
        #
        # NOTE: This recursive implementation mirrors the paper. In production,
        # each recursive call would be an RPC to a different machine. The
        # number of RPCs = number of hops = O(log N).
        raise NotImplementedError("TODO(human): Implement ChordNode.find_successor")

    def join(self, existing_node: ChordNode | None) -> None:
        # TODO(human): Join the Chord ring via an existing node.
        #
        # If existing_node is None, this node is creating a new ring
        # (it's the first and only node):
        #   - Set self.predecessor = None (or self, conventions vary)
        #   - Set self.successor = self (it's its own successor)
        #   - Fill all finger table entries with self.
        #
        # If existing_node is provided, this node is joining an existing ring:
        #   - Set self.predecessor = None (will be set by stabilize later)
        #   - Ask the existing node to find this node's successor:
        #       self.successor = existing_node.find_successor(self.id)
        #   - Leave other finger entries as None (fix_fingers will populate them)
        #
        # In a real system, the join also involves transferring keys from
        # the successor to the new node (keys in (predecessor, self.id]).
        # We handle that in transfer_keys_from_successor() below.
        #
        # The join protocol in Chord is designed to be safe even with
        # concurrent joins: stabilize() will eventually fix any
        # inconsistencies. The only requirement is that finger[0]
        # (successor) is correct after join.
        raise NotImplementedError("TODO(human): Implement ChordNode.join")

    def stabilize(self) -> None:
        # TODO(human): Periodic stabilization protocol.
        #
        # This is Chord's self-healing mechanism. It runs periodically on
        # every node and ensures that successor/predecessor pointers are
        # correct, even after nodes join or leave.
        #
        # Algorithm (from Stoica et al. 2001, Figure 7):
        #
        # 1. Let x = self.successor.predecessor
        #    (Ask our successor who it thinks its predecessor is.)
        #
        # 2. If x is not None and x.id is in the interval (self.id, self.successor.id)
        #    (exclusive both sides), then x is a better successor for us.
        #    This happens when a new node joined between us and our successor.
        #    Update: self.successor = x
        #
        #    Use in_interval(x.id, self.id, self.successor.id,
        #                    inclusive_left=False, inclusive_right=False)
        #
        # 3. Notify our successor that we exist:
        #    self.successor.notify(self)
        #
        # Why this works:
        #   When node X joins between A and B (A -> B becomes A -> X -> B),
        #   X sets its successor to B. When A stabilizes, it asks B for B's
        #   predecessor, which B will eventually report as X. A then updates
        #   its successor to X. Now A -> X -> B is correct.
        raise NotImplementedError("TODO(human): Implement ChordNode.stabilize")

    def notify(self, candidate: ChordNode) -> None:
        # TODO(human): Notification from a node that thinks it might be our predecessor.
        #
        # Called by candidate.stabilize() as: self.successor.notify(self).
        #
        # Algorithm:
        # If self.predecessor is None, OR candidate.id is in the interval
        # (self.predecessor.id, self.id) (exclusive both sides), then
        # candidate is a better predecessor. Update: self.predecessor = candidate.
        #
        # Use in_interval(candidate.id, self.predecessor.id, self.id,
        #                 inclusive_left=False, inclusive_right=False)
        #
        # Why this is separate from stabilize:
        #   stabilize() fixes successor pointers (forward links).
        #   notify() fixes predecessor pointers (backward links).
        #   Together, they maintain the bidirectional ring invariant.
        raise NotImplementedError("TODO(human): Implement ChordNode.notify")

    def fix_fingers(self) -> None:
        # TODO(human): Recompute all finger table entries.
        #
        # In real Chord, fix_fingers is called periodically and fixes ONE
        # random finger per invocation (to spread the cost over time).
        # For simplicity, we fix ALL fingers in one call.
        #
        # Algorithm:
        # For i in range(M):
        #   target = (self.id + 2**i) % RING_MOD
        #   self.finger[i] = self.find_successor(target)
        #
        # The finger table entry i should point to the successor of
        # (self.id + 2^i) mod RING_MOD. This is the first node at or
        # after the position that is 2^i steps clockwise from self.
        #
        # Note: finger[0] is the immediate successor, which should already
        # be correct from stabilize(). fix_fingers ensures that entries
        # 1 through M-1 are also up to date.
        #
        # Why fix all fingers?
        #   In this simplified single-process simulation, there's no cost
        #   to fixing all fingers at once. In a real distributed system,
        #   each fix requires an RPC (find_successor), so spreading them
        #   out reduces per-round network traffic.
        raise NotImplementedError("TODO(human): Implement ChordNode.fix_fingers")

    def store(self, key_id: int, value: str) -> ChordNode:
        """Store a key-value pair in the DHT.

        Provided -- no TODO. Routes the key to the responsible node via
        find_successor, then stores the value there. Returns the node
        that stored it.
        """
        responsible = self.find_successor(key_id)
        responsible.data[key_id] = value
        return responsible

    def lookup(self, key_id: int) -> tuple[str | None, int]:
        """Look up a key in the DHT, counting hops.

        Provided -- no TODO. Returns (value_or_None, hop_count).
        Reimplements find_successor iteratively to count hops.
        """
        hops = 0
        current = self

        # Iterative version of find_successor for hop counting
        while True:
            if current.successor is None:
                return None, hops

            if in_interval(key_id, current.id, current.successor.id):
                # Found: successor is responsible
                value = current.successor.data.get(key_id)
                return value, hops + 1

            # Find closest preceding finger
            forwarded = False
            for i in range(M - 1, -1, -1):
                f = current.finger[i]
                if f is not None and in_interval(
                    f.id, current.id, key_id,
                    inclusive_left=False, inclusive_right=False,
                ):
                    current = f
                    hops += 1
                    forwarded = True
                    break

            if not forwarded:
                # Fallback to successor
                value = current.successor.data.get(key_id) if current.successor else None
                return value, hops + 1

    def __repr__(self) -> str:
        succ = self.successor.id if self.successor else None
        pred = self.predecessor.id if self.predecessor else None
        return f"ChordNode(id={self.id}, succ={succ}, pred={pred}, keys={len(self.data)})"


# ---------------------------------------------------------------------------
# Ring builder: create and stabilize a Chord ring
# ---------------------------------------------------------------------------

def build_chord_ring(node_names: list[str], stabilize_rounds: int = 10) -> list[ChordNode]:
    """Create a Chord ring from a list of node names.

    Provided -- no TODO. Uses your ChordNode implementation.

    1. Hash each name to get a node ID.
    2. Create the first node (creates a new ring).
    3. Each subsequent node joins via the first node.
    4. Run stabilize_rounds rounds of stabilize + fix_fingers on all nodes.
    5. Return the list of nodes sorted by ID.
    """
    if not node_names:
        return []

    # Create nodes with hashed IDs
    nodes: list[ChordNode] = []
    seen_ids: set[int] = set()
    for name in node_names:
        nid = chord_hash(name)
        # Handle ID collisions (rare with good hash, but possible with small M)
        while nid in seen_ids:
            nid = (nid + 1) % RING_MOD
        seen_ids.add(nid)
        nodes.append(ChordNode(nid))

    # Sort by ID for deterministic ring order
    nodes.sort(key=lambda n: n.id)

    # First node creates the ring
    nodes[0].join(None)

    # Remaining nodes join via the first node
    for node in nodes[1:]:
        node.join(nodes[0])

    # Stabilization rounds
    for _ in range(stabilize_rounds):
        for node in nodes:
            node.stabilize()
        for node in nodes:
            node.fix_fingers()

    return nodes


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_chord_ring(nodes: list[ChordNode], title: str = "Chord DHT Ring") -> plt.Figure:
    """Visualize the Chord ring showing nodes, successor links, and finger table arcs.

    Provided -- no TODO. Creates a clear visualization of the ring topology.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Draw the ring
    theta = np.linspace(0, 2 * np.pi, 360)
    ring_r = 1.0
    ax.plot(ring_r * np.cos(theta), ring_r * np.sin(theta), "k-", linewidth=1, alpha=0.2)

    # Node positions
    node_angles = {}
    for node in nodes:
        angle = 2 * np.pi * node.id / RING_MOD
        node_angles[node.id] = angle

    # Draw finger table arcs (faint, for one node as example)
    example_node = nodes[0]
    for i in range(M):
        f = example_node.finger[i]
        if f and f.id != example_node.id:
            a1 = node_angles[example_node.id]
            a2 = node_angles.get(f.id)
            if a2 is not None:
                x1, y1 = ring_r * np.cos(a1), ring_r * np.sin(a1)
                x2, y2 = ring_r * np.cos(a2), ring_r * np.sin(a2)
                ax.annotate(
                    "", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="orange", alpha=0.3, lw=0.8),
                )

    # Draw successor arcs (thicker)
    for node in nodes:
        if node.successor and node.successor.id != node.id:
            a1 = node_angles[node.id]
            a2 = node_angles.get(node.successor.id)
            if a2 is not None:
                x1, y1 = ring_r * np.cos(a1), ring_r * np.sin(a1)
                x2, y2 = ring_r * np.cos(a2), ring_r * np.sin(a2)
                ax.annotate(
                    "", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="blue", alpha=0.5, lw=1.5),
                )

    # Draw nodes
    for node in nodes:
        angle = node_angles[node.id]
        x, y = ring_r * np.cos(angle), ring_r * np.sin(angle)
        ax.plot(x, y, "o", color="red", markersize=12, markeredgecolor="black", zorder=5)

        # Label
        label_r = ring_r + 0.15
        ax.text(
            label_r * np.cos(angle), label_r * np.sin(angle),
            f"{node.id}", fontsize=9, fontweight="bold",
            ha="center", va="center",
        )

    # Legend
    ax.plot([], [], "o", color="red", markersize=8, label="Chord Node")
    ax.plot([], [], "-", color="blue", alpha=0.5, linewidth=1.5, label="Successor link")
    ax.plot([], [], "-", color="orange", alpha=0.3, linewidth=0.8, label=f"Finger arcs (node {example_node.id})")
    ax.legend(loc="upper right", fontsize=10)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\n{len(nodes)} nodes, m={M}, ring size={RING_MOD}", fontsize=13)
    ax.axis("off")

    fig.tight_layout()
    return fig


def plot_hop_distribution(hop_counts: list[int], num_nodes: int) -> plt.Figure:
    """Plot the distribution of lookup hop counts.

    Provided -- no TODO.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    max_hops = max(hop_counts) if hop_counts else 1
    bins = range(0, max_hops + 2)
    ax.hist(hop_counts, bins=bins, color="steelblue", edgecolor="black", alpha=0.8, align="left")

    avg_hops = sum(hop_counts) / len(hop_counts) if hop_counts else 0
    theoretical = np.log2(max(num_nodes, 1)) / 2  # Expected: ~(log2 N)/2
    ax.axvline(avg_hops, color="red", linestyle="--", linewidth=2, label=f"Avg: {avg_hops:.2f}")
    ax.axvline(theoretical, color="green", linestyle=":", linewidth=2,
               label=f"Theoretical ~log2(N)/2: {theoretical:.2f}")

    ax.set_xlabel("Number of Hops")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Chord Lookup Hop Distribution ({num_nodes} nodes, {len(hop_counts)} lookups)")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()

    print("=" * 60)
    print("Exercise 3: Chord DHT with Finger Tables")
    print("=" * 60)

    # --- Part A: Create the ring ---
    print(f"\n--- Part A: Creating Chord Ring (m={M}, ring size={RING_MOD}) ---")
    node_names = [f"node-{i}" for i in range(16)]
    nodes = build_chord_ring(node_names, stabilize_rounds=20)

    print(f"Created {len(nodes)} nodes on the ring:")
    for node in nodes:
        print(f"  {node}")

    # --- Part B: Verify ring structure ---
    print("\n--- Part B: Verify Ring Structure ---")
    print("Checking successor/predecessor chain...")
    for i, node in enumerate(nodes):
        expected_succ = nodes[(i + 1) % len(nodes)]
        expected_pred = nodes[(i - 1) % len(nodes)]
        succ_ok = node.successor is not None and node.successor.id == expected_succ.id
        pred_ok = node.predecessor is not None and node.predecessor.id == expected_pred.id
        status = "OK" if (succ_ok and pred_ok) else "ISSUE"
        if not succ_ok or not pred_ok:
            print(f"  Node {node.id}: {status} (succ={node.successor}, pred={node.predecessor})")
    print("  Ring structure verified.")

    # --- Part C: Finger table inspection ---
    print(f"\n--- Part C: Finger Table for Node {nodes[0].id} ---")
    example = nodes[0]
    print(f"  {'Entry':>5} | {'Target':>8} | {'Finger ID':>10}")
    print("  " + "-" * 35)
    for i in range(M):
        target = (example.id + 2**i) % RING_MOD
        finger = example.finger[i]
        fid = finger.id if finger else "None"
        print(f"  {i:>5} | {target:>8} | {fid:>10}")

    # --- Part D: Store and lookup keys ---
    print("\n--- Part D: Store and Lookup Keys ---")
    test_keys = [(chord_hash(f"key-{i}"), f"value-{i}") for i in range(20)]

    # Store keys from node 0
    for key_id, value in test_keys:
        responsible = nodes[0].store(key_id, value)
        print(f"  Stored key {key_id:>3} -> node {responsible.id:>3}")

    # Lookup keys from various nodes
    print("\nLookup from different starting nodes:")
    all_hops: list[int] = []
    for key_id, expected_value in test_keys[:5]:
        start = random.choice(nodes)
        value, hops = start.lookup(key_id)
        all_hops.append(hops)
        status = "OK" if value == expected_value else f"MISMATCH (got {value})"
        print(f"  key={key_id:>3} from node {start.id:>3}: hops={hops}, {status}")

    # --- Part E: Hop count analysis ---
    print("\n--- Part E: Hop Count Analysis (1000 lookups) ---")
    hop_counts: list[int] = []
    for i in range(1000):
        key_id = chord_hash(f"lookup-key-{i}")
        start = random.choice(nodes)
        _, hops = start.lookup(key_id)
        hop_counts.append(hops)

    avg = sum(hop_counts) / len(hop_counts)
    max_h = max(hop_counts)
    theoretical = np.log2(len(nodes)) / 2
    print(f"  Average hops:     {avg:.2f}")
    print(f"  Maximum hops:     {max_h}")
    print(f"  Theoretical avg:  ~{theoretical:.2f} (log2(N)/2 = log2({len(nodes)})/2)")
    print(f"  O(log N) bound:   {np.log2(len(nodes)):.2f}")

    # --- Part F: Visualizations ---
    print("\n--- Part F: Visualizations ---")
    fig1 = visualize_chord_ring(nodes)
    save_plot(fig1, "03_chord_ring.png")
    plt.close(fig1)

    fig2 = plot_hop_distribution(hop_counts, len(nodes))
    save_plot(fig2, "03_hop_distribution.png")
    plt.close(fig2)

    print("\n" + "=" * 60)
    print("Exercise 3 complete. Check plots/ for visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
