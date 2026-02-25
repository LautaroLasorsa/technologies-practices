#!/usr/bin/env python3
"""
Exercise 1: Consistent Hash Ring with Virtual Nodes (~25 min).

Implement the core consistent hashing data structure: a ring of virtual node
positions with O(log N) binary-search lookup. This is the foundation used by
Amazon DynamoDB, Apache Cassandra, Memcached (libketama), and Akamai CDN.

The ring maps both keys and server identifiers to positions on a circular
space of size 2^32. Each physical server is placed at multiple positions
("virtual nodes") for load balance. To find a key's server, find the nearest
position clockwise via bisect_right.
"""
from __future__ import annotations

import bisect
import sys
from collections import defaultdict
from pathlib import Path

# Ensure sibling modules are importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hash_utils import (  # noqa: E402
    RING_SIZE,
    hash_to_ring,
    format_ring_pos,
    load_balance_stats,
    draw_ring,
    draw_load_bar_chart,
    save_plot,
    ensure_dirs,
)


# ---------------------------------------------------------------------------
# ConsistentHashRing
# ---------------------------------------------------------------------------

# TODO(human): Implement the ConsistentHashRing class.
#
# This is the foundational data structure for consistent hashing. You are
# building the exact algorithm described in Karger et al. (1997) extended
# with virtual nodes (as used in Amazon Dynamo, 2007).
#
# Internal data structures you need:
#
#   self.num_virtual_nodes: int
#       Number of virtual nodes per physical server. More vnodes = better
#       load balance but more memory and slightly slower lookup. Production
#       systems like Dynamo use ~150.
#
#   self._ring_positions: list[int]
#       Sorted list of all virtual node positions on the ring.
#       Kept sorted for O(log N) binary search via bisect.
#
#   self._position_to_node: dict[int, str]
#       Maps each ring position -> the physical node name that owns it.
#       When a lookup finds the nearest clockwise position, this dict
#       tells you which physical server it belongs to.
#
#   self._node_positions: dict[str, list[int]]
#       Maps each physical node name -> list of its virtual node positions.
#       Needed for remove_node (must know which positions to delete).
#
# How virtual nodes work:
#   For a physical node "server-A" with V=3 virtual nodes, hash three
#   distinct strings: hash("server-A:0"), hash("server-A:1"), hash("server-A:2").
#   Each produces a different position on the ring. This scatters the node's
#   "ownership" across the ring, smoothing out the arc-length distribution.
#
# Lookup algorithm (get_node):
#   1. Hash the key to a ring position: pos = hash_to_ring(key)
#   2. Use bisect_right(self._ring_positions, pos) to find the index of
#      the first ring position STRICTLY GREATER than pos.
#   3. If the index equals len(self._ring_positions), wrap around to index 0
#      (the ring is circular -- keys past the last position belong to the
#      first position, like 11:55 PM wrapping to 12:00 AM on a clock).
#   4. Return self._position_to_node[self._ring_positions[index]].
#
# Why bisect_right and not bisect_left?
#   bisect_right returns the insertion point AFTER any existing equal values.
#   If a key hashes to exactly the same position as a vnode, bisect_right
#   moves past it, which means the key is assigned to the NEXT clockwise
#   node. This matches the convention that each node "owns" the arc
#   (predecessor, self] -- i.e., keys up to and including the node's position
#   belong to it, so a key AT a node position is assigned to that node's
#   successor. In practice, exact collisions are astronomically rare with
#   32-bit hashes, but correctness matters.


class ConsistentHashRing:
    """Consistent hash ring with virtual nodes.

    Usage:
        ring = ConsistentHashRing(num_virtual_nodes=150)
        ring.add_node("server-A")
        ring.add_node("server-B")
        ring.add_node("server-C")
        server = ring.get_node("user:12345")  # -> "server-B"
    """

    def __init__(self, num_virtual_nodes: int = 150) -> None:
        # TODO(human): Initialize the ring data structures.
        #
        # Store num_virtual_nodes and create four empty structures:
        #   - self._ring_positions: list[int] = []
        #   - self._position_to_node: dict[int, str] = {}
        #   - self._node_positions: dict[str, list[int]] = {}
        #   - self.num_virtual_nodes = num_virtual_nodes
        #
        # The ring starts empty -- nodes are added via add_node().
        raise NotImplementedError("TODO(human): Implement ConsistentHashRing.__init__")

    def add_node(self, node: str) -> None:
        # TODO(human): Add a physical node to the ring.
        #
        # Steps:
        # 1. Generate num_virtual_nodes positions for this node by hashing
        #    f"{node}:{i}" for i in range(self.num_virtual_nodes).
        #    Use hash_to_ring() from hash_utils.
        #
        # 2. For each position:
        #    a. Use bisect.insort() to insert into self._ring_positions
        #       (maintains sorted order for O(log N) lookup).
        #    b. Map position -> node in self._position_to_node.
        #
        # 3. Store all positions in self._node_positions[node].
        #
        # Edge case: if two virtual nodes from different physical nodes
        # hash to the same position, the second one overwrites the first
        # in _position_to_node. With 32-bit hashes and < 1000 vnodes total,
        # collision probability is negligible (birthday paradox: ~50% at
        # ~77,000 entries). For production systems, handle this or use
        # 128-bit hashes.
        raise NotImplementedError("TODO(human): Implement ConsistentHashRing.add_node")

    def remove_node(self, node: str) -> None:
        # TODO(human): Remove a physical node and all its virtual nodes from the ring.
        #
        # Steps:
        # 1. Look up the node's positions in self._node_positions[node].
        # 2. For each position:
        #    a. Remove from self._position_to_node.
        #    b. Remove from self._ring_positions.
        #       IMPORTANT: self._ring_positions.remove(pos) is O(N) per call.
        #       For a cleaner approach, collect all positions to remove, then
        #       rebuild _ring_positions = sorted list of remaining positions.
        #       Or remove one by one -- for this exercise either is fine.
        # 3. Delete the entry from self._node_positions.
        #
        # After removal, keys that were assigned to this node will
        # automatically be reassigned to the next clockwise node --
        # this is the key migration property. Only keys in the removed
        # node's arcs move; all other assignments stay the same.
        raise NotImplementedError("TODO(human): Implement ConsistentHashRing.remove_node")

    def get_node(self, key: str) -> str | None:
        # TODO(human): Find which physical node owns the given key.
        #
        # Algorithm:
        # 1. If the ring is empty, return None.
        # 2. Hash the key: pos = hash_to_ring(key)
        # 3. Find the insertion point: idx = bisect.bisect_right(self._ring_positions, pos)
        # 4. Wrap around: if idx == len(self._ring_positions), set idx = 0
        # 5. Return self._position_to_node[self._ring_positions[idx]]
        #
        # This is O(log M) where M = total virtual nodes on the ring.
        # For 10 servers with 150 vnodes each, that's log2(1500) ~ 11 comparisons.
        raise NotImplementedError("TODO(human): Implement ConsistentHashRing.get_node")

    def get_distribution(self, keys: list[str]) -> dict[str, int]:
        """Count how many keys map to each physical node.

        Provided for you -- no TODO. Uses get_node() which you implement above.
        """
        counts: dict[str, int] = defaultdict(int)
        for key in keys:
            node = self.get_node(key)
            if node:
                counts[node] += 1
        return dict(counts)

    @property
    def nodes(self) -> list[str]:
        """Return list of physical node names currently on the ring."""
        return list(self._node_positions.keys())

    @property
    def total_positions(self) -> int:
        """Return total number of virtual node positions on the ring."""
        return len(self._ring_positions)

    def __repr__(self) -> str:
        n = len(self._node_positions)
        total = self.total_positions
        return f"ConsistentHashRing(nodes={n}, vnodes_per_node={self.num_virtual_nodes}, total_positions={total})"


# ---------------------------------------------------------------------------
# Visualization: Ring with key assignments
# ---------------------------------------------------------------------------

# TODO(human): Implement visualize_ring().
#
# This function creates a visual representation of the hash ring, showing
# where nodes sit on the ring and how keys are distributed among them.
# Seeing the ring makes the abstract concept concrete -- you can observe
# how virtual nodes scatter across the ring and how keys cluster.
#
# Steps:
# 1. Create a ConsistentHashRing with the given nodes and num_virtual_nodes.
# 2. Assign each key to its node via ring.get_node().
# 3. Use draw_ring() from hash_utils to produce the matplotlib figure.
# 4. Use draw_load_bar_chart() to show the load distribution.
# 5. Save both plots with save_plot().
# 6. Return the ring for further use.
#
# This visualization is the "aha" moment for consistent hashing --
# you can literally see that adding a node only affects its neighbors.

def visualize_ring(
    nodes: list[str],
    keys: list[str],
    num_virtual_nodes: int = 150,
    plot_prefix: str = "01",
) -> ConsistentHashRing:
    # TODO(human): Build the ring, assign keys, visualize.
    #
    # 1. Create ring = ConsistentHashRing(num_virtual_nodes)
    # 2. Add all nodes with ring.add_node(node) for each node.
    # 3. Build key_assignments = {key: ring.get_node(key) for key in keys}
    # 4. Build node_positions = ring._node_positions (the dict of node -> positions)
    # 5. Call draw_ring(node_positions, key_assignments, title=...) -> fig
    # 6. save_plot(fig, f"{plot_prefix}_ring.png")
    # 7. plt.close(fig)
    #
    # 8. Get distribution = ring.get_distribution(keys)
    # 9. Call draw_load_bar_chart(distribution, title=...) -> fig2
    # 10. save_plot(fig2, f"{plot_prefix}_load_distribution.png")
    # 11. plt.close(fig2)
    #
    # 12. Return ring
    raise NotImplementedError("TODO(human): Implement visualize_ring")


# ---------------------------------------------------------------------------
# Experiment: Effect of virtual node count on balance
# ---------------------------------------------------------------------------

def experiment_vnode_count(
    nodes: list[str],
    keys: list[str],
    vnode_counts: list[int] | None = None,
) -> None:
    """Test how the number of virtual nodes affects load balance.

    Provided for you -- uses ConsistentHashRing (which you implement).
    Creates a multi-panel plot showing CV (coefficient of variation) vs
    number of virtual nodes.
    """
    if vnode_counts is None:
        vnode_counts = [1, 5, 10, 25, 50, 100, 150, 200, 300, 500]

    cvs: list[float] = []
    max_mean_ratios: list[float] = []

    print(f"\n{'Vnodes':>8} | {'CV':>8} | {'Max/Mean':>10} | {'Std Dev':>10}")
    print("-" * 48)

    for v in vnode_counts:
        ring = ConsistentHashRing(num_virtual_nodes=v)
        for node in nodes:
            ring.add_node(node)

        dist = ring.get_distribution(keys)
        stats = load_balance_stats(dist)
        cvs.append(stats["cv"])
        max_mean_ratios.append(stats["max_mean_ratio"])
        print(f"{v:>8} | {stats['cv']:>8.4f} | {stats['max_mean_ratio']:>10.3f}x | {stats['std']:>10.1f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(vnode_counts, cvs, "bo-", linewidth=2, markersize=6)
    ax1.set_xlabel("Virtual Nodes per Server")
    ax1.set_ylabel("Coefficient of Variation (lower = better)")
    ax1.set_title("Load Balance vs Virtual Node Count")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="green", linestyle="--", alpha=0.5, label="Perfect balance")
    ax1.legend()

    ax2.plot(vnode_counts, max_mean_ratios, "rs-", linewidth=2, markersize=6)
    ax2.set_xlabel("Virtual Nodes per Server")
    ax2.set_ylabel("Max / Mean Load Ratio (1.0 = perfect)")
    ax2.set_title("Max Overload vs Virtual Node Count")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Perfect balance")
    ax2.legend()

    fig.suptitle(f"Effect of Virtual Nodes ({len(nodes)} servers, {len(keys)} keys)", fontsize=14)
    fig.tight_layout()
    save_plot(fig, "01_vnode_balance.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()

    print("=" * 60)
    print("Exercise 1: Consistent Hash Ring with Virtual Nodes")
    print("=" * 60)

    # --- Part A: Basic ring operations ---
    print("\n--- Part A: Basic Ring Operations ---")
    servers = [f"server-{chr(65 + i)}" for i in range(5)]  # server-A .. server-E
    print(f"Servers: {servers}")

    ring = ConsistentHashRing(num_virtual_nodes=150)
    for s in servers:
        ring.add_node(s)
    print(f"Ring: {ring}")

    # Look up some keys
    test_keys = [f"user:{i}" for i in range(10)]
    print("\nKey -> Server assignments:")
    for key in test_keys:
        node = ring.get_node(key)
        print(f"  {key:>12} -> {node}")

    # --- Part B: Load distribution ---
    print("\n--- Part B: Load Distribution (10,000 keys) ---")
    keys = [f"key-{i}" for i in range(10_000)]
    dist = ring.get_distribution(keys)
    stats = load_balance_stats(dist)

    for s in sorted(dist.keys()):
        pct = dist[s] / len(keys) * 100
        print(f"  {s}: {dist[s]:>5} keys ({pct:.1f}%)")

    print(f"\n  CV = {stats['cv']:.4f} (0 = perfect balance)")
    print(f"  Max/Mean = {stats['max_mean_ratio']:.3f}x")

    # --- Part C: Visualize the ring ---
    print("\n--- Part C: Ring Visualization ---")
    # Use fewer keys for the ring diagram (readability)
    visualize_ring(servers, keys[:200], num_virtual_nodes=150)

    # --- Part D: Node removal and key migration ---
    print("\n--- Part D: Node Removal ---")
    assignments_before = {key: ring.get_node(key) for key in keys}

    removed = servers[2]  # Remove server-C
    print(f"Removing {removed}...")
    ring.remove_node(removed)
    print(f"Ring after removal: {ring}")

    assignments_after = {key: ring.get_node(key) for key in keys}
    moved = sum(1 for k in keys if assignments_before[k] != assignments_after[k])
    pct_moved = moved / len(keys) * 100

    print(f"Keys moved: {moved}/{len(keys)} ({pct_moved:.1f}%)")
    print(f"Theoretical ideal: {len(keys) / len(servers):.0f} ({100 / len(servers):.1f}%)")

    # --- Part E: Virtual node count experiment ---
    print("\n--- Part E: Effect of Virtual Node Count on Balance ---")
    experiment_vnode_count(servers, keys)

    print("\n" + "=" * 60)
    print("Exercise 1 complete. Check plots/ for visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
