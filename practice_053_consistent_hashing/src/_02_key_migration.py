#!/usr/bin/env python3
"""
Exercise 2: Key Migration Analysis (~20 min).

Quantify the key migration cost when servers join or leave the cluster.
Compare naive modulo hashing (hash(key) % N) against consistent hashing
to empirically verify the theoretical migration bounds:

  - Naive modulo:      ~K * (N-1)/N keys move when adding 1 server to N
  - Consistent hashing: ~K / (N+1) keys move when adding 1 server to N

This is the core value proposition of consistent hashing -- the dramatic
reduction in data movement during topology changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure sibling modules are importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hash_utils import (  # noqa: E402
    RING_SIZE,
    hash_to_ring,
    save_plot,
    ensure_dirs,
    DATA_DIR,
)
from consistent_hash_ring import ConsistentHashRing  # noqa: E402


# ---------------------------------------------------------------------------
# Naive modulo hashing
# ---------------------------------------------------------------------------

def naive_modulo_assign(key: str, num_servers: int) -> int:
    """Assign a key to a server index using hash(key) % N.

    Provided for you -- this is the naive approach that consistent hashing
    improves upon. Note how the server index changes for almost every key
    when num_servers changes by 1.
    """
    return hash_to_ring(key) % num_servers


# ---------------------------------------------------------------------------
# Key migration measurement
# ---------------------------------------------------------------------------

# TODO(human): Implement measure_key_migration().
#
# This function is the heart of the exercise. It answers: "When we add one
# server to a cluster of N, what fraction of keys must move?"
#
# For NAIVE modulo hashing:
#   1. Compute assignment_before = {key: hash_to_ring(key) % N for each key}
#   2. Compute assignment_after  = {key: hash_to_ring(key) % (N+1) for each key}
#   3. Count how many keys have a different assignment.
#   4. The fraction should be approximately (N) / (N+1), which is close to 1
#      for large N. For N=10, that's ~90.9% of keys moving!
#
# For CONSISTENT hashing:
#   1. Create a ConsistentHashRing with servers ["node-0", ..., "node-(N-1)"]
#      and the given num_virtual_nodes.
#   2. Record assignment_before = {key: ring.get_node(key) for each key}
#   3. Add one more server ("node-N") to the ring.
#   4. Record assignment_after = {key: ring.get_node(key) for each key}
#   5. Count how many keys moved (different assignment).
#   6. The fraction should be approximately 1/(N+1). For N=10, that's ~9.1%.
#
# Why this matters:
#   In a distributed cache with 10 servers and 1 million keys, adding one
#   server with naive hashing invalidates ~909,000 cache entries (thundering
#   herd to the backend). Consistent hashing invalidates only ~91,000 --
#   a 10x reduction. For databases, the difference is even more critical:
#   909,000 key migrations vs 91,000 means 10x less network traffic,
#   10x less disk I/O, and 10x faster rebalancing.
#
# Return format:
#   Return a dict with these keys:
#     "naive_moved": int       -- number of keys moved under naive hashing
#     "naive_fraction": float  -- fraction of keys moved (0.0 to 1.0)
#     "consistent_moved": int  -- number of keys moved under consistent hashing
#     "consistent_fraction": float  -- fraction of keys moved
#     "num_keys": int          -- total number of keys
#     "num_servers_before": int -- N
#     "num_servers_after": int  -- N + 1

def measure_key_migration(
    num_servers: int,
    keys: list[str],
    num_virtual_nodes: int = 150,
) -> dict[str, int | float]:
    # TODO(human): Measure key migration for both naive and consistent hashing.
    #
    # Follow the algorithm described above. Specifically:
    #
    # NAIVE:
    #   before_naive = {key: naive_modulo_assign(key, num_servers) for key in keys}
    #   after_naive  = {key: naive_modulo_assign(key, num_servers + 1) for key in keys}
    #   naive_moved  = sum(1 for k in keys if before_naive[k] != after_naive[k])
    #
    # CONSISTENT:
    #   Create ring with num_servers nodes, record assignments.
    #   Add one node, record assignments again. Count differences.
    #
    # Return the result dict described above.
    raise NotImplementedError("TODO(human): Implement measure_key_migration")


# ---------------------------------------------------------------------------
# Multi-scenario migration comparison
# ---------------------------------------------------------------------------

# TODO(human): Implement compare_naive_vs_consistent().
#
# Run measure_key_migration for multiple cluster sizes to show that the
# migration benefit grows with cluster size. Specifically:
#
# For each N in server_counts (e.g., [3, 5, 10, 15, 20, 30, 50]):
#   1. Call measure_key_migration(N, keys, num_virtual_nodes)
#   2. Collect the results.
#   3. Also compute the THEORETICAL predictions:
#      - Naive theoretical:      N / (N + 1)
#      - Consistent theoretical: 1 / (N + 1)
#
# Then generate a comparison bar chart with:
#   - X axis: number of servers (N)
#   - Y axis: fraction of keys moved
#   - Four bars per N: naive (measured), naive (theoretical),
#     consistent (measured), consistent (theoretical)
#
# Why multiple cluster sizes?
#   The relative advantage of consistent hashing increases with N.
#   At N=3: naive moves 75% vs consistent's 25% (3x better).
#   At N=50: naive moves 98% vs consistent's 2% (49x better).
#   Seeing this progression makes the scaling argument visceral.
#
# Save the plot to plots/02_migration_comparison.png.
# Save the raw data to data/02_migration_results.json.

def compare_naive_vs_consistent(
    keys: list[str],
    server_counts: list[int] | None = None,
    num_virtual_nodes: int = 150,
) -> list[dict[str, int | float]]:
    # TODO(human): Run migration analysis for multiple cluster sizes and plot.
    #
    # Steps:
    # 1. If server_counts is None, use [3, 5, 10, 15, 20, 30, 50].
    # 2. For each N in server_counts, call measure_key_migration(N, keys, num_virtual_nodes).
    #    Collect all results in a list.
    # 3. Print a summary table:
    #      N | Naive Moved% | Consistent Moved% | Naive Theory | Consistent Theory
    # 4. Generate the comparison bar chart:
    #    - Use matplotlib grouped bar chart (4 bars per cluster size).
    #    - Color naive bars red/light-red, consistent bars blue/light-blue.
    #    - Include a legend.
    #    - Title: "Key Migration: Naive Modulo vs Consistent Hashing"
    # 5. save_plot(fig, "02_migration_comparison.png")
    # 6. Save results to data/02_migration_results.json using json.dump.
    # 7. Return the list of result dicts.
    #
    # Hint for grouped bar chart:
    #   x = np.arange(len(server_counts))
    #   width = 0.2
    #   ax.bar(x - 1.5*width, naive_measured, width, label="Naive (measured)")
    #   ax.bar(x - 0.5*width, naive_theory, width, label="Naive (theoretical)")
    #   ax.bar(x + 0.5*width, consistent_measured, width, label="Consistent (measured)")
    #   ax.bar(x + 1.5*width, consistent_theory, width, label="Consistent (theoretical)")
    raise NotImplementedError("TODO(human): Implement compare_naive_vs_consistent")


# ---------------------------------------------------------------------------
# Experiment: Migration when removing a node
# ---------------------------------------------------------------------------

def experiment_node_removal(
    keys: list[str],
    num_servers: int = 10,
    num_virtual_nodes: int = 150,
) -> dict[str, int | float]:
    """Measure migration when REMOVING a server (not just adding).

    Provided for you -- uses ConsistentHashRing (which you implemented).

    When a server is removed from a consistent hash ring, only the keys
    that were assigned to that server need to move. They migrate to the
    next clockwise neighbor. The number of moved keys equals the number
    of keys the removed server was responsible for: approximately K/N.

    For naive modulo hashing, removing a server (N -> N-1) again causes
    nearly all keys to be reshuffled.
    """
    servers = [f"node-{i}" for i in range(num_servers)]

    # Consistent hashing: remove one node
    ring = ConsistentHashRing(num_virtual_nodes=num_virtual_nodes)
    for s in servers:
        ring.add_node(s)

    before_consistent = {key: ring.get_node(key) for key in keys}
    ring.remove_node(servers[-1])  # Remove last server
    after_consistent = {key: ring.get_node(key) for key in keys}

    consistent_moved = sum(1 for k in keys if before_consistent[k] != after_consistent[k])

    # Naive: N -> N-1
    before_naive = {key: naive_modulo_assign(key, num_servers) for key in keys}
    after_naive = {key: naive_modulo_assign(key, num_servers - 1) for key in keys}
    naive_moved = sum(1 for k in keys if before_naive[k] != after_naive[k])

    return {
        "num_servers_before": num_servers,
        "num_servers_after": num_servers - 1,
        "num_keys": len(keys),
        "naive_moved": naive_moved,
        "naive_fraction": naive_moved / len(keys),
        "consistent_moved": consistent_moved,
        "consistent_fraction": consistent_moved / len(keys),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()

    print("=" * 60)
    print("Exercise 2: Key Migration Analysis")
    print("=" * 60)

    keys = [f"key-{i}" for i in range(100_000)]

    # --- Part A: Single migration measurement ---
    print("\n--- Part A: Single Migration (10 -> 11 servers) ---")
    result = measure_key_migration(10, keys)
    print(f"  Naive modulo:      {result['naive_moved']:>6} keys moved ({result['naive_fraction']:.1%})")
    print(f"  Consistent hashing: {result['consistent_moved']:>6} keys moved ({result['consistent_fraction']:.1%})")
    print(f"  Theoretical naive:      {10/11:.1%}")
    print(f"  Theoretical consistent: {1/11:.1%}")
    print(f"  Improvement: {result['naive_fraction'] / max(result['consistent_fraction'], 1e-10):.1f}x fewer keys moved")

    # --- Part B: Multi-scenario comparison ---
    print("\n--- Part B: Multi-Scenario Comparison ---")
    results = compare_naive_vs_consistent(keys)

    # --- Part C: Node removal ---
    print("\n--- Part C: Node Removal (10 -> 9 servers) ---")
    removal = experiment_node_removal(keys, num_servers=10)
    print(f"  Naive modulo:      {removal['naive_moved']:>6} keys moved ({removal['naive_fraction']:.1%})")
    print(f"  Consistent hashing: {removal['consistent_moved']:>6} keys moved ({removal['consistent_fraction']:.1%})")

    # Verify consistent hashing moved ~K/N keys
    expected = len(keys) / removal["num_servers_before"]
    actual = removal["consistent_moved"]
    print(f"  Expected ~{expected:.0f} keys moved (K/N), got {actual}")

    print("\n" + "=" * 60)
    print("Exercise 2 complete. Check plots/ and data/ for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
