#!/usr/bin/env python3
"""
Exercise 4: Comparing Naive Modulo vs Consistent vs Jump vs Rendezvous Hashing (~15 min).

Implement jump consistent hashing (Lamping & Veach, 2014) and rendezvous hashing
(Thaler & Ravishankar, 1998), then benchmark all three approaches head-to-head on:

  1. Load balance      -- How evenly are keys distributed across servers?
  2. Lookup latency    -- How fast is a single key lookup?
  3. Key migration     -- What fraction of keys move when adding a server?
  4. Memory usage      -- How much state does the algorithm need?

This exercise crystallizes WHEN to use each algorithm in practice:
  - Consistent hashing (ring + vnodes): General-purpose, any topology change.
  - Jump consistent hash: Fastest, balanced in expectation, but sequential-only changes.
  - Rendezvous hashing: Simplest, no data structure, but O(N) per lookup.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

# Ensure sibling modules are importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hash_utils import (  # noqa: E402
    hash_to_ring,
    hash_pair,
    load_balance_stats,
    save_plot,
    ensure_dirs,
    DATA_DIR,
)
from consistent_hash_ring import ConsistentHashRing  # noqa: E402


# ---------------------------------------------------------------------------
# Jump Consistent Hashing (Lamping & Veach, 2014)
# ---------------------------------------------------------------------------

# TODO(human): Implement JumpConsistentHash.
#
# Jump consistent hash is a ~5-line algorithm from Google that maps a 64-bit
# key to one of N buckets with PERFECT balance and minimal disruption.
#
# From the paper (arXiv:1406.2294):
#
#   int32_t JumpConsistentHash(uint64_t key, int32_t num_buckets) {
#       int64_t b = -1, j = 0;
#       while (j < num_buckets) {
#           b = j;
#           key = key * 2862933555777941757ULL + 1;
#           j = (b + 1) * (double(1LL << 31) / double((key >> 33) + 1));
#       }
#       return b;
#   }
#
# How it works (intuition):
#   Imagine adding buckets one at a time (0, 1, 2, ..., N-1). For each new
#   bucket j, a key "jumps" to j with probability 1/j (reservoir sampling
#   argument). The LCG (linear congruential generator) seeded by the key
#   produces deterministic pseudo-random numbers for each step. The clever
#   part: instead of testing every j, the algorithm computes the NEXT j to
#   jump to, skipping over buckets where no jump happens. This yields
#   O(ln N) expected iterations instead of O(N).
#
# Properties:
#   - Balanced in expectation: each bucket has probability exactly 1/N (CV->0 as K->inf)
#   - O(ln N) time, O(1) space (no ring, no sorted array)
#   - Minimal disruption: going from N to N+1 buckets moves exactly 1/(N+1) keys
#   - Limitation: only supports adding/removing the LAST bucket (N-1 -> N or N -> N-1)
#     You cannot remove an arbitrary server -- all buckets must be numbered 0..N-1
#
# Implementation notes:
#   - Python doesn't have unsigned 64-bit arithmetic, so use & 0xFFFFFFFFFFFFFFFF
#     to simulate 64-bit overflow.
#   - The magic constant 2862933555777941757 is the LCG multiplier from Knuth.
#   - (1 << 31) / ((key >> 33) + 1) computes the probability threshold.
#   - Convert the key string to a 64-bit int by hashing it (SHA-256 truncated to 8 bytes).


class JumpConsistentHash:
    """Jump consistent hashing (Lamping & Veach, 2014).

    Usage:
        jch = JumpConsistentHash(num_buckets=10)
        bucket = jch.get_bucket("user:12345")  # -> int in [0, 10)
    """

    def __init__(self, num_buckets: int) -> None:
        # TODO(human): Store num_buckets.
        # That's it -- jump consistent hash needs NO other state.
        # This is the "zero memory" property: the entire algorithm is
        # a pure function of (key, num_buckets), with no data structure.

        self.num_buckets = num_buckets

    def get_bucket(self, key: str) -> int:
        # TODO(human): Implement the jump consistent hash algorithm.
        #
        # Steps:
        # 1. Convert the string key to a 64-bit unsigned integer:
        #    import hashlib
        #    key_int = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        #
        # 2. Implement the core loop from the paper:
        #    b, j = -1, 0
        #    while j < self.num_buckets:
        #        b = j
        #        key_int = ((key_int * 2862933555777941757) + 1) & 0xFFFFFFFFFFFFFFFF
        #        j = int((b + 1) * ((1 << 31) / ((key_int >> 33) + 1)))
        #    return int(b)
        #
        # The & 0xFFFFFFFFFFFFFFFF masks to 64 bits, simulating C's unsigned overflow.
        # Each iteration of the loop is one "jump" -- on average there are ln(N) jumps.
        #
        # Verify: get_bucket should return an int in [0, self.num_buckets).

        key_int = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"big")
        b, j = -1, 0
        while j < self.num_buckets:
            b = j
            key_int = ((key_int * 2862933555777941757) + 1) & 0xFFFFFFFFFFFFFFFF
            j = int((b + 1) * ((1 << 31) / ((key_int >> 33) + 1)))
        return b

    def get_distribution(self, keys: list[str]) -> dict[int, int]:
        """Count how many keys map to each bucket.

        Provided -- no TODO.
        """
        counts: dict[int, int] = {i: 0 for i in range(self.num_buckets)}
        for key in keys:
            bucket = self.get_bucket(key)
            counts[bucket] += 1
        return counts

    def get_server(self, key) -> str:
        return str(self.get_bucket(key))
# ---------------------------------------------------------------------------
# Rendezvous Hashing / Highest Random Weight (Thaler & Ravishankar, 1998)
# ---------------------------------------------------------------------------

# TODO(human): Implement RendezvousHash.
#
# Rendezvous hashing is the simplest of the three algorithms. For a given key,
# compute hash(key, server) for EVERY server in the cluster. The server with
# the highest hash value "wins" and is assigned the key.
#
# Algorithm:
#   def get_server(key, servers):
#       return max(servers, key=lambda s: hash(key + ":" + s))
#
# That's the entire algorithm. No ring, no sorted array, no finger table.
#
# Properties:
#   - Balanced in expectation (each server has probability 1/N; CV->0 as K->inf)
#   - Minimal disruption: when adding/removing a server, only O(K/N) keys move.
#     A key only moves if the new server has a higher hash than the current winner.
#   - Supports ARBITRARY topology changes: any server can be added or removed.
#   - O(N) per lookup: must compute hash against every server.
#     This is the main disadvantage -- for N=1000, that's 1000 hash computations per lookup.
#
# Why it has minimal disruption:
#   When server X is added, a key moves to X only if hash(key, X) > hash(key, current_winner).
#   Since the hash is uniform, this happens with probability 1/(N+1). So on average,
#   K/(N+1) keys move -- the same optimal bound as consistent hashing.
#
# Use case: Small clusters (N < 100) where simplicity matters more than lookup speed.
# Used in Microsoft's CARP (Cache Array Routing Protocol).


class RendezvousHash:
    """Rendezvous hashing / Highest Random Weight (Thaler & Ravishankar, 1998).

    Usage:
        rh = RendezvousHash(["server-A", "server-B", "server-C"])
        server = rh.get_server("user:12345")  # -> "server-B"
    """

    def __init__(self, servers: list[str]) -> None:
        # TODO(human): Store a COPY of the servers list.
        #
        # Store self.servers = list(servers) to avoid aliasing.
        # That's all the state rendezvous hashing needs -- just the list of servers.
        self.servers = list(servers)

    def get_server(self, key: str) -> str:
        # TODO(human): Find the server with the highest hash for this key.
        #
        # For each server in self.servers, compute hash_pair(key, server)
        # (imported from hash_utils). Return the server with the maximum hash.
        #
        # One-liner version:
        #   return max(self.servers, key=lambda s: hash_pair(key, s))
        #
        # This is O(N) where N = len(self.servers). Every lookup must
        # compute N hashes. For N=10, that's trivial. For N=10000, it's
        # 10000 SHA-256 computations per lookup -- significant.
        return max(self.servers, key=lambda s: hash_pair(key, s))

    def add_server(self, server: str) -> None:
        """Add a server to the pool.

        Provided -- no TODO.
        """
        if server not in self.servers:
            self.servers.append(server)

    def remove_server(self, server: str) -> None:
        """Remove a server from the pool.

        Provided -- no TODO.
        """
        self.servers.remove(server)

    def get_distribution(self, keys: list[str]) -> dict[str, int]:
        """Count how many keys map to each server.

        Provided -- no TODO.
        """
        counts: dict[str, int] = {s: 0 for s in self.servers}
        for key in keys:
            server = self.get_server(key)
            counts[server] += 1
        return counts


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------

# TODO(human): Implement benchmark_comparison().
#
# Run all three hashing algorithms head-to-head and generate a multi-panel
# comparison chart. This is the synthesis exercise that ties everything together.
#
# Metrics to measure:
#
# 1. LOAD BALANCE (bar chart per algorithm):
#    - For each algorithm with N servers and K keys, measure the CV
#      (coefficient of variation) of keys per server.
#    - Consistent hashing: use ConsistentHashRing(num_virtual_nodes=150)
#    - Jump consistent hash: use JumpConsistentHash(N)
#    - Rendezvous: use RendezvousHash(servers)
#    - Lower CV = better balance. Jump should be ~0, consistent ~0.02-0.05,
#      rendezvous ~0.01.
#
# 2. LOOKUP LATENCY (bar chart, microseconds per lookup):
#    - Time 10,000 lookups for each algorithm.
#    - Consistent: ring.get_node(key)
#    - Jump: jch.get_bucket(key)
#    - Rendezvous: rh.get_server(key)
#    - Jump should be fastest (pure arithmetic), rendezvous slowest (N hashes).
#
# 3. KEY MIGRATION (bar chart):
#    - For each algorithm, measure fraction of keys that move when going
#      from N to N+1 servers.
#    - Consistent: add a new node to the ring, compare before/after.
#    - Jump: change num_buckets from N to N+1, compare before/after.
#    - Rendezvous: add a new server, compare before/after.
#    - All three should be close to the theoretical minimum of 1/(N+1).
#
# 4. MEMORY USAGE (bar chart, approximate):
#    - Consistent: N * num_virtual_nodes * (sizeof(int) + pointer) ~= N * 150 * 12 bytes
#    - Jump: 0 (just num_buckets int)
#    - Rendezvous: N * avg_server_name_length
#
# Generate a 2x2 subplot figure with all four metrics. Save to
# plots/04_algorithm_comparison.png.
#
# Also save the raw benchmark data to data/04_comparison_results.json.

def benchmark_comparison(
    num_servers: int = 10,
    num_keys: int = 100_000,
    num_virtual_nodes: int = 150,
) -> dict[str, dict[str, float]]:
    # TODO(human): Benchmark all three algorithms and generate comparison plots.
    #
    # Steps:
    #
    # 1. Setup:
    #    servers = [f"server-{i}" for i in range(num_servers)]
    #    keys = [f"key-{i}" for i in range(num_keys)]
    #    ring = ConsistentHashRing(num_virtual_nodes); add all servers
    #    jch = JumpConsistentHash(num_servers)
    #    rh = RendezvousHash(servers)
    #
    # 2. Load balance:
    #    For each algorithm, get the distribution (keys per server/bucket).
    #    Compute CV using load_balance_stats().
    #    Note: for JumpConsistentHash, bucket numbers are ints (0..N-1),
    #    so convert to str keys for load_balance_stats if needed.
    #
    # 3. Lookup latency:
    #    For each algorithm, time num_keys lookups:
    #      start = time.perf_counter()
    #      for key in keys: algorithm.get_xxx(key)
    #      elapsed = time.perf_counter() - start
    #      us_per_lookup = elapsed / num_keys * 1_000_000
    #
    # 4. Key migration:
    #    For each algorithm:
    #      a. Record assignments with N servers.
    #      b. Add one more server (N+1).
    #      c. Record assignments with N+1 servers.
    #      d. Count fraction that changed.
    #    For consistent: ring2 = new ring with N+1 servers, compare.
    #    For jump: jch2 = JumpConsistentHash(num_servers + 1), compare.
    #    For rendezvous: rh2 with extra server, compare.
    #
    # 5. Memory (approximate, bytes):
    #    consistent_mem = num_servers * num_virtual_nodes * 12  # position + pointer overhead
    #    jump_mem = 8  # just an int
    #    rendezvous_mem = sum(len(s.encode()) for s in servers)  # server name strings
    #
    # 6. Generate 2x2 subplot figure:
    #    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    #    Panel (0,0): Load Balance (CV) -- bar chart, 3 bars
    #    Panel (0,1): Lookup Latency (us/lookup) -- bar chart, 3 bars
    #    Panel (1,0): Key Migration (% moved) -- bar chart with theoretical line
    #    Panel (1,1): Memory Usage (bytes) -- bar chart, log scale
    #
    # 7. save_plot(fig, "04_algorithm_comparison.png")
    #
    # 8. Save results dict to data/04_comparison_results.json
    #
    # 9. Return results dict with structure:
    #    {
    #      "consistent": {"cv": ..., "latency_us": ..., "migration_pct": ..., "memory_bytes": ...},
    #      "jump":       {"cv": ..., "latency_us": ..., "migration_pct": ..., "memory_bytes": ...},
    #      "rendezvous": {"cv": ..., "latency_us": ..., "migration_pct": ..., "memory_bytes": ...},
    #    }

    servers = [f"server-{i}" for i in range(num_servers)]
    keys = [f"key-{i}" for i in range(num_keys)]

    ring = ConsistentHashRing(num_virtual_nodes)
    for server in servers: ring.add_node(server)
    jch = JumpConsistentHash(num_servers)
    rh = RendezvousHash(servers)

    ring_d = ring.get_distribution(keys)
    jch_d = {keys[k]: j for (k,j) in jch.get_distribution(keys).items()}
    rh_d = rh.get_distribution(keys)

    ring_stats = load_balance_stats(ring_d)
    jch_stats = load_balance_stats(jch_d)
    rh_stats = load_balance_stats(rh_d)

    times = []
    for algo in (ring, jch, rh):
        start = time.perf_counter()
        for key in keys: algo.get_server(key)
        elapsed = time.perf_counter() - start
        times.append(elapsed/num_keys * 1_000_000)

    ring2 = ConsistentHashRing(num_virtual_nodes)
    for server in servers: ring2.add_node(server)
    ring2.add_node(f"server-{num_servers}")
    jch2 = JumpConsistentHash(num_servers+1)
    rh2 = RendezvousHash(servers+[f"server-{num_servers}"])

    changes = []

    for (before, after) in ((ring,ring2),(jch,jch2),(rh,rh2)):
        changed = 0
        for key in keys:
            if before.get_server(key) != after.get_server(key):
                changed += 1
        changes.append(changed/len(keys) * 100)

    consistent_mem = num_servers * num_virtual_nodes * 12
    jump_mem = 8
    rendezvous_mem = sum(len(s.encode()) for s in servers)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs : list[list[Axes]]
    x = ["Consistent Hash Ring", "Jump Consistent Hash", "Rendezvous Hash"]
    axs[0][0].bar(
        x,
        [ring_stats["cv"],jch_stats["cv"],rh_stats["cv"]],
    )
    axs[0][0].set_title("Load Balance (CV)")

    axs[0][1].bar(
        x,
        times,
    )
    axs[0][1].set_title("Lookup Latency (us/lookup)")

    axs[1][0].bar(
        x,
        changes
    )
    axs[1][0].set_title("Key Migration (% moved)")

    axs[1][1].bar(
        x,
        [consistent_mem, jump_mem, rendezvous_mem],
    )
    axs[1][1].set_title("Memory Usage (bytes)")

    save_plot(fig,"04_algorithm_comparison.png")

    results = {
              "consistent": {"cv": ring_stats["cv"], "latency_us": times[0], "migration_pct": changes[0], "memory_bytes": consistent_mem},
              "jump":       {"cv": jch_stats["cv"], "latency_us": times[1], "migration_pct": changes[1], "memory_bytes": jump_mem},
              "rendezvous": {"cv": rh_stats["cv"], "latency_us": times[2], "migration_pct": changes[2], "memory_bytes": rendezvous_mem},
    }

    with open("data/04_comparison_results.json", "w") as fout:
        fout.write(json.dumps(results))

    return results


# ---------------------------------------------------------------------------
# Scaling experiment: how algorithms behave as N grows
# ---------------------------------------------------------------------------

def scaling_experiment(
    max_servers: int = 100,
    num_keys: int = 50_000,
    server_counts: list[int] | None = None,
) -> None:
    """Measure how lookup latency scales with number of servers.

    Provided -- uses your implementations. Generates a scaling plot.
    """
    if server_counts is None:
        server_counts = [5, 10, 20, 30, 50, 75, 100]

    consistent_times: list[float] = []
    jump_times: list[float] = []
    rendezvous_times: list[float] = []

    keys = [f"key-{i}" for i in range(num_keys)]

    print(f"\n{'N':>5} | {'Consistent':>12} | {'Jump':>12} | {'Rendezvous':>12}  (us/lookup)")
    print("-" * 58)

    for n in server_counts:
        servers = [f"server-{i}" for i in range(n)]

        # Consistent hashing
        ring = ConsistentHashRing(num_virtual_nodes=150)
        for s in servers:
            ring.add_node(s)
        t0 = time.perf_counter()
        for key in keys:
            ring.get_node(key)
        consistent_us = (time.perf_counter() - t0) / num_keys * 1_000_000
        consistent_times.append(consistent_us)

        # Jump
        jch = JumpConsistentHash(n)
        t0 = time.perf_counter()
        for key in keys:
            jch.get_bucket(key)
        jump_us = (time.perf_counter() - t0) / num_keys * 1_000_000
        jump_times.append(jump_us)

        # Rendezvous
        rh = RendezvousHash(servers)
        t0 = time.perf_counter()
        for key in keys:
            rh.get_server(key)
        rendezvous_us = (time.perf_counter() - t0) / num_keys * 1_000_000
        rendezvous_times.append(rendezvous_us)

        print(f"{n:>5} | {consistent_us:>10.2f}us | {jump_us:>10.2f}us | {rendezvous_us:>10.2f}us")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(server_counts, consistent_times, "bo-", linewidth=2, markersize=6, label="Consistent (ring + vnodes)")
    ax.plot(server_counts, jump_times, "gs-", linewidth=2, markersize=6, label="Jump (Lamping-Veach)")
    ax.plot(server_counts, rendezvous_times, "r^-", linewidth=2, markersize=6, label="Rendezvous (HRW)")

    ax.set_xlabel("Number of Servers (N)")
    ax.set_ylabel("Lookup Latency (us/lookup)")
    ax.set_title("Lookup Latency vs Cluster Size")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_plot(fig, "04_scaling_latency.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()

    print("=" * 60)
    print("Exercise 4: Algorithm Comparison")
    print("=" * 60)

    # --- Part A: Quick sanity checks ---
    print("\n--- Part A: Sanity Checks ---")

    # Jump consistent hash
    jch = JumpConsistentHash(10)
    print("Jump consistent hash (10 buckets):")
    for i in range(5):
        key = f"test-key-{i}"
        bucket = jch.get_bucket(key)
        print(f"  {key} -> bucket {bucket}")

    # Rendezvous hash
    servers = [f"server-{i}" for i in range(5)]
    rh = RendezvousHash(servers)
    print("\nRendezvous hash (5 servers):")
    for i in range(5):
        key = f"test-key-{i}"
        server = rh.get_server(key)
        print(f"  {key} -> {server}")

    # --- Part B: Full benchmark ---
    print("\n--- Part B: Full Benchmark (10 servers, 100k keys) ---")
    results = benchmark_comparison(num_servers=10, num_keys=100_000)

    print("\nResults summary:")
    print(f"  {'Metric':<20} | {'Consistent':>12} | {'Jump':>12} | {'Rendezvous':>12}")
    print("  " + "-" * 62)
    for metric in ["cv", "latency_us", "migration_pct", "memory_bytes"]:
        c = results["consistent"][metric]
        j = results["jump"][metric]
        r = results["rendezvous"][metric]
        if metric == "memory_bytes":
            print(f"  {metric:<20} | {c:>10.0f}B | {j:>10.0f}B | {r:>10.0f}B")
        elif metric == "migration_pct":
            print(f"  {metric:<20} | {c:>11.2f}% | {j:>11.2f}% | {r:>11.2f}%")
        else:
            print(f"  {metric:<20} | {c:>12.4f} | {j:>12.4f} | {r:>12.4f}")

    # --- Part C: Scaling experiment ---
    print("\n--- Part C: Scaling Experiment ---")
    scaling_experiment()

    # --- Part D: Trade-off summary ---
    print("\n--- Part D: Algorithm Trade-off Summary ---")
    print("""
    +-------------------+----------------+----------------+----------------+
    | Property          | Consistent     | Jump           | Rendezvous     |
    |                   | (Ring+Vnodes)  | (Lamping 2014) | (HRW 1998)     |
    +-------------------+----------------+----------------+----------------+
    | Balance           | Good (CV~0.03) | Unbiased 1/N   | Unbiased 1/N   |
    | Lookup time       | O(log M)       | O(ln N)        | O(N)           |
    | Memory            | O(N * V)       | O(1)           | O(N)           |
    | Migration (add)   | ~K/(N+1)       | K/(N+1) exact  | ~K/(N+1)       |
    | Arbitrary remove? | YES            | NO (last only) | YES            |
    | Use case          | General        | Sequential     | Small clusters |
    +-------------------+----------------+----------------+----------------+

    Choose:
      - Consistent hashing: General-purpose, production default (DynamoDB, Cassandra)
      - Jump: When servers are numbered sequentially, unbiased balance (sharded DBs)
      - Rendezvous: Small N, simplicity > performance (config routing, DNS)
    """)

    print("=" * 60)
    print("Exercise 4 complete. Check plots/ for visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
