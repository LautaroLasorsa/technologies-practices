#!/usr/bin/env python3
"""
Hash function utilities and visualization helpers for consistent hashing.

This module is FULLY SCAFFOLDED -- no TODO(human) here.
It provides:
  - hash_to_ring(): Maps a string key to a position on the hash ring [0, RING_SIZE)
  - hash_to_ring_md5(): MD5-based variant for comparison
  - RING_SIZE / RING_BITS: Constants defining the hash ring size
  - format_ring_pos(): Human-readable hex representation of ring positions
  - draw_ring(): matplotlib helper for visualizing ring topology
  - load_balance_stats(): Compute standard deviation and max/min ratio of load distribution
  - save_plot(): Save a matplotlib figure to the plots/ directory
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Ring constants
# ---------------------------------------------------------------------------

RING_BITS = 32
RING_SIZE = 2**RING_BITS  # 4,294,967,296 positions on the ring


# ---------------------------------------------------------------------------
# Hash functions
# ---------------------------------------------------------------------------

def hash_to_ring(key: str) -> int:
    """Hash a string key to a position on the ring [0, RING_SIZE).

    Uses SHA-256, truncated to RING_BITS bits. SHA-256 provides excellent
    uniformity, which is critical for even distribution of both keys and
    virtual nodes on the ring.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # Take first 4 bytes (32 bits) as big-endian unsigned int
    return int.from_bytes(digest[:4], "big") % RING_SIZE


def hash_to_ring_md5(key: str) -> int:
    """Hash a string key using MD5, truncated to RING_BITS bits.

    MD5 is faster but cryptographically broken. For consistent hashing
    the only requirement is uniformity, not collision resistance, so MD5
    is fine in practice. Provided for comparison.
    """
    digest = hashlib.md5(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % RING_SIZE


def hash_pair(key: str, server: str) -> int:
    """Hash a (key, server) pair for rendezvous hashing.

    Combines key and server into a single hash. The concatenation order
    and separator must be deterministic so all clients agree.
    """
    combined = f"{key}:{server}"
    digest = hashlib.sha256(combined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_ring_pos(pos: int) -> str:
    """Format a ring position as hex for human-readable output.

    Example: format_ring_pos(255) -> '0x000000ff'
    """
    return f"0x{pos:08x}"


def ring_distance(a: int, b: int) -> int:
    """Clockwise distance from position a to position b on the ring.

    On a ring of size RING_SIZE, the clockwise distance from a to b is:
      (b - a) % RING_SIZE

    This is always >= 0 and < RING_SIZE.
    """
    return (b - a) % RING_SIZE


# ---------------------------------------------------------------------------
# Load balance statistics
# ---------------------------------------------------------------------------

def load_balance_stats(
    counts: dict[str, int],
) -> dict[str, float]:
    """Compute load balance statistics from a mapping of server -> key count.

    Returns a dict with:
      - 'mean': average keys per server
      - 'std': standard deviation
      - 'cv': coefficient of variation (std/mean), lower is more balanced
      - 'max_min_ratio': max_count / min_count, 1.0 = perfectly balanced
      - 'max_mean_ratio': max_count / mean, 1.0 = perfectly balanced
    """
    if not counts:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0, "max_min_ratio": 0.0, "max_mean_ratio": 0.0}

    values = list(counts.values())
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)

    min_val = min(values)
    max_val = max(values)

    return {
        "mean": mean,
        "std": std,
        "cv": std / mean if mean > 0 else 0.0,
        "max_min_ratio": max_val / min_val if min_val > 0 else float("inf"),
        "max_mean_ratio": max_val / mean if mean > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(fig: plt.Figure, filename: str) -> Path:
    """Save a matplotlib figure to the plots/ directory.

    Returns the path to the saved file.
    """
    ensure_dirs()
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {path.relative_to(Path(__file__).resolve().parent.parent)}")
    return path


def draw_ring(
    node_positions: dict[str, list[int]],
    key_assignments: dict[str, str] | None = None,
    title: str = "Consistent Hash Ring",
    figsize: tuple[float, float] = (10, 10),
    show_keys: bool = True,
    max_keys_shown: int = 50,
) -> plt.Figure:
    """Draw a visual representation of the consistent hash ring.

    Args:
        node_positions: Maps physical_node_name -> list of positions on the ring.
        key_assignments: Optional mapping of key_name -> assigned_node_name.
        title: Plot title.
        figsize: Figure dimensions.
        show_keys: Whether to show key positions on the ring.
        max_keys_shown: Max number of keys to draw (for readability).

    Returns:
        The matplotlib Figure object (not shown -- caller decides to show or save).
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Draw the ring
    theta = np.linspace(0, 2 * np.pi, 360)
    ring_r = 1.0
    ax.plot(ring_r * np.cos(theta), ring_r * np.sin(theta), "k-", linewidth=1.5, alpha=0.3)

    # Color palette for nodes
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(node_positions), 1)))
    node_color_map: dict[str, tuple] = {}

    # Draw node positions
    for i, (node_name, positions) in enumerate(sorted(node_positions.items())):
        color = colors[i % len(colors)]
        node_color_map[node_name] = color

        for j, pos in enumerate(positions):
            angle = 2 * np.pi * pos / RING_SIZE
            x = ring_r * np.cos(angle)
            y = ring_r * np.sin(angle)

            # Draw the vnode marker
            ax.plot(x, y, "o", color=color, markersize=8, markeredgecolor="black",
                    markeredgewidth=0.5, zorder=5)

            # Label only the first vnode for each physical node (to avoid clutter)
            if j == 0:
                label_r = ring_r + 0.15
                ax.annotate(
                    node_name,
                    xy=(x, y),
                    xytext=(label_r * np.cos(angle), label_r * np.sin(angle)),
                    fontsize=8, fontweight="bold", color=color,
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=color, alpha=0.5),
                    zorder=6,
                )

    # Draw key assignments
    if show_keys and key_assignments:
        keys_to_show = list(key_assignments.items())[:max_keys_shown]
        for key_name, assigned_node in keys_to_show:
            pos = hash_to_ring(key_name)
            angle = 2 * np.pi * pos / RING_SIZE
            key_r = ring_r - 0.08
            x = key_r * np.cos(angle)
            y = key_r * np.sin(angle)

            color = node_color_map.get(assigned_node, "gray")
            ax.plot(x, y, ".", color=color, markersize=4, alpha=0.6, zorder=3)

    # Legend
    legend_patches = [
        mpatches.Patch(color=node_color_map[name], label=name)
        for name in sorted(node_color_map)
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9, framealpha=0.8)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    fig.tight_layout()
    return fig


def draw_load_bar_chart(
    counts: dict[str, int],
    title: str = "Load Distribution",
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Draw a bar chart of keys per server.

    Includes a horizontal line showing the ideal (perfectly balanced) load.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    servers = sorted(counts.keys())
    loads = [counts[s] for s in servers]
    total = sum(loads)
    ideal = total / len(servers) if servers else 0

    colors = plt.cm.Set2(np.linspace(0, 1, len(servers)))
    bars = ax.bar(servers, loads, color=colors, edgecolor="black", linewidth=0.5)

    # Ideal line
    ax.axhline(y=ideal, color="red", linestyle="--", linewidth=1.5, label=f"Ideal: {ideal:.0f}")

    # Labels on bars
    for bar, load in zip(bars, loads):
        deviation = ((load - ideal) / ideal * 100) if ideal > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(loads) * 0.01,
            f"{load}\n({deviation:+.1f}%)",
            ha="center", va="bottom", fontsize=8,
        )

    stats = load_balance_stats(counts)
    ax.set_title(f"{title}\nCV={stats['cv']:.3f}  Max/Mean={stats['max_mean_ratio']:.2f}x", fontsize=12)
    ax.set_xlabel("Server")
    ax.set_ylabel("Number of Keys")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Verify hash utilities work correctly."""
    print("=" * 60)
    print("Hash Utilities -- Self-Test")
    print("=" * 60)

    # Test 1: hash_to_ring produces values in [0, RING_SIZE)
    print("\n[Test 1] hash_to_ring range check...")
    for key in ["test", "hello", "server-A:vnode42", ""]:
        h = hash_to_ring(key)
        assert 0 <= h < RING_SIZE, f"Out of range: {h}"
        print(f"  hash_to_ring({key!r}) = {format_ring_pos(h)} ({h})")

    # Test 2: hash_to_ring is deterministic
    print("\n[Test 2] Determinism...")
    h1 = hash_to_ring("consistent")
    h2 = hash_to_ring("consistent")
    assert h1 == h2, "Not deterministic!"
    print(f"  OK: same input -> same output ({format_ring_pos(h1)})")

    # Test 3: hash_to_ring_md5 works
    print("\n[Test 3] MD5 variant...")
    h_sha = hash_to_ring("test")
    h_md5 = hash_to_ring_md5("test")
    print(f"  SHA-256: {format_ring_pos(h_sha)}")
    print(f"  MD5:     {format_ring_pos(h_md5)}")
    assert h_sha != h_md5, "Different hash functions should give different results"
    print("  OK: different hash functions produce different positions")

    # Test 4: hash_pair works for rendezvous hashing
    print("\n[Test 4] hash_pair for rendezvous hashing...")
    h1 = hash_pair("key1", "server-A")
    h2 = hash_pair("key1", "server-B")
    h3 = hash_pair("key2", "server-A")
    print(f"  hash_pair('key1', 'server-A') = {h1}")
    print(f"  hash_pair('key1', 'server-B') = {h2}")
    print(f"  hash_pair('key2', 'server-A') = {h3}")
    assert h1 != h2, "Different servers should give different hashes"
    assert h1 != h3, "Different keys should give different hashes"
    print("  OK: all distinct")

    # Test 5: ring_distance
    print("\n[Test 5] ring_distance (clockwise)...")
    assert ring_distance(0, 100) == 100
    assert ring_distance(100, 0) == RING_SIZE - 100
    assert ring_distance(0, 0) == 0
    print("  OK: clockwise distance is correct")

    # Test 6: load_balance_stats
    print("\n[Test 6] load_balance_stats...")
    perfect = {"A": 100, "B": 100, "C": 100}
    stats = load_balance_stats(perfect)
    assert stats["cv"] == 0.0, f"Expected CV=0 for perfect balance, got {stats['cv']}"
    assert stats["max_min_ratio"] == 1.0
    print(f"  Perfect balance: CV={stats['cv']:.3f}, max/min={stats['max_min_ratio']:.2f}")

    skewed = {"A": 200, "B": 100, "C": 50}
    stats2 = load_balance_stats(skewed)
    assert stats2["cv"] > 0, "Skewed distribution should have CV > 0"
    print(f"  Skewed balance:  CV={stats2['cv']:.3f}, max/min={stats2['max_min_ratio']:.2f}")

    # Test 7: Distribution uniformity (statistical check)
    print("\n[Test 7] Distribution uniformity (10,000 keys)...")
    num_buckets = 10
    bucket_counts = [0] * num_buckets
    for i in range(10_000):
        h = hash_to_ring(f"key-{i}")
        bucket = h * num_buckets // RING_SIZE
        bucket_counts[bucket] += 1

    mean = 10_000 / num_buckets
    max_deviation = max(abs(c - mean) / mean for c in bucket_counts)
    print(f"  Bucket counts: {bucket_counts}")
    print(f"  Max deviation from mean: {max_deviation:.1%}")
    assert max_deviation < 0.15, f"Hash distribution too skewed: {max_deviation:.1%}"
    print("  OK: distribution is reasonably uniform")

    # Test 8: Directories exist
    print("\n[Test 8] Output directories...")
    ensure_dirs()
    assert PLOTS_DIR.exists(), f"plots/ dir not created: {PLOTS_DIR}"
    assert DATA_DIR.exists(), f"data/ dir not created: {DATA_DIR}"
    print(f"  plots/ -> {PLOTS_DIR}")
    print(f"  data/  -> {DATA_DIR}")

    print("\n" + "-" * 60)
    print("All hash utility self-tests passed.")
    print("-" * 60)


if __name__ == "__main__":
    _self_test()
