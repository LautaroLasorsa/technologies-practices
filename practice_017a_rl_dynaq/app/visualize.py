"""Plotting utilities for RL learning curves.

Fully implemented -- no TODOs here. Used by compare.py to visualize results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOT_DIR = Path(__file__).parent.parent / "plots"


def smooth(values: list[float], window: int = 20) -> np.ndarray:
    """Apply a rolling average to smooth noisy reward curves."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_learning_curves(
    results: dict[str, list[float]],
    title: str = "Learning Curves: Q-Learning vs Dyna-Q",
    window: int = 20,
    filename: str = "learning_curves.png",
) -> None:
    """Plot smoothed reward curves for multiple agents.

    Args:
        results: dict mapping agent name -> list of episode rewards.
        title: plot title.
        window: smoothing window size for rolling average.
        filename: output filename (saved in plots/ directory).
    """
    fig, (ax_raw, ax_smooth) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12"]

    for idx, (name, rewards) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        episodes = np.arange(len(rewards))

        ax_raw.plot(episodes, rewards, alpha=0.3, color=color, linewidth=0.5)
        ax_raw.set_title("Raw Episode Rewards")
        ax_raw.set_xlabel("Episode")
        ax_raw.set_ylabel("Total Reward")

        smoothed = smooth(rewards, window)
        smoothed_episodes = np.arange(len(smoothed)) + window // 2
        ax_smooth.plot(smoothed_episodes, smoothed, label=name, color=color, linewidth=2)

    ax_smooth.set_title(f"Smoothed (window={window})")
    ax_smooth.set_xlabel("Episode")
    ax_smooth.set_ylabel("Total Reward")
    ax_smooth.legend(loc="lower right")
    ax_smooth.axhline(y=-13, color="gray", linestyle="--", alpha=0.5, label="Optimal (-13)")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    PLOT_DIR.mkdir(exist_ok=True)
    output_path = PLOT_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def plot_planning_steps_comparison(
    results: dict[int, list[float]],
    title: str = "Effect of Planning Steps on Dyna-Q",
    window: int = 20,
    filename: str = "planning_steps.png",
) -> None:
    """Plot learning curves for different numbers of planning steps.

    Args:
        results: dict mapping n_planning_steps -> list of episode rewards.
        title: plot title.
        window: smoothing window.
        filename: output filename.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6"]

    for idx, (n_steps, rewards) in enumerate(sorted(results.items())):
        color = colors[idx % len(colors)]
        smoothed = smooth(rewards, window)
        episodes = np.arange(len(smoothed)) + window // 2
        label = f"n={n_steps}" if n_steps > 0 else "Q-Learning (n=0)"
        ax.plot(episodes, smoothed, label=label, color=color, linewidth=2)

    ax.axhline(y=-13, color="gray", linestyle="--", alpha=0.5)
    ax.text(5, -12, "Optimal (-13)", color="gray", fontsize=9, alpha=0.7)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward (smoothed)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")

    fig.tight_layout()

    PLOT_DIR.mkdir(exist_ok=True)
    output_path = PLOT_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close(fig)
