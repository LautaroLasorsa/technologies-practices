"""Compare Q-Learning vs Dyna-Q with varying planning steps.

Fully implemented -- no TODOs. Trains all agents and produces comparison plots.

Usage:
    uv run python app/compare.py
"""

import numpy as np

from app.dyna_q import DynaQParams, train as train_dyna_q
from app.q_learning import QParams, train as train_q_learning
from app.visualize import plot_learning_curves, plot_planning_steps_comparison


SHARED_SEED = 42
N_EPISODES = 300


def run_q_learning_baseline() -> list[float]:
    """Train vanilla Q-learning and return reward history."""
    print("=" * 60)
    print("Training Q-Learning baseline...")
    print("=" * 60)
    params = QParams(n_episodes=N_EPISODES, seed=SHARED_SEED)
    _, rewards = train_q_learning(params)
    return rewards


def run_dyna_q(n_planning_steps: int) -> list[float]:
    """Train Dyna-Q with given planning steps and return reward history."""
    print("=" * 60)
    print(f"Training Dyna-Q (n={n_planning_steps})...")
    print("=" * 60)
    params = DynaQParams(
        n_episodes=N_EPISODES,
        seed=SHARED_SEED,
        n_planning_steps=n_planning_steps,
    )
    _, rewards = train_dyna_q(params)
    return rewards


def compare_q_vs_dyna() -> None:
    """Side-by-side comparison: Q-Learning vs Dyna-Q (n=10)."""
    q_rewards = run_q_learning_baseline()
    dyna_rewards = run_dyna_q(n_planning_steps=10)

    plot_learning_curves(
        results={
            "Q-Learning": q_rewards,
            "Dyna-Q (n=10)": dyna_rewards,
        },
        title="Q-Learning vs Dyna-Q on CliffWalking",
    )

    print_summary({"Q-Learning": q_rewards, "Dyna-Q (n=10)": dyna_rewards})


def compare_planning_steps() -> None:
    """Sweep over different planning step counts."""
    planning_steps = [0, 5, 10, 50]
    results: dict[int, list[float]] = {}

    for n in planning_steps:
        results[n] = run_dyna_q(n)

    plot_planning_steps_comparison(results)

    labeled = {f"n={n}" if n > 0 else "Q-Learning (n=0)": r for n, r in results.items()}
    print_summary(labeled)


def print_summary(results: dict[str, list[float]]) -> None:
    """Print a summary table of final performance."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Agent':<25} {'Avg last 50':>12} {'Best':>8} {'Episodes to -20':>16}")
    print("-" * 60)

    for name, rewards in results.items():
        avg_last_50 = np.mean(rewards[-50:])
        best = max(rewards)
        episodes_to_threshold = find_convergence_episode(rewards, threshold=-20.0)
        threshold_str = str(episodes_to_threshold) if episodes_to_threshold else "never"
        print(f"{name:<25} {avg_last_50:>12.1f} {best:>8.0f} {threshold_str:>16}")


def find_convergence_episode(
    rewards: list[float],
    threshold: float = -20.0,
    window: int = 20,
) -> int | None:
    """Find the first episode where the rolling average exceeds threshold."""
    if len(rewards) < window:
        return None
    for i in range(window, len(rewards)):
        avg = np.mean(rewards[i - window : i])
        if avg >= threshold:
            return i
    return None


def main() -> None:
    """Run both comparisons."""
    print("Part 1: Q-Learning vs Dyna-Q")
    print()
    compare_q_vs_dyna()

    print("\n\n")

    print("Part 2: Effect of Planning Steps")
    print()
    compare_planning_steps()

    print("\nDone! Check plots/ directory for visualizations.")


if __name__ == "__main__":
    main()
