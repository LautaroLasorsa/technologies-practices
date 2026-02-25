"""Exercise 4: Evaluate RL Agent vs Baseline Policies.

Load the trained RL model and systematically compare it against
round-robin, random, and least-connections baselines. Collect
per-step metrics across multiple episodes and generate comparison
visualizations.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Import from previous exercises (filenames start with digits).
_src = Path(__file__).resolve().parent
sys.path.insert(0, str(_src))
_env_mod = importlib.import_module("00_load_balancer_env")
LoadBalancerEnv = _env_mod.LoadBalancerEnv
_bl_mod = importlib.import_module("01_baseline_policies")
RoundRobinPolicy = _bl_mod.RoundRobinPolicy
RandomPolicy = _bl_mod.RandomPolicy
LeastConnectionsPolicy = _bl_mod.LeastConnectionsPolicy


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


class RLPolicy:
    """Adapter that wraps a trained SB3 model as a Policy."""

    def __init__(self, model_path: str | Path) -> None:
        self._model = PPO.load(str(model_path))

    def select_server(self, observation: np.ndarray, num_servers: int) -> int:
        action, _ = self._model.predict(observation, deterministic=True)
        return int(action)

    def reset(self) -> None:
        pass  # SB3 models are stateless between episodes


# TODO(human): Implement compare_policies
#
# Run each policy for multiple episodes on the same environment,
# collecting PER-STEP metrics. Return a DataFrame suitable for
# statistical analysis and plotting.
#
# You need to:
# 1. Initialize an empty list to collect rows.
# 2. For each (policy_name, policy) in policies_dict.items():
#    a. For each episode in range(num_episodes):
#       - env.reset(seed=episode) for reproducibility across policies.
#         Using the SAME seed per episode ensures all policies face the
#         same sequence of server conditions — a fair comparison.
#       - policy.reset()
#       - For each step in range(num_steps):
#         * action = policy.select_server(obs, env.num_servers)
#         * obs, reward, terminated, truncated, info = env.step(action)
#         * Append a row: {"policy": policy_name, "episode": episode,
#           "step": step, "latency": info["latency"],
#           "queue_total": info["queue_total"],
#           "overflow": info["overflow"], "reward": reward}
#         * If terminated or truncated, break
# 3. Return pd.DataFrame(rows).
#
# Key insight: collecting per-step data (not just episode summaries)
# lets you compute any aggregate later — p99, rolling means, per-
# episode variance. This is more flexible than pre-aggregating.
# Using the same random seed per episode ensures a FAIR comparison:
# all policies face identical server configurations and traffic.
#
# Signature:
#   def compare_policies(
#       env: LoadBalancerEnv,
#       policies_dict: dict[str, object],
#       num_episodes: int = 10,
#       num_steps: int = 500,
#   ) -> pd.DataFrame:

def compare_policies(
    env: LoadBalancerEnv,
    policies_dict: dict[str, object],
    num_episodes: int = 10,
    num_steps: int = 500,
) -> pd.DataFrame:
    """Run all policies on the same episodes, return per-step DataFrame."""
    raise NotImplementedError(
        "TODO(human): Evaluation loop collecting per-step metrics "
        "for each policy across multiple episodes."
    )


# TODO(human): Implement plot_comparison
#
# Create a multi-panel figure comparing policy performance.
# Visualization is essential for RL evaluation — numbers alone
# hide distributional differences.
#
# You need to create a figure with 3 subplots (1 row x 3 columns
# or 3 rows x 1 column):
#
# 1. BOX PLOT of latency distributions (one box per policy):
#    - Group results_df by "policy", extract "latency" column.
#    - Use ax.boxplot() or pandas .plot(kind="box").
#    - Title: "Latency Distribution by Policy"
#    - This shows median, quartiles, and outliers — much more
#      informative than mean alone. The RL agent should have a
#      tighter distribution (fewer extreme latencies).
#
# 2. TIME-SERIES of rolling mean latency (one line per policy):
#    - For each policy, compute a rolling mean over steps (window=50).
#    - Average across episodes: group by (policy, step), mean latency,
#      then rolling(50).mean().
#    - Use ax.plot() with a legend.
#    - Title: "Rolling Mean Latency Over Time"
#    - This reveals whether a policy improves or degrades over time
#      and how quickly it adapts to the environment.
#
# 3. BAR CHART of p99 latencies (one bar per policy):
#    - Compute np.percentile(latencies, 99) per policy.
#    - Use ax.bar().
#    - Title: "p99 Latency by Policy"
#    - p99 is the metric SREs care about most — it represents the
#      worst experience for 1% of requests.
#
# Save the figure to save_path. Use plt.tight_layout().
#
# Signature:
#   def plot_comparison(results_df: pd.DataFrame, save_path: Path) -> None:

def plot_comparison(results_df: pd.DataFrame, save_path: Path) -> None:
    """Create multi-panel comparison figure and save to disk."""
    raise NotImplementedError(
        "TODO(human): Create 3-panel figure — box plot, rolling mean, "
        "p99 bar chart — and save to save_path."
    )


# ---------------------------------------------------------------------------
# Main: load model, run comparison, generate plots
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Exercise 4: RL vs Baseline Evaluation")
    print("=" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load trained model ---
    model_path = MODELS_DIR / "ppo_load_balancer"
    if not model_path.with_suffix(".zip").exists():
        print(f"\nERROR: Trained model not found at {model_path}.zip")
        print("Run src/02_train_agent.py first to train the agent.")
        return

    # --- Create environment and policies ---
    env = LoadBalancerEnv(num_servers=5, max_queue=20, max_steps=500)

    policies = {
        "Round-Robin": RoundRobinPolicy(),
        "Random": RandomPolicy(seed=42),
        "Least-Connections": LeastConnectionsPolicy(),
        "RL (PPO)": RLPolicy(model_path),
    }

    # --- Run comparison ---
    print("\nRunning comparison (10 episodes x 500 steps per policy)...")
    results_df = compare_policies(env, policies, num_episodes=10, num_steps=500)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    summary = results_df.groupby("policy")["latency"].agg(
        mean_latency="mean",
        std_latency="std",
        p50_latency=lambda x: np.percentile(x, 50),
        p99_latency=lambda x: np.percentile(x, 99),
    )
    summary["overflow_rate"] = results_df.groupby("policy")["overflow"].mean()
    summary["mean_reward"] = results_df.groupby("policy")["reward"].mean()
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    # --- Generate plots ---
    plot_path = PLOTS_DIR / "rl_vs_baselines.png"
    print(f"\nGenerating comparison plot: {plot_path}")
    plot_comparison(results_df, plot_path)
    print("Plot saved.")

    env.close()
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
