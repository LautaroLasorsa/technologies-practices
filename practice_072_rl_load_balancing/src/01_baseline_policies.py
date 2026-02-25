"""Exercise 2: Baseline Load Balancing Policies.

Implement three classic load balancing algorithms and an evaluation harness.
These baselines serve as the performance floor that the RL agent must beat.

Each policy is a callable: given the current observation, it returns an
action (server index). The evaluation function runs each policy for
multiple episodes and collects latency/throughput metrics.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

# Import the environment from the previous exercise.
# Because filenames start with digits (00_, 01_, ...), we use importlib.
_src = Path(__file__).resolve().parent
sys.path.insert(0, str(_src))
_env_mod = importlib.import_module("00_load_balancer_env")
LoadBalancerEnv = _env_mod.LoadBalancerEnv


# ---------------------------------------------------------------------------
# Policy interface
# ---------------------------------------------------------------------------

class Policy(Protocol):
    """Protocol for load balancing policies."""

    def select_server(self, observation: np.ndarray, num_servers: int) -> int:
        """Choose a server index given the current observation."""
        ...

    def reset(self) -> None:
        """Reset any internal state for a new episode."""
        ...


# TODO(human): Implement RoundRobinPolicy
#
# Round-robin is the simplest load balancing strategy: cycle through
# servers in order (0, 1, 2, ..., K-1, 0, 1, 2, ...).
#
# You need to:
# 1. Track a `self._current_index` that starts at 0.
# 2. In select_server(): return self._current_index, then increment
#    it modulo num_servers. This ensures perfect rotation.
# 3. In reset(): set self._current_index back to 0.
#
# Key insight: round-robin completely IGNORES the observation. It
# doesn't matter if server 2 has a full queue — round-robin sends
# the next request there anyway. This is its fundamental weakness
# in dynamic environments, but it's perfectly fair in homogeneous
# steady-state conditions.
#
# class RoundRobinPolicy:
#     def __init__(self) -> None: ...
#     def select_server(self, observation: np.ndarray, num_servers: int) -> int: ...
#     def reset(self) -> None: ...

class RoundRobinPolicy:
    """Round-robin: cycle through servers in fixed order."""

    def __init__(self) -> None:
        raise NotImplementedError(
            "TODO(human): Initialize round-robin state (current index)."
        )

    def select_server(self, observation: np.ndarray, num_servers: int) -> int:
        raise NotImplementedError(
            "TODO(human): Return current server, advance index modulo num_servers."
        )

    def reset(self) -> None:
        raise NotImplementedError(
            "TODO(human): Reset the rotation index to 0."
        )


# TODO(human): Implement LeastConnectionsPolicy
#
# Least-connections always routes to the server with the lowest
# current queue length. This is a REACTIVE policy — it uses real-time
# state to make decisions, unlike round-robin.
#
# You need to:
# 1. Extract queue lengths from the observation. Recall the observation
#    layout: for each server i, features are at indices [i*3, i*3+1, i*3+2]
#    corresponding to [queue_length_norm, cpu_util, avg_latency_norm].
#    The queue length (normalized) is at index i*3.
# 2. Return the server index with the smallest queue length:
#    np.argmin(queue_lengths). If tied, argmin returns the first — fine.
# 3. reset() does nothing — this policy is stateless.
#
# Key insight: least-connections is the strongest simple baseline.
# It adapts to load imbalances because overloaded servers naturally
# accumulate longer queues and receive fewer new requests. However,
# it cannot anticipate FUTURE load — it only reacts to current state.
# It also treats all queue items as equal (a cheap request counts the
# same as an expensive one).
#
# class LeastConnectionsPolicy:
#     def select_server(self, observation: np.ndarray, num_servers: int) -> int: ...
#     def reset(self) -> None: ...

class LeastConnectionsPolicy:
    """Least-connections: route to the server with the shortest queue."""

    def select_server(self, observation: np.ndarray, num_servers: int) -> int:
        raise NotImplementedError(
            "TODO(human): Extract queue lengths from observation, return argmin."
        )

    def reset(self) -> None:
        raise NotImplementedError(
            "TODO(human): No-op — this policy is stateless."
        )


# TODO(human): Implement RandomPolicy
#
# Random policy picks a server uniformly at random. Despite its
# simplicity, it has theoretical backing: for large server pools,
# the law of large numbers ensures roughly even distribution.
#
# You need to:
# 1. Store a numpy RNG in __init__ (np.random.default_rng(seed=42)).
# 2. In select_server(): return self._rng.integers(0, num_servers).
# 3. In reset(): re-seed the RNG for reproducibility, or leave as-is.
#
# Key insight: random is the exploration baseline. If the RL agent
# does worse than random, training has failed. If it does better
# than random but worse than least-connections, it has learned
# something but not enough. The gap between random and least-
# connections shows the value of using state information.
#
# class RandomPolicy:
#     def __init__(self, seed: int = 42) -> None: ...
#     def select_server(self, observation: np.ndarray, num_servers: int) -> int: ...
#     def reset(self) -> None: ...

class RandomPolicy:
    """Random: pick a server uniformly at random."""

    def __init__(self, seed: int = 42) -> None:
        raise NotImplementedError(
            "TODO(human): Initialize numpy RNG with the given seed."
        )

    def select_server(self, observation: np.ndarray, num_servers: int) -> int:
        raise NotImplementedError(
            "TODO(human): Return a random integer in [0, num_servers)."
        )

    def reset(self) -> None:
        raise NotImplementedError(
            "TODO(human): Optionally re-seed the RNG."
        )


# TODO(human): Implement evaluate_policy
#
# Run a policy in the environment for multiple episodes and collect
# per-step metrics. This function is the evaluation harness you'll
# reuse in exercises 3 and 4.
#
# You need to:
# 1. For each episode (num_episodes iterations):
#    a. Call env.reset(seed=episode_idx) for reproducibility.
#    b. Call policy.reset() to clear any internal state.
#    c. For each step (num_steps iterations):
#       - Get action = policy.select_server(obs, env.num_servers)
#         (you'll need env to expose num_servers as an attribute).
#       - Call obs, reward, terminated, truncated, info = env.step(action)
#       - Record: latency (from info["latency"]), overflow (info["overflow"]),
#         queue_total (info["queue_total"]), reward.
#       - If terminated or truncated, reset and break inner loop.
# 2. After all episodes, compute aggregate metrics:
#    - mean_latency: np.mean of all recorded latencies
#    - p99_latency: np.percentile(latencies, 99)
#    - throughput: total non-overflow steps / total steps
#    - overflow_rate: total overflows / total steps
#    - mean_reward: np.mean of all rewards
# 3. Return a dict with these keys.
#
# Signature:
#   def evaluate_policy(
#       env: LoadBalancerEnv,
#       policy: Policy,
#       num_episodes: int = 10,
#       num_steps: int = 500,
#   ) -> dict[str, float]:

def evaluate_policy(
    env: LoadBalancerEnv,
    policy: Policy,
    num_episodes: int = 10,
    num_steps: int = 500,
) -> dict[str, float]:
    """Run a policy for multiple episodes, return aggregate metrics."""
    raise NotImplementedError(
        "TODO(human): Evaluation loop — run episodes, collect metrics, "
        "compute mean/p99 latency, throughput, overflow rate."
    )


# ---------------------------------------------------------------------------
# Main: run all baselines and print comparison
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Exercise 2: Baseline Load Balancing Policies")
    print("=" * 60)

    env = LoadBalancerEnv(num_servers=5, max_queue=20, max_steps=500)

    policies: dict[str, Policy] = {
        "Round-Robin": RoundRobinPolicy(),
        "Least-Connections": LeastConnectionsPolicy(),
        "Random": RandomPolicy(seed=42),
    }

    results: list[dict[str, object]] = []

    for name, policy in policies.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_policy(env, policy, num_episodes=10, num_steps=500)
        metrics["policy"] = name
        results.append(metrics)
        print(f"  Mean latency:  {metrics['mean_latency']:.3f}")
        print(f"  p99 latency:   {metrics['p99_latency']:.3f}")
        print(f"  Throughput:    {metrics['throughput']:.3f}")
        print(f"  Overflow rate: {metrics['overflow_rate']:.4f}")
        print(f"  Mean reward:   {metrics['mean_reward']:.3f}")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    df = pd.DataFrame(results)
    df = df.set_index("policy")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))

    env.close()


if __name__ == "__main__":
    main()
