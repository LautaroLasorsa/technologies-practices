"""Exercise 1: Custom Gymnasium Environment for Load Balancer Simulation.

This module implements a LoadBalancerEnv that simulates routing incoming
requests to K backend servers. Each server has its own processing rate,
queue, and latency characteristics. The RL agent observes the state of
all servers and decides which one should handle the next request.

The environment follows the standard Gymnasium interface:
  - observation_space: Box of shape (num_servers * 3,) with per-server
    features [queue_length, cpu_utilization, avg_latency]
  - action_space: Discrete(num_servers) — pick a server index
  - reward: negative latency (lower is better), penalty for overflow
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class LoadBalancerEnv(gym.Env):
    """Simulates a load balancer distributing requests across backend servers.

    Each server has:
      - A processing rate (requests/step) drawn from a configurable range
      - A bounded queue (max_queue requests)
      - Tracked CPU utilization and rolling average latency

    At each step, the agent routes one incoming request to a server.
    Servers process their queues stochastically based on their processing rate.
    """

    metadata = {"render_modes": []}

    # --- Default simulation parameters ---
    DEFAULT_NUM_SERVERS: int = 5
    DEFAULT_MAX_QUEUE: int = 20
    DEFAULT_MAX_STEPS: int = 500
    # Processing rate range: each server processes between min and max requests/step
    DEFAULT_MIN_PROC_RATE: float = 1.0
    DEFAULT_MAX_PROC_RATE: float = 3.0
    # Latency parameters
    DEFAULT_BASE_LATENCY: float = 1.0  # base latency per request (ms-like units)
    DEFAULT_QUEUE_LATENCY_FACTOR: float = 0.5  # additional latency per queued request
    # Reward parameters
    DEFAULT_OVERFLOW_PENALTY: float = -10.0

    # TODO(human): Implement __init__
    #
    # Initialize the load balancer environment. This is where you define the
    # MDP structure — what the agent observes and what it can do.
    #
    # You need to:
    # 1. Store the simulation parameters (num_servers, max_queue, max_steps,
    #    processing rate range, latency factors, overflow penalty).
    # 2. Define self.action_space as Discrete(num_servers). The agent picks
    #    which server receives the next request — a classic discrete choice.
    # 3. Define self.observation_space as a Box with shape (num_servers * 3,).
    #    Each server contributes 3 features: queue_length (0 to max_queue),
    #    cpu_utilization (0.0 to 1.0), avg_latency (0.0 to some upper bound).
    #    Use np.float32 dtype. Set low/high bounds appropriately — SB3 uses
    #    these bounds for observation normalization.
    # 4. Initialize internal state arrays (will be properly set in reset()):
    #    - self._queue_lengths: np.ndarray of shape (num_servers,)
    #    - self._processing_rates: np.ndarray of shape (num_servers,)
    #    - self._cpu_utils: np.ndarray of shape (num_servers,)
    #    - self._avg_latencies: np.ndarray of shape (num_servers,)
    #    - self._step_count: int
    # 5. Call self.reset() at the end to initialize everything properly.
    #
    # Signature:
    #   def __init__(
    #       self,
    #       num_servers: int = DEFAULT_NUM_SERVERS,
    #       max_queue: int = DEFAULT_MAX_QUEUE,
    #       max_steps: int = DEFAULT_MAX_STEPS,
    #       min_proc_rate: float = DEFAULT_MIN_PROC_RATE,
    #       max_proc_rate: float = DEFAULT_MAX_PROC_RATE,
    #       base_latency: float = DEFAULT_BASE_LATENCY,
    #       queue_latency_factor: float = DEFAULT_QUEUE_LATENCY_FACTOR,
    #       overflow_penalty: float = DEFAULT_OVERFLOW_PENALTY,
    #   ) -> None:
    def __init__(
        self,
        num_servers: int = DEFAULT_NUM_SERVERS,
        max_queue: int = DEFAULT_MAX_QUEUE,
        max_steps: int = DEFAULT_MAX_STEPS,
        min_proc_rate: float = DEFAULT_MIN_PROC_RATE,
        max_proc_rate: float = DEFAULT_MAX_PROC_RATE,
        base_latency: float = DEFAULT_BASE_LATENCY,
        queue_latency_factor: float = DEFAULT_QUEUE_LATENCY_FACTOR,
        overflow_penalty: float = DEFAULT_OVERFLOW_PENALTY,
    ) -> None:
        super().__init__()
        raise NotImplementedError(
            "TODO(human): Initialize the environment — define action_space, "
            "observation_space, and internal state arrays."
        )

    # TODO(human): Implement reset
    #
    # Reset the environment to a clean initial state for a new episode.
    # This is called before each episode and must return (observation, info).
    #
    # You need to:
    # 1. Call super().reset(seed=seed) to properly seed the RNG. After this,
    #    self.np_random is available as a seeded numpy RNG.
    # 2. Assign each server a random processing rate uniformly sampled from
    #    [min_proc_rate, max_proc_rate] using self.np_random.uniform().
    #    This creates heterogeneous servers — some faster, some slower.
    # 3. Zero out all queue lengths, CPU utilizations, and average latencies.
    # 4. Reset step counter to 0.
    # 5. Return (self._get_observation(), {}) — the empty dict is the info.
    #
    # Signature:
    #   def reset(
    #       self,
    #       seed: int | None = None,
    #       options: dict | None = None,
    #   ) -> tuple[np.ndarray, dict]:
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        raise NotImplementedError(
            "TODO(human): Reset server states, randomize processing rates, "
            "return initial observation."
        )

    # TODO(human): Implement step
    #
    # Execute one timestep: route a request to the chosen server, simulate
    # processing across all servers, compute reward.
    #
    # This is the core simulation logic. At each step:
    #
    # 1. ROUTE REQUEST: Add 1 to the chosen server's queue. If the queue is
    #    already at max_queue, the request is "dropped" — set a boolean flag
    #    `overflow = True` but do NOT add to the queue.
    #
    # 2. COMPUTE LATENCY for the routed request: latency depends on how
    #    loaded the target server is. A simple model:
    #      latency = base_latency + queue_latency_factor * queue_length / processing_rate
    #    This captures the intuition that latency grows with queue depth
    #    and decreases with faster servers.
    #
    # 3. PROCESS QUEUES: Each server processes some of its queued requests.
    #    The number processed is drawn from a Poisson distribution:
    #      processed = min(self.np_random.poisson(rate), queue_length)
    #    Poisson introduces realistic stochasticity — sometimes a server
    #    processes more, sometimes fewer than its average rate.
    #    Subtract processed from queue_length (clamp to 0).
    #
    # 4. UPDATE METRICS:
    #    - cpu_util[i] = queue_lengths[i] / max_queue (fraction of capacity used)
    #    - avg_latency[i]: exponential moving average with alpha=0.1:
    #        avg_latency[i] = 0.9 * avg_latency[i] + 0.1 * (base_latency + factor * queue / rate)
    #
    # 5. COMPUTE REWARD:
    #    - If overflow: reward = overflow_penalty
    #    - Else: reward = -latency (negative because we minimize latency)
    #
    # 6. CHECK TERMINATION:
    #    - terminated = False (no natural terminal state)
    #    - truncated = (step_count >= max_steps)
    #
    # 7. Build info dict with: {"latency": latency, "overflow": overflow,
    #    "queue_total": sum(queue_lengths)}
    #
    # Return (observation, reward, terminated, truncated, info)
    #
    # Signature:
    #   def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        raise NotImplementedError(
            "TODO(human): Route request, simulate processing, compute reward."
        )

    # TODO(human): Implement _get_observation
    #
    # Construct the observation array from the current internal state.
    # The observation is a flat numpy array of shape (num_servers * 3,)
    # with dtype np.float32.
    #
    # For each server i, pack 3 features in order:
    #   [queue_length_normalized, cpu_utilization, avg_latency_normalized]
    #
    # Normalize queue_length by dividing by max_queue (so it's in [0, 1]).
    # Normalize avg_latency by dividing by a reasonable upper bound (e.g.,
    # base_latency + queue_latency_factor * max_queue). This keeps all
    # features in roughly [0, 1], which helps neural network training.
    #
    # Use np.stack or np.concatenate to build the flat array.
    # Cast to np.float32 — Gymnasium and SB3 expect float32 observations.
    #
    # Signature:
    #   def _get_observation(self) -> np.ndarray:
    def _get_observation(self) -> np.ndarray:
        raise NotImplementedError(
            "TODO(human): Build flat observation array from server states."
        )


# ---------------------------------------------------------------------------
# Verification: run env_checker and a few random steps
# ---------------------------------------------------------------------------

def main() -> None:
    """Verify the environment with Gymnasium's env_checker and random rollout."""
    from gymnasium.utils.env_checker import check_env

    print("=" * 60)
    print("LoadBalancerEnv — Environment Verification")
    print("=" * 60)

    env = LoadBalancerEnv(num_servers=5, max_queue=20, max_steps=200)

    # --- Gymnasium compliance check ---
    print("\nRunning gymnasium env_checker...")
    try:
        check_env(env, warn=True, skip_render_check=True)
        print("  env_checker passed!")
    except Exception as e:
        print(f"  env_checker FAILED: {e}")
        return

    # --- Random rollout ---
    print(f"\nRunning 100-step random rollout...")
    obs, info = env.reset(seed=42)
    print(f"  Initial obs shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")

    total_reward = 0.0
    latencies: list[float] = []
    overflows = 0

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        latencies.append(info.get("latency", 0.0))
        if info.get("overflow", False):
            overflows += 1
        if terminated or truncated:
            obs, info = env.reset()

    print(f"\n  Results after 100 random steps:")
    print(f"    Total reward:    {total_reward:.2f}")
    print(f"    Mean latency:    {np.mean(latencies):.3f}")
    print(f"    Max latency:     {np.max(latencies):.3f}")
    print(f"    Overflows:       {overflows}")
    print(f"    Final queue sum: {info.get('queue_total', 'N/A')}")

    env.close()
    print("\nEnvironment verification complete.")


if __name__ == "__main__":
    main()
