"""Exercise 5: Dynamic Scenarios — Server Crashes, Traffic Spikes, Degradation.

Extend the base environment with scheduled dynamic events to test how
well each policy adapts to changing conditions. RL agents should shine
here because they observe state changes and adjust; static policies cannot.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
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
_eval_mod = importlib.import_module("03_evaluation")
RLPolicy = _eval_mod.RLPolicy


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


@dataclass
class DynamicEvent:
    """A scheduled event that modifies the environment at a given timestep.

    Attributes:
        timestep: When the event triggers (step number).
        event_type: One of "server_crash", "traffic_spike", "slow_server",
                    "recover_server".
        target_server: Which server is affected (index). Ignored for
                       traffic_spike.
        duration: How many steps the event lasts (for traffic_spike).
        factor: Multiplier for slow_server (e.g., 0.5 = half speed).
    """

    timestep: int
    event_type: str
    target_server: int = 0
    duration: int = 50
    factor: float = 0.5


# TODO(human): Implement DynamicLoadBalancerEnv
#
# Subclass LoadBalancerEnv to inject dynamic events during episodes.
# Events modify server behavior at scheduled timesteps, creating a
# non-stationary environment that tests adaptability.
#
# You need to:
#
# 1. In __init__(self, events, **kwargs):
#    - Call super().__init__(**kwargs) to set up the base environment.
#    - Store self._events as a list of DynamicEvent objects.
#    - Store self._original_rates (will be set in reset) to allow
#      recovering servers after crashes.
#    - Store self._active_spike: bool = False for traffic spike tracking.
#
# 2. Override reset(self, seed=None, options=None):
#    - Call super().reset(seed=seed, options=options) to get (obs, info).
#    - Save self._original_rates = self._processing_rates.copy() so
#      we can restore rates when "recover_server" fires.
#    - Return (obs, info).
#
# 3. Override step(self, action):
#    - BEFORE calling super().step(action), check if any event triggers
#      at the current self._step_count:
#
#      "server_crash": Set self._processing_rates[target] = 0.0.
#        This server stops processing — its queue will grow unboundedly.
#        A smart agent should notice the rising queue and stop routing there.
#
#      "traffic_spike": For the next `duration` steps, the environment
#        should route EXTRA requests. Implementation approach: call
#        super().step() multiple times (e.g., 3x) for each agent step,
#        using the agent's action for the first and random actions for
#        the extras. Or simpler: directly add to random server queues.
#        The simplest approach: just increase queue lengths of random
#        servers by 2-3 each step for `duration` steps.
#
#      "slow_server": Set self._processing_rates[target] *= factor.
#        This server is degraded (e.g., disk failing, high load from
#        other tenants). Its queue will grow more slowly than a crash
#        but still causes higher latency.
#
#      "recover_server": Restore self._processing_rates[target] to
#        self._original_rates[target]. This tests whether the agent
#        can re-include a recovered server — static policies do this
#        automatically, but does the RL agent adapt back?
#
#    - Call obs, reward, terminated, truncated, info = super().step(action)
#    - Add event info to the info dict: info["active_events"] = [...]
#    - Return (obs, reward, terminated, truncated, info).
#
# The key insight: the RL agent's OBSERVATION includes queue lengths
# and CPU utilization. When a server crashes (rate=0), its queue grows
# and CPU maxes out — the agent "sees" this and should route elsewhere.
# Round-robin is blind to it. Least-connections will react but only
# after the queue has already grown. The RL agent, having trained on
# similar dynamics, should react faster.
#
# class DynamicLoadBalancerEnv(LoadBalancerEnv):
#     def __init__(self, events: list[DynamicEvent], **kwargs) -> None: ...
#     def reset(self, seed=None, options=None): ...
#     def step(self, action: int): ...

class DynamicLoadBalancerEnv(LoadBalancerEnv):
    """LoadBalancerEnv with scheduled dynamic events."""

    def __init__(self, events: list[DynamicEvent], **kwargs) -> None:
        raise NotImplementedError(
            "TODO(human): Call super().__init__, store events, "
            "initialize event-tracking state."
        )

    def reset(self, seed=None, options=None):
        raise NotImplementedError(
            "TODO(human): Call super().reset(), save original processing "
            "rates for recovery events."
        )

    def step(self, action: int):
        raise NotImplementedError(
            "TODO(human): Check for triggered events, modify server state, "
            "then call super().step()."
        )


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, list[DynamicEvent]] = {
    "Server Crash & Recovery": [
        DynamicEvent(timestep=100, event_type="server_crash", target_server=2),
        DynamicEvent(timestep=300, event_type="recover_server", target_server=2),
    ],
    "Traffic Spike": [
        DynamicEvent(timestep=150, event_type="traffic_spike", duration=80),
    ],
    "Progressive Degradation": [
        DynamicEvent(timestep=50, event_type="slow_server", target_server=0, factor=0.5),
        DynamicEvent(timestep=150, event_type="slow_server", target_server=1, factor=0.3),
        DynamicEvent(timestep=250, event_type="slow_server", target_server=3, factor=0.5),
    ],
    "Chaos (Multiple Events)": [
        DynamicEvent(timestep=50, event_type="slow_server", target_server=1, factor=0.5),
        DynamicEvent(timestep=100, event_type="server_crash", target_server=3),
        DynamicEvent(timestep=200, event_type="traffic_spike", duration=60),
        DynamicEvent(timestep=300, event_type="recover_server", target_server=3),
        DynamicEvent(timestep=350, event_type="slow_server", target_server=0, factor=0.3),
    ],
}


# ---------------------------------------------------------------------------
# Main: run scenarios and compare
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Exercise 5: Dynamic Scenarios")
    print("=" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load trained model
    model_path = MODELS_DIR / "ppo_load_balancer"
    if not model_path.with_suffix(".zip").exists():
        print(f"\nERROR: Trained model not found at {model_path}.zip")
        print("Run src/02_train_agent.py first.")
        return

    policies = {
        "Round-Robin": RoundRobinPolicy(),
        "Random": RandomPolicy(seed=42),
        "Least-Connections": LeastConnectionsPolicy(),
        "RL (PPO)": RLPolicy(model_path),
    }

    all_results: list[pd.DataFrame] = []

    for scenario_name, events in SCENARIOS.items():
        print(f"\n--- Scenario: {scenario_name} ---")
        print(f"  Events: {len(events)}")
        for e in events:
            print(f"    t={e.timestep}: {e.event_type} "
                  f"(server={e.target_server})")

        env = DynamicLoadBalancerEnv(
            events=events,
            num_servers=5,
            max_queue=20,
            max_steps=500,
        )

        for policy_name, policy in policies.items():
            obs, _ = env.reset(seed=42)
            policy.reset()

            latencies = []
            for step_idx in range(500):
                action = policy.select_server(obs, env.num_servers)
                obs, reward, terminated, truncated, info = env.step(action)
                latencies.append(info["latency"])
                if terminated or truncated:
                    break

            mean_lat = np.mean(latencies)
            p99_lat = np.percentile(latencies, 99)
            print(f"  {policy_name:20s} | mean={mean_lat:.3f} | p99={p99_lat:.3f}")

            scenario_df = pd.DataFrame({
                "scenario": scenario_name,
                "policy": policy_name,
                "step": range(len(latencies)),
                "latency": latencies,
            })
            all_results.append(scenario_df)

        env.close()

    # --- Combined results ---
    if all_results:
        results = pd.concat(all_results, ignore_index=True)

        print("\n" + "=" * 60)
        print("Aggregate Results Across All Scenarios")
        print("=" * 60)

        summary = results.groupby(["scenario", "policy"])["latency"].agg(
            mean="mean", p99=lambda x: np.percentile(x, 99)
        ).round(4)
        print(summary.to_string())

        # --- Plot: latency time-series per scenario ---
        fig, axes = plt.subplots(
            len(SCENARIOS), 1,
            figsize=(12, 4 * len(SCENARIOS)),
            sharex=True,
        )
        if len(SCENARIOS) == 1:
            axes = [axes]

        for ax, scenario_name in zip(axes, SCENARIOS.keys()):
            scenario_data = results[results["scenario"] == scenario_name]
            for policy_name in policies.keys():
                policy_data = scenario_data[
                    scenario_data["policy"] == policy_name
                ]
                if not policy_data.empty:
                    rolling = policy_data["latency"].rolling(
                        window=20, min_periods=1
                    ).mean()
                    ax.plot(
                        policy_data["step"].values,
                        rolling.values,
                        label=policy_name,
                        alpha=0.8,
                    )

            # Mark events
            for event in SCENARIOS[scenario_name]:
                ax.axvline(
                    x=event.timestep,
                    color="red",
                    linestyle="--",
                    alpha=0.5,
                )
                ax.annotate(
                    event.event_type,
                    xy=(event.timestep, ax.get_ylim()[1] * 0.9),
                    fontsize=7,
                    rotation=45,
                    color="red",
                )

            ax.set_title(f"Scenario: {scenario_name}")
            ax.set_ylabel("Rolling Mean Latency")
            ax.legend(loc="upper left", fontsize=8)

        axes[-1].set_xlabel("Step")
        plt.tight_layout()
        plot_path = PLOTS_DIR / "dynamic_scenarios.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nPlot saved: {plot_path}")

    print("\nDynamic scenario evaluation complete.")


if __name__ == "__main__":
    main()
