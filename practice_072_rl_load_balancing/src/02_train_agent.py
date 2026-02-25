"""Exercise 3: Train an RL Agent for Load Balancing.

Use Stable-Baselines3 to train a PPO or DQN agent on the custom
LoadBalancerEnv. This exercise covers:
  - Wrapping the environment for SB3 compatibility
  - Configuring algorithm hyperparameters
  - Monitoring training progress with callbacks
  - Saving and loading trained models
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

# Import the environment from exercise 1.
_src = Path(__file__).resolve().parent
sys.path.insert(0, str(_src))
_env_mod = importlib.import_module("00_load_balancer_env")
LoadBalancerEnv = _env_mod.LoadBalancerEnv


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


# TODO(human): Implement create_training_env
#
# Create and wrap the LoadBalancerEnv for use with Stable-Baselines3.
# SB3 requires environments wrapped in specific ways for training.
#
# You need to:
# 1. Create a LoadBalancerEnv with the given num_servers and max_queue.
# 2. Wrap it with Monitor(env, str(LOGS_DIR / "train")) to record
#    episode rewards and lengths. Monitor is SB3's logging wrapper —
#    it tracks per-episode statistics that appear in training logs.
# 3. Alternatively, use make_vec_env() with a lambda that creates the
#    env. Vectorized envs allow parallel data collection (n_envs > 1),
#    which speeds up training. For simplicity, n_envs=1 is fine.
# 4. Return the wrapped environment.
#
# Key insight: SB3's algorithms expect either a Gymnasium env or a
# VecEnv. The Monitor wrapper records episode stats to a CSV file
# in the logs directory, which you can later plot to visualize the
# training curve. Without Monitor, you only see aggregate loss metrics.
#
# Signature:
#   def create_training_env(
#       num_servers: int = 5,
#       max_queue: int = 20,
#       max_steps: int = 500,
#   ) -> Monitor | gym.Env:

def create_training_env(
    num_servers: int = 5,
    max_queue: int = 20,
    max_steps: int = 500,
):
    """Create a Monitor-wrapped LoadBalancerEnv for SB3 training."""
    raise NotImplementedError(
        "TODO(human): Create LoadBalancerEnv, wrap with Monitor, return."
    )


# TODO(human): Implement train_agent
#
# Create and train a Stable-Baselines3 agent (PPO or DQN) on the
# load balancer environment.
#
# You need to:
# 1. Choose an algorithm class: PPO or DQN (both work for Discrete
#    action spaces). PPO is generally more stable for custom envs;
#    DQN can be more sample-efficient but needs a replay buffer.
#
# 2. Instantiate the model:
#      model = PPO(
#          "MlpPolicy",          # Multi-layer perceptron policy
#          env,                   # The wrapped environment
#          learning_rate=hyperparams.get("learning_rate", 3e-4),
#          n_steps=hyperparams.get("n_steps", 2048),
#          batch_size=hyperparams.get("batch_size", 64),
#          n_epochs=hyperparams.get("n_epochs", 10),
#          gamma=hyperparams.get("gamma", 0.99),
#          verbose=1,             # Print training progress
#          tensorboard_log=str(LOGS_DIR),
#      )
#    For DQN, the signature differs slightly:
#      model = DQN(
#          "MlpPolicy", env,
#          learning_rate=..., batch_size=..., gamma=...,
#          exploration_fraction=0.2,  # fraction of training for epsilon decay
#          verbose=1, tensorboard_log=str(LOGS_DIR),
#      )
#
# 3. Optionally create an EvalCallback to periodically evaluate the
#    agent on a separate env and save the best model:
#      eval_env = LoadBalancerEnv(num_servers=..., max_queue=...)
#      eval_callback = EvalCallback(
#          eval_env,
#          best_model_save_path=str(MODELS_DIR),
#          log_path=str(LOGS_DIR),
#          eval_freq=5000,
#          deterministic=True,
#      )
#
# 4. Train: model.learn(total_timesteps=total_timesteps, callback=eval_callback)
#
# 5. Return the trained model.
#
# Key hyperparameter intuitions:
#   - learning_rate: 3e-4 is a safe default. Too high → unstable, too low → slow.
#   - n_steps (PPO): how many steps to collect before each policy update.
#     Larger = more stable gradients but slower updates.
#   - batch_size: mini-batch size for gradient updates. 64 is standard.
#   - gamma: discount factor. 0.99 means the agent values future rewards.
#     For load balancing, 0.95-0.99 works well.
#
# Signature:
#   def train_agent(
#       env,
#       algorithm: str = "PPO",
#       total_timesteps: int = 50_000,
#       hyperparams: dict | None = None,
#   ) -> PPO | DQN:

def train_agent(
    env,
    algorithm: str = "PPO",
    total_timesteps: int = 50_000,
    hyperparams: dict | None = None,
) -> PPO | DQN:
    """Create and train an SB3 agent on the load balancer environment."""
    raise NotImplementedError(
        "TODO(human): Instantiate PPO or DQN with hyperparams, call .learn(), "
        "return the trained model."
    )


# ---------------------------------------------------------------------------
# Main: train and save the agent
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Exercise 3: Train RL Agent for Load Balancing")
    print("=" * 60)

    # Ensure output directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Create environment ---
    print("\nCreating training environment...")
    env = create_training_env(num_servers=5, max_queue=20, max_steps=500)

    # --- Train ---
    print("\nTraining PPO agent (50,000 timesteps)...")
    print("  This may take a few minutes.\n")

    hyperparams = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
    }

    model = train_agent(
        env,
        algorithm="PPO",
        total_timesteps=50_000,
        hyperparams=hyperparams,
    )

    # --- Save ---
    save_path = MODELS_DIR / "ppo_load_balancer"
    model.save(str(save_path))
    print(f"\nModel saved to: {save_path}")

    # --- Quick test ---
    print("\nQuick test: 10 steps with trained agent...")
    test_env = LoadBalancerEnv(num_servers=5, max_queue=20, max_steps=500)
    obs, _ = test_env.reset(seed=99)
    for _ in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(int(action))
        print(f"  Action: server {action}, Latency: {info['latency']:.3f}, "
              f"Reward: {reward:.3f}, Queue total: {info['queue_total']}")
        if terminated or truncated:
            break

    test_env.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
