"""Collect transitions from CartPole using a random policy.

This module is fully implemented. Run it to generate a dataset of
(state, action, reward, next_state, done) transitions that will be used
to train the neural world model in a supervised fashion.

Usage:
    uv run python -m app.data_collection
"""

from pathlib import Path

import gymnasium as gym
import numpy as np

from app.config import DATA_COLLECTION, ENV
from app.replay_buffer import ReplayBuffer, Transition


def collect_random_transitions(
    num_episodes: int,
    max_steps: int,
) -> ReplayBuffer:
    """Collect transitions by acting randomly in the environment.

    A random policy is sufficient for initial data collection because CartPole
    has a small state space and random actions naturally explore the dynamics
    (the pole falls in diverse ways from diverse states).

    Returns a ReplayBuffer filled with the collected transitions.
    """
    env = gym.make(ENV.name)
    buffer = ReplayBuffer(capacity=num_episodes * max_steps)
    total_transitions = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        for _step in range(max_steps):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(
                Transition(
                    state=np.array(state, dtype=np.float32),
                    action=action,
                    reward=float(reward),
                    next_state=np.array(next_state, dtype=np.float32),
                    done=done,
                )
            )
            total_transitions += 1

            if done:
                break
            state = next_state

    env.close()
    print(f"Collected {total_transitions} transitions over {num_episodes} episodes.")
    return buffer


def save_buffer(buffer: ReplayBuffer, path: str) -> None:
    """Save replay buffer contents to a compressed NumPy file."""
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    batch = buffer.all_data()
    np.savez_compressed(
        filepath,
        states=batch.states,
        actions=batch.actions,
        rewards=batch.rewards,
        next_states=batch.next_states,
        dones=batch.dones,
    )
    print(f"Saved {buffer.size} transitions to {filepath}")


def load_buffer(path: str, capacity: int | None = None) -> ReplayBuffer:
    """Load transitions from a saved NumPy file into a ReplayBuffer."""
    data = np.load(path)
    n = len(data["states"])
    buffer = ReplayBuffer(capacity=capacity or n)
    buffer.load_from_arrays(
        states=data["states"],
        actions=data["actions"],
        rewards=data["rewards"],
        next_states=data["next_states"],
        dones=data["dones"],
    )
    print(f"Loaded {buffer.size} transitions from {path}")
    return buffer


def main() -> None:
    """Collect random transitions and save to disk."""
    buffer = collect_random_transitions(
        num_episodes=DATA_COLLECTION.num_episodes,
        max_steps=DATA_COLLECTION.max_steps_per_episode,
    )
    save_buffer(buffer, DATA_COLLECTION.save_path)

    # Print some statistics about the collected data
    batch = buffer.all_data()
    print(f"\nDataset statistics:")
    print(f"  States  -- mean: {batch.states.mean(axis=0)}, std: {batch.states.std(axis=0)}")
    print(f"  Rewards -- mean: {batch.rewards.mean():.2f}, std: {batch.rewards.std():.2f}")
    print(f"  Done fraction: {batch.dones.mean():.3f}")


if __name__ == "__main__":
    main()
