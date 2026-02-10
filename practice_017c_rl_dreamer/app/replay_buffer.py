"""Simple episode replay buffer for Dreamer.

Stores complete episodes and samples random sub-sequences for training.
Dreamer needs sequences (not individual transitions) because the RSSM
is a recurrent model that processes observations over time.

This module is fully implemented -- no TODOs.
"""

import random
from dataclasses import dataclass, field

import numpy as np
import torch

from app.config import DreamerConfig


@dataclass
class Episode:
    """A single episode stored as numpy arrays."""

    observations: np.ndarray   # (T, obs_dim)
    actions: np.ndarray        # (T,) integer actions
    rewards: np.ndarray        # (T,)

    @property
    def length(self) -> int:
        return len(self.observations)


class ReplayBuffer:
    """Stores episodes and samples batched subsequences."""

    def __init__(self, config: DreamerConfig) -> None:
        self.config = config
        self.episodes: list[Episode] = []

    def add_episode(self, episode: Episode) -> None:
        """Add an episode, evicting the oldest if at capacity."""
        self.episodes.append(episode)
        while len(self.episodes) > self.config.buffer_capacity:
            self.episodes.pop(0)

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample a batch of subsequences from stored episodes.

        Returns tensors ready for world-model training:
            "observations": (batch, seq_len, obs_dim)
            "actions": (batch, seq_len, action_dim)  -- one-hot
            "rewards": (batch, seq_len)
        """
        batch_obs = []
        batch_actions = []
        batch_rewards = []

        eligible = [ep for ep in self.episodes if ep.length >= self.config.sequence_length]
        if not eligible:
            eligible = self.episodes

        for _ in range(self.config.batch_size):
            episode = random.choice(eligible)
            max_start = max(0, episode.length - self.config.sequence_length)
            start = random.randint(0, max_start)
            end = start + min(self.config.sequence_length, episode.length)

            obs_seq = episode.observations[start:end]
            act_seq = episode.actions[start:end]
            rew_seq = episode.rewards[start:end]

            # Pad if shorter than sequence_length
            actual_len = len(obs_seq)
            if actual_len < self.config.sequence_length:
                pad_len = self.config.sequence_length - actual_len
                obs_seq = np.concatenate([
                    obs_seq,
                    np.zeros((pad_len, self.config.obs_dim)),
                ])
                act_seq = np.concatenate([act_seq, np.zeros(pad_len)])
                rew_seq = np.concatenate([rew_seq, np.zeros(pad_len)])

            batch_obs.append(obs_seq)
            batch_actions.append(act_seq)
            batch_rewards.append(rew_seq)

        obs_tensor = torch.tensor(np.array(batch_obs), dtype=torch.float32)
        rewards_tensor = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

        # One-hot encode actions
        actions_int = torch.tensor(np.array(batch_actions), dtype=torch.long)
        actions_onehot = torch.nn.functional.one_hot(
            actions_int, num_classes=self.config.action_dim,
        ).float()

        return {
            "observations": obs_tensor,
            "actions": actions_onehot,
            "rewards": rewards_tensor,
        }

    @property
    def total_steps(self) -> int:
        return sum(ep.length for ep in self.episodes)

    def __len__(self) -> int:
        return len(self.episodes)
