"""Hyperparameters for simplified Dreamer on CartPole-v1."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DreamerConfig:
    """All hyperparameters in one place.

    Grouped by component so it's clear which values feed into which module.
    """

    # --- Environment ---
    env_name: str = "CartPole-v1"
    obs_dim: int = 4       # CartPole observation space
    action_dim: int = 2    # CartPole discrete actions

    # --- RSSM dimensions ---
    deterministic_dim: int = 200    # GRU hidden state size (h)
    stochastic_dim: int = 30       # Stochastic latent size (z)
    embedding_dim: int = 64        # Encoder output size

    # --- Network hidden sizes ---
    encoder_hidden: int = 128
    decoder_hidden: int = 128
    rssm_hidden: int = 200         # MLP hidden size inside RSSM sub-models
    actor_hidden: int = 128
    critic_hidden: int = 128

    # --- Training ---
    seed: int = 42
    total_epochs: int = 300
    collect_episodes_per_epoch: int = 1
    train_steps_per_epoch: int = 10
    sequence_length: int = 50      # Length of sequences sampled from replay
    batch_size: int = 16

    # --- World model training ---
    world_model_lr: float = 3e-4
    kl_weight: float = 1.0         # Beta for KL term in world model loss
    kl_free_nats: float = 0.0      # Free nats (0 = no free nats, simplified)

    # --- Actor-critic ---
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    imagination_horizon: int = 15  # Steps to imagine into the future
    gamma: float = 0.99            # Discount factor
    lambda_gae: float = 0.95       # Lambda for generalized advantage / lambda-returns
    entropy_weight: float = 1e-3   # Entropy bonus for actor

    # --- Replay buffer ---
    buffer_capacity: int = 100     # Max episodes stored

    # --- Derived (computed) ---
    @property
    def latent_dim(self) -> int:
        """Full latent state dimension: deterministic + stochastic concatenated."""
        return self.deterministic_dim + self.stochastic_dim
