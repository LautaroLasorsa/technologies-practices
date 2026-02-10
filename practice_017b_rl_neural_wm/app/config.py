"""Shared configuration for the neural world model practice."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    """CartPole-v1 environment parameters."""

    name: str = "CartPole-v1"
    state_dim: int = 4  # cart_pos, cart_vel, pole_angle, pole_angular_vel
    action_dim: int = 2  # left (0), right (1)


@dataclass(frozen=True)
class WorldModelConfig:
    """Hyperparameters for the learned dynamics model."""

    hidden_dim: int = 128
    num_hidden_layers: int = 2
    learning_rate: float = 1e-3
    batch_size: int = 64
    train_epochs: int = 50
    validation_split: float = 0.1
    done_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    state_loss_weight: float = 1.0


@dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters for the DQN policy network."""

    hidden_dim: int = 128
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_freq: int = 10  # episodes between target network sync


@dataclass(frozen=True)
class AgentConfig:
    """Hyperparameters for the Dyna-style model-based agent."""

    total_episodes: int = 300
    max_steps_per_episode: int = 500
    replay_buffer_capacity: int = 50_000
    min_buffer_size: int = 500  # collect this many transitions before training
    simulated_steps_per_real_step: int = 5  # K in Dyna: model rollouts per real step
    imagined_rollout_length: int = 1  # how many steps to unroll the model
    model_train_freq: int = 10  # retrain world model every N episodes
    model_train_epochs: int = 10  # epochs per retrain
    batch_size: int = 64


@dataclass(frozen=True)
class DataCollectionConfig:
    """Parameters for the initial random data collection phase."""

    num_episodes: int = 100
    max_steps_per_episode: int = 500
    save_path: str = "data/transitions.npz"


ENV = EnvConfig()
WORLD_MODEL = WorldModelConfig()
DQN = DQNConfig()
AGENT = AgentConfig()
DATA_COLLECTION = DataCollectionConfig()
