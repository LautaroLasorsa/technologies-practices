"""Dyna-style model-based agent combining real experience with simulated rollouts.

This is the main training script. It implements a DQN agent that also trains
a neural world model and uses it to generate additional (simulated) training
data -- the core idea behind Dyna-Q extended to neural function approximation.

The training loop:
    1. Act in the real environment (epsilon-greedy on Q-values)
    2. Store transition in replay buffer
    3. Train DQN on one real mini-batch
    4. (Every N episodes) Retrain the world model on all collected data
    5. For each real step, do K simulated steps:
       a. Sample a real state from the buffer
       b. Imagine a short rollout with the world model
       c. Train DQN on the simulated transitions

Usage:
    uv run python -m app.agent
"""

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from app.config import AGENT, DQN, ENV, WORLD_MODEL
from app.replay_buffer import ReplayBuffer, Transition, TransitionBatch
from app.train_model import split_buffer, train_world_model
from app.world_model import WorldModel


# ---------------------------------------------------------------------------
# Q-Network (DQN)
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    """Simple MLP Q-network: state -> Q-values for each action.

    This is a standard fully-connected network. It is fully implemented
    so you can focus on the RL training logic.
    """

    def __init__(
        self,
        state_dim: int = ENV.state_dim,
        action_dim: int = ENV.action_dim,
        hidden_dim: int = DQN.hidden_dim,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions given a batch of states.

        Args:
            state: Shape (batch, state_dim).

        Returns:
            Q-values, shape (batch, action_dim).
        """
        return self.net(state)


# ---------------------------------------------------------------------------
# DQN Agent (action selection + training)
# ---------------------------------------------------------------------------


class DQNAgent:
    """DQN agent with epsilon-greedy exploration and target network.

    Attributes:
        q_net: Online Q-network (updated every step).
        target_net: Target Q-network (synced periodically for stability).
        optimizer: Adam optimizer for q_net.
        epsilon: Current exploration rate.
        device: Torch device.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.q_net = QNetwork().to(device)
        self.target_net = QNetwork().to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = Adam(self.q_net.parameters(), lr=DQN.learning_rate)
        self.epsilon = DQN.epsilon_start

    def select_action(self, state: np.ndarray) -> int:
        """Choose an action using epsilon-greedy policy on Q-values.

        Args:
            state: Current state as numpy array, shape (state_dim,).

        Returns:
            Action index (int).
        """
        # TODO(human): Implement epsilon-greedy action selection.
        #
        # Steps:
        #   1. With probability self.epsilon, return a random action:
        #      np.random.randint(ENV.action_dim)
        #   2. Otherwise, compute Q-values using self.q_net:
        #      a. Convert state to tensor: torch.tensor(state, dtype=torch.float32, device=self.device)
        #      b. Add batch dim: state_tensor.unsqueeze(0)
        #      c. Forward pass (inside torch.no_grad()): q_values = self.q_net(state_tensor)
        #      d. Return the action with highest Q-value: q_values.argmax(dim=1).item()
        raise NotImplementedError("TODO(human): Implement epsilon-greedy action selection")

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(DQN.epsilon_end, self.epsilon * DQN.epsilon_decay)

    def sync_target_network(self) -> None:
        """Copy online Q-network weights to target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train_on_batch(self, batch: TransitionBatch) -> float:
        """Update Q-network on a mini-batch of transitions.

        Uses the standard DQN loss:
            target = reward + gamma * max_a' Q_target(next_state, a') * (1 - done)
            loss   = MSE(Q(state, action), target)

        Args:
            batch: A TransitionBatch of experience (real or simulated).

        Returns:
            Loss value as float.
        """
        # TODO(human): Implement the DQN training step.
        #
        # Steps:
        #   1. Convert batch arrays to tensors on self.device:
        #      states      = torch.tensor(batch.states, ..., device=self.device)
        #      actions     = torch.tensor(batch.actions, dtype=torch.long, ...)
        #      rewards     = torch.tensor(batch.rewards, ...)
        #      next_states = torch.tensor(batch.next_states, ...)
        #      dones       = torch.tensor(batch.dones, ...)
        #
        #   2. Current Q-values for the taken actions:
        #      q_values = self.q_net(states)                    # (batch, action_dim)
        #      q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch,)
        #
        #   3. Target Q-values (no gradient!):
        #      with torch.no_grad():
        #          next_q = self.target_net(next_states).max(dim=1).values  # (batch,)
        #          targets = rewards + DQN.gamma * next_q * (1.0 - dones)   # (batch,)
        #
        #   4. Loss and backprop:
        #      loss = F.mse_loss(q_values, targets)
        #      self.optimizer.zero_grad()
        #      loss.backward()
        #      self.optimizer.step()
        #
        #   5. Return loss.item()
        raise NotImplementedError("TODO(human): Implement DQN training step")


# ---------------------------------------------------------------------------
# Dyna-style training loop
# ---------------------------------------------------------------------------


def train_dyna_agent(
    use_model: bool = True,
) -> list[float]:
    """Train a DQN agent with optional Dyna-style model-based augmentation.

    Args:
        use_model: If True, train a world model and use simulated rollouts.
                   If False, train a pure model-free DQN (baseline).

    Returns:
        List of episode rewards (total reward per episode).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(ENV.name)
    agent = DQNAgent(device)
    buffer = ReplayBuffer(AGENT.replay_buffer_capacity)

    world_model: WorldModel | None = None
    if use_model:
        world_model = WorldModel().to(device)
        wm_optimizer = Adam(world_model.parameters(), lr=WORLD_MODEL.learning_rate)

    episode_rewards: list[float] = []

    for episode in range(AGENT.total_episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0.0

        for _step in range(AGENT.max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32)

            buffer.push(Transition(state, action, float(reward), next_state, done))
            episode_reward += reward

            # --- Train DQN on real data ---
            if buffer.size >= AGENT.min_buffer_size:
                real_batch = buffer.sample(AGENT.batch_size)
                agent.train_on_batch(real_batch)

                # --- Simulated rollouts (Dyna step) ---
                if use_model and world_model is not None:
                    _train_on_simulated_data(agent, world_model, buffer, device)

            if done:
                break
            state = next_state

        episode_rewards.append(episode_reward)
        agent.decay_epsilon()

        # --- Sync target network ---
        if (episode + 1) % DQN.target_update_freq == 0:
            agent.sync_target_network()

        # --- Retrain world model periodically ---
        if use_model and world_model is not None and (episode + 1) % AGENT.model_train_freq == 0:
            if buffer.size >= AGENT.min_buffer_size:
                _retrain_world_model(world_model, buffer)

        # --- Logging ---
        if (episode + 1) % 20 == 0:
            recent = episode_rewards[-20:]
            mode = "Dyna" if use_model else "DQN"
            print(
                f"[{mode}] Episode {episode+1:3d}  "
                f"avg_reward={np.mean(recent):.1f}  "
                f"epsilon={agent.epsilon:.3f}"
            )

    env.close()
    return episode_rewards


def _train_on_simulated_data(
    agent: DQNAgent,
    world_model: WorldModel,
    buffer: ReplayBuffer,
    device: torch.device,
) -> None:
    """Generate simulated transitions from the world model and train the DQN.

    For each of K simulated steps:
        1. Sample a real state from the buffer
        2. Pick an action (using the agent's current policy)
        3. Predict (next_state, reward, done) with the world model
        4. Create a TransitionBatch from the simulated transition
        5. Train the DQN on it

    Args:
        agent: The DQN agent.
        world_model: Trained world model.
        buffer: Replay buffer (to sample starting states).
        device: Torch device.
    """
    # TODO(human): Implement Dyna-style simulated training.
    #
    # Steps:
    #   for _ in range(AGENT.simulated_steps_per_real_step):
    #       1. Sample a batch of starting states from the buffer:
    #          start_states = buffer.sample_states(AGENT.batch_size)
    #
    #       2. For each state in the batch, use agent.select_action() to pick an action
    #          actions = np.array([agent.select_action(s) for s in start_states])
    #
    #       3. Use the world model to predict next states, rewards, dones:
    #          - Convert start_states to tensor
    #          - Convert actions to tensor (long)
    #          - Forward pass: pred_ns, pred_r, pred_d = world_model(states_t, actions_t)
    #          - Convert predictions to numpy
    #
    #       4. Build a TransitionBatch from the simulated data:
    #          sim_batch = TransitionBatch(
    #              states=start_states,
    #              actions=actions,
    #              rewards=pred_rewards_np,
    #              next_states=pred_next_states_np,
    #              dones=pred_dones_np,   # threshold at 0.5
    #          )
    #
    #       5. Train the agent: agent.train_on_batch(sim_batch)
    #
    # Hint: Use torch.no_grad() for the world model forward pass (we don't
    #       want gradients flowing through the world model during DQN training).
    raise NotImplementedError("TODO(human): Implement Dyna-style simulated training")


def _retrain_world_model(world_model: WorldModel, buffer: ReplayBuffer) -> None:
    """Retrain the world model on the current replay buffer contents."""
    train_data, val_data = split_buffer(buffer, WORLD_MODEL.validation_split)
    train_world_model(
        world_model,
        train_data,
        val_data,
        epochs=AGENT.model_train_epochs,
        batch_size=AGENT.batch_size,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def smooth(values: list[float], window: int = 20) -> np.ndarray:
    """Apply a moving average to smooth noisy reward curves."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_comparison(
    dqn_rewards: list[float],
    dyna_rewards: list[float],
    save_path: str,
) -> None:
    """Plot learning curves: model-free DQN vs. Dyna-style DQN."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(smooth(dqn_rewards), label="Model-Free DQN", linewidth=2, alpha=0.8)
    ax.plot(smooth(dyna_rewards), label="Dyna DQN (model-based)", linewidth=2, alpha=0.8)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward (smoothed)")
    ax.set_title("Model-Free vs. Model-Based DQN on CartPole-v1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Train both a model-free and model-based agent, then compare."""
    print("=" * 60)
    print("Training Model-Free DQN (baseline)")
    print("=" * 60)
    dqn_rewards = train_dyna_agent(use_model=False)

    print()
    print("=" * 60)
    print("Training Dyna DQN (model-based)")
    print("=" * 60)
    dyna_rewards = train_dyna_agent(use_model=True)

    plot_comparison(dqn_rewards, dyna_rewards, "plots/dqn_vs_dyna.png")

    print("\nFinal results (last 50 episodes):")
    print(f"  Model-Free DQN: {np.mean(dqn_rewards[-50:]):.1f} avg reward")
    print(f"  Dyna DQN:       {np.mean(dyna_rewards[-50:]):.1f} avg reward")


if __name__ == "__main__":
    main()
