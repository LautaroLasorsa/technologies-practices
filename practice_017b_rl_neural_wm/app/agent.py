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

    def _batch_to_tensors(
        self, batch: TransitionBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a TransitionBatch to tensors on self.device."""
        return (
            torch.tensor(batch.states, dtype=torch.float32, device=self.device),
            torch.tensor(batch.actions, dtype=torch.long, device=self.device),
            torch.tensor(batch.rewards, dtype=torch.float32, device=self.device),
            torch.tensor(batch.next_states, dtype=torch.float32, device=self.device),
            torch.tensor(batch.dones, dtype=torch.float32, device=self.device),
        )

    def compute_td_target(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the TD target using the (frozen) target network.

        The Bellman target for DQN is:
            target = reward + gamma * max_a' Q_target(next_state, a') * (1 - done)

        Returns a tensor of shape (batch,). Must be computed without gradients
        so backprop does not flow into the target network.
        """
        # ── Exercise Context ──────────────────────────────────────────────────
        # This is the bootstrapped target DQN trains against. Using a *frozen*
        # target network (instead of self.q_net) stabilises training — otherwise
        # the target moves every step and Q can diverge.

        # TODO(human): Compute the TD target.
        #
        # Steps:
        #   1. Wrap the whole computation in `with torch.no_grad():`
        #   2. next_q = self.target_net(next_states).max(dim=1).values  # (batch,)
        #   3. targets = rewards + DQN.gamma * next_q * (1.0 - dones)   # (batch,)
        #   4. Return targets
        raise NotImplementedError("TODO(human): Implement compute_td_target")

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
        states, actions, rewards, next_states, dones = self._batch_to_tensors(batch)

        # ── Exercise Context ──────────────────────────────────────────────────
        # This is the core DQN update: predict Q(s,a), compare against the TD
        # target, and take a gradient step. The gather(...) trick selects, for
        # each sample in the batch, the Q-value of the action actually taken.

        # TODO(human): Implement the DQN gradient step.
        #
        # Steps:
        #   1. Current Q-values for the taken actions:
        #      q_all = self.q_net(states)                               # (batch, action_dim)
        #      q_values = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch,)
        #
        #   2. Targets (no gradient): use self.compute_td_target(rewards, next_states, dones)
        #
        #   3. Loss and backprop:
        #      loss = F.mse_loss(q_values, targets)
        #      self.optimizer.zero_grad()
        #      loss.backward()
        #      self.optimizer.step()
        #
        #   4. Return loss.item()
        raise NotImplementedError("TODO(human): Implement DQN gradient step")


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


def _simulate_batch(
    agent: DQNAgent,
    world_model: WorldModel,
    buffer: ReplayBuffer,
    device: torch.device,
) -> TransitionBatch:
    """Generate one simulated mini-batch using the world model.

    Sample real starting states from the buffer, pick actions with the agent's
    current policy, and roll one step forward through the learned dynamics.
    Return a TransitionBatch that looks just like a real batch so the DQN
    update path doesn't need to know the difference.
    """
    # ── Exercise Context ──────────────────────────────────────────────────
    # This is the "imagination" step of Dyna — fabricating additional training
    # data with the learned model instead of paying for real environment steps.
    # The world model is used in no_grad mode: we are *using* it, not training
    # it here.

    # TODO(human): Build and return a simulated TransitionBatch.
    #
    # Steps:
    #   1. Sample starting states from the buffer:
    #      start_states = buffer.sample_states(AGENT.batch_size)
    #
    #   2. Pick an action per state using the agent's policy:
    #      actions = np.array([agent.select_action(s) for s in start_states])
    #
    #   3. Predict next_state / reward / done with the world model:
    #      states_t  = torch.tensor(start_states, dtype=torch.float32, device=device)
    #      actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    #      with torch.no_grad():
    #          pred_ns, pred_r, pred_d = world_model(states_t, actions_t)
    #
    #   4. Convert predictions to numpy and assemble a TransitionBatch:
    #      next_states = pred_ns.cpu().numpy()
    #      rewards     = pred_r.squeeze(-1).cpu().numpy()
    #      dones       = (pred_d.squeeze(-1).cpu().numpy() > 0.5).astype(np.float32)
    #      return TransitionBatch(states=start_states, actions=actions,
    #                             rewards=rewards, next_states=next_states, dones=dones)
    raise NotImplementedError("TODO(human): Implement _simulate_batch")


def _train_on_simulated_data(
    agent: DQNAgent,
    world_model: WorldModel,
    buffer: ReplayBuffer,
    device: torch.device,
) -> None:
    """Run K Dyna updates: simulate a batch, train the DQN on it, repeat.

    K = AGENT.simulated_steps_per_real_step. This is the knob that controls
    the real/simulated mixing ratio — the central design choice of Dyna.
    """
    # TODO(human): Loop K times, simulate a batch, train the agent on it.
    #
    # for _ in range(AGENT.simulated_steps_per_real_step):
    #     sim_batch = _simulate_batch(agent, world_model, buffer, device)
    #     agent.train_on_batch(sim_batch)
    raise NotImplementedError("TODO(human): Implement the Dyna K-step loop")


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
