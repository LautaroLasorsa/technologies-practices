"""Tabular Q-Learning agent for discrete environments.

This is the model-free baseline. The agent learns Q(s, a) values purely from
real environment interactions -- no model, no planning.

Reference: Sutton & Barto, Section 6.5 (Q-learning: Off-Policy TD Control)
    Q(s, a) += alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class QParams:
    """Hyperparameters for Q-learning."""

    env_id: str = "CliffWalking-v1"
    n_episodes: int = 500
    alpha: float = 0.1          # learning rate
    gamma: float = 0.99         # discount factor
    epsilon: float = 0.1        # exploration rate (epsilon-greedy)
    seed: int = 42


class QLearningAgent:
    """Tabular Q-learning agent.

    Attributes:
        q_table: np.ndarray of shape (n_states, n_actions) holding Q-values.
    """

    def __init__(self, n_states: int, n_actions: int, params: QParams) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.params = params
        self.q_table = np.zeros((n_states, n_actions))
        self.rng = np.random.default_rng(params.seed)

    def select_action(self, state: int) -> int:
        """Choose an action using epsilon-greedy policy.

        TODO(human): Implement epsilon-greedy action selection.

        With probability epsilon, pick a random action (exploration).
        Otherwise, pick the action with the highest Q-value for this state (exploitation).
        Use self.rng for randomness (self.rng.random() for uniform, self.rng.integers(self.n_actions) for random action).

        Hint: This is the same explore/exploit tradeoff from multi-armed bandits.
        Think of it like a CP heuristic: sometimes try a random branch, usually go greedy.
        """
        raise NotImplementedError("TODO(human): implement epsilon-greedy action selection")

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """Apply one-step Q-learning update.

        TODO(human): Implement the Q-learning update rule.

        Formula: Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        When done is True, the target is just `r` (no future rewards from terminal state).

        Hint: np.max(self.q_table[next_state]) gives max_a' Q(s', a').
        """
        raise NotImplementedError("TODO(human): implement Q-learning update rule")

    def train_episode(self, env: gym.Env) -> float:
        """Run one episode of Q-learning, returning total reward.

        TODO(human): Implement the episode training loop.

        Steps:
        1. Reset the environment: state, _ = env.reset()
        2. Loop until done (or truncated):
           a. Select action with self.select_action(state)
           b. Step the environment: next_state, reward, terminated, truncated, _ = env.step(action)
           c. Update Q-values with self.update(state, action, reward, next_state, terminated)
           d. Accumulate reward
           e. Set done = terminated or truncated, advance state = next_state
        3. Return total episode reward

        Hint: CliffWalking optimal reward is -13 (13 steps at -1 each).
        """
        raise NotImplementedError("TODO(human): implement episode training loop")


def train(params: QParams | None = None) -> tuple[QLearningAgent, list[float]]:
    """Train a Q-learning agent and return (agent, reward_history).

    This is fully implemented -- it calls your train_episode in a loop.
    """
    params = params or QParams()
    env = gym.make(params.env_id)
    env.reset(seed=params.seed)

    n_states = env.observation_space.n  # type: ignore[attr-defined]
    n_actions = env.action_space.n  # type: ignore[attr-defined]

    agent = QLearningAgent(n_states, n_actions, params)
    reward_history: list[float] = []

    for episode in range(params.n_episodes):
        episode_reward = agent.train_episode(env)
        reward_history.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(reward_history[-100:])
            print(f"Q-Learning | Episode {episode + 1:>4d} | Avg reward (last 100): {avg:.1f}")

    env.close()
    return agent, reward_history


if __name__ == "__main__":
    agent, rewards = train()
    print(f"\nFinal avg reward (last 50): {np.mean(rewards[-50:]):.1f}")
    print(f"Best episode reward: {max(rewards):.0f}")
