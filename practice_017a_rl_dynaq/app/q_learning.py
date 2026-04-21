"""Tabular Q-Learning agent for discrete environments.

The model-free baseline.  The agent learns Q(s, a) values purely from real
environment interactions -- no model, no planning.

Three small focused TODOs:
  - `select_action`: epsilon-greedy exploration
  - `update`      : one-step Q-learning TD update
  - `train_episode`: roll out one full episode

The training harness (`train`) and the __main__ runner are scaffolded.

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

    # -- TODO 1 ----------------------------------------------------------------

    def select_action(self, state: int) -> int:
        """Return an epsilon-greedy action for *state*.

        With probability `self.params.epsilon` pick a uniform random action
        (explore); otherwise pick `argmax_a Q(state, a)` (exploit).  Use
        `self.rng.random()` and `self.rng.integers(self.n_actions)` for
        reproducibility.
        """
        # TODO(human): coin-flip epsilon, then either random integer action
        # or np.argmax over self.q_table[state].
        raise NotImplementedError("Implement select_action()")

    # -- TODO 2 ----------------------------------------------------------------

    def update(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> None:
        """Apply one-step Q-learning update in place on `self.q_table`.

        Target = `reward` when `done`, else `reward + gamma * max_a' Q(s', a')`.
        TD error = target - Q(s, a).  Update: Q(s, a) += alpha * TD_error.
        """
        # TODO(human): compute the bootstrapped target (respecting terminal
        # states) and do the in-place Q(s, a) += alpha * (target - Q(s, a)) update.
        raise NotImplementedError("Implement update()")

    # -- TODO 3 ----------------------------------------------------------------

    def train_episode(self, env: gym.Env) -> float:
        """Run one episode end-to-end, returning the total (undiscounted) reward.

        Reset the env, then loop: select an action, step the env, call
        `self.update(...)`, accumulate reward, advance state, and stop when
        `terminated or truncated`.
        """
        # TODO(human): implement the action -> step -> update -> advance loop.
        raise NotImplementedError("Implement train_episode()")


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
