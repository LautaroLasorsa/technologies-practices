"""Dyna-Q agent: Q-learning + learned model + planning steps.

Dyna-Q extends Q-learning by adding two components:
1. A tabular environment model that memorizes transitions
2. Planning steps that sample from the model to do extra Q-value updates

The key insight: each real interaction updates Q once (direct RL) AND feeds the
model. Then the model is sampled n times for additional Q updates (planning).
This makes the agent much more sample-efficient -- it extracts more learning
from each real experience.

Reference: Sutton & Barto, Chapter 8, Figure 8.4 (Tabular Dyna-Q)
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from app.environment_model import EnvironmentModel
from app.q_learning import QLearningAgent, QParams


@dataclass
class DynaQParams(QParams):
    """Hyperparameters for Dyna-Q (extends Q-learning params)."""

    n_planning_steps: int = 10  # number of model-based updates per real step


class DynaQAgent(QLearningAgent):
    """Dyna-Q agent: Q-learning + model + planning.

    Inherits Q-table, select_action, and update from QLearningAgent.
    Adds a learned model and planning loop.
    """

    def __init__(self, n_states: int, n_actions: int, params: DynaQParams) -> None:
        super().__init__(n_states, n_actions, params)
        self.model = EnvironmentModel(seed=params.seed)
        self.n_planning_steps = params.n_planning_steps

    def train_episode(self, env: gym.Env) -> float:
        """Run one episode of Dyna-Q, returning total reward.

        TODO(human): Implement the Dyna-Q episode loop.

        This is like Q-learning's train_episode, but after each real step you add:
        1. Model update: self.model.update(state, action, reward, next_state, terminated)
        2. Planning loop: repeat self.n_planning_steps times:
           a. Sample from model: s, a, r, s_next, d = self.model.sample()
           b. Q-update on sampled data: self.update(s, a, r, s_next, d)

        Full step order per timestep:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            self.update(state, action, reward, next_state, terminated)     # direct RL
            self.model.update(state, action, reward, next_state, terminated)  # model learning
            for _ in range(self.n_planning_steps):                         # planning
                s, a, r, s_next, d = self.model.sample()
                self.update(s, a, r, s_next, d)

        Hint: The only difference from Q-learning is lines 4-7 above.
        If n_planning_steps=0, this is identical to Q-learning.
        """
        raise NotImplementedError("TODO(human): implement Dyna-Q episode loop with planning")


def train(params: DynaQParams | None = None) -> tuple[DynaQAgent, list[float]]:
    """Train a Dyna-Q agent and return (agent, reward_history).

    This is fully implemented -- it calls your train_episode in a loop.
    """
    params = params or DynaQParams()
    env = gym.make(params.env_id)
    env.reset(seed=params.seed)

    n_states = env.observation_space.n  # type: ignore[attr-defined]
    n_actions = env.action_space.n  # type: ignore[attr-defined]

    agent = DynaQAgent(n_states, n_actions, params)
    reward_history: list[float] = []

    for episode in range(params.n_episodes):
        episode_reward = agent.train_episode(env)
        reward_history.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(reward_history[-100:])
            n = params.n_planning_steps
            print(f"Dyna-Q (n={n}) | Episode {episode + 1:>4d} | Avg reward (last 100): {avg:.1f}")

    env.close()
    return agent, reward_history


if __name__ == "__main__":
    agent, rewards = train()
    print(f"\nFinal avg reward (last 50): {np.mean(rewards[-50:]):.1f}")
    print(f"Best episode reward: {max(rewards):.0f}")
    print(f"Model size: {agent.model.size} stored transitions")
