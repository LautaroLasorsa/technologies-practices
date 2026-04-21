"""Dyna-Q agent: Q-learning + learned model + planning steps.

Dyna-Q extends Q-learning with two additions that make each real interaction
"count more":

1. A tabular `EnvironmentModel` that memorizes observed transitions.
2. A planning loop that, after every real step, samples `n` stored transitions
   and does extra Q-updates on them ("dreams").

The structure of one real step becomes:

    action = self.select_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    self.update(state, action, reward, next_state, terminated)          # direct RL
    self.model.update(state, action, reward, next_state, terminated)    # model learn
    self.plan()                                                          # n planning steps

Two small focused TODOs:
  - `plan()`       : the planning loop -- n samples from the model, n Q-updates
  - `train_episode`: full real-step loop that also feeds the model and calls `plan()`

`__main__` and `train(...)` are scaffolded.

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
    Adds a learned model and a planning loop.
    """

    def __init__(self, n_states: int, n_actions: int, params: DynaQParams) -> None:
        super().__init__(n_states, n_actions, params)
        self.model = EnvironmentModel(seed=params.seed)
        self.n_planning_steps = params.n_planning_steps

    # -- TODO 1 ----------------------------------------------------------------

    def plan(self) -> None:
        """Run `self.n_planning_steps` imagined Q-updates from the model.

        Each step samples a random previously-observed transition via
        `self.model.sample()` and applies the regular `self.update(...)` on
        it -- as if it were real.  If the model is empty (nothing stored yet)
        this is a no-op.  With `n_planning_steps=0` this does nothing and the
        agent is equivalent to Q-learning.
        """
        # TODO(human): if the model has at least one transition, loop
        # n_planning_steps times, sample, and Q-update on the sampled tuple.
        raise NotImplementedError("Implement DynaQAgent.plan()")

    # -- TODO 2 ----------------------------------------------------------------

    def train_episode(self, env: gym.Env) -> float:
        """Run one Dyna-Q episode, returning total reward.

        Same shape as `QLearningAgent.train_episode`, but each real step is
        followed by two extra operations:
          1. `self.model.update(state, action, reward, next_state, terminated)`
             -- feed the model so planning has something to sample.
          2. `self.plan()` -- n simulated Q-updates from the model.

        With `n_planning_steps=0` this reduces to vanilla Q-learning.
        """
        # TODO(human): real-step loop with model feeding + planning after each step.
        raise NotImplementedError("Implement DynaQAgent.train_episode()")


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
