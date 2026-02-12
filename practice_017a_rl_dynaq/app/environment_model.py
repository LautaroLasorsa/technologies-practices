"""Tabular environment model for Dyna-Q.

The model memorizes observed transitions: given (state, action), it predicts
(reward, next_state, done). This is the simplest possible "world model" --
a lookup table. No neural networks, no generalization. It only knows about
(state, action) pairs it has actually seen.

Assumption: the environment is deterministic. Each (s, a) maps to exactly one
(r, s', done). For stochastic environments, you would need to store distributions.

Reference: Sutton & Barto, Chapter 8 -- Model(s, a) = (r, s')
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Transition:
    """A single observed transition."""

    reward: float
    next_state: int
    done: bool


class EnvironmentModel:
    """Tabular model that stores observed (s, a) -> (r, s', done) transitions.

    Internally uses a dict mapping (state, action) -> Transition.
    Also tracks which states have been visited and which actions were taken
    in each state, so we can sample uniformly for planning.

    Attributes:
        transitions: dict mapping (state, action) tuples to Transition objects.
        observed_states: list of states that have been visited at least once.
        observed_actions: dict mapping each observed state to its list of taken actions.
    """

    def __init__(self, seed: int = 42) -> None:
        self.transitions: dict[tuple[int, int], Transition] = {}
        self.observed_states: list[int] = []
        self.observed_actions: dict[int, list[int]] = {}
        self.rng = np.random.default_rng(seed)

    @property
    def size(self) -> int:
        """Number of stored transitions."""
        return len(self.transitions)

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """Record an observed transition.

        # ── Exercise Context ──────────────────────────────────────────────────
        # This teaches model-based RL's first component: learning a model of the environment.
        # By memorizing transitions, the agent can later "replay" them mentally (planning)
        # without needing real environment interaction — the key to sample efficiency.

        TODO(human): Implement model update.

        Steps:
        1. Store the transition: self.transitions[(state, action)] = Transition(reward, next_state, done)
        2. If state not in self.observed_states, append it
        3. If state not in self.observed_actions, initialize it to an empty list
        4. If action not in self.observed_actions[state], append it

        This gives us the bookkeeping needed for uniform sampling in `sample()`.

        Hint: This is just a dictionary write + two "add if not present" operations.
        Think of it as building an adjacency list: for each visited node, track which edges we've explored.
        """
        raise NotImplementedError("TODO(human): implement model update")

    def sample(self) -> tuple[int, int, float, int, bool]:
        """Sample a random previously-observed (s, a) and return (s, a, r, s', done).

        # ── Exercise Context ──────────────────────────────────────────────────
        # This teaches planning via mental simulation. By sampling past transitions,
        # the agent "dreams" about what could happen and updates Q-values on imagined
        # experience — extracting more learning from each real interaction.

        TODO(human): Implement random sampling from the model.

        Steps:
        1. Pick a random state from self.observed_states
        2. Pick a random action from self.observed_actions[state]
        3. Look up the transition: self.transitions[(state, action)]
        4. Return (state, action, transition.reward, transition.next_state, transition.done)

        Use self.rng.integers(len(collection)) to pick random indices.

        Hint: This is how the agent "dreams" -- it replays random past experiences
        from memory to get extra Q-value updates for free (no real env interaction).
        """
        raise NotImplementedError("TODO(human): implement model sampling")
