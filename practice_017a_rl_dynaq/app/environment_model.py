"""Tabular environment model for Dyna-Q.

The simplest possible "world model": a dict keyed by (state, action) that
returns the last observed (reward, next_state, done).  No neural nets, no
generalization -- the model only knows about (s, a) pairs it has actually
seen.  We assume the environment is deterministic, so remembering the most
recent transition is enough.

Two small focused TODOs:
  - `update`: record an observed transition + book-keep for sampling
  - `sample`: pick a uniformly random previously-observed (s, a) and replay it

Reference: Sutton & Barto, Chapter 8 -- Model(s, a) = (r, s')
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    """A single observed transition."""

    reward: float
    next_state: int
    done: bool


class EnvironmentModel:
    """Tabular model storing observed (s, a) -> (r, s', done) transitions.

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

    # -- TODO 1 ----------------------------------------------------------------

    def update(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> None:
        """Record an observed transition and update the sampling bookkeeping.

        Overwrite `self.transitions[(state, action)]`.  Append `state` to
        `observed_states` on first sight, and append `action` to
        `observed_actions[state]` on first sight.  Think of it as an
        adjacency list: for every visited node, remember which edges we've
        explored.
        """
        # TODO(human): dict write + two "add if not present" operations.
        raise NotImplementedError("Implement EnvironmentModel.update()")

    # -- TODO 2 ----------------------------------------------------------------

    def sample(self) -> tuple[int, int, float, int, bool]:
        """Return a random previously-observed (s, a, r, s', done).

        Pick a uniformly random state from `observed_states`, then a uniformly
        random action from `observed_actions[state]`, then look up the stored
        `Transition`.  Use `self.rng.integers(len(collection))` for indices.
        This is how the agent "dreams" -- replaying random past experiences
        for free Q-updates with no real environment interaction.
        """
        # TODO(human): sample a state, sample an action for it, look up the
        # Transition, and return the 5-tuple.
        raise NotImplementedError("Implement EnvironmentModel.sample()")
