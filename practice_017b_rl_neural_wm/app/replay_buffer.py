"""Experience replay buffer for storing and sampling transitions.

A transition is (state, action, reward, next_state, done).
This module is fully implemented -- no TODOs here.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Transition:
    """A single environment transition."""

    state: NDArray[np.float32]
    action: int
    reward: float
    next_state: NDArray[np.float32]
    done: bool


class ReplayBuffer:
    """Fixed-capacity circular buffer of transitions with uniform sampling."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._states: list[NDArray[np.float32]] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._next_states: list[NDArray[np.float32]] = []
        self._dones: list[bool] = []
        self._position = 0
        self._full = False

    @property
    def size(self) -> int:
        return self._capacity if self._full else self._position

    def push(self, transition: Transition) -> None:
        """Add a transition to the buffer, overwriting oldest if full."""
        if self._full:
            self._states[self._position] = transition.state
            self._actions[self._position] = transition.action
            self._rewards[self._position] = transition.reward
            self._next_states[self._position] = transition.next_state
            self._dones[self._position] = transition.done
        else:
            self._states.append(transition.state)
            self._actions.append(transition.action)
            self._rewards.append(transition.reward)
            self._next_states.append(transition.next_state)
            self._dones.append(transition.done)

        self._position = (self._position + 1) % self._capacity
        if self._position == 0 and not self._full:
            self._full = True

    def sample(self, batch_size: int) -> "TransitionBatch":
        """Sample a random batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return TransitionBatch(
            states=np.array([self._states[i] for i in indices], dtype=np.float32),
            actions=np.array([self._actions[i] for i in indices], dtype=np.int64),
            rewards=np.array([self._rewards[i] for i in indices], dtype=np.float32),
            next_states=np.array(
                [self._next_states[i] for i in indices], dtype=np.float32
            ),
            dones=np.array([self._dones[i] for i in indices], dtype=np.float32),
        )

    def sample_states(self, batch_size: int) -> NDArray[np.float32]:
        """Sample random states from the buffer (used as starting points for imagined rollouts)."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return np.array([self._states[i] for i in indices], dtype=np.float32)

    def all_data(self) -> "TransitionBatch":
        """Return all stored transitions as a single batch."""
        n = self.size
        return TransitionBatch(
            states=np.array(self._states[:n], dtype=np.float32),
            actions=np.array(self._actions[:n], dtype=np.int64),
            rewards=np.array(self._rewards[:n], dtype=np.float32),
            next_states=np.array(self._next_states[:n], dtype=np.float32),
            dones=np.array(self._dones[:n], dtype=np.float32),
        )

    def load_from_arrays(
        self,
        states: NDArray[np.float32],
        actions: NDArray[np.int64],
        rewards: NDArray[np.float32],
        next_states: NDArray[np.float32],
        dones: NDArray[np.float32],
    ) -> None:
        """Bulk-load transitions from NumPy arrays (e.g., from saved file)."""
        for i in range(len(states)):
            self.push(
                Transition(
                    state=states[i],
                    action=int(actions[i]),
                    reward=float(rewards[i]),
                    next_state=next_states[i],
                    done=bool(dones[i]),
                )
            )


@dataclass
class TransitionBatch:
    """A batch of transitions as NumPy arrays, ready for conversion to tensors."""

    states: NDArray[np.float32]       # (batch, state_dim)
    actions: NDArray[np.int64]        # (batch,)
    rewards: NDArray[np.float32]      # (batch,)
    next_states: NDArray[np.float32]  # (batch, state_dim)
    dones: NDArray[np.float32]        # (batch,)  0.0 or 1.0
