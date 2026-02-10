"""Neural world model: an MLP that learns environment dynamics.

The world model learns the mapping:
    f(state, action) -> (next_state, reward, done)

This is a supervised learning problem: given collected transitions from the real
environment, train the MLP to predict what happens next.

Architecture:
    Input:  [state (4,) | one_hot_action (2,)] -> 6-dim vector
    Shared encoder: Linear -> ReLU -> ... -> hidden_dim
    State head:  Linear -> predicted next_state (4,)
    Reward head: Linear -> predicted reward (1,)
    Done head:   Linear -> Sigmoid -> predicted done probability (1,)

Usage:
    uv run python -m app.world_model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import ENV, WORLD_MODEL


class WorldModel(nn.Module):
    """Learned dynamics model predicting (next_state, reward, done) from (state, action).

    The model uses a shared encoder trunk followed by three separate prediction
    heads. This multi-task architecture shares features across predictions while
    allowing each output to specialize.

    Attributes:
        state_dim: Dimensionality of the environment state.
        action_dim: Number of discrete actions (one-hot encoded as input).
        hidden_dim: Width of each hidden layer in the shared encoder.
        num_hidden_layers: Number of hidden layers in the shared encoder.
    """

    def __init__(
        self,
        state_dim: int = ENV.state_dim,
        action_dim: int = ENV.action_dim,
        hidden_dim: int = WORLD_MODEL.hidden_dim,
        num_hidden_layers: int = WORLD_MODEL.num_hidden_layers,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        input_dim = state_dim + action_dim  # state concatenated with one-hot action

        # TODO(human): Build the shared encoder.
        #
        # Create a nn.Sequential called self.encoder with:
        #   - An input Linear layer: input_dim -> hidden_dim
        #   - A ReLU activation
        #   - For each additional hidden layer (num_hidden_layers - 1):
        #       - Linear: hidden_dim -> hidden_dim
        #       - ReLU
        #
        # Hint: You can build a list of layers and unpack with nn.Sequential(*layers).
        #
        # self.encoder = nn.Sequential(...)
        raise NotImplementedError("TODO(human): Build the shared encoder trunk")

        # TODO(human): Build the three prediction heads.
        #
        # self.state_head:  Linear(hidden_dim, state_dim)   -- predicts delta or absolute next state
        # self.reward_head: Linear(hidden_dim, 1)           -- predicts reward (scalar)
        # self.done_head:   Linear(hidden_dim, 1)           -- predicts done logit (use sigmoid in forward)
        #
        # Note: We predict the next state directly (not delta). For CartPole this works fine.
        #       For more complex environments, predicting delta (s' - s) and adding to s is more stable.
        raise NotImplementedError("TODO(human): Build the three prediction heads")

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next_state, reward, and done probability.

        Args:
            state: Batch of states, shape (batch, state_dim).
            action: Batch of action indices, shape (batch,) with dtype long.

        Returns:
            pred_next_state: Shape (batch, state_dim).
            pred_reward: Shape (batch, 1).
            pred_done: Shape (batch, 1), values in [0, 1] (probability).
        """
        # TODO(human): Implement the forward pass.
        #
        # Steps:
        #   1. One-hot encode the action: F.one_hot(action, self.action_dim).float()
        #   2. Concatenate state and one-hot action along dim=-1
        #   3. Pass through self.encoder to get shared features
        #   4. Pass features through each head:
        #      - state_head  -> pred_next_state
        #      - reward_head -> pred_reward
        #      - done_head   -> torch.sigmoid(...) -> pred_done
        #   5. Return (pred_next_state, pred_reward, pred_done)
        raise NotImplementedError("TODO(human): Implement forward pass")

    def compute_loss(
        self,
        pred_next_state: torch.Tensor,
        pred_reward: torch.Tensor,
        pred_done: torch.Tensor,
        target_next_state: torch.Tensor,
        target_reward: torch.Tensor,
        target_done: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the combined training loss.

        Args:
            pred_next_state: Model prediction, shape (batch, state_dim).
            pred_reward: Model prediction, shape (batch, 1).
            pred_done: Model prediction (probability), shape (batch, 1).
            target_next_state: Ground truth, shape (batch, state_dim).
            target_reward: Ground truth, shape (batch,) or (batch, 1).
            target_done: Ground truth, shape (batch,) or (batch, 1), values 0.0/1.0.

        Returns:
            total_loss: Scalar tensor, weighted sum of the three losses.
            loss_dict: Breakdown of individual losses for logging.
        """
        # TODO(human): Compute the three losses and combine them.
        #
        # 1. State loss: F.mse_loss(pred_next_state, target_next_state)
        #
        # 2. Reward loss: F.mse_loss(pred_reward.squeeze(-1), target_reward)
        #    (squeeze pred_reward from (batch,1) to (batch,) to match target shape)
        #
        # 3. Done loss: F.binary_cross_entropy(pred_done.squeeze(-1), target_done)
        #    (BCE because done is a binary classification: 0 or 1)
        #
        # 4. Combine: total = state_weight * state_loss
        #                    + reward_weight * reward_loss
        #                    + done_weight * done_loss
        #    Use WORLD_MODEL.state_loss_weight, .reward_loss_weight, .done_loss_weight
        #
        # 5. Return (total_loss, {"state": ..., "reward": ..., "done": ..., "total": ...})
        #    where each value is a float (use .item()) for logging.
        raise NotImplementedError("TODO(human): Compute combined loss")

    @torch.no_grad()
    def predict(
        self,
        state: torch.Tensor,
        action: int,
    ) -> tuple[torch.Tensor, float, bool]:
        """Single-step prediction for planning (no gradient tracking).

        Args:
            state: Single state tensor, shape (state_dim,).
            action: Scalar action index.

        Returns:
            next_state: Predicted next state, shape (state_dim,).
            reward: Predicted reward (float).
            done: Predicted done (bool, thresholded at 0.5).
        """
        state_batch = state.unsqueeze(0)
        action_batch = torch.tensor([action], dtype=torch.long, device=state.device)

        pred_state, pred_reward, pred_done = self.forward(state_batch, action_batch)

        return (
            pred_state.squeeze(0),
            pred_reward.item(),
            pred_done.item() > 0.5,
        )


def _test_shapes() -> None:
    """Verify that the model produces correct output shapes."""
    model = WorldModel()
    batch_size = 8

    states = torch.randn(batch_size, ENV.state_dim)
    actions = torch.randint(0, ENV.action_dim, (batch_size,))

    pred_state, pred_reward, pred_done = model(states, actions)

    assert pred_state.shape == (batch_size, ENV.state_dim), (
        f"Expected ({batch_size}, {ENV.state_dim}), got {pred_state.shape}"
    )
    assert pred_reward.shape == (batch_size, 1), (
        f"Expected ({batch_size}, 1), got {pred_reward.shape}"
    )
    assert pred_done.shape == (batch_size, 1), (
        f"Expected ({batch_size}, 1), got {pred_done.shape}"
    )
    assert (pred_done >= 0).all() and (pred_done <= 1).all(), (
        "Done predictions must be in [0, 1] (sigmoid output)"
    )

    print("All shape checks passed!")
    print(f"  pred_state:  {pred_state.shape}")
    print(f"  pred_reward: {pred_reward.shape}")
    print(f"  pred_done:   {pred_done.shape}")


if __name__ == "__main__":
    _test_shapes()
