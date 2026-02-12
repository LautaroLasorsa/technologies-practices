"""Use the trained world model to generate imagined rollouts and evaluate accuracy.

This module tests how well the learned dynamics model predicts the real
environment's behavior over multiple steps. It demonstrates the compounding
error problem: even if single-step predictions are accurate, errors accumulate
over multi-step rollouts.

Usage:
    uv run python -m app.data_collection   # collect data
    uv run python -m app.train_model        # train world model
    uv run python -m app.plan_with_model    # evaluate model predictions
"""

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from app.config import ENV
from app.world_model import WorldModel


def imagine_rollout(
    model: WorldModel,
    start_state: torch.Tensor,
    actions: list[int],
    device: torch.device,
) -> dict[str, list]:
    """Generate an imagined trajectory using the world model.

    Starting from a real state, repeatedly apply the world model to predict
    the next state, reward, and done flag for each action in the sequence.

    Args:
        model: Trained WorldModel.
        start_state: Initial state tensor, shape (state_dim,).
        actions: Sequence of actions to take in the imagined rollout.
        device: Torch device.

    Returns:
        Dictionary with keys:
            "states": List of state tensors (including start_state), length len(actions)+1
            "rewards": List of predicted rewards, length len(actions)
            "dones": List of predicted done flags, length len(actions)
    """
    # ── Exercise Context ──────────────────────────────────────────────────
    # This teaches planning via model-based rollouts: using the learned dynamics model
    # to simulate future trajectories. It demonstrates how models enable "mental simulation"
    # and reveals the compounding error problem (errors accumulate over time).

    # TODO(human): Implement the imagined rollout.
    #
    # Steps:
    #   1. Initialize: current_state = start_state, states = [start_state], rewards = [], dones = []
    #   2. For each action in the actions list:
    #      a. Use model.predict(current_state, action) to get (next_state, reward, done)
    #      b. Append next_state to states, reward to rewards, done to dones
    #      c. If done is True, stop the rollout early (break)
    #      d. Otherwise, set current_state = next_state
    #   3. Return {"states": states, "rewards": rewards, "dones": dones}
    #
    # Note: model.predict() already handles unsqueezing and torch.no_grad().
    raise NotImplementedError("TODO(human): Implement imagined rollout")


def collect_real_rollout(
    env: gym.Env,
    start_state: np.ndarray,
    actions: list[int],
) -> dict[str, list]:
    """Collect a real trajectory from the environment for comparison.

    Note: We reset the environment and then set its internal state to match
    start_state. For CartPole, we can do this via env.unwrapped.state.

    Args:
        env: Gymnasium environment (not wrapped in render mode).
        start_state: Initial state as numpy array.
        actions: Sequence of actions to execute.

    Returns:
        Same format as imagine_rollout: {"states", "rewards", "dones"}.
    """
    env.reset()
    env.unwrapped.state = start_state.copy()

    states = [start_state.copy()]
    rewards = []
    dones = []

    for action in actions:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        states.append(np.array(next_state, dtype=np.float32))
        rewards.append(float(reward))
        dones.append(done)
        if done:
            break

    return {"states": states, "rewards": rewards, "dones": dones}


def evaluate_model_accuracy(
    model: WorldModel,
    env: gym.Env,
    num_rollouts: int,
    rollout_length: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Compare model predictions to real trajectories across many rollouts.

    For each rollout:
        1. Reset the environment to get a real starting state
        2. Sample random actions
        3. Run the same actions through both the real env and the world model
        4. Compute per-step MSE between predicted and real states

    Args:
        model: Trained WorldModel.
        env: Gymnasium environment.
        num_rollouts: Number of rollouts to average over.
        rollout_length: Maximum number of steps per rollout.
        device: Torch device.

    Returns:
        Dictionary with:
            "per_step_mse": Array of shape (rollout_length,) -- average state MSE at each step
            "cumulative_error": Array of shape (rollout_length,) -- cumulative error over steps
    """
    # TODO(human): Implement model accuracy evaluation.
    #
    # Steps:
    #   1. Initialize: all_step_errors = list of empty lists, one per step
    #   2. For each rollout in range(num_rollouts):
    #      a. Reset env, get start_state
    #      b. Sample random actions: [env.action_space.sample() for _ in range(rollout_length)]
    #      c. Collect real rollout: collect_real_rollout(env, start_state, actions)
    #      d. Collect imagined rollout: imagine_rollout(model, start_state_tensor, actions, device)
    #      e. For each step where both rollouts have data:
    #         - Compute MSE between real_states[step+1] and imagined_states[step+1]
    #           (convert tensors to numpy if needed, use np.mean((real - pred)**2))
    #         - Append to all_step_errors[step]
    #   3. Compute per_step_mse: np.array([np.mean(errors) for errors in all_step_errors if errors])
    #   4. Compute cumulative_error: np.cumsum(per_step_mse)
    #   5. Return {"per_step_mse": per_step_mse, "cumulative_error": cumulative_error}
    #
    # Hint: len(real["states"]) may be < rollout_length+1 if the episode ended early.
    #       Use min(len(real["states"]), len(imagined["states"])) - 1 as the actual length.
    raise NotImplementedError("TODO(human): Implement model accuracy evaluation")


def plot_trajectory_comparison(
    real: dict[str, list],
    imagined: dict[str, list],
    save_path: str,
) -> None:
    """Plot real vs. predicted state trajectories side by side."""
    state_labels = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Vel"]
    real_states = np.array([s if isinstance(s, np.ndarray) else s.cpu().numpy() for s in real["states"]])
    imag_states = np.array([s if isinstance(s, np.ndarray) else s.cpu().numpy() for s in imagined["states"]])

    min_len = min(len(real_states), len(imag_states))
    real_states = real_states[:min_len]
    imag_states = imag_states[:min_len]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, (ax, label) in enumerate(zip(axes, state_labels)):
        steps = range(min_len)
        ax.plot(steps, real_states[:, i], label="Real", linewidth=2)
        ax.plot(steps, imag_states[:, i], label="Predicted", linewidth=2, linestyle="--")
        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Real vs. Predicted Trajectory", fontsize=14)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Trajectory comparison saved to {save_path}")


def plot_compounding_error(
    accuracy: dict[str, np.ndarray],
    save_path: str,
) -> None:
    """Plot per-step and cumulative prediction error."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    steps = range(1, len(accuracy["per_step_mse"]) + 1)

    ax1.plot(steps, accuracy["per_step_mse"], linewidth=2, color="tab:blue")
    ax1.set_xlabel("Rollout Step")
    ax1.set_ylabel("Mean Squared Error")
    ax1.set_title("Per-Step Prediction Error")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, accuracy["cumulative_error"], linewidth=2, color="tab:red")
    ax2.set_xlabel("Rollout Step")
    ax2.set_ylabel("Cumulative MSE")
    ax2.set_title("Compounding Error Over Rollout")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("World Model Accuracy vs. Rollout Length", fontsize=14)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Compounding error plot saved to {save_path}")


def main() -> None:
    """Load trained model, evaluate on real environment, generate plots."""
    model_path = Path("checkpoints/world_model.pt")
    if not model_path.exists():
        print(f"No trained model found at {model_path}.")
        print("Run `uv run python -m app.train_model` first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WorldModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded world model from {model_path}")

    env = gym.make(ENV.name)

    # --- Single trajectory comparison ---
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    rollout_len = 50
    actions = [env.action_space.sample() for _ in range(rollout_len)]

    real_traj = collect_real_rollout(env, state, actions)
    imagined_traj = imagine_rollout(
        model,
        torch.tensor(state, dtype=torch.float32, device=device),
        actions,
        device,
    )
    plot_trajectory_comparison(real_traj, imagined_traj, "plots/trajectory_comparison.png")

    # --- Compounding error analysis ---
    accuracy = evaluate_model_accuracy(
        model, env, num_rollouts=50, rollout_length=30, device=device
    )
    plot_compounding_error(accuracy, "plots/compounding_error.png")

    print(f"\nPer-step MSE (first 5 steps): {accuracy['per_step_mse'][:5]}")
    print(f"Per-step MSE (last 5 steps):  {accuracy['per_step_mse'][-5:]}")

    env.close()


if __name__ == "__main__":
    main()
