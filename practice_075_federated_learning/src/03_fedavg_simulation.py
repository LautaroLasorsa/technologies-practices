"""Exercise 3 -- FedAvg Simulation with N=10 Clients.

This exercise implements the full FedAvg protocol as a manual simulation loop.
Instead of using Flower's Ray-based simulation engine (which has limited Windows
support), we orchestrate the protocol ourselves:

  For each round:
    1. Create/reset clients with the global model
    2. Select a fraction of clients
    3. Call fit() on each selected client (local training)
    4. Aggregate updated weights using weighted average (FedAvg)
    5. Evaluate the new global model on the test set

This teaches the mechanics of FL at the deepest level -- you implement exactly
what Flower's simulation engine does internally.
"""

import sys
import random
from pathlib import Path

# Ensure practice root is on sys.path for cross-module imports
_PRACTICE_DIR = Path(__file__).resolve().parent.parent
if str(_PRACTICE_DIR) not in sys.path:
    sys.path.insert(0, str(_PRACTICE_DIR))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from src.00_centralized_baseline import (
    MNISTNet,
    evaluate_model,
    get_device,
    load_mnist_train,
    load_mnist_test,
)
from src.02_flower_client import MNISTFlowerClient, set_parameters

PRACTICE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PRACTICE_DIR / "data"
PLOTS_DIR = PRACTICE_DIR / "plots"


# ---------------------------------------------------------------------------
# TODO(human): Implement client factory
# ---------------------------------------------------------------------------

def client_fn(
    client_id: int,
    partition_indices: np.ndarray,
    train_dataset: torch.utils.data.Dataset,
    test_loader: DataLoader,
    device: torch.device,
) -> MNISTFlowerClient:
    """Factory function: create a FlowerClient for a given client ID.

    This function creates one federated client with its own data partition.
    In a real FL system, each client would be a separate device with its own
    local data. Here we simulate this by giving each client a Subset of MNIST.

    Steps:
      1. Create a Subset of train_dataset using partition_indices.
         This gives the client only its assigned data.
      2. Wrap the Subset in a DataLoader with batch_size=32 and shuffle=True.
      3. Create a fresh MNISTNet model (each client starts with a new model;
         the global weights will be loaded via fit()).
      4. Return an MNISTFlowerClient initialized with the model, train loader,
         test_loader (shared global test set), and device.

    Args:
        client_id: Integer ID of this client (for logging, not used in logic).
        partition_indices: Numpy array of dataset indices assigned to this client.
        train_dataset: The full MNIST training dataset.
        test_loader: DataLoader for the global test set (shared across all clients).
        device: torch.device for training.

    Returns:
        An MNISTFlowerClient ready to participate in FL rounds.
    """
    # TODO(human): Implement client creation.
    # This is straightforward: Subset -> DataLoader -> MNISTNet -> MNISTFlowerClient.
    # Each client gets a fresh model because the server will send global weights via fit().
    raise NotImplementedError("Implement client_fn")


# ---------------------------------------------------------------------------
# TODO(human): Implement the federated simulation loop
# ---------------------------------------------------------------------------

def run_federated_simulation(
    num_clients: int,
    num_rounds: int,
    fraction_fit: float,
    partition_name: str,
    local_epochs: int = 1,
) -> dict[str, list[float]]:
    """Run a FedAvg federated learning simulation and return metrics per round.

    This implements the full FedAvg protocol manually:

    Setup:
      1. Load partition indices from data/partitions/{partition_name}/.
      2. Load MNIST train and test datasets.
      3. Initialize a global model (MNISTNet) -- this is the server's model.
      4. Extract initial global parameters: list of numpy arrays from model.state_dict().

    For each round r = 1..num_rounds:
      1. CLIENT SELECTION: Randomly select K = max(1, int(fraction_fit * num_clients))
         client indices from range(num_clients).
      2. LOCAL TRAINING: For each selected client index:
         a. Create a FlowerClient using client_fn() with that client's partition.
         b. Call client.fit(global_parameters, {"epochs": local_epochs}).
         c. Collect the returned (updated_params, num_samples, metrics).
      3. AGGREGATION (FedAvg weighted average):
         a. Compute total_samples = sum of all num_samples.
         b. For each parameter index i, compute:
            new_param[i] = sum(num_samples_k / total_samples * params_k[i] for each client k)
         c. This is the FedAvg formula: w = sum(n_k/n * w_k).
         d. Update global_parameters with the aggregated weights.
      4. EVALUATION: Load global_parameters into a model and evaluate on the test set.
         Record the accuracy for this round.
      5. Print round metrics (round number, loss, accuracy).

    Args:
        num_clients: Total number of clients in the federation.
        num_rounds: Number of communication rounds.
        fraction_fit: Fraction of clients selected per round (e.g., 0.5 = 50%).
        partition_name: Name of the partition to load (e.g., "iid", "non_iid_alpha_0.1").
        local_epochs: Number of local training epochs per client per round.

    Returns:
        Dict with keys "round", "loss", "accuracy" -- each a list of per-round values.
    """
    # TODO(human): Implement the FedAvg simulation loop.
    #
    # Key implementation details:
    # - Extract global params: [val.cpu().numpy() for val in model.state_dict().values()]
    # - FedAvg aggregation: weighted average of numpy arrays by dataset size
    # - Use set_parameters() to load aggregated weights back into the model
    # - Track accuracy per round for plotting
    #
    # This is the most important function in the practice. Take time to understand
    # each step -- this IS the FedAvg algorithm from McMahan et al. 2017.
    raise NotImplementedError("Implement run_federated_simulation")


# ---------------------------------------------------------------------------
# Scaffolded: plotting and orchestration
# ---------------------------------------------------------------------------

def plot_accuracy_curve(
    results: dict[str, list[float]],
    centralized_acc: float,
    save_path: Path,
    title: str = "FedAvg Training Curve",
) -> None:
    """Plot accuracy per round with centralized baseline reference line."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results["round"], results["accuracy"], "b-o", markersize=4, label="FedAvg")
    ax.axhline(y=centralized_acc, color="r", linestyle="--", label=f"Centralized ({centralized_acc:.4f})")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {save_path}")


def load_centralized_accuracy() -> float:
    """Load the centralized baseline accuracy from exercise 0."""
    acc_path = DATA_DIR / "centralized_accuracy.txt"
    if not acc_path.exists():
        print("WARNING: Centralized accuracy not found. Run exercise 0 first.")
        print("         Using default value of 0.99.")
        return 0.99
    return float(acc_path.read_text().strip())


def main() -> None:
    print("=" * 60)
    print("Exercise 3: FedAvg Simulation")
    print("=" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    centralized_acc = load_centralized_accuracy()
    print(f"\nCentralized baseline accuracy: {centralized_acc:.4f}")

    # --- Run FedAvg with IID data ---
    print("\n--- FedAvg with IID Data (10 clients, 20 rounds) ---\n")
    results_iid = run_federated_simulation(
        num_clients=10,
        num_rounds=20,
        fraction_fit=0.5,
        partition_name="iid",
        local_epochs=1,
    )

    final_acc = results_iid["accuracy"][-1]
    print(f"\n  Final FedAvg accuracy (IID): {final_acc:.4f}")
    print(f"  Gap from centralized: {centralized_acc - final_acc:.4f}")

    plot_accuracy_curve(
        results_iid,
        centralized_acc,
        PLOTS_DIR / "fedavg_iid_accuracy.png",
        title="FedAvg with IID Data (10 clients, 20 rounds)",
    )

    # --- Run FedAvg with non-IID data ---
    print("\n--- FedAvg with Non-IID Data (alpha=0.1, 10 clients, 20 rounds) ---\n")
    results_non_iid = run_federated_simulation(
        num_clients=10,
        num_rounds=20,
        fraction_fit=0.5,
        partition_name="non_iid_alpha_0.1",
        local_epochs=1,
    )

    final_acc_non_iid = results_non_iid["accuracy"][-1]
    print(f"\n  Final FedAvg accuracy (non-IID): {final_acc_non_iid:.4f}")
    print(f"  Gap from centralized: {centralized_acc - final_acc_non_iid:.4f}")

    plot_accuracy_curve(
        results_non_iid,
        centralized_acc,
        PLOTS_DIR / "fedavg_non_iid_accuracy.png",
        title="FedAvg with Non-IID Data (alpha=0.1, 10 clients, 20 rounds)",
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Centralized:     {centralized_acc:.4f}")
    print(f"  FedAvg (IID):    {final_acc:.4f}  (gap: {centralized_acc - final_acc:+.4f})")
    print(f"  FedAvg (non-IID):{final_acc_non_iid:.4f}  (gap: {centralized_acc - final_acc_non_iid:+.4f})")


if __name__ == "__main__":
    main()
