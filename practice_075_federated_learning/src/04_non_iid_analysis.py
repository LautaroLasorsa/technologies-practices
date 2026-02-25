"""Exercise 4 -- Non-IID Analysis and FedProx Strategy.

This exercise compares FedAvg and FedProx across IID and non-IID data distributions.
FedProx adds a proximal term to the local loss function to reduce client drift:

    L_local = CrossEntropy(y, y_hat) + (mu/2) * ||w - w_global||^2

The proximal term penalizes local weights that diverge from the global model,
which is critical when clients have very different data distributions.

You'll implement:
  1. A FedProx-aware training function
  2. A comparison experiment runner
  3. A convergence plot that shows all configurations side-by-side
"""

import sys
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
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.00_centralized_baseline import (
    MNISTNet,
    evaluate_model,
    get_device,
    load_mnist_train,
    load_mnist_test,
)
from src.02_flower_client import MNISTFlowerClient, set_parameters
from src.03_fedavg_simulation import client_fn, load_centralized_accuracy

PRACTICE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PRACTICE_DIR / "data"
PLOTS_DIR = PRACTICE_DIR / "plots"


# ---------------------------------------------------------------------------
# FedProx client -- extends MNISTFlowerClient with proximal term
# ---------------------------------------------------------------------------

class FedProxFlowerClient(MNISTFlowerClient):
    """Flower client with FedProx proximal regularization.

    Identical to MNISTFlowerClient except fit() adds a proximal term
    to the loss: (mu/2) * ||w_local - w_global||^2. This keeps the
    local model close to the global model during training, reducing
    the client drift that non-IID data causes.

    The proximal_mu parameter controls regularization strength:
      - mu=0 -> equivalent to FedAvg (no regularization)
      - mu=0.01 -> light regularization
      - mu=0.1 -> moderate regularization
      - mu=1.0 -> strong regularization (may underfit local data)
    """

    def __init__(
        self,
        model: MNISTNet,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        proximal_mu: float = 0.0,
    ) -> None:
        super().__init__(model, train_loader, test_loader, device)
        self.proximal_mu = proximal_mu


# ---------------------------------------------------------------------------
# TODO(human): Implement FedProx training
# ---------------------------------------------------------------------------

def train_with_proximal_term(
    client: FedProxFlowerClient,
    parameters: list[np.ndarray],
    config: dict,
) -> tuple[list[np.ndarray], int, dict]:
    """Train locally with FedProx proximal regularization.

    This replaces the standard fit() method with one that adds the proximal
    penalty term. The key idea: after loading global weights and BEFORE
    training, save a copy of the global parameters. During training, add
    the L2 penalty between current weights and the saved global weights.

    Algorithm:
      1. Load global parameters into client.model using set_parameters().
      2. SAVE a copy of the global weights as a list of tensors (detached, on device).
         global_params = [p.clone().detach() for p in client.model.parameters()]
      3. Create optimizer (Adam, lr=0.001) and criterion (CrossEntropyLoss).
      4. For each epoch (config.get("epochs", 1)):
           For each batch in client.train_loader:
             a. Forward pass: outputs = model(images)
             b. Standard loss: ce_loss = criterion(outputs, labels)
             c. Compute proximal term:
                proximal_loss = sum(
                    torch.sum((w - w_global) ** 2)
                    for w, w_global in zip(model.parameters(), global_params)
                )
             d. Total loss = ce_loss + (client.proximal_mu / 2) * proximal_loss
             e. Backward pass and optimizer step.
      5. Return (updated_params, num_samples, {"train_loss": avg_loss, "proximal_mu": mu})

    Args:
        client: A FedProxFlowerClient with proximal_mu set.
        parameters: Global model weights as list of numpy arrays.
        config: Dict with "epochs" key.

    Returns:
        Same 3-tuple as NumPyClient.fit(): (weights, num_samples, metrics).
    """
    # TODO(human): Implement FedProx training loop.
    # The ONLY difference from standard training is steps 2 and 4c-4d:
    # saving global params and adding the proximal penalty to the loss.
    # When proximal_mu=0, this should behave identically to FedAvg.
    raise NotImplementedError("Implement train_with_proximal_term")


# ---------------------------------------------------------------------------
# TODO(human): Implement comparison experiment runner
# ---------------------------------------------------------------------------

def run_comparison_experiment(
    configs: list[dict],
    num_rounds: int = 20,
    num_clients: int = 10,
    fraction_fit: float = 0.5,
    local_epochs: int = 1,
) -> dict[str, list[float]]:
    """Run multiple FL simulations with different configurations and collect results.

    Each config dict has keys:
      - "name": str -- label for this configuration (e.g., "IID + FedAvg")
      - "partition_name": str -- which partition to load (e.g., "iid", "non_iid_alpha_0.1")
      - "proximal_mu": float -- 0.0 for FedAvg, >0 for FedProx

    For each configuration:
      1. Load partition indices from data/partitions/{partition_name}/.
      2. Initialize a fresh global model.
      3. For each round:
         a. Select clients (same as exercise 3).
         b. If proximal_mu == 0: use standard client.fit() (FedAvg).
            If proximal_mu > 0: use train_with_proximal_term() (FedProx).
         c. Aggregate using FedAvg weighted average.
         d. Evaluate and record accuracy.
      4. Store the per-round accuracy list under results[config_name].

    This is essentially run_federated_simulation from exercise 3, but
    parameterized to support FedProx and run multiple configs back-to-back.

    Args:
        configs: List of configuration dicts.
        num_rounds: Communication rounds per simulation.
        num_clients: Total clients.
        fraction_fit: Fraction selected per round.
        local_epochs: Local epochs per client.

    Returns:
        Dict mapping config_name -> list of per-round accuracies.
        Example: {"IID + FedAvg": [0.85, 0.90, ...], "Non-IID + FedProx": [0.80, ...]}
    """
    # TODO(human): Implement the comparison experiment.
    # This is a generalization of run_federated_simulation from exercise 3.
    # The key addition: check proximal_mu to decide whether to use standard
    # fit() or train_with_proximal_term().
    # Tip: Create a FedProxFlowerClient instead of MNISTFlowerClient when mu > 0.
    raise NotImplementedError("Implement run_comparison_experiment")


# ---------------------------------------------------------------------------
# TODO(human): Implement convergence comparison plot
# ---------------------------------------------------------------------------

def plot_convergence_comparison(
    results: dict[str, list[float]],
    centralized_acc: float,
    save_path: Path,
) -> None:
    """Plot accuracy curves for all configurations on a single figure.

    Creates a line plot with:
      - X-axis: Communication round (1, 2, ..., num_rounds)
      - Y-axis: Test accuracy (0.0 to 1.0)
      - One line per configuration, with different colors/markers
      - A horizontal dashed red line for the centralized baseline
      - Legend showing all config names + centralized baseline
      - Grid for readability
      - Title: "Convergence Comparison: FedAvg vs FedProx"

    The plot should clearly show:
      - IID converges faster than non-IID with FedAvg
      - FedProx helps non-IID converge faster and reach higher accuracy
      - All federated approaches converge below the centralized baseline

    Args:
        results: Dict mapping config_name -> list of per-round accuracies.
        centralized_acc: The centralized baseline accuracy (horizontal line).
        save_path: Path to save the plot.
    """
    # TODO(human): Implement the convergence comparison plot.
    # Use plt.plot() for each config with different markers/colors.
    # Use plt.axhline() for the centralized baseline.
    # Remember: plt.savefig(), plt.close() -- don't leave figures open.
    raise NotImplementedError("Implement plot_convergence_comparison")


# ---------------------------------------------------------------------------
# Scaffolded: orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Exercise 4: Non-IID Analysis and FedProx")
    print("=" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    centralized_acc = load_centralized_accuracy()
    print(f"\nCentralized baseline accuracy: {centralized_acc:.4f}")

    # Define experiment configurations
    configs = [
        {
            "name": "IID + FedAvg",
            "partition_name": "iid",
            "proximal_mu": 0.0,
        },
        {
            "name": "Non-IID (a=0.1) + FedAvg",
            "partition_name": "non_iid_alpha_0.1",
            "proximal_mu": 0.0,
        },
        {
            "name": "Non-IID (a=0.1) + FedProx (mu=0.01)",
            "partition_name": "non_iid_alpha_0.1",
            "proximal_mu": 0.01,
        },
        {
            "name": "Non-IID (a=0.1) + FedProx (mu=0.1)",
            "partition_name": "non_iid_alpha_0.1",
            "proximal_mu": 0.1,
        },
    ]

    print(f"\nRunning {len(configs)} experiment configurations...\n")

    results = run_comparison_experiment(
        configs=configs,
        num_rounds=20,
        num_clients=10,
        fraction_fit=0.5,
        local_epochs=1,
    )

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Configuration':<40} {'Final Acc':>10} {'Best Acc':>10} {'Gap':>10}")
    print("-" * 70)
    for name, accuracies in results.items():
        final = accuracies[-1]
        best = max(accuracies)
        gap = centralized_acc - final
        print(f"{name:<40} {final:>10.4f} {best:>10.4f} {gap:>+10.4f}")
    print(f"{'Centralized Baseline':<40} {centralized_acc:>10.4f}")

    # --- Convergence plot ---
    plot_convergence_comparison(
        results,
        centralized_acc,
        PLOTS_DIR / "convergence_comparison.png",
    )
    print(f"\nConvergence plot saved to {PLOTS_DIR / 'convergence_comparison.png'}")


if __name__ == "__main__":
    main()
