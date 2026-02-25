"""Exercise 2 -- Flower Client Implementation.

The Flower client is the core FL abstraction: it defines how a node (device,
hospital, phone) participates in federated training. You implement the three
methods of NumPyClient that define the client protocol:

  - get_parameters: serialize model weights to numpy arrays
  - fit: receive global weights, train locally, return updated weights
  - evaluate: receive global weights, evaluate locally, return metrics

The key challenge is converting between PyTorch tensors (model.state_dict())
and numpy arrays (what Flower transmits over the wire).
"""

import sys
from collections import OrderedDict
from pathlib import Path

# Ensure practice root is on sys.path for cross-module imports
_PRACTICE_DIR = Path(__file__).resolve().parent.parent
if str(_PRACTICE_DIR) not in sys.path:
    sys.path.insert(0, str(_PRACTICE_DIR))

import numpy as np
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.00_centralized_baseline import (
    MNISTNet,
    get_device,
    load_mnist_train,
    load_mnist_test,
    MNIST_TRANSFORM,
)

PRACTICE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PRACTICE_DIR / "data"


# ---------------------------------------------------------------------------
# Helpers for numpy <-> PyTorch conversion (used inside the client)
# ---------------------------------------------------------------------------

def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Set model weights from a list of numpy arrays.

    Flower communicates weights as list[np.ndarray]. This function converts
    them back to PyTorch tensors and loads them into the model's state_dict.
    The order of arrays must match the order of model.state_dict().keys().
    """
    state_dict = OrderedDict(
        {k: torch.from_numpy(v) for k, v in zip(model.state_dict().keys(), parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------------
# TODO(human): Implement the Flower client
# ---------------------------------------------------------------------------

class MNISTFlowerClient(fl.client.NumPyClient):
    """Flower client for federated MNIST training.

    This client wraps a local MNISTNet model and a local data partition.
    It implements the three methods of the NumPyClient protocol that the
    FL server calls during each communication round.

    Constructor receives:
        model: An MNISTNet instance (the local model).
        train_loader: DataLoader for this client's training data partition.
        test_loader: DataLoader for evaluation (can be the global test set
                     or a local validation split).
        device: torch.device to train on.

    Methods to implement:

    1. get_parameters(self, config: dict) -> list[np.ndarray]
       Return the current model weights as a list of numpy arrays.
       Iterate over model.state_dict().values(), detach each tensor,
       move to CPU, and convert to numpy: val.cpu().numpy().
       The *config* dict is unused here but part of the Flower protocol.

    2. fit(self, parameters: list[np.ndarray], config: dict) -> tuple[list[np.ndarray], int, dict]
       This is called when the server wants this client to train.
       Steps:
         a. Load the global parameters into the local model using set_parameters().
         b. Create an optimizer (Adam, lr=0.001) and loss function (CrossEntropyLoss).
         c. Train for config.get("epochs", 1) local epochs using standard PyTorch training.
         d. Return a 3-tuple:
            - Updated model weights as list[np.ndarray] (call self.get_parameters({}))
            - Number of training samples (len(self.train_loader.dataset))
            - A metrics dict, e.g. {"train_loss": avg_loss}

    3. evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple[float, int, dict]
       This is called when the server wants this client to evaluate the global model.
       Steps:
         a. Load the parameters into the local model using set_parameters().
         b. Evaluate on self.test_loader (no gradient computation).
         c. Return a 3-tuple:
            - Average loss (float)
            - Number of evaluation samples (int)
            - A metrics dict, e.g. {"accuracy": acc}
    """

    def __init__(
        self,
        model: MNISTNet,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model.to(self.device)

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        """Return model weights as a list of numpy arrays.

        Iterate over self.model.state_dict().values(). For each tensor:
          - Call .cpu() to move to CPU (in case model is on GPU)
          - Call .detach() to remove from computation graph
          - Call .numpy() to convert to numpy array
        Return the list of all these arrays.
        """
        # TODO(human): Implement get_parameters.
        # This is ~2 lines: a list comprehension over state_dict values.
        raise NotImplementedError("Implement get_parameters")

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[list[np.ndarray], int, dict]:
        """Train on local data and return updated weights.

        Steps:
          1. Call set_parameters(self.model, parameters) to load global weights.
          2. Read epochs from config: epochs = config.get("epochs", 1)
          3. Create optimizer (Adam, lr=0.001) and criterion (CrossEntropyLoss).
          4. Standard PyTorch training loop for *epochs* epochs over self.train_loader.
          5. Return (self.get_parameters({}), len(self.train_loader.dataset), {"train_loss": avg_loss})

        The number of samples returned is used by FedAvg to weight this client's
        contribution: clients with more data get more weight in the average.
        """
        # TODO(human): Implement fit.
        # This is the standard PyTorch training loop you know well.
        # The only FL-specific parts: set_parameters at the start, get_parameters at the end.
        raise NotImplementedError("Implement fit")

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[float, int, dict]:
        """Evaluate global model on local test data.

        Steps:
          1. Call set_parameters(self.model, parameters) to load global weights.
          2. Run inference on self.test_loader with torch.no_grad().
          3. Compute average loss and accuracy.
          4. Return (loss, len(self.test_loader.dataset), {"accuracy": accuracy})

        This tells the server how well the global model performs on this client's data.
        In non-IID settings, accuracy varies significantly across clients.
        """
        # TODO(human): Implement evaluate.
        # Standard PyTorch evaluation loop with no_grad() context.
        raise NotImplementedError("Implement evaluate")


# ---------------------------------------------------------------------------
# Scaffolded: local test of the client
# ---------------------------------------------------------------------------

def test_client_locally() -> None:
    """Create a single client and run one fit + evaluate cycle."""
    device = get_device()
    print(f"Device: {device}")

    # Load data -- use a small subset to test quickly
    train_dataset = load_mnist_train()
    test_dataset = load_mnist_test()

    # Use first 1000 samples as a "client partition" for testing
    small_train = Subset(train_dataset, list(range(1000)))
    train_loader = DataLoader(small_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create client
    model = MNISTNet()
    client = MNISTFlowerClient(model, train_loader, test_loader, device)

    # Test get_parameters
    params = client.get_parameters({})
    print(f"\nget_parameters: {len(params)} arrays")
    for i, p in enumerate(params):
        print(f"  param[{i}]: shape={p.shape}, dtype={p.dtype}")

    # Test fit (1 epoch)
    print("\nRunning fit (1 epoch)...")
    updated_params, num_samples, fit_metrics = client.fit(params, {"epochs": 1})
    print(f"  Trained on {num_samples} samples")
    print(f"  Train loss: {fit_metrics.get('train_loss', 'N/A'):.4f}")

    # Test evaluate
    print("\nRunning evaluate...")
    loss, num_eval, eval_metrics = client.evaluate(updated_params, {})
    print(f"  Evaluated on {num_eval} samples")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {eval_metrics.get('accuracy', 'N/A'):.4f}")

    print("\n--- Client protocol test passed ---")


def main() -> None:
    print("=" * 60)
    print("Exercise 2: Flower Client Implementation")
    print("=" * 60)
    test_client_locally()


if __name__ == "__main__":
    main()
