"""Exercise 0 -- Centralized MNIST Baseline (fully scaffolded).

Trains a simple CNN on the full MNIST dataset for 5 epochs.
This establishes the upper bound accuracy that federated approaches aim to match.
The model architecture and helper functions defined here are reused by all exercises.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PRACTICE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PRACTICE_DIR / "data"
MODELS_DIR = PRACTICE_DIR / "models"

# ---------------------------------------------------------------------------
# Model definition -- shared across all exercises
# ---------------------------------------------------------------------------

class MNISTNet(nn.Module):
    """Simple CNN for MNIST: 2 conv layers + 2 fully-connected layers.

    Architecture:
        conv1(1->32, 3x3, pad=1) -> ReLU -> MaxPool(2)   => 32x14x14
        conv2(32->64, 3x3, pad=1) -> ReLU -> MaxPool(2)  => 64x7x7
        fc1(64*7*7 -> 128) -> ReLU -> Dropout(0.25)
        fc2(128 -> 10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # (B, 64, 7, 7)
        x = x.view(x.size(0), -1)             # (B, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Training and evaluation helpers -- shared across all exercises
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device (CUDA > CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train *model* for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate *model* and return (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    n = len(dataloader.dataset)
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# MNIST data loading helpers -- shared across all exercises
# ---------------------------------------------------------------------------

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def load_mnist_train() -> datasets.MNIST:
    """Download (if needed) and return the MNIST training set."""
    return datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=MNIST_TRANSFORM)


def load_mnist_test() -> datasets.MNIST:
    """Download (if needed) and return the MNIST test set."""
    return datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=MNIST_TRANSFORM)


# ---------------------------------------------------------------------------
# Main -- train centralized baseline
# ---------------------------------------------------------------------------

def train_centralized_baseline(epochs: int = 5, batch_size: int = 64, lr: float = 0.001) -> float:
    """Train CNN on full MNIST and return test accuracy."""
    device = get_device()
    print(f"Device: {device}")

    # Data
    train_dataset = load_mnist_train()
    test_dataset = load_mnist_test()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = MNISTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"  Epoch {epoch}/{epochs} -- train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    # Final evaluation
    _, final_acc = evaluate_model(model, test_loader, criterion, device)
    return final_acc


def save_results(accuracy: float) -> None:
    """Persist centralized accuracy for comparison in later exercises."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    acc_path = DATA_DIR / "centralized_accuracy.txt"
    acc_path.write_text(f"{accuracy:.6f}\n")
    print(f"\nCentralized accuracy saved to {acc_path}")


def main() -> None:
    print("=" * 60)
    print("Exercise 0: Centralized MNIST Baseline")
    print("=" * 60)
    print("\nTraining CNN on full MNIST (upper bound for FL experiments)...\n")

    accuracy = train_centralized_baseline()

    print(f"\n--- Centralized baseline accuracy: {accuracy:.4f} ---")
    save_results(accuracy)


if __name__ == "__main__":
    main()
