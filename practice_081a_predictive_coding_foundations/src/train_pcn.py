"""
Simple PCN training script on Fashion-MNIST.

Trains a PredictiveCodingNetwork, prints epoch stats, and saves a checkpoint.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.predictive_coding_network import PredictiveCodingNetwork

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"

# Hyperparameters
LAYER_DIMS = [784, 256, 128, 10]
NUM_EPOCHS = 10
BATCH_SIZE = 64
INFERENCE_STEPS = 20
INFERENCE_RATE = 0.1
LEARNING_RATE = 0.001


def load_fashion_mnist(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Load Fashion-MNIST with normalization."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_set = datasets.FashionMNIST(
        root=str(DATA_DIR), train=True, download=True, transform=transform,
    )
    test_set = datasets.FashionMNIST(
        root=str(DATA_DIR), train=False, download=True, transform=transform,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate(pcn: PredictiveCodingNetwork, test_loader: DataLoader, device: torch.device) -> float:
    """Evaluate PCN on test set using forward pass only."""
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)
        outputs = pcn.forward(images)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return correct / total


def save_checkpoint(pcn: PredictiveCodingNetwork, epoch: int, accuracy: float) -> None:
    """Save PCN weights to a checkpoint file."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "accuracy": accuracy,
        "layer_dims": pcn.layer_dims,
        "weights": [layer.synaptic_weights.cpu().clone() for layer in pcn.cortical_layers],
    }
    path = CHECKPOINT_DIR / f"pcn_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, str(path))
    print(f"  Checkpoint saved: {path.name}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: {LAYER_DIMS}")
    print(f"Inference steps: {INFERENCE_STEPS}, rate: {INFERENCE_RATE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    train_loader, test_loader = load_fashion_mnist(BATCH_SIZE)

    pcn = PredictiveCodingNetwork(
        layer_dims=LAYER_DIMS,
        inference_rate=INFERENCE_RATE,
        learning_rate=LEARNING_RATE,
        num_inference_steps=INFERENCE_STEPS,
    )
    pcn.to(device)

    for epoch in range(NUM_EPOCHS):
        total_energy = 0.0
        correct = 0
        total = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=False)
        for images, labels in pbar:
            images = images.view(images.size(0), -1).to(device)
            one_hot = torch.zeros(images.size(0), 10, device=device)
            one_hot.scatter_(1, labels.to(device).unsqueeze(1), 1.0)

            metrics = pcn.train_step(images, one_hot)

            total_energy += metrics["free_energy"]
            correct += int(metrics["accuracy"] * images.size(0))
            total += images.size(0)
            num_batches += 1
            pbar.set_postfix(energy=f"{total_energy / num_batches:.4f}", acc=f"{correct / total:.4f}")

        epoch_energy = total_energy / num_batches
        epoch_train_acc = correct / total
        epoch_test_acc = evaluate(pcn, test_loader, device)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} -- "
            f"Energy: {epoch_energy:.4f}, "
            f"Train Acc: {epoch_train_acc:.4f}, "
            f"Test Acc: {epoch_test_acc:.4f}"
        )

        save_checkpoint(pcn, epoch + 1, epoch_test_acc)

    print("\nTraining complete.")
    final_acc = evaluate(pcn, test_loader, device)
    print(f"Final test accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()
