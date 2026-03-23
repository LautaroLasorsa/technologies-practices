"""
3-dataset benchmark: PCN vs Backprop MLP.

Trains both models on MNIST, Fashion-MNIST, and KMNIST using identical
architectures, then produces comparison plots.
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.backprop_baseline import BackpropMLP, train_backprop_model
from src.predictive_coding_network import PredictiveCodingNetwork
from src.visualization import plot_comparison, plot_benchmark_summary

DATASETS = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
    "kmnist": datasets.KMNIST,
}

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def load_dataset(name: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Load a torchvision dataset and return train/test DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset_cls = DATASETS[name]
    train_set = dataset_cls(root=str(DATA_DIR), train=True, download=True, transform=transform)
    test_set = dataset_cls(root=str(DATA_DIR), train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def evaluate_pcn(pcn: PredictiveCodingNetwork, test_loader: DataLoader, device: torch.device) -> float:
    """Evaluate PCN accuracy on a test set (forward pass only, no inference)."""
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


def evaluate_backprop(model: BackpropMLP, test_loader: DataLoader, device: torch.device) -> float:
    """Evaluate backprop MLP accuracy on a test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total


def train_pcn_model(
    pcn: PredictiveCodingNetwork,
    train_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
) -> dict:
    """Train a PCN and return training history.

    Args:
        pcn: PredictiveCodingNetwork instance.
        train_loader: DataLoader yielding (images, labels) batches.
        num_epochs: Number of training epochs.
        device: torch.device for computation.

    Returns:
        Dict with 'free_energy' and 'accuracy' lists (one entry per epoch).
    """
    history = {"free_energy": [], "accuracy": []}

    for epoch in range(num_epochs):
        total_energy = 0.0
        correct = 0
        total = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"  [PCN] Epoch {epoch + 1}/{num_epochs}", leave=False)
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
        epoch_acc = correct / total
        history["free_energy"].append(epoch_energy)
        history["accuracy"].append(epoch_acc)
        print(f"  [PCN] Epoch {epoch + 1}/{num_epochs} -- Energy: {epoch_energy:.4f}, Acc: {epoch_acc:.4f}")

    return history


def run_benchmark(
    dataset_names: list[str],
    layer_dims: list[int],
    num_epochs: int,
    batch_size: int,
    inference_steps: int,
    inference_rate: float,
    learning_rate: float,
) -> dict:
    """Run full benchmark on specified datasets.

    Returns:
        Dict mapping dataset_name -> {
            'backprop_history': {...},
            'pcn_history': {...},
            'backprop_test_acc': float,
            'pcn_test_acc': float,
        }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}

    for ds_name in dataset_names:
        print(f"\n{'=' * 60}")
        print(f"  Dataset: {ds_name.upper()}")
        print(f"{'=' * 60}")

        train_loader, test_loader = load_dataset(ds_name, batch_size)

        # --- Backprop MLP ---
        print(f"\nTraining Backprop MLP on {ds_name}...")
        bp_model = BackpropMLP(layer_dims)
        bp_history = train_backprop_model(
            bp_model, train_loader, num_epochs=num_epochs,
            learning_rate=learning_rate, device=device,
        )
        bp_test_acc = evaluate_backprop(bp_model, test_loader, device)
        print(f"  [Backprop] Test accuracy: {bp_test_acc:.4f}")

        # --- PCN ---
        print(f"\nTraining PCN on {ds_name}...")
        pcn = PredictiveCodingNetwork(
            layer_dims=layer_dims,
            inference_rate=inference_rate,
            learning_rate=learning_rate,
            num_inference_steps=inference_steps,
        )
        pcn.to(device)
        pcn_history = train_pcn_model(pcn, train_loader, num_epochs, device)
        pcn_test_acc = evaluate_pcn(pcn, test_loader, device)
        print(f"  [PCN] Test accuracy: {pcn_test_acc:.4f}")

        # --- Plot comparison ---
        plot_comparison(bp_history, pcn_history, ds_name, str(OUTPUT_DIR))

        results[ds_name] = {
            "backprop_history": bp_history,
            "pcn_history": pcn_history,
            "backprop_test_acc": bp_test_acc,
            "pcn_test_acc": pcn_test_acc,
        }

    # --- Summary ---
    print_summary(results)
    if len(results) > 1:
        plot_benchmark_summary(results, str(OUTPUT_DIR))

    return results


def print_summary(results: dict) -> None:
    """Print a summary table of final accuracies."""
    print(f"\n{'=' * 60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Dataset':<20} {'Backprop':>10} {'PCN':>10}")
    print(f"  {'-' * 40}")
    for ds_name, res in results.items():
        bp_acc = res["backprop_test_acc"]
        pcn_acc = res["pcn_test_acc"]
        print(f"  {ds_name:<20} {bp_acc:>10.4f} {pcn_acc:>10.4f}")
    print(f"{'=' * 60}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCN vs Backprop Benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["mnist", "fashion_mnist", "kmnist", "all"],
        help="Dataset to benchmark (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--inference-steps", type=int, default=20, help="PCN inference steps (default: 20)")
    parser.add_argument("--inference-rate", type=float, default=0.1, help="PCN inference rate gamma (default: 0.1)")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate alpha (default: 0.001)")
    parser.add_argument(
        "--layer-dims",
        type=int,
        nargs="+",
        default=[784, 256, 128, 10],
        help="Layer dimensions (default: 784 256 128 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset == "all":
        dataset_names = list(DATASETS.keys())
    else:
        dataset_names = [args.dataset]

    run_benchmark(
        dataset_names=dataset_names,
        layer_dims=args.layer_dims,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        inference_steps=args.inference_steps,
        inference_rate=args.inference_rate,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
