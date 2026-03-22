"""Experiment runner for Session B exercises."""

import argparse
import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.backprop_baseline import BackpropMLP
from src.predictive_coding_network import PredictiveCodingNetwork
from src.visualization import (
    plot_continual_learning,
    plot_generated_images,
    plot_missing_data_reconstruction,
    plot_noise_robustness,
    plot_retinotopic_results,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "checkpoints"


def get_fashion_mnist(batch_size: int = 64):
    """Load Fashion-MNIST dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.FashionMNIST("data", train=True, download=True, transform=transform)
    test = datasets.FashionMNIST("data", train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size),
    )


def get_cifar10(batch_size: int = 64):
    """Load CIFAR-10 dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    test = datasets.CIFAR10("data", train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size),
    )


def train_or_load_pcn(layer_dims=None, num_epochs=10, batch_size=64):
    """Train a PCN on Fashion-MNIST or load from checkpoint."""
    layer_dims = layer_dims or [784, 256, 128, 10]
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "pcn_fashion_mnist.pt")

    pcn = PredictiveCodingNetwork(layer_dims=layer_dims).to(DEVICE)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        for i, layer in enumerate(pcn.cortical_layers):
            layer.synaptic_weights = state[f"layer_{i}_weights"].to(DEVICE)
        return pcn

    print("Training PCN on Fashion-MNIST...")
    train_loader, _ = get_fashion_mnist(batch_size)

    for epoch in range(num_epochs):
        total_acc = 0
        count = 0
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(DEVICE)
            one_hot = torch.zeros(images.size(0), 10, device=DEVICE)
            one_hot.scatter_(1, labels.to(DEVICE).unsqueeze(1), 1.0)
            metrics = pcn.train_step(images, one_hot)
            total_acc += metrics["accuracy"] * images.size(0)
            count += images.size(0)
        print(f"  Epoch {epoch + 1}/{num_epochs} -- Acc: {total_acc / count:.4f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    state = {f"layer_{i}_weights": l.synaptic_weights for i, l in enumerate(pcn.cortical_layers)}
    torch.save(state, checkpoint_path)
    return pcn


def run_generative(pcn):
    """Phase 1: Generate images from labels."""
    from src.generative_inference import generate_all_classes

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n=== Phase 1: Generative Mode ===")
    print("Generating images for all 10 Fashion-MNIST classes...")
    results = generate_all_classes(pcn, num_classes=10, samples_per_class=8, device=DEVICE)
    plot_generated_images(results, os.path.join(OUTPUT_DIR, "generated_images.png"))
    print(f"Saved to {OUTPUT_DIR}/generated_images.png")


def run_missing_data(pcn):
    """Phase 2: Missing data inference."""
    from src.missing_data import create_masks, infer_missing_data

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n=== Phase 2: Missing Data Inference ===")
    _, test_loader = get_fashion_mnist(batch_size=8)
    images, labels = next(iter(test_loader))
    images = images.view(8, -1).to(DEVICE)

    mask_types = ["top_half", "bottom_half", "left_half", "random_50", "random_75"]
    results = {}
    for mask_type in mask_types:
        mask = create_masks(784, mask_type).to(DEVICE)
        mask_batch = mask.unsqueeze(0).expand_as(images)
        partial = images * mask_batch
        reconstructed, pred_labels = infer_missing_data(
            pcn, partial, mask_batch, device=DEVICE,
        )
        results[mask_type] = {
            "original": images,
            "partial": partial,
            "reconstructed": reconstructed,
            "mask": mask,
            "pred_labels": pred_labels,
            "true_labels": labels,
        }
        print(f"  {mask_type}: predicted {pred_labels.cpu().tolist()}, actual {labels.tolist()}")

    plot_missing_data_reconstruction(results, os.path.join(OUTPUT_DIR, "missing_data.png"))
    print(f"Saved to {OUTPUT_DIR}/missing_data.png")


def run_precision(pcn):
    """Phase 3: Precision weighting experiment."""
    from src.precision_weighting import noise_robustness_experiment

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n=== Phase 3: Precision Weighting ===")
    _, test_loader = get_fashion_mnist(batch_size=256)
    images, labels = next(iter(test_loader))
    images = images.view(images.size(0), -1)

    results = noise_robustness_experiment(pcn, images, labels, device=DEVICE)
    plot_noise_robustness(results, os.path.join(OUTPUT_DIR, "precision_robustness.png"))
    print(f"Saved to {OUTPUT_DIR}/precision_robustness.png")


def run_continual():
    """Phase 4: Continual learning experiment."""
    from src.continual_learning import continual_learning_experiment

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n=== Phase 4: Continual Learning ===")

    # Split Fashion-MNIST: task1 = classes 0-4 (tops), task2 = classes 5-9 (bottoms/accessories)
    transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.FashionMNIST("data", train=True, download=True, transform=transform)
    full_test = datasets.FashionMNIST("data", train=False, download=True, transform=transform)

    task1_train_idx = [i for i, (_, l) in enumerate(full_train) if l < 5]
    task2_train_idx = [i for i, (_, l) in enumerate(full_train) if l >= 5]
    task1_test_idx = [i for i, (_, l) in enumerate(full_test) if l < 5]
    task2_test_idx = [i for i, (_, l) in enumerate(full_test) if l >= 5]

    task1_train = DataLoader(Subset(full_train, task1_train_idx), batch_size=64, shuffle=True)
    task2_train = DataLoader(Subset(full_train, task2_train_idx), batch_size=64, shuffle=True)
    task1_test = DataLoader(Subset(full_test, task1_test_idx), batch_size=256)
    task2_test = DataLoader(Subset(full_test, task2_test_idx), batch_size=256)

    # Fresh PCN for fair comparison
    pcn = PredictiveCodingNetwork(layer_dims=[784, 256, 128, 10]).to(DEVICE)
    pcn_results = continual_learning_experiment(
        pcn, task1_train, task2_train, task1_test, task2_test, device=DEVICE,
    )

    # Compare with backprop MLP
    mlp = BackpropMLP(layer_dims=[784, 256, 128, 10]).to(DEVICE)
    mlp_results = _backprop_continual(mlp, task1_train, task2_train, task1_test, task2_test)

    print(
        f"\n  PCN: Task1 baseline={pcn_results['task1_baseline']:.4f}, "
        f"after Task2={pcn_results['task1_after_task2']:.4f}, "
        f"forgetting={pcn_results['forgetting']:.4f}"
    )
    print(
        f"  MLP: Task1 baseline={mlp_results['task1_baseline']:.4f}, "
        f"after Task2={mlp_results['task1_after_task2']:.4f}, "
        f"forgetting={mlp_results['forgetting']:.4f}"
    )

    plot_continual_learning(pcn_results, mlp_results, os.path.join(OUTPUT_DIR, "continual_learning.png"))


def _backprop_continual(mlp, task1_train, task2_train, task1_test, task2_test, num_epochs=5):
    """Pre-built backprop continual learning for comparison."""
    import torch.nn as nn
    import torch.optim as optim

    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    def train_epoch(loader):
        for imgs, lbls in loader:
            imgs = imgs.view(imgs.size(0), -1).to(DEVICE)
            lbls = lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(mlp(imgs), lbls)
            loss.backward()
            optimizer.step()

    def evaluate(loader):
        correct = total = 0
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.view(imgs.size(0), -1).to(DEVICE)
                preds = mlp(imgs).argmax(1)
                correct += (preds == lbls.to(DEVICE)).sum().item()
                total += lbls.size(0)
        return correct / total

    for _ in range(num_epochs):
        train_epoch(task1_train)
    baseline = evaluate(task1_test)

    for _ in range(num_epochs):
        train_epoch(task2_train)

    task1_after = evaluate(task1_test)
    return {
        "task1_baseline": baseline,
        "task1_after_task2": task1_after,
        "task2_accuracy": evaluate(task2_test),
        "forgetting": (baseline - task1_after) / max(baseline, 1e-6),
    }


def run_retinotopic():
    """Phase 5: Convolutional PCN on CIFAR-10."""
    print("\n=== Phase 5: Retinotopic Cortical Layers (CIFAR-10) ===")
    print("Testing convolutional PCN layers...")

    from src.retinotopic_layer import RetinotopicCorticalLayer

    # Build a simple conv PCN: 3 conv layers
    # Architecture mirrors V1 -> V2 -> V4 with increasing channels
    layers = [
        RetinotopicCorticalLayer(3, 32, kernel_size=3, padding=1),    # V1: 3->32 channels
        RetinotopicCorticalLayer(32, 64, kernel_size=3, padding=1),   # V2: 32->64
        RetinotopicCorticalLayer(64, 128, kernel_size=3, stride=2, padding=1),  # V4: spatial reduction
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Verify the layers work with a sample batch
    sample_batch = torch.randn(4, 3, 32, 32)
    try:
        pred1 = layers[0].compute_prediction(sample_batch)
        print(f"  V1 output shape: {pred1.shape}")
        pred2 = layers[1].compute_prediction(pred1)
        print(f"  V2 output shape: {pred2.shape}")
        pred3 = layers[2].compute_prediction(pred2)
        print(f"  V4 output shape: {pred3.shape}")

        # Test error computation
        error = layers[0].compute_prediction_error(pred1, pred1)
        print(f"  Zero error check (should be 0): {error.abs().sum().item():.6f}")

        # Test Hebbian update
        error_signal = torch.randn_like(pred1) * 0.01
        layers[0].hebbian_update(sample_batch, error_signal)
        print("  Hebbian update: OK")

        print("  Retinotopic layers working!")
    except NotImplementedError as e:
        print(f"  TODO(human) not yet implemented: {e}")
        print("  Implement RetinotopicCorticalLayer first.")

    plot_retinotopic_results(None, os.path.join(OUTPUT_DIR, "retinotopic.png"))


def main():
    parser = argparse.ArgumentParser(description="PCN Divergence Experiments")
    parser.add_argument(
        "--experiment",
        choices=["generative", "missing-data", "precision", "continual", "retinotopic", "all"],
        default="all",
    )
    args = parser.parse_args()

    pcn = None
    if args.experiment in ("generative", "missing-data", "precision", "all"):
        pcn = train_or_load_pcn()

    if args.experiment == "generative" or args.experiment == "all":
        run_generative(pcn)
    if args.experiment == "missing-data" or args.experiment == "all":
        run_missing_data(pcn)
    if args.experiment == "precision" or args.experiment == "all":
        run_precision(pcn)
    if args.experiment == "continual" or args.experiment == "all":
        run_continual()
    if args.experiment == "retinotopic" or args.experiment == "all":
        run_retinotopic()


if __name__ == "__main__":
    main()
