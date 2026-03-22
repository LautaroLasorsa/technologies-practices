"""
Visualization utilities for PCN divergence experiments (Session B).

All plots save to the specified output path and display with plt.show().
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_generated_images(
    results: dict[int, torch.Tensor],
    output_path: str,
    image_size: int = 28,
) -> None:
    """Plot grid of generated images per class.

    Args:
        results: Dict mapping label (int) -> generated images (batch, input_dim)
        output_path: Path to save the plot.
        image_size: Side length of square images (default 28 for MNIST).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_classes = len(results)
    samples_per_class = results[0].shape[0] if 0 in results else 8

    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(samples_per_class * 1.5, num_classes * 1.5))

    fashion_labels = [
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Boot",
    ]

    for label in range(num_classes):
        images = results[label].detach().cpu()
        for j in range(min(samples_per_class, images.shape[0])):
            ax = axes[label, j] if num_classes > 1 else axes[j]
            img = images[j].view(image_size, image_size).numpy()
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if j == 0:
                lbl = fashion_labels[label] if label < len(fashion_labels) else str(label)
                ax.set_ylabel(lbl, fontsize=8, rotation=0, labelpad=40, va="center")

    fig.suptitle("Generated Images (PCN Generative Mode)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.show()


def plot_missing_data_reconstruction(
    results: dict[str, dict],
    output_path: str,
    image_size: int = 28,
    max_samples: int = 4,
) -> None:
    """Plot original, masked, and reconstructed images for each mask type.

    Args:
        results: Dict mapping mask_type -> {original, partial, reconstructed, mask, pred_labels, true_labels}
        output_path: Path to save the plot.
        image_size: Side length of square images.
        max_samples: Number of sample images to show per mask type.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mask_types = list(results.keys())
    n_masks = len(mask_types)
    n_samples = min(max_samples, results[mask_types[0]]["original"].shape[0])

    # Layout: 3 rows (original, masked, reconstructed) x (n_masks * n_samples) columns
    fig, axes = plt.subplots(3, n_masks * n_samples, figsize=(n_masks * n_samples * 1.2, 4))

    row_labels = ["Original", "Masked", "Reconstructed"]

    for mi, mask_type in enumerate(mask_types):
        data = results[mask_type]
        for si in range(n_samples):
            col = mi * n_samples + si
            for row, key in enumerate(["original", "partial", "reconstructed"]):
                ax = axes[row, col]
                img = data[key][si].detach().cpu().view(image_size, image_size).numpy()
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                if si == 0 and mi == 0:
                    ax.set_ylabel(row_labels[row], fontsize=9)
            # Title on top row
            if si == n_samples // 2:
                axes[0, col].set_title(mask_type.replace("_", " "), fontsize=8)

    fig.suptitle("Missing Data Reconstruction (PCN Inference)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.show()


def plot_noise_robustness(
    results: dict,
    output_path: str,
) -> None:
    """Line plot of noise level vs accuracy for standard and precision-weighted PCN.

    Args:
        results: Dict with "noise_levels", "pcn_standard", "pcn_precision" lists.
        output_path: Path to save the plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        results["noise_levels"], results["pcn_standard"],
        "b-o", label="PCN (forward pass)", markersize=6,
    )
    ax.plot(
        results["noise_levels"], results["pcn_precision"],
        "r-s", label="PCN (precision-weighted)", markersize=6,
    )

    ax.set_xlabel("Noise Standard Deviation", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)
    ax.set_title("Noise Robustness: Standard vs Precision-Weighted PCN", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    plt.show()


def plot_continual_learning(
    pcn_results: dict,
    mlp_results: dict,
    output_path: str,
) -> None:
    """Grouped bar chart comparing PCN vs MLP continual learning.

    Args:
        pcn_results: Dict with task1_baseline, task1_after_task2, task2_accuracy, forgetting.
        mlp_results: Same structure for MLP.
        output_path: Path to save the plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    labels = ["Task 1\n(baseline)", "Task 1\n(after Task 2)", "Task 2\n(after training)", "Forgetting\n(lower=better)"]
    pcn_vals = [
        pcn_results["task1_baseline"],
        pcn_results["task1_after_task2"],
        pcn_results["task2_accuracy"],
        pcn_results["forgetting"],
    ]
    mlp_vals = [
        mlp_results["task1_baseline"],
        mlp_results["task1_after_task2"],
        mlp_results["task2_accuracy"],
        mlp_results["forgetting"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_pcn = ax.bar(x - width / 2, pcn_vals, width, label="PCN", color="indianred")
    bars_mlp = ax.bar(x + width / 2, mlp_vals, width, label="Backprop MLP", color="steelblue")

    ax.set_ylabel("Accuracy / Forgetting Rate", fontsize=12)
    ax.set_title("Continual Learning: PCN vs Backprop MLP", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Value labels
    for bar in list(bars_pcn) + list(bars_mlp):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 0.01,
            f"{height:.3f}", ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    plt.show()


def plot_retinotopic_results(
    results: dict | None,
    output_path: str,
) -> None:
    """Placeholder plot for retinotopic (conv) PCN results.

    Args:
        results: Optional results dict (None if layers not yet implemented).
        output_path: Path to save the plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    if results is None:
        ax.text(
            0.5, 0.5,
            "Retinotopic layers implemented.\nRun full training to see results.",
            transform=ax.transAxes, ha="center", va="center", fontsize=14,
        )
        ax.set_title("Retinotopic (Convolutional) PCN -- Phase 5", fontsize=13)
    else:
        # Plot whatever metrics are available
        if "accuracy" in results:
            ax.plot(results["accuracy"], "g-o", markersize=4)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("Retinotopic PCN Training on CIFAR-10", fontsize=13)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    plt.show()
