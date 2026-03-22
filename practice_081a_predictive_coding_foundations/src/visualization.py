"""
Visualization utilities for PCN vs Backprop comparison.

All plots save to the outputs/ directory and display with plt.show().
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_comparison(
    backprop_history: dict,
    pcn_history: dict,
    dataset_name: str,
    output_dir: str,
) -> None:
    """Plot side-by-side accuracy curves for backprop and PCN.

    Args:
        backprop_history: Dict with 'loss' and 'accuracy' lists.
        pcn_history: Dict with 'free_energy' and 'accuracy' lists.
        dataset_name: Name of the dataset (for title).
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs_bp = range(1, len(backprop_history["accuracy"]) + 1)
    epochs_pcn = range(1, len(pcn_history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Accuracy
    axes[0].plot(epochs_bp, backprop_history["accuracy"], "b-o", label="Backprop", markersize=4)
    axes[0].plot(epochs_pcn, pcn_history["accuracy"], "r-s", label="PCN", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Accuracy")
    axes[0].set_title(f"{dataset_name.upper()} -- Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: Loss / Energy
    ax_loss = axes[1]
    ax_loss.plot(epochs_bp, backprop_history["loss"], "b-o", label="Backprop Loss", markersize=4)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Backprop Loss", color="b")
    ax_loss.tick_params(axis="y", labelcolor="b")

    ax_energy = ax_loss.twinx()
    ax_energy.plot(epochs_pcn, pcn_history["free_energy"], "r-s", label="PCN Free Energy", markersize=4)
    ax_energy.set_ylabel("PCN Free Energy", color="r")
    ax_energy.tick_params(axis="y", labelcolor="r")

    axes[1].set_title(f"{dataset_name.upper()} -- Loss / Energy")

    # Combined legend
    lines_1, labels_1 = ax_loss.get_legend_handles_labels()
    lines_2, labels_2 = ax_energy.get_legend_handles_labels()
    ax_loss.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    ax_loss.grid(True, alpha=0.3)

    fig.tight_layout()
    path = Path(output_dir) / f"comparison_{dataset_name}.png"
    fig.savefig(str(path), dpi=150)
    print(f"  Saved: {path}")
    plt.show()


def plot_prediction_errors(
    neural_activities: list[torch.Tensor],
    cortical_layers: list,
    output_dir: str,
    max_samples: int = 8,
) -> None:
    """Plot heatmaps of prediction errors at each layer for a batch of samples.

    Args:
        neural_activities: List [a_0, ..., a_L] of activities.
        cortical_layers: List of CorticalLayer instances.
        output_dir: Directory to save the plot.
        max_samples: Max number of samples to show.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_layers = len(cortical_layers)
    n_samples = min(max_samples, neural_activities[0].shape[0])

    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 4))
    if num_layers == 1:
        axes = [axes]

    for ell, layer in enumerate(cortical_layers):
        prediction = layer.compute_prediction(neural_activities[ell])
        epsilon = layer.compute_prediction_error(neural_activities[ell + 1], prediction)
        error_data = epsilon[:n_samples].detach().cpu().numpy()

        im = axes[ell].imshow(error_data, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        axes[ell].set_title(f"Layer {ell} -> {ell + 1} errors")
        axes[ell].set_xlabel("Neuron index")
        axes[ell].set_ylabel("Sample")
        fig.colorbar(im, ax=axes[ell], fraction=0.046, pad=0.04)

    fig.suptitle("Prediction Errors by Layer", fontsize=14)
    fig.tight_layout()
    path = Path(output_dir) / "prediction_errors.png"
    fig.savefig(str(path), dpi=150)
    print(f"  Saved: {path}")
    plt.show()


def plot_energy_trace(
    energy_trace: list[float],
    output_dir: str,
    title: str = "Free Energy During Inference",
) -> None:
    """Plot free energy convergence during a single inference phase.

    Args:
        energy_trace: List of energy values at each inference step.
        output_dir: Directory to save the plot.
        title: Plot title.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    steps = range(1, len(energy_trace) + 1)
    ax.plot(steps, energy_trace, "g-o", markersize=4, linewidth=2)
    ax.set_xlabel("Inference Step")
    ax.set_ylabel("Free Energy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Annotate convergence
    if len(energy_trace) >= 2:
        reduction = (energy_trace[0] - energy_trace[-1]) / max(abs(energy_trace[0]), 1e-8) * 100
        ax.annotate(
            f"Reduction: {reduction:.1f}%",
            xy=(len(energy_trace), energy_trace[-1]),
            xytext=(len(energy_trace) * 0.6, (energy_trace[0] + energy_trace[-1]) / 2),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=10,
            color="gray",
        )

    fig.tight_layout()
    path = Path(output_dir) / "energy_trace.png"
    fig.savefig(str(path), dpi=150)
    print(f"  Saved: {path}")
    plt.show()


def plot_benchmark_summary(
    results: dict,
    output_dir: str,
) -> None:
    """Bar chart comparing final test accuracies across datasets.

    Args:
        results: Dict mapping dataset_name -> {
            'backprop_test_acc': float,
            'pcn_test_acc': float,
        }
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset_names = list(results.keys())
    bp_accs = [results[ds]["backprop_test_acc"] for ds in dataset_names]
    pcn_accs = [results[ds]["pcn_test_acc"] for ds in dataset_names]

    x = range(len(dataset_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_bp = ax.bar([i - width / 2 for i in x], bp_accs, width, label="Backprop", color="steelblue")
    bars_pcn = ax.bar([i + width / 2 for i in x], pcn_accs, width, label="PCN", color="indianred")

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("PCN vs Backprop -- Test Accuracy Comparison")
    ax.set_xticks(list(x))
    ax.set_xticklabels([ds.upper() for ds in dataset_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Add value labels on bars
    for bar in bars_bp:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.3f}",
                ha="center", va="bottom", fontsize=9, color="steelblue")
    for bar in bars_pcn:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.3f}",
                ha="center", va="bottom", fontsize=9, color="indianred")

    fig.tight_layout()
    path = Path(output_dir) / "benchmark_summary.png"
    fig.savefig(str(path), dpi=150)
    print(f"  Saved: {path}")
    plt.show()
