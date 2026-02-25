"""Exercise 2: Train a Neural Network to Predict Query Latency.

This exercise builds the core ML component of a learned query optimizer:
a model that predicts how long a query will take to execute based on its
execution plan features.

Traditional optimizers use hand-tuned cost formulas. Here we replace that
with a data-driven model trained on actual execution times collected in
Exercise 1.

TODO(human) functions are in src/cost_model.py:
  - class LatencyPredictor(nn.Module)
  - train_model()
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.shared import DATA_DIR, MODELS_DIR, PLOTS_DIR
from src.cost_model import LatencyPredictor, train_model


# ---------------------------------------------------------------------------
# Evaluation helpers (scaffolded)
# ---------------------------------------------------------------------------

def evaluate_model(
    model: LatencyPredictor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> dict:
    """Evaluate model on test set, returning metrics."""
    model.eval()
    X_norm = (X_test - feature_mean) / feature_std
    X_t = torch.tensor(X_norm, dtype=torch.float32)

    with torch.no_grad():
        y_pred_log = model(X_t).squeeze().numpy()

    y_pred = np.expm1(y_pred_log)  # inverse of log1p
    y_actual = y_test

    # Metrics
    errors = np.abs(y_pred - y_actual)
    rel_errors = errors / np.maximum(y_actual, 0.01)

    return {
        "mae_ms": float(np.mean(errors)),
        "median_ae_ms": float(np.median(errors)),
        "mean_relative_error": float(np.mean(rel_errors)),
        "median_relative_error": float(np.median(rel_errors)),
        "y_actual": y_actual,
        "y_pred": y_pred,
    }


def plot_loss_curves(history: dict[str, list[float]]) -> None:
    """Plot training and validation loss curves."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="Train Loss", alpha=0.8)
    ax.plot(history["val_loss"], label="Val Loss", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log-latency)")
    ax.set_title("Latency Predictor Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = PLOTS_DIR / "loss_curves.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss curves saved to {path}")


def plot_predictions(y_actual: np.ndarray, y_pred: np.ndarray) -> None:
    """Plot actual vs predicted latency scatter plot."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_actual, y_pred, alpha=0.5, s=30)

    # Perfect prediction line
    lo = min(y_actual.min(), y_pred.min()) * 0.9
    hi = max(y_actual.max(), y_pred.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "r--", alpha=0.7, label="Perfect prediction")

    ax.set_xlabel("Actual Latency (ms)")
    ax.set_ylabel("Predicted Latency (ms)")
    ax.set_title("Actual vs Predicted Query Latency")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    path = PLOTS_DIR / "predictions.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Predictions plot saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Exercise 2: Train Latency Prediction Model")
    print("=" * 60)

    # Load data from Exercise 1
    features_path = DATA_DIR / "plan_features.csv"
    if not features_path.exists():
        print(f"\nERROR: {features_path} not found.")
        print("Run Exercise 1 first: uv run python src/01_explain_parser.py")
        return

    print("\nLoading plan features...")
    df = pd.read_csv(features_path)
    print(f"  Loaded {len(df)} samples")

    # Separate features from metadata
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feat_cols].values.astype(np.float32)
    y = df["latency_ms"].values.astype(np.float32)

    print(f"  Feature vector size: {X.shape[1]}")
    print(f"  Latency range: {y.min():.2f} - {y.max():.2f} ms")
    print(f"  Latency mean: {y.mean():.2f} ms, median: {np.median(y):.2f} ms")

    # Train/val/test split (60/20/20)
    n = len(X)
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\n  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train model
    print("\nTraining latency predictor...")
    print("-" * 40)
    n_features = X_train.shape[1]
    model = LatencyPredictor(n_features)
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=200, lr=1e-3)

    # Plot loss curves
    print("\nPlotting loss curves...")
    plot_loss_curves(history)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    print("-" * 40)

    # Compute normalization stats from training data (same as in train_model)
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)
    feature_std[feature_std == 0] = 1.0

    metrics = evaluate_model(model, X_test, y_test, feature_mean, feature_std)
    print(f"  MAE: {metrics['mae_ms']:.2f} ms")
    print(f"  Median AE: {metrics['median_ae_ms']:.2f} ms")
    print(f"  Mean Relative Error: {metrics['mean_relative_error']:.2%}")
    print(f"  Median Relative Error: {metrics['median_relative_error']:.2%}")

    # Plot predictions
    print("\nPlotting predictions...")
    plot_predictions(metrics["y_actual"], metrics["y_pred"])

    # Save model and normalization stats
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "latency_predictor.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": n_features,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }, model_path)
    print(f"\n  Model saved to {model_path}")

    print("\nExercise 2 complete!")


if __name__ == "__main__":
    main()
