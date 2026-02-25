"""Exercise 4: Evaluation Framework with ROC/PR Curves.

This exercise builds a comprehensive evaluation framework that compares
all detectors from the previous exercises using threshold-independent
metrics (ROC AUC, PR AUC) and threshold-specific metrics (precision,
recall, F1).

In production AIOps, choosing a detector is not just about accuracy --
it's about the tradeoff between false positives (alert fatigue) and
false negatives (missed incidents). ROC and PR curves visualize this
tradeoff across all possible thresholds.

After completing this exercise you should understand:
- Why ROC AUC alone is insufficient for imbalanced anomaly detection
- How PR AUC better reflects detector quality when anomalies are rare
- How to generate and interpret ROC and PR curve plots
- The practical meaning of precision/recall in alerting: "should we page?"
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "metrics.csv"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"

SERVICES = [
    "api-gateway",
    "user-service",
    "order-service",
    "payment-service",
    "inventory-service",
]

FEATURE_COLS = ["latency_ms", "error_rate", "cpu_percent", "memory_percent"]
CONTAMINATION = 0.05


# ---------------------------------------------------------------------------
# TODO(human): Implement compute_metrics
# ---------------------------------------------------------------------------
#
# Compute a comprehensive set of evaluation metrics for an anomaly detector.
#
# Metrics to compute:
#
#   1. ROC AUC (from scores):
#        from sklearn.metrics import roc_auc_score
#        roc_auc = roc_auc_score(y_true, y_scores)
#
#      ROC AUC measures the detector's ability to RANK anomalies higher than
#      normal points, across all possible thresholds. 1.0 = perfect ranking,
#      0.5 = random. It answers: "If I pick a random anomaly and a random
#      normal point, what's the probability the detector scores the anomaly
#      higher?"
#
#   2. PR AUC (from scores):
#        from sklearn.metrics import precision_recall_curve, auc
#        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
#        pr_auc = auc(recall_vals, precision_vals)
#
#      PR AUC focuses on the minority class (anomalies). It's more informative
#      than ROC AUC when classes are highly imbalanced. A detector that catches
#      most anomalies with few false positives will have high PR AUC. In alerting
#      terms: "can we fire alerts that are mostly real without missing incidents?"
#
#   3. Precision (from binary predictions):
#        from sklearn.metrics import precision_score
#        precision = precision_score(y_true, y_pred, zero_division=0)
#
#      "Of the alerts we fired, how many were real anomalies?"
#      Low precision = alert fatigue (too many false alarms).
#
#   4. Recall (from binary predictions):
#        from sklearn.metrics import recall_score
#        recall = recall_score(y_true, y_pred, zero_division=0)
#
#      "Of the real anomalies, how many did we catch?"
#      Low recall = missed incidents (silent failures).
#
#   5. F1 score (from binary predictions):
#        from sklearn.metrics import f1_score
#        f1 = f1_score(y_true, y_pred, zero_division=0)
#
#      Harmonic mean of precision and recall. Balances both concerns.
#
# Edge case handling:
#   When ALL labels are the same class (e.g., a service has no anomalies,
#   or the detector flags everything), roc_auc_score will raise a ValueError.
#   Wrap the ROC AUC computation in a try/except and return 0.0 on failure.
#   Similarly, precision_recall_curve may produce unexpected results with
#   a single class -- handle with try/except.
#
# Parameters:
#   y_true: np.ndarray       -- ground truth labels, shape (n,), bool or int (1=anomaly)
#   y_scores: np.ndarray     -- continuous anomaly scores, shape (n,), higher = more anomalous
#   y_pred: np.ndarray       -- binary predictions, shape (n,), bool or int (1=anomaly)
#
# Returns:
#   dict with keys: "roc_auc", "pr_auc", "precision", "recall", "f1"
#   All values are floats rounded to 4 decimal places.
#
# Example:
#   >>> y_true = np.array([0, 0, 1, 0, 1, 0, 0, 1])
#   >>> y_scores = np.array([0.1, 0.2, 0.9, 0.15, 0.85, 0.3, 0.1, 0.7])
#   >>> y_pred = np.array([0, 0, 1, 0, 1, 0, 0, 1])
#   >>> compute_metrics(y_true, y_scores, y_pred)
#   {"roc_auc": 0.9333, "pr_auc": 0.9167, "precision": 1.0, "recall": 1.0, "f1": 1.0}

def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    raise NotImplementedError(
        "TODO(human): Compute ROC AUC, PR AUC, precision, recall, and F1. "
        "Use sklearn.metrics functions. Handle edge cases where all labels "
        "are the same class (roc_auc_score raises ValueError)."
    )


# ---------------------------------------------------------------------------
# TODO(human): Implement plot_roc_pr_curves
# ---------------------------------------------------------------------------
#
# Generate a side-by-side plot with ROC curves (left) and PR curves (right)
# for all detectors. Save the plot to the given save_path.
#
# What to plot:
#
#   LEFT subplot: ROC curves
#     For each detector in results_dict:
#       from sklearn.metrics import roc_curve
#       fpr, tpr, _ = roc_curve(y_true, y_scores)
#       ax1.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
#     Also plot the diagonal (random baseline): ax1.plot([0,1], [0,1], 'k--', alpha=0.3)
#     Labels: xlabel="False Positive Rate", ylabel="True Positive Rate"
#     Title: "ROC Curves"
#
#   RIGHT subplot: PR curves
#     For each detector in results_dict:
#       from sklearn.metrics import precision_recall_curve
#       prec, rec, _ = precision_recall_curve(y_true, y_scores)
#       ax2.plot(rec, prec, label=f"{name} (AUC={pr_auc:.3f})")
#     Labels: xlabel="Recall", ylabel="Precision"
#     Title: "Precision-Recall Curves"
#
# Implementation hints:
#   import matplotlib
#   matplotlib.use("Agg")  # Non-interactive backend (no GUI window)
#   import matplotlib.pyplot as plt
#
#   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
#
#   # ... plot on ax1 and ax2 ...
#
#   ax1.legend(loc="lower right")
#   ax2.legend(loc="lower left")
#   fig.suptitle("Anomaly Detection: ROC and PR Curve Comparison")
#   fig.tight_layout()
#   fig.savefig(save_path, dpi=150, bbox_inches="tight")
#   plt.close(fig)
#
# Parameters:
#   results_dict: dict[str, dict]
#     Mapping detector name to a dict containing:
#       "y_true": np.ndarray    -- ground truth labels
#       "y_scores": np.ndarray  -- continuous anomaly scores
#       "roc_auc": float        -- pre-computed ROC AUC (for legend)
#       "pr_auc": float         -- pre-computed PR AUC (for legend)
#
#   save_path: Path
#     Where to save the PNG file (e.g., plots/roc_pr_comparison.png)
#
# Returns:
#   None (saves the plot to disk)
#
# Edge case handling:
#   If a detector's y_scores are all identical (e.g., detector failed),
#   skip it in the plot. Check: if np.std(y_scores) == 0, skip.
#
# Why both ROC and PR:
#   ROC curves can be misleadingly optimistic for imbalanced data. When
#   anomalies are 5% of data, even a detector with many false positives
#   will have a low FPR (because there are so many normal points).
#   PR curves expose this: a detector that fires 100 alerts to catch
#   10 anomalies will show precision=0.10 on the PR curve, making its
#   weakness obvious. Always look at BOTH curves together.

def plot_roc_pr_curves(
    results_dict: dict[str, dict],
    save_path: Path,
) -> None:
    raise NotImplementedError(
        "TODO(human): Create a 1x2 matplotlib figure with ROC curves (left) "
        "and PR curves (right). Plot one curve per detector with legend showing "
        "AUC values. Save to save_path."
    )


# ---------------------------------------------------------------------------
# Scaffolded: Run all detectors and collect results for evaluation
# ---------------------------------------------------------------------------


def get_all_detector_results(df: pd.DataFrame) -> dict[str, dict]:
    """Run all detectors on the full dataset and collect scores/predictions.

    Returns a dict mapping detector name to:
      {"y_true": ..., "y_scores": ..., "y_pred": ..., "roc_auc": ..., "pr_auc": ...}
    """
    results: dict[str, dict] = {}

    # Collect all services' data
    all_y_true: list[np.ndarray] = []
    all_X_scaled: list[np.ndarray] = []

    for service in SERVICES:
        svc_df = df[df["service"] == service].reset_index(drop=True)
        X = svc_df[FEATURE_COLS].values
        y = svc_df["is_anomaly"].values.astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        all_y_true.append(y)
        all_X_scaled.append(X_scaled)

    combined_y = np.concatenate(all_y_true)
    combined_X = np.concatenate(all_X_scaled)

    print(f"  Combined dataset: {combined_X.shape[0]} points, "
          f"{combined_y.sum()} anomalies ({combined_y.mean() * 100:.1f}%)")

    # --- Detector 1: Isolation Forest (sklearn) ---
    print("  Running Isolation Forest...")
    iso = IsolationForest(n_estimators=100, contamination=CONTAMINATION, random_state=42)
    iso.fit(combined_X)
    iso_scores = -iso.decision_function(combined_X)  # Negate: higher = more anomalous
    iso_preds = (iso.predict(combined_X) == -1).astype(int)  # -1 -> anomaly

    iso_metrics = compute_metrics(combined_y, iso_scores, iso_preds)
    results["IsolationForest"] = {
        "y_true": combined_y,
        "y_scores": iso_scores,
        "y_pred": iso_preds,
        **iso_metrics,
    }
    print(f"    ROC AUC={iso_metrics['roc_auc']:.4f}  PR AUC={iso_metrics['pr_auc']:.4f}  "
          f"F1={iso_metrics['f1']:.4f}")

    # --- Detectors 2-5: PyOD models ---
    # Import PyOD detectors (same as Exercise 3)
    try:
        from pyod.models.ecod import ECOD
        from pyod.models.copod import COPOD
        from pyod.models.knn import KNN
        from pyod.models.auto_encoder import AutoEncoder
    except ImportError as e:
        print(f"  ERROR: PyOD import failed: {e}")
        print("  Run 'uv sync' to install pyod.")
        return results

    pyod_detectors = {
        "ECOD": ECOD(contamination=CONTAMINATION),
        "COPOD": COPOD(contamination=CONTAMINATION),
        "KNN": KNN(n_neighbors=10, contamination=CONTAMINATION),
        "AutoEncoder": AutoEncoder(
            hidden_neurons=[16, 8, 8, 16],
            epochs=30,
            batch_size=32,
            contamination=CONTAMINATION,
            preprocessing=True,
        ),
    }

    for name, detector in pyod_detectors.items():
        print(f"  Running {name}...")
        try:
            detector.fit(combined_X)
            scores = detector.decision_function(combined_X)
            preds = detector.predict(combined_X)

            metrics = compute_metrics(combined_y, scores, preds)
            results[name] = {
                "y_true": combined_y,
                "y_scores": scores,
                "y_pred": preds,
                **metrics,
            }
            print(f"    ROC AUC={metrics['roc_auc']:.4f}  PR AUC={metrics['pr_auc']:.4f}  "
                  f"F1={metrics['f1']:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[name] = {
                "y_true": combined_y,
                "y_scores": np.zeros(len(combined_y)),
                "y_pred": np.zeros(len(combined_y), dtype=int),
                "roc_auc": 0.0,
                "pr_auc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

    return results


def print_comparison_table(results: dict[str, dict]) -> None:
    """Print a formatted comparison table of all detectors."""
    print(f"\n{'Detector':<18s} {'ROC AUC':>8s} {'PR AUC':>8s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    print("-" * 55)

    # Sort by PR AUC descending (most informative metric for imbalanced data)
    sorted_names = sorted(results.keys(), key=lambda n: results[n].get("pr_auc", 0), reverse=True)

    for name in sorted_names:
        r = results[name]
        print(f"{name:<18s} {r.get('roc_auc', 0):8.4f} {r.get('pr_auc', 0):8.4f} "
              f"{r.get('precision', 0):6.3f} {r.get('recall', 0):6.3f} {r.get('f1', 0):6.3f}")


def main() -> None:
    print("=" * 60)
    print("Exercise 4: Evaluation Framework")
    print("=" * 60)

    # Load data
    if not DATA_FILE.exists():
        print(f"\nERROR: {DATA_FILE} not found.")
        print("Run src/00_generate_metrics.py first to generate the dataset.")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"\nLoaded {len(df):,} data points")

    # Run all detectors
    print("\nRunning all detectors on combined dataset...")
    results = get_all_detector_results(df)

    # Print comparison
    print_comparison_table(results)

    # Generate plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOTS_DIR / "roc_pr_comparison.png"

    print(f"\nGenerating ROC/PR curves...")
    plot_roc_pr_curves(results, save_path)
    print(f"Saved to {save_path}")

    # Summary insights
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print("Key takeaways:")
    print("  1. ROC AUC can be misleadingly high for imbalanced data.")
    print("     PR AUC gives a more honest picture of detector quality.")
    print("  2. No single detector dominates all scenarios.")
    print("     ECOD/COPOD are fast baselines; KNN adapts to density;")
    print("     AutoEncoder captures nonlinear patterns.")
    print("  3. In production, the choice depends on your alert budget:")
    print("     - High precision (few false alarms) -> conservative threshold")
    print("     - High recall (catch everything) -> aggressive threshold")
    print("     - The ROC/PR curves show this tradeoff at every threshold.")


if __name__ == "__main__":
    main()
