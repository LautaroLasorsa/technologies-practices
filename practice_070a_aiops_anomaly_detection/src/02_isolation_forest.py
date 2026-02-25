"""Exercise 2: Isolation Forest on Multivariate Metrics.

This exercise transitions from univariate statistical methods to a
multivariate ML detector. Isolation Forest considers ALL four metrics
(latency_ms, error_rate, cpu_percent, memory_percent) simultaneously,
enabling detection of collective anomalies where no single metric
exceeds its individual threshold but the combination is abnormal.

How Isolation Forest works:
  1. Build an ensemble of isolation trees (default: 100 trees)
  2. Each tree randomly selects a feature and a random split value
  3. Recursively partition data until each point is isolated in its own leaf
  4. Anomalies are "few and different" -- they need fewer splits to isolate,
     resulting in shorter path lengths from root to leaf
  5. Anomaly score = average path length across all trees (shorter = more anomalous)

After completing this exercise you should understand:
- How to prepare multivariate feature matrices for anomaly detection
- The meaning of IsolationForest's contamination, n_estimators, max_samples
- The difference between sklearn's -1/+1 labels and boolean anomaly labels
- Why multivariate detection catches anomalies that univariate misses
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "metrics.csv"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

SERVICES = [
    "api-gateway",
    "user-service",
    "order-service",
    "payment-service",
    "inventory-service",
]

FEATURE_COLS = ["latency_ms", "error_rate", "cpu_percent", "memory_percent"]

# Contamination: expected proportion of anomalies. Set based on the dataset's
# actual anomaly rate (~3-8%). If unknown, start with 0.05 and tune.
CONTAMINATION = 0.05


# ---------------------------------------------------------------------------
# TODO(human): Implement train_isolation_forest
# ---------------------------------------------------------------------------
#
# Create, configure, and fit a scikit-learn IsolationForest model on the
# given training data. Return the fitted model.
#
# What IsolationForest's key parameters mean:
#
#   n_estimators (int, default=100):
#     Number of isolation trees in the ensemble. More trees = more stable
#     scores, but slower training. 100-200 is typical for most datasets.
#
#   max_samples (int or float, default="auto"):
#     Number of samples drawn to build each tree. "auto" uses min(256, n_samples).
#     Smaller subsamples make each tree more diverse (good for anomaly detection)
#     because anomalies are easier to isolate in smaller subsets. Using
#     "auto" is recommended unless you have domain-specific reasons to change it.
#
#   contamination (float or "auto"):
#     Expected proportion of outliers in the training data. This sets the
#     decision threshold for predict(): the threshold is chosen so that
#     contamination * 100% of training samples are labeled as anomalies.
#     - Use a float (e.g., 0.05) when you have a rough estimate of the anomaly rate
#     - Use "auto" to use the offset-based threshold from the original paper
#     For this exercise, pass the contamination parameter from the function argument.
#
#   random_state (int):
#     For reproducibility. Use 42.
#
# Implementation:
#   1. Create: model = IsolationForest(n_estimators=..., contamination=..., random_state=42)
#   2. Fit: model.fit(X_train)
#   3. Return the fitted model
#
# Parameters:
#   X_train: np.ndarray      -- shape (n_samples, n_features), the training feature matrix
#   contamination: float     -- expected anomaly proportion (e.g., 0.05)
#
# Returns:
#   IsolationForest          -- the fitted model
#
# Note on scaling:
#   Isolation Forest is tree-based and does NOT require feature scaling
#   (unlike distance-based methods). However, we scale anyway in the
#   scaffolded code because it makes the feature matrix easier to interpret
#   and is good practice for pipeline consistency.

def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float,
) -> IsolationForest:
    raise NotImplementedError(
        "TODO(human): Create an IsolationForest with n_estimators=100, "
        "the given contamination, and random_state=42. Fit it on X_train "
        "and return the fitted model."
    )


# ---------------------------------------------------------------------------
# TODO(human): Implement detect_anomalies
# ---------------------------------------------------------------------------
#
# Use the fitted Isolation Forest model to predict anomalies on new data.
# Return both boolean predictions and raw anomaly scores.
#
# scikit-learn's IsolationForest output conventions:
#
#   model.predict(X):
#     Returns an array of +1 (inlier/normal) and -1 (outlier/anomaly).
#     This is the OPPOSITE of most anomaly detection libraries (including PyOD),
#     where 1 = anomaly. You need to convert: anomaly = (predictions == -1).
#
#   model.decision_function(X):
#     Returns raw anomaly scores. In sklearn's convention:
#     - NEGATIVE scores = more anomalous (further from normal)
#     - POSITIVE scores = more normal
#     This is also OPPOSITE to PyOD (where higher = more anomalous).
#     For consistency with the evaluation framework in Exercise 4,
#     negate the scores: anomaly_scores = -model.decision_function(X)
#     so that higher scores = more anomalous (matching PyOD convention).
#
# Implementation:
#   1. raw_labels = model.predict(X)               # array of +1/-1
#   2. is_anomaly = (raw_labels == -1)              # boolean array
#   3. raw_scores = model.decision_function(X)      # negative = anomalous
#   4. anomaly_scores = -raw_scores                 # flip: positive = anomalous
#   5. Return (is_anomaly, anomaly_scores)
#
# Parameters:
#   model: IsolationForest   -- a fitted IsolationForest model
#   X: np.ndarray            -- shape (n_samples, n_features), data to predict on
#
# Returns:
#   tuple[np.ndarray, np.ndarray]:
#     - is_anomaly: bool array, shape (n_samples,), True = anomaly
#     - anomaly_scores: float array, shape (n_samples,), higher = more anomalous
#
# Why we negate the scores:
#   PyOD (Exercise 3) uses the convention "higher score = more anomalous."
#   scikit-learn uses the opposite. By negating sklearn's scores here,
#   the evaluation framework in Exercise 4 can treat all detectors uniformly.

def detect_anomalies(
    model: IsolationForest,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError(
        "TODO(human): Use model.predict() and model.decision_function() "
        "to get binary predictions and anomaly scores. Convert sklearn's "
        "-1/+1 to boolean, negate scores so higher = more anomalous."
    )


# ---------------------------------------------------------------------------
# Scaffolded: Feature preparation, per-service training, results collection
# ---------------------------------------------------------------------------


def prepare_features(
    df: pd.DataFrame,
    service: str,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Extract and scale feature matrix for a single service.

    Returns:
        X_scaled: StandardScaler-transformed feature matrix
        X_raw: original (unscaled) feature matrix (for reference)
        y_true: ground-truth anomaly labels
    """
    svc_df = df[df["service"] == service].reset_index(drop=True)
    X_raw = svc_df[FEATURE_COLS].values
    y_true = svc_df["is_anomaly"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, X_raw, y_true


def evaluate_detector(
    y_true: pd.Series,
    y_pred: np.ndarray,
    name: str,
) -> dict:
    """Compute detection statistics."""
    y_true_bool = y_true.values.astype(bool)
    y_pred_bool = y_pred.astype(bool)

    tp = int(np.sum(y_pred_bool & y_true_bool))
    fp = int(np.sum(y_pred_bool & ~y_true_bool))
    fn = int(np.sum(~y_pred_bool & y_true_bool))
    tn = int(np.sum(~y_pred_bool & ~y_true_bool))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "detector": name,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main() -> None:
    print("=" * 60)
    print("Exercise 2: Isolation Forest on Multivariate Metrics")
    print("=" * 60)

    # Load data
    if not DATA_FILE.exists():
        print(f"\nERROR: {DATA_FILE} not found.")
        print("Run src/00_generate_metrics.py first to generate the dataset.")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"\nLoaded {len(df):,} data points from {DATA_FILE.name}")
    print(f"Features: {FEATURE_COLS}")
    print(f"Contamination: {CONTAMINATION}")

    # Train and evaluate per service
    print(f"\n{'Service':<25s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    print("-" * 65)

    all_results: list[dict] = []
    all_scores: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for service in SERVICES:
        X_scaled, X_raw, y_true = prepare_features(df, service)

        # Train
        model = train_isolation_forest(X_scaled, contamination=CONTAMINATION)

        # Detect
        is_anomaly, scores = detect_anomalies(model, X_scaled)

        # Evaluate
        result = evaluate_detector(y_true, is_anomaly, service)
        all_results.append(result)
        all_scores.append(scores)
        all_preds.append(is_anomaly)
        all_labels.append(y_true.values.astype(bool))

        print(
            f"{service:<25s} {result['TP']:4d} {result['FP']:4d} {result['FN']:4d} "
            f"{result['precision']:6.3f} {result['recall']:6.3f} {result['f1']:6.3f}"
        )

    # Aggregate
    total_tp = sum(r["TP"] for r in all_results)
    total_fp = sum(r["FP"] for r in all_results)
    total_fn = sum(r["FN"] for r in all_results)
    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    agg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    agg_f1 = (2 * agg_precision * agg_recall / (agg_precision + agg_recall)
              if (agg_precision + agg_recall) > 0 else 0.0)

    print("-" * 65)
    print(f"{'TOTAL':<25s} {total_tp:4d} {total_fp:4d} {total_fn:4d} "
          f"{agg_precision:6.3f} {agg_recall:6.3f} {agg_f1:6.3f}")

    # Save model info
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nNote: For production use, you would serialize the model with joblib.")
    print(f"Models directory: {MODELS_DIR}")

    print("\nComparison with Exercise 1:")
    print("  Statistical baselines only looked at latency_ms (1 metric).")
    print("  Isolation Forest uses all 4 metrics simultaneously.")
    print("  Correlation anomalies (high latency + low CPU) should be better detected.")


if __name__ == "__main__":
    main()
