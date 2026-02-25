"""Exercise 3: PyOD Model Comparison.

This exercise uses PyOD's unified API to compare four fundamentally different
anomaly detection algorithms on the same microservice metrics data:

  - ECOD: Empirical Cumulative Distribution-based (statistical, parameter-free)
  - COPOD: Copula-Based Outlier Detection (statistical, parameter-free)
  - KNN: K-Nearest Neighbors distance-based (classical ML)
  - AutoEncoder: Neural network reconstruction error (deep learning)

The key insight: PyOD provides a consistent fit/predict/decision_function API
across all 50+ algorithms. Swapping detectors requires changing only the
constructor -- all downstream code stays identical. This is the power of a
well-designed library API.

After completing this exercise you should understand:
- PyOD's unified API pattern (fit, predict, decision_function, decision_scores_)
- How contamination affects the decision threshold
- The different algorithmic philosophies (distribution, distance, reconstruction)
- How to build a detector comparison framework
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "metrics.csv"

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
# TODO(human): Implement build_detector_suite
# ---------------------------------------------------------------------------
#
# Create and return a dictionary mapping detector names to PyOD detector
# instances. Each detector should be configured with the given contamination
# parameter and reasonable defaults for other hyperparameters.
#
# Detectors to include:
#
# 1. ECOD (Empirical Cumulative Distribution-based Outlier Detection):
#      from pyod.models.ecod import ECOD
#      ECOD(contamination=contamination)
#
#    How ECOD works:
#      - For each feature dimension, computes the empirical CDF
#      - Estimates tail probabilities for each data point per dimension
#      - Aggregates tail probabilities across dimensions into a single score
#      - Points in the extreme tails (across multiple dimensions) get high scores
#    Key property: PARAMETER-FREE beyond contamination. No tuning needed.
#    This makes it an excellent baseline detector.
#
# 2. COPOD (Copula-Based Outlier Detection):
#      from pyod.models.copod import COPOD
#      COPOD(contamination=contamination)
#
#    How COPOD works:
#      - Models the multivariate distribution using empirical copulas
#      - Copulas capture dependencies between features (unlike ECOD which
#        treats features independently)
#      - Estimates tail probabilities considering cross-feature relationships
#    Key property: Also PARAMETER-FREE. Slightly more sophisticated than ECOD
#    because it considers feature correlations.
#
# 3. KNN (K-Nearest Neighbors Outlier Detection):
#      from pyod.models.knn import KNN
#      KNN(n_neighbors=10, contamination=contamination)
#
#    How KNN works:
#      - For each point, computes distance to its k-th nearest neighbor
#      - Points that are far from their neighbors are anomalies
#      - Several methods: "largest" (k-th neighbor distance), "mean" (avg of
#        k neighbors), "median" (median of k neighbors)
#    Key parameter: n_neighbors (k). Larger k = smoother decisions, smaller k =
#    more sensitive to local density variations. 5-20 is typical.
#    Warning: O(n^2) distance computation -- slow on very large datasets.
#
# 4. AutoEncoder:
#      from pyod.models.auto_encoder import AutoEncoder
#      AutoEncoder(
#          hidden_neurons=[16, 8, 8, 16],
#          epochs=30,
#          batch_size=32,
#          contamination=contamination,
#          preprocessing=True,
#      )
#
#    How AutoEncoder works:
#      - Neural network: encoder compresses input (4 features) through layers
#        [16, 8] down to 8 neurons, then decoder expands back [8, 16]
#      - Trained to minimize reconstruction error on the training data
#      - Anomalies produce high reconstruction error because the network
#        learned to reconstruct "normal" patterns
#      - hidden_neurons defines the architecture: [16, 8, 8, 16] means
#        input -> 16 -> 8 (bottleneck) -> 8 -> 16 -> output
#      - preprocessing=True applies StandardScaler internally
#    Note: AutoEncoder may print training progress -- this is normal.
#    Suppress with verbose=0 if available, or just let it print.
#
# Parameters:
#   contamination: float   -- expected anomaly proportion (e.g., 0.05)
#
# Returns:
#   dict[str, object]      -- mapping detector name (str) to PyOD detector instance
#
# Example return:
#   {"ECOD": ECOD(contamination=0.05), "COPOD": COPOD(...), "KNN": KNN(...), ...}
#
# Important: All PyOD detectors share the same API:
#   detector.fit(X)                    -- train on data
#   detector.decision_function(X)      -- anomaly scores (higher = more anomalous)
#   detector.predict(X)                -- binary labels (0=normal, 1=outlier)
#   detector.decision_scores_          -- scores for training data (after fit)
#   detector.labels_                   -- labels for training data (after fit)
#   detector.threshold_                -- decision threshold (after fit)
#
# Note the OPPOSITE convention from sklearn's IsolationForest:
#   - PyOD: predict() returns 1 for outliers, 0 for inliers
#   - sklearn: predict() returns -1 for outliers, +1 for inliers
#   - PyOD: higher decision_function score = more anomalous
#   - sklearn: lower decision_function score = more anomalous

def build_detector_suite(contamination: float) -> dict[str, object]:
    raise NotImplementedError(
        "TODO(human): Import and instantiate ECOD, COPOD, KNN, and AutoEncoder "
        "from pyod.models. Return a dict mapping names to detector instances. "
        "See the comments above for exact import paths and parameters."
    )


# ---------------------------------------------------------------------------
# TODO(human): Implement fit_and_score
# ---------------------------------------------------------------------------
#
# Fit a PyOD detector on training data, then generate anomaly scores and
# binary predictions on test data. Return (scores, predictions).
#
# PyOD's unified API makes this straightforward:
#
#   1. detector.fit(X_train)
#      - Trains the model on the training data
#      - After fit(), detector.decision_scores_ contains scores for X_train
#      - After fit(), detector.threshold_ is the decision boundary
#
#   2. scores = detector.decision_function(X_test)
#      - Returns anomaly scores for each row in X_test
#      - Higher score = more anomalous (PyOD convention)
#      - These are continuous values -- useful for ROC/PR curves
#
#   3. predictions = detector.predict(X_test)
#      - Returns binary labels: 0 = normal, 1 = outlier
#      - Uses detector.threshold_ as the cutoff
#      - Equivalent to: (scores > detector.threshold_).astype(int)
#
# Parameters:
#   detector: object         -- a PyOD detector instance (already constructed, not yet fit)
#   X_train: np.ndarray      -- training features, shape (n_train, n_features)
#   X_test: np.ndarray       -- test features, shape (n_test, n_features)
#
# Returns:
#   tuple[np.ndarray, np.ndarray]:
#     - scores: float array, shape (n_test,), anomaly scores (higher = more anomalous)
#     - predictions: int array, shape (n_test,), binary labels (0=normal, 1=outlier)
#
# Error handling:
#   Some detectors (especially AutoEncoder) may fail on very small datasets
#   or produce warnings. Wrap the fit/predict in a try/except and return
#   arrays of zeros if the detector fails, so the comparison framework
#   continues running even if one detector has issues.
#
# Example:
#   >>> from pyod.models.ecod import ECOD
#   >>> detector = ECOD(contamination=0.05)
#   >>> scores, preds = fit_and_score(detector, X_train, X_test)
#   >>> scores.shape  # (n_test,)
#   >>> preds.shape   # (n_test,), values are 0 or 1

def fit_and_score(
    detector: object,
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError(
        "TODO(human): Fit the detector on X_train, then use decision_function() "
        "and predict() on X_test to get scores and binary predictions."
    )


# ---------------------------------------------------------------------------
# Scaffolded: Comparison framework
# ---------------------------------------------------------------------------


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute precision, recall, F1 from binary predictions."""
    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_pred.astype(bool)

    tp = int(np.sum(y_pred_bool & y_true_bool))
    fp = int(np.sum(y_pred_bool & ~y_true_bool))
    fn = int(np.sum(~y_pred_bool & y_true_bool))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main() -> None:
    print("=" * 60)
    print("Exercise 3: PyOD Model Comparison")
    print("=" * 60)

    # Load data
    if not DATA_FILE.exists():
        print(f"\nERROR: {DATA_FILE} not found.")
        print("Run src/00_generate_metrics.py first to generate the dataset.")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"\nLoaded {len(df):,} data points")
    print(f"Features: {FEATURE_COLS}")
    print(f"Contamination: {CONTAMINATION}")

    # Build detector suite
    detectors = build_detector_suite(CONTAMINATION)
    detector_names = list(detectors.keys())
    print(f"Detectors: {detector_names}")

    # Results collection: {detector_name: {service: {metrics}}}
    all_results: dict[str, list[dict]] = {name: [] for name in detector_names}

    # Per-service comparison
    for service in SERVICES:
        print(f"\n--- {service} ---")

        # Prepare features
        svc_df = df[df["service"] == service].reset_index(drop=True)
        X = svc_df[FEATURE_COLS].values
        y_true = svc_df["is_anomaly"].values.astype(bool)

        # Scale features (important for KNN and AutoEncoder)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Use the entire dataset for both train and test (unsupervised setting).
        # In production, you would train on a known-clean period and test on new data.
        X_train = X_scaled
        X_test = X_scaled

        # Run each detector
        for det_name, detector_template in detectors.items():
            # Create a fresh detector for each service (avoid reusing fitted state)
            import copy
            detector = copy.deepcopy(detector_template)

            scores, preds = fit_and_score(detector, X_train, X_test)
            metrics = evaluate_predictions(y_true, preds)
            metrics["detector"] = det_name
            metrics["service"] = service
            all_results[det_name].append(metrics)

            print(f"  {det_name:<15s}  P={metrics['precision']:.3f}  "
                  f"R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")

    # Aggregate per detector
    print(f"\n{'=' * 60}")
    print("Aggregated Results (all services)")
    print(f"{'=' * 60}")
    print(f"{'Detector':<15s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    print("-" * 55)

    for det_name in detector_names:
        results_list = all_results[det_name]
        total_tp = sum(r["TP"] for r in results_list)
        total_fp = sum(r["FP"] for r in results_list)
        total_fn = sum(r["FN"] for r in results_list)
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"{det_name:<15s} {total_tp:4d} {total_fp:4d} {total_fn:4d} "
              f"{precision:6.3f} {recall:6.3f} {f1:6.3f}")

    print("\nObservations to look for:")
    print("  - ECOD and COPOD are fast and parameter-free -- good baselines")
    print("  - KNN adapts to local density -- may perform well on clustered data")
    print("  - AutoEncoder captures nonlinear patterns but needs more data/epochs")
    print("  - No single detector dominates -- ensemble approaches combine strengths")
    print("\nNext: Exercise 4 compares all detectors with ROC/PR curves")


if __name__ == "__main__":
    main()
