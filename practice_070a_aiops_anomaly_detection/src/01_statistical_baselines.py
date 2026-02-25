"""Exercise 1: Statistical Baseline Detectors.

This exercise implements two foundational anomaly detection methods that
underpin most production alerting systems:

1. Rolling Z-score: measures how many standard deviations each point is
   from the local rolling mean. Simple, fast, threshold-based.
2. Moving average band: computes a rolling mean +/- k * rolling_std
   corridor. Points outside the band are flagged as anomalies.

Both operate on univariate time series (one metric at a time, per service).
This is how tools like Prometheus alerting rules, Datadog monitors, and
AWS CloudWatch anomaly detection work at their core.

After completing this exercise you should understand:
- How rolling statistics adapt to non-stationary time series
- The tradeoff between window size (sensitivity vs. stability)
- Why univariate methods miss multi-metric anomalies (motivation for Ex. 2)
"""

from pathlib import Path

import pandas as pd

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

# Detector parameters -- experiment with these after implementing
ZSCORE_WINDOW = 50
ZSCORE_THRESHOLD = 3.0
MA_WINDOW = 50
MA_NUM_STD = 2.5


# ---------------------------------------------------------------------------
# TODO(human): Implement detect_zscore
# ---------------------------------------------------------------------------
#
# Compute a rolling Z-score for a pandas Series and return a boolean Series
# indicating which points are anomalous (|z-score| > threshold).
#
# How the rolling Z-score works:
#   For each point x_t in the series, compute the mean and standard deviation
#   over the preceding `window` points (a rolling window). Then:
#
#       z_t = (x_t - rolling_mean_t) / rolling_std_t
#
#   If |z_t| > threshold, the point is anomalous.
#
# Implementation hints:
#   - Use pandas rolling: series.rolling(window=window, min_periods=1)
#   - .mean() and .std() give you the rolling statistics
#   - Compute z_scores = (series - rolling_mean) / rolling_std
#   - Handle division by zero: if rolling_std is 0 (constant window),
#     the z-score should be 0 (not anomalous). Use .fillna(0) or replace.
#   - Return (z_scores.abs() > threshold) -- a boolean Series
#
# Parameters:
#   series: pd.Series       -- the metric values (e.g., latency_ms for one service)
#   window: int             -- rolling window size (number of past points)
#   threshold: float        -- Z-score threshold; flag points where |z| > threshold
#
# Returns:
#   pd.Series of bool       -- True where the point is detected as anomalous
#
# Why this matters:
#   Z-score is the simplest anomaly detector and the mental model behind most
#   static alerting rules. Its weakness: it assumes roughly Gaussian data within
#   each window, and a single outlier in the window inflates the std, masking
#   subsequent anomalies (the "swamping" effect). You'll see this in results.
#
# Example:
#   >>> series = pd.Series([10, 11, 12, 100, 11, 10])
#   >>> detect_zscore(series, window=3, threshold=2.0)
#   0    False
#   1    False
#   2    False
#   3     True    # 100 is far from the rolling mean of ~11
#   4    False
#   5    False

def detect_zscore(series: pd.Series, window: int, threshold: float) -> pd.Series:
    raise NotImplementedError(
        "TODO(human): Implement rolling Z-score anomaly detection. "
        "Compute rolling mean and std over the window, calculate z-scores, "
        "and return a boolean Series where |z| > threshold."
    )


# ---------------------------------------------------------------------------
# TODO(human): Implement detect_moving_average
# ---------------------------------------------------------------------------
#
# Compute a moving average band and flag points outside the band as anomalies.
#
# How the moving average band works:
#   For each point x_t, compute:
#       upper_band_t = rolling_mean_t + num_std * rolling_std_t
#       lower_band_t = rolling_mean_t - num_std * rolling_std_t
#
#   If x_t > upper_band_t OR x_t < lower_band_t, the point is anomalous.
#
# Implementation hints:
#   - Same rolling computation as Z-score: series.rolling(window=window, min_periods=1)
#   - Compute upper_band = rolling_mean + num_std * rolling_std
#   - Compute lower_band = rolling_mean - num_std * rolling_std
#   - Return (series > upper_band) | (series < lower_band)
#
# Parameters:
#   series: pd.Series       -- the metric values
#   window: int             -- rolling window size
#   num_std: float          -- number of standard deviations for the band width
#
# Returns:
#   pd.Series of bool       -- True where the point is outside the band
#
# Why this matters:
#   Moving average bands are conceptually identical to Bollinger Bands in finance
#   and are the default anomaly detection method in many monitoring tools.
#   The advantage over raw Z-score: the band is visually interpretable (you can
#   plot it), and the num_std parameter maps directly to "how sensitive do we
#   want the alerts to be?" Production teams tune num_std: smaller = more
#   sensitive (more alerts), larger = fewer but higher-confidence alerts.
#
# Difference from Z-score:
#   Mathematically, detect_moving_average(series, window, num_std) produces
#   the same results as detect_zscore(series, window, threshold=num_std).
#   The conceptual difference is in how teams think about and tune them:
#   - Z-score: "flag if more than N standard deviations from mean"
#   - MA band: "flag if outside the normal corridor"
#   Both perspectives are useful. In practice, the MA band is more commonly
#   visualized on dashboards (Grafana, Datadog), while Z-score is used in
#   programmatic alerting rules.
#
# Example:
#   >>> series = pd.Series([10, 11, 9, 10, 50, 10, 11])
#   >>> detect_moving_average(series, window=4, num_std=2.0)
#   0    False
#   1    False
#   2    False
#   3    False
#   4     True    # 50 is far above the upper band
#   5    False
#   6    False

def detect_moving_average(series: pd.Series, window: int, num_std: float) -> pd.Series:
    raise NotImplementedError(
        "TODO(human): Implement moving average band anomaly detection. "
        "Compute rolling mean +/- num_std * rolling_std, and return a boolean "
        "Series where the value is outside the band."
    )


# ---------------------------------------------------------------------------
# Scaffolded: Apply detectors and collect results
# ---------------------------------------------------------------------------


def evaluate_detector(
    y_true: pd.Series,
    y_pred: pd.Series,
    name: str,
) -> dict:
    """Compute basic detection statistics."""
    tp = ((y_pred == True) & (y_true == True)).sum()
    fp = ((y_pred == True) & (y_true == False)).sum()
    fn = ((y_pred == False) & (y_true == True)).sum()
    tn = ((y_pred == False) & (y_true == False)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "detector": name,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def run_statistical_baselines(df: pd.DataFrame) -> list[dict]:
    """Apply Z-score and moving average detectors to each service's latency_ms."""
    all_results: list[dict] = []

    for service in SERVICES:
        svc_df = df[df["service"] == service].copy()
        svc_df = svc_df.reset_index(drop=True)
        y_true = svc_df["is_anomaly"]
        series = svc_df["latency_ms"]

        # Z-score detector
        zscore_preds = detect_zscore(series, window=ZSCORE_WINDOW, threshold=ZSCORE_THRESHOLD)
        result = evaluate_detector(y_true, zscore_preds, f"zscore|{service}")
        all_results.append(result)

        # Moving average detector
        ma_preds = detect_moving_average(series, window=MA_WINDOW, num_std=MA_NUM_STD)
        result = evaluate_detector(y_true, ma_preds, f"ma_band|{service}")
        all_results.append(result)

    return all_results


def main() -> None:
    print("=" * 60)
    print("Exercise 1: Statistical Baseline Detectors")
    print("=" * 60)

    # Load data
    if not DATA_FILE.exists():
        print(f"\nERROR: {DATA_FILE} not found.")
        print("Run src/00_generate_metrics.py first to generate the dataset.")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"\nLoaded {len(df):,} data points from {DATA_FILE.name}")
    print(f"Services: {df['service'].nunique()}, Anomalies: {df['is_anomaly'].sum()}")

    # Run detectors
    print(f"\nRunning detectors (window={ZSCORE_WINDOW}, z_thresh={ZSCORE_THRESHOLD}, ma_std={MA_NUM_STD})...")
    print("-" * 60)

    results = run_statistical_baselines(df)

    # Print results table
    print(f"\n{'Detector':<35s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['detector']:<35s} {r['TP']:4d} {r['FP']:4d} {r['FN']:4d} "
            f"{r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f}"
        )

    # Aggregate across services
    print("\n" + "-" * 60)
    print("Aggregated results (all services):")
    for detector_type in ["zscore", "ma_band"]:
        type_results = [r for r in results if r["detector"].startswith(detector_type)]
        total_tp = sum(r["TP"] for r in type_results)
        total_fp = sum(r["FP"] for r in type_results)
        total_fn = sum(r["FN"] for r in type_results)
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"  {detector_type:<15s}  TP={total_tp:4d}  FP={total_fp:4d}  FN={total_fn:4d}  "
              f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")

    print("\nNote: These baselines only look at latency_ms (univariate).")
    print("Correlation anomalies and multi-metric patterns are likely missed.")
    print("Exercise 2 (Isolation Forest) addresses this with multivariate detection.")


if __name__ == "__main__":
    main()
