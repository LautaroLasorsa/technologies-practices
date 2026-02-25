#!/usr/bin/env python3
"""Exercise 8-9: Frequency-based anomaly detection on log clusters.

Pipeline:
  1. Load parsed logs with timestamps and cluster assignments
  2. Build frequency vectors: count cluster occurrences per time window
  3. Detect anomalous windows with IsolationForest
  4. Compare detected anomalies with ground truth (injected anomaly burst)
  5. Visualize with a frequency heatmap and highlighted anomalous windows
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
PLOTS_DIR = Path(__file__).parent.parent / "plots"


# ---------------------------------------------------------------------------
# TODO(human): Exercise 8 -- Build frequency vectors per time window
# ---------------------------------------------------------------------------
def build_frequency_vectors(
    parsed_df: pd.DataFrame,
    time_window: str = "5min",
) -> pd.DataFrame:
    """Group log lines into time windows and count occurrences of each cluster.

    TODO(human): Convert the stream of parsed log events into a time series
    of cluster-frequency vectors, one per time window.

    Steps:
      1. Ensure the "timestamp" column is parsed as datetime:
         parsed_df["timestamp"] = pd.to_datetime(parsed_df["timestamp"])
      2. Set a time-window column by flooring timestamps to the window boundary:
         parsed_df["window"] = parsed_df["timestamp"].dt.floor(time_window)
         For time_window="5min", this groups logs into 5-minute buckets:
         08:00-08:05, 08:05-08:10, etc.
      3. Create a pivot table counting occurrences:
         freq_df = parsed_df.pivot_table(
             index="window",
             columns="cluster_id",
             aggfunc="size",
             fill_value=0,
         )
         This produces a DataFrame where:
           - rows = time windows (datetime index)
           - columns = cluster IDs (int column names)
           - values = count of log lines in that cluster during that window
      4. Optionally normalize each row by the total logs in that window:
         freq_norm = freq_df.div(freq_df.sum(axis=1), axis=0)
         Normalization is important because a window with 2x logs isn't necessarily
         anomalous -- what matters is the *proportion* of clusters. A window where
         70% of logs are ERROR clusters is anomalous even if total volume is normal.
      5. Print shape: f"  Frequency matrix: {freq_df.shape[0]} windows x {freq_df.shape[1]} clusters"
      6. Return the NORMALIZED frequency DataFrame (proportions summing to 1 per row).

    Design consideration: The choice of time_window affects sensitivity.
      - "5min": Detects short bursts (good for the injected 5-min anomaly).
      - "15min": Smooths over short spikes, catches sustained shifts.
      - "1min": Very granular, but may have many empty/sparse windows.
    Start with "5min" to match the injected anomaly duration.

    Args:
        parsed_df: DataFrame with columns [timestamp, cluster_id, ...].
        time_window: Pandas offset alias for time bucketing (e.g., "5min", "15min").

    Returns:
        pd.DataFrame with datetime index (windows) and cluster ID columns,
        values = normalized frequency (proportion of logs in each cluster per window).
    """
    raise NotImplementedError("TODO(human): Build frequency vectors")


# ---------------------------------------------------------------------------
# TODO(human): Exercise 9 -- Detect anomalous windows with IsolationForest
# ---------------------------------------------------------------------------
def detect_anomalous_windows(
    frequency_df: pd.DataFrame,
    contamination: float = 0.05,
) -> np.ndarray:
    """Use IsolationForest to detect anomalous time windows based on cluster frequencies.

    TODO(human): Apply IsolationForest to the frequency vectors to find
    time windows with unusual cluster distributions.

    Steps:
      1. Import IsolationForest from sklearn.ensemble.
      2. Create and fit the model:
         model = IsolationForest(
             contamination=contamination,
             random_state=42,
             n_estimators=200,
         )
         model.fit(frequency_df.values)

         How IsolationForest works:
         It builds an ensemble of random trees that recursively partition the
         feature space. Anomalous points (unusual frequency distributions) are
         isolated in fewer splits because they lie in sparse regions of the
         feature space. The model scores each point by average path length:
         shorter path = more anomalous.

      3. Predict anomalies:
         predictions = model.predict(frequency_df.values)
         Returns: +1 for normal windows, -1 for anomalous windows.
      4. Convert to boolean: anomalies = (predictions == -1)
      5. Print summary:
         n_anomalous = anomalies.sum()
         f"  Detected {n_anomalous} anomalous windows out of {len(anomalies)} total"
         f"  Anomalous windows: {frequency_df.index[anomalies].tolist()}"
      6. Return the boolean array.

    The contamination parameter sets the expected fraction of anomalous windows.
    0.05 = expect ~5% of windows to be anomalous. If you know the approximate
    anomaly rate (e.g., a 5-min burst in 2 hours = ~4%), set contamination
    accordingly. Setting it too high flags normal windows; too low misses real
    anomalies.

    What makes a window anomalous? Not just high volume -- the cluster *proportions*
    must be unusual. For example:
      - Normal: 60% INFO-auth, 30% INFO-order, 10% WARN-gateway
      - Anomalous: 40% ERROR-payment-critical (clusters that are normally rare)

    Args:
        frequency_df: DataFrame from build_frequency_vectors (windows x clusters).
        contamination: Expected proportion of anomalous windows (0.0 to 0.5).

    Returns:
        np.ndarray of booleans, shape (n_windows,). True = anomalous.
    """
    raise NotImplementedError("TODO(human): Detect anomalous windows")


# ---------------------------------------------------------------------------
# Scaffolded: visualization and evaluation
# ---------------------------------------------------------------------------
def plot_frequency_heatmap(
    frequency_df: pd.DataFrame,
    anomalies: np.ndarray,
    save_path: Path,
) -> None:
    """Plot a heatmap of cluster frequencies with anomalous windows highlighted."""
    fig, (ax_heat, ax_bar) = plt.subplots(
        2, 1, figsize=(16, 10), height_ratios=[4, 1],
        gridspec_kw={"hspace": 0.05},
    )

    # Heatmap of cluster frequencies
    window_labels = [t.strftime("%H:%M") for t in frequency_df.index]
    im = ax_heat.imshow(
        frequency_df.values.T,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    ax_heat.set_ylabel("Cluster ID")
    ax_heat.set_title("Log Cluster Frequency Over Time (normalized proportions)")

    # Only show every Nth x-tick to avoid crowding
    n_ticks = min(20, len(window_labels))
    tick_step = max(1, len(window_labels) // n_ticks)
    ax_heat.set_xticks(range(0, len(window_labels), tick_step))
    ax_heat.set_xticklabels(
        [window_labels[i] for i in range(0, len(window_labels), tick_step)],
        rotation=45, ha="right", fontsize=8,
    )

    # Y-axis: cluster IDs
    cluster_ids = frequency_df.columns.tolist()
    if len(cluster_ids) <= 30:
        ax_heat.set_yticks(range(len(cluster_ids)))
        ax_heat.set_yticklabels(cluster_ids, fontsize=7)
    else:
        tick_step_y = max(1, len(cluster_ids) // 20)
        ax_heat.set_yticks(range(0, len(cluster_ids), tick_step_y))
        ax_heat.set_yticklabels(
            [cluster_ids[i] for i in range(0, len(cluster_ids), tick_step_y)],
            fontsize=7,
        )

    fig.colorbar(im, ax=ax_heat, label="Proportion", shrink=0.8)

    # Anomaly bar: red for anomalous windows, green for normal
    colors = ["#d32f2f" if a else "#4caf50" for a in anomalies]
    ax_bar.bar(range(len(anomalies)), [1] * len(anomalies), color=colors, width=1.0)
    ax_bar.set_xlim(-0.5, len(anomalies) - 0.5)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Time Window")
    ax_bar.set_ylabel("Anomaly")

    # Match x-ticks
    ax_bar.set_xticks(range(0, len(window_labels), tick_step))
    ax_bar.set_xticklabels(
        [window_labels[i] for i in range(0, len(window_labels), tick_step)],
        rotation=45, ha="right", fontsize=8,
    )

    # Legend for anomaly bar
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4caf50", label="Normal"),
        Patch(facecolor="#d32f2f", label="Anomalous"),
    ]
    ax_bar.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_against_ground_truth(
    frequency_df: pd.DataFrame,
    anomalies: np.ndarray,
    metadata_path: Path,
) -> None:
    """Compare detected anomalies with ground truth from the metadata CSV."""
    if not metadata_path.exists():
        print("  Ground truth metadata not found, skipping evaluation.")
        return

    metadata = pd.read_csv(metadata_path)
    metadata["timestamp"] = pd.to_datetime(metadata["timestamp"])

    # Find windows that contain anomaly burst lines
    anomaly_lines = metadata[metadata["is_anomaly"] == True]  # noqa: E712
    if anomaly_lines.empty:
        print("  No anomaly lines in ground truth.")
        return

    anomaly_start = anomaly_lines["timestamp"].min()
    anomaly_end = anomaly_lines["timestamp"].max()
    print(f"\n  Ground truth anomaly burst: {anomaly_start} to {anomaly_end}")
    print(f"  Anomaly lines: {len(anomaly_lines)}")

    # Which windows overlap with the anomaly burst?
    gt_anomalous_windows = set()
    for window_start in frequency_df.index:
        # Each window covers [window_start, window_start + 5min)
        window_end = window_start + pd.Timedelta("5min")
        if window_start <= anomaly_end and window_end >= anomaly_start:
            gt_anomalous_windows.add(window_start)

    # Compare
    detected_windows = set(frequency_df.index[anomalies])

    true_positives = detected_windows & gt_anomalous_windows
    false_positives = detected_windows - gt_anomalous_windows
    false_negatives = gt_anomalous_windows - detected_windows

    print(f"\n  Ground truth anomalous windows: {len(gt_anomalous_windows)}")
    print(f"  Detected anomalous windows:     {len(detected_windows)}")
    print(f"  True positives:                  {len(true_positives)}")
    print(f"  False positives:                 {len(false_positives)}")
    print(f"  False negatives:                 {len(false_negatives)}")

    if gt_anomalous_windows:
        recall = len(true_positives) / len(gt_anomalous_windows)
        print(f"  Recall:                          {recall:.2%}")
    if detected_windows:
        precision = len(true_positives) / len(detected_windows)
        print(f"  Precision:                       {precision:.2%}")

    if true_positives:
        print(f"\n  Successfully detected windows:")
        for w in sorted(true_positives):
            print(f"    {w}")

    if false_negatives:
        print(f"\n  Missed anomalous windows:")
        for w in sorted(false_negatives):
            print(f"    {w}")


def main() -> None:
    """Run the anomaly detection pipeline."""
    parsed_path = DATA_DIR / "parsed_logs.csv"
    metadata_path = DATA_DIR / "logs_metadata.csv"

    if not parsed_path.exists():
        print("ERROR: data/parsed_logs.csv not found. Run 01_drain_parsing.py first.")
        return

    # Load parsed logs and join with metadata for timestamps
    print("Loading parsed logs and metadata...")
    parsed_df = pd.read_csv(parsed_path)
    metadata = pd.read_csv(metadata_path)

    # Merge timestamp from metadata into parsed logs (by line index)
    parsed_df["timestamp"] = metadata["timestamp"]
    parsed_df["timestamp"] = pd.to_datetime(parsed_df["timestamp"])
    print(f"  Loaded {len(parsed_df)} parsed log lines")

    time_range = parsed_df["timestamp"].max() - parsed_df["timestamp"].min()
    print(f"  Time range: {parsed_df['timestamp'].min()} to {parsed_df['timestamp'].max()}")
    print(f"  Duration: {time_range}")

    # Build frequency vectors
    print(f"\nBuilding cluster frequency vectors (5-min windows)...")
    frequency_df = build_frequency_vectors(parsed_df, time_window="5min")
    print(f"  Result: {frequency_df.shape[0]} windows x {frequency_df.shape[1]} clusters")

    # Detect anomalies
    print(f"\nRunning IsolationForest anomaly detection...")
    anomalies = detect_anomalous_windows(frequency_df, contamination=0.05)

    # Evaluate against ground truth
    print(f"\n{'='*80}")
    print("EVALUATION AGAINST GROUND TRUTH")
    print(f"{'='*80}")
    evaluate_against_ground_truth(frequency_df, anomalies, metadata_path)

    # Visualize
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    heatmap_path = PLOTS_DIR / "anomaly_heatmap.png"
    print(f"\nGenerating frequency heatmap with anomaly highlights...")
    plot_frequency_heatmap(frequency_df, anomalies, heatmap_path)
    print(f"  Saved heatmap to {heatmap_path}")

    # Save anomaly results
    anomaly_results_path = DATA_DIR / "anomaly_results.csv"
    results = pd.DataFrame({
        "window": frequency_df.index,
        "is_anomalous": anomalies,
        "total_logs": frequency_df.sum(axis=1).values if frequency_df.values.max() <= 1.0
        else frequency_df.sum(axis=1).values,
    })
    results.to_csv(anomaly_results_path, index=False)
    print(f"  Saved anomaly results to {anomaly_results_path}")

    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"  Frequency matrix: data/anomaly_results.csv")
    print(f"  Heatmap: plots/anomaly_heatmap.png")
    print(f"  Check if the detected anomalous windows align with the injected burst (~09:12-09:18).")


if __name__ == "__main__":
    main()
