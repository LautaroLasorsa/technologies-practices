"""Generate synthetic microservice metrics with injected anomalies.

This script creates realistic time-series metrics for 5 microservices,
then injects three types of anomalies: spikes, drifts, and correlation
breaks. The output is saved to data/metrics.csv with ground-truth labels.

This file is FULLY SCAFFOLDED -- no TODO(human) blocks. Run it directly
to generate the dataset that all subsequent exercises use.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
N_POINTS = 1000  # Time points per service

SERVICES = [
    "api-gateway",
    "user-service",
    "order-service",
    "payment-service",
    "inventory-service",
]

# Normal metric distributions per service (mean, std)
# Each service has a different baseline to simulate realistic heterogeneity.
METRIC_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "api-gateway": {
        "latency_ms": (45.0, 8.0),
        "error_rate": (0.005, 0.002),
        "cpu_percent": (35.0, 5.0),
        "memory_percent": (50.0, 4.0),
    },
    "user-service": {
        "latency_ms": (30.0, 5.0),
        "error_rate": (0.003, 0.001),
        "cpu_percent": (25.0, 4.0),
        "memory_percent": (40.0, 3.0),
    },
    "order-service": {
        "latency_ms": (80.0, 15.0),
        "error_rate": (0.008, 0.003),
        "cpu_percent": (55.0, 8.0),
        "memory_percent": (65.0, 5.0),
    },
    "payment-service": {
        "latency_ms": (120.0, 20.0),
        "error_rate": (0.010, 0.004),
        "cpu_percent": (45.0, 7.0),
        "memory_percent": (55.0, 6.0),
    },
    "inventory-service": {
        "latency_ms": (25.0, 4.0),
        "error_rate": (0.002, 0.001),
        "cpu_percent": (20.0, 3.0),
        "memory_percent": (35.0, 3.0),
    },
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "metrics.csv"


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_normal_metrics(
    rng: np.random.Generator,
    service: str,
    n_points: int,
) -> pd.DataFrame:
    """Generate normal (non-anomalous) metrics for a single service.

    Adds slight autocorrelation to simulate realistic time series:
    each point is 70% of the random draw + 30% of the previous point.
    """
    profile = METRIC_PROFILES[service]
    data: dict[str, np.ndarray] = {}

    for metric, (mean, std) in profile.items():
        raw = rng.normal(mean, std, size=n_points)
        # Add autocorrelation (smoothing)
        smoothed = np.empty(n_points)
        smoothed[0] = raw[0]
        for i in range(1, n_points):
            smoothed[i] = 0.7 * raw[i] + 0.3 * smoothed[i - 1]
        # Clip to realistic ranges
        if metric == "error_rate":
            smoothed = np.clip(smoothed, 0.0, 1.0)
        elif metric.endswith("_percent"):
            smoothed = np.clip(smoothed, 0.0, 100.0)
        else:
            smoothed = np.clip(smoothed, 0.0, None)
        data[metric] = smoothed

    start_time = pd.Timestamp("2025-01-01")
    timestamps = pd.date_range(start=start_time, periods=n_points, freq="min")

    df = pd.DataFrame(data)
    df.insert(0, "timestamp", timestamps)
    df.insert(1, "service", service)
    df["is_anomaly"] = False

    return df


def inject_spike_anomalies(
    rng: np.random.Generator,
    df: pd.DataFrame,
    n_spikes: int = 5,
) -> pd.DataFrame:
    """Inject spike anomalies: sudden 5-10x increase for 1-3 points.

    Spikes simulate sudden failures like a database timeout causing a
    latency burst, or a deployment bug causing an error rate jump.
    """
    df = df.copy()
    n_rows = len(df)

    for _ in range(n_spikes):
        # Pick a random point and a random metric
        idx = rng.integers(50, n_rows - 10)
        duration = rng.integers(1, 4)  # 1-3 consecutive points
        metric = rng.choice(["latency_ms", "error_rate", "cpu_percent"])
        multiplier = rng.uniform(5.0, 10.0)

        for offset in range(duration):
            pos = idx + offset
            if pos < n_rows:
                original = df.at[pos, metric]
                df.at[pos, metric] = original * multiplier
                df.at[pos, "is_anomaly"] = True

    return df


def inject_drift_anomalies(
    rng: np.random.Generator,
    df: pd.DataFrame,
    n_drifts: int = 3,
) -> pd.DataFrame:
    """Inject drift anomalies: gradual increase over 10-20 points.

    Drifts simulate issues like memory leaks, connection pool exhaustion,
    or gradual degradation under increasing load.
    """
    df = df.copy()
    n_rows = len(df)

    for _ in range(n_drifts):
        idx = rng.integers(100, n_rows - 30)
        duration = rng.integers(10, 21)  # 10-20 points
        metric = rng.choice(["latency_ms", "memory_percent", "cpu_percent"])
        # Gradual increase from 1x to 3-5x
        peak_multiplier = rng.uniform(3.0, 5.0)

        for offset in range(duration):
            pos = idx + offset
            if pos < n_rows:
                progress = offset / duration
                multiplier = 1.0 + (peak_multiplier - 1.0) * progress
                original = df.at[pos, metric]
                df.at[pos, metric] = original * multiplier
                df.at[pos, "is_anomaly"] = True

    return df


def inject_correlation_anomalies(
    rng: np.random.Generator,
    df: pd.DataFrame,
    n_breaks: int = 3,
) -> pd.DataFrame:
    """Inject correlation anomalies: break the natural metric relationships.

    Normally, high CPU correlates with high latency and high memory usage.
    A correlation anomaly is when latency spikes but CPU drops -- indicating
    something unusual like a network issue rather than a compute bottleneck.
    These are hard to catch with univariate detectors.
    """
    df = df.copy()
    n_rows = len(df)

    for _ in range(n_breaks):
        idx = rng.integers(50, n_rows - 10)
        duration = rng.integers(3, 8)  # 3-7 points

        for offset in range(duration):
            pos = idx + offset
            if pos < n_rows:
                # High latency + LOW cpu (opposite of normal correlation)
                latency_mean = METRIC_PROFILES[df.at[pos, "service"]]["latency_ms"][0]
                cpu_mean = METRIC_PROFILES[df.at[pos, "service"]]["cpu_percent"][0]
                df.at[pos, "latency_ms"] = latency_mean * rng.uniform(3.0, 5.0)
                df.at[pos, "cpu_percent"] = cpu_mean * rng.uniform(0.1, 0.3)
                df.at[pos, "is_anomaly"] = True

    return df


def generate_all_metrics() -> pd.DataFrame:
    """Generate complete metrics dataset for all services with anomalies."""
    rng = np.random.default_rng(SEED)
    all_dfs: list[pd.DataFrame] = []

    for service in SERVICES:
        print(f"  Generating metrics for {service}...")
        df = generate_normal_metrics(rng, service, N_POINTS)
        df = inject_spike_anomalies(rng, df)
        df = inject_drift_anomalies(rng, df)
        df = inject_correlation_anomalies(rng, df)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics of the generated dataset."""
    total = len(df)
    n_anomalies = df["is_anomaly"].sum()
    anomaly_rate = n_anomalies / total * 100

    print(f"\n{'=' * 60}")
    print("Dataset Summary")
    print(f"{'=' * 60}")
    print(f"  Total data points:   {total:,}")
    print(f"  Services:            {df['service'].nunique()}")
    print(f"  Points per service:  {total // df['service'].nunique():,}")
    print(f"  Anomalies:           {n_anomalies:,} ({anomaly_rate:.1f}%)")
    print(f"  Normal:              {total - n_anomalies:,} ({100 - anomaly_rate:.1f}%)")

    print(f"\n  Per-service anomaly counts:")
    for service in SERVICES:
        svc_df = df[df["service"] == service]
        svc_anomalies = svc_df["is_anomaly"].sum()
        print(f"    {service:25s}  {svc_anomalies:4d} anomalies / {len(svc_df)} points")

    print(f"\n  Metric ranges (all services):")
    for col in ["latency_ms", "error_rate", "cpu_percent", "memory_percent"]:
        print(f"    {col:20s}  min={df[col].min():.4f}  mean={df[col].mean():.4f}  max={df[col].max():.4f}")


def main() -> None:
    print("=" * 60)
    print("Generate Synthetic Microservice Metrics")
    print("=" * 60)

    # Generate
    print("\nGenerating metrics for 5 microservices...")
    df = generate_all_metrics()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")

    # Summary
    print_summary(df)

    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
