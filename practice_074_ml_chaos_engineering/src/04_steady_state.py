"""Steady state detection and chaos impact verification.

Defines what "normal" looks like using statistical baselines, then checks
whether simulated chaos injections violate steady state. This is the
foundational capability that makes chaos experiments meaningful: without
a steady-state definition, you cannot tell if an experiment succeeded.

Outputs:
- data/steady_state_bounds.json -- per-service metric bounds
- data/chaos_impact_report.csv -- which services violated steady state
- plots/steady_state_check.png -- timeline showing violations
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
PLOTS_DIR = Path(__file__).parent.parent / "plots"

SEED = 42
METRICS = ["latency_ms", "error_rate", "throughput_rps", "cpu_percent"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_baseline_metrics(path: Path) -> pd.DataFrame:
    """Load baseline metrics CSV."""
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# TODO(human): Define steady state
# ---------------------------------------------------------------------------

def define_steady_state(
    metrics_df: pd.DataFrame,
    service: str,
    window_size: int = 20,
    num_std: float = 2.5,
) -> dict[str, tuple[float, float]]:
    """Define steady-state bounds for a service from baseline metrics.

    Given a time series of normal-operation metrics for a service, compute
    statistical bounds that define "steady state". Any future observation
    outside these bounds is a potential anomaly.

    Approach -- rolling statistics vs global statistics:
    - Global: compute mean and std over ALL baseline data points. Simple but
      misses time-of-day patterns (diurnal cycles in traffic, latency, etc.).
    - Rolling: compute mean and std over a SLIDING WINDOW of `window_size`
      points, then take the overall min(lower_bound) and max(upper_bound)
      across all windows. This adapts to local variability.

    Use the ROLLING approach:
    1. Filter metrics_df to rows where service == service.
    2. For each metric in METRICS ("latency_ms", "error_rate",
       "throughput_rps", "cpu_percent"):
       a. Extract the time series (sorted by timestamp).
       b. Compute rolling mean and rolling std with window=window_size.
       c. Compute per-window bounds: rolling_mean +/- num_std * rolling_std.
       d. The steady-state lower bound = min of all per-window lower bounds.
          The steady-state upper bound = max of all per-window upper bounds.
       e. This gives the WIDEST envelope that still represents normal behavior.
    3. Return a dict mapping metric_name -> (lower_bound, upper_bound).

    Design choice rationale: Rolling windows capture the widest range of
    "normal" behavior, including peaks and troughs. This avoids false
    positives from natural variation (e.g., higher latency during peak hours).
    The trade-off is that bounds are wider = less sensitive to small anomalies.
    Adjusting num_std controls this trade-off.

    Args:
        metrics_df: Full baseline metrics DataFrame (all services).
        service: Name of the service to compute bounds for.
        window_size: Rolling window size in data points.
        num_std: Number of standard deviations for the bounds.

    Returns:
        Dict mapping metric name to (lower_bound, upper_bound) tuple.
    """
    # TODO(human): Implement rolling-window steady state bounds. Use pandas
    # .rolling(window=window_size) on the sorted time series for each metric.
    # The rolling() method returns a Rolling object that supports .mean()
    # and .std(). Don't forget to dropna() after rolling (first window_size-1
    # values will be NaN).
    raise NotImplementedError("Implement define_steady_state")


# ---------------------------------------------------------------------------
# TODO(human): Check steady state
# ---------------------------------------------------------------------------

def check_steady_state(
    metrics_df: pd.DataFrame,
    service: str,
    bounds: dict[str, tuple[float, float]],
    window: int = 5,
) -> tuple[bool, list[dict]]:
    """Check whether a service is in steady state over a recent time window.

    For each metric in bounds, check if the ROLLING MEAN over the given
    window stays within the defined bounds. A single metric exceeding bounds
    is a steady-state violation.

    Steps:
    1. Filter metrics_df to the given service, sorted by timestamp.
    2. Take the LAST `window` data points (most recent observations).
    3. For each metric in bounds:
       a. Compute the mean of the last `window` values.
       b. Compare against (lower_bound, upper_bound).
       c. If the mean is outside bounds, record a violation:
          {
              "metric": metric_name,
              "value": the computed mean,
              "lower_bound": lower_bound,
              "upper_bound": upper_bound,
              "direction": "below" or "above"
          }
    4. Return (is_steady, violations):
       - is_steady = True if violations list is empty
       - violations = list of violation dicts

    Why rolling mean instead of individual points? Individual points can have
    transient spikes that are normal. Checking the rolling mean over a window
    smooths out noise and detects SUSTAINED deviation, which is what matters
    for steady-state assessment.

    Args:
        metrics_df: Metrics DataFrame (may contain injected chaos data).
        service: Name of the service to check.
        bounds: Dict from define_steady_state (metric -> (lower, upper)).
        window: Number of recent data points to average.

    Returns:
        Tuple of (is_steady, violations_list).
    """
    # TODO(human): Implement the steady-state check. Filter to the service,
    # get the last `window` rows, compute mean per metric, compare to bounds.
    # Return (True, []) if all metrics within bounds, else (False, [...]).
    raise NotImplementedError("Implement check_steady_state")


# ---------------------------------------------------------------------------
# Chaos injection simulation (scaffolded)
# ---------------------------------------------------------------------------

def inject_chaos(
    metrics_df: pd.DataFrame,
    target_service: str,
    chaos_type: str = "latency_spike",
) -> pd.DataFrame:
    """Simulate chaos injection by modifying a service's recent metrics.

    Creates a copy of the metrics with degraded values for the target service
    in the last 10 time points.
    """
    rng = np.random.default_rng(SEED)
    df = metrics_df.copy()

    service_mask = df["service"] == target_service
    last_timestamps = sorted(df.loc[service_mask, "timestamp"].unique())[-10:]
    chaos_mask = service_mask & df["timestamp"].isin(last_timestamps)

    if chaos_type == "latency_spike":
        # 5x latency increase
        df.loc[chaos_mask, "latency_ms"] *= 5.0
        # Error rate jumps to 15-25%
        df.loc[chaos_mask, "error_rate"] = rng.uniform(0.15, 0.25, size=chaos_mask.sum())
    elif chaos_type == "throughput_drop":
        # Throughput drops to 10% of normal
        df.loc[chaos_mask, "throughput_rps"] *= 0.1
        # CPU drops (service not doing work)
        df.loc[chaos_mask, "cpu_percent"] *= 0.2
    elif chaos_type == "resource_exhaustion":
        # CPU maxes out, latency spikes, errors increase
        df.loc[chaos_mask, "cpu_percent"] = rng.uniform(92, 99, size=chaos_mask.sum())
        df.loc[chaos_mask, "latency_ms"] *= 8.0
        df.loc[chaos_mask, "error_rate"] = rng.uniform(0.10, 0.40, size=chaos_mask.sum())
    else:
        print(f"  Unknown chaos type: {chaos_type}")

    return df


# ---------------------------------------------------------------------------
# Visualization (scaffolded)
# ---------------------------------------------------------------------------

def plot_steady_state_check(
    baseline_df: pd.DataFrame,
    chaos_df: pd.DataFrame,
    service: str,
    bounds: dict[str, tuple[float, float]],
    path: Path,
) -> None:
    """Plot metrics timeline showing baseline, chaos injection, and bounds."""
    path.parent.mkdir(parents=True, exist_ok=True)

    baseline_svc = baseline_df[baseline_df["service"] == service].sort_values("timestamp")
    chaos_svc = chaos_df[chaos_df["service"] == service].sort_values("timestamp")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Steady State Check: {service}", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes.flat, METRICS):
        lower, upper = bounds.get(metric, (None, None))

        # Baseline
        ax.plot(
            baseline_svc["timestamp"], baseline_svc[metric],
            color="#2196F3", alpha=0.5, linewidth=1, label="Baseline",
        )
        # Chaos-injected
        ax.plot(
            chaos_svc["timestamp"], chaos_svc[metric],
            color="#F44336", alpha=0.8, linewidth=1.5, label="Under Chaos",
        )

        # Bounds
        if lower is not None and upper is not None:
            ax.axhline(lower, color="green", linestyle="--", alpha=0.6, label="Lower bound")
            ax.axhline(upper, color="green", linestyle="--", alpha=0.6, label="Upper bound")
            ax.fill_between(
                chaos_svc["timestamp"], lower, upper,
                color="green", alpha=0.05,
            )

        ax.set_title(metric, fontsize=11)
        ax.set_xlabel("Timestamp", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved steady state check plot to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Steady State Detection & Chaos Impact Verification")
    print("=" * 60)

    # Load baseline metrics
    metrics_path = DATA_DIR / "baseline_metrics.csv"
    if not metrics_path.exists():
        print("ERROR: baseline_metrics.csv not found. Run src/00_system_model.py first.")
        return

    baseline_df = load_baseline_metrics(metrics_path)
    services = sorted(baseline_df["service"].unique())
    print(f"Loaded baseline metrics: {len(baseline_df)} rows, {len(services)} services")

    # Define steady state for all services
    print("\nDefining steady state bounds for each service...")
    all_bounds = {}
    for svc in services:
        bounds = define_steady_state(baseline_df, svc)
        all_bounds[svc] = {k: list(v) for k, v in bounds.items()}  # JSON-serializable
        print(f"  {svc:30s} -> latency=[{bounds['latency_ms'][0]:.1f}, {bounds['latency_ms'][1]:.1f}] ms, "
              f"error=[{bounds['error_rate'][0]:.4f}, {bounds['error_rate'][1]:.4f}]")

    # Save bounds
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "steady_state_bounds.json", "w") as f:
        json.dump(all_bounds, f, indent=2)
    print(f"\nSaved bounds to {DATA_DIR / 'steady_state_bounds.json'}")

    # Verify: all services should be in steady state on baseline data
    print("\n--- Verifying baseline is in steady state ---")
    all_steady = True
    for svc in services:
        bounds = {k: tuple(v) for k, v in all_bounds[svc].items()}
        is_steady, violations = check_steady_state(baseline_df, svc, bounds)
        if not is_steady:
            print(f"  WARNING: {svc} has {len(violations)} violations in baseline!")
            all_steady = False
    if all_steady:
        print("  All services are in steady state on baseline data.")

    # Inject chaos and check impact
    chaos_scenarios = [
        ("postgres-primary", "latency_spike"),
        ("api-gateway", "resource_exhaustion"),
        ("payment-service", "throughput_drop"),
    ]

    print("\n--- Injecting Chaos & Checking Steady State ---")
    impact_records = []

    for target_service, chaos_type in chaos_scenarios:
        print(f"\n  Chaos: {chaos_type} on {target_service}")
        chaos_df = inject_chaos(baseline_df, target_service, chaos_type)

        # Check if the target service is still in steady state
        bounds = {k: tuple(v) for k, v in all_bounds[target_service].items()}
        is_steady, violations = check_steady_state(chaos_df, target_service, bounds)

        status = "STEADY" if is_steady else "VIOLATED"
        print(f"    {target_service}: {status}")
        if violations:
            for v in violations:
                print(f"      - {v['metric']}: {v['value']:.4f} ({v['direction']} "
                      f"bound [{v['lower_bound']:.4f}, {v['upper_bound']:.4f}])")

        impact_records.append({
            "target_service": target_service,
            "chaos_type": chaos_type,
            "is_steady": is_steady,
            "num_violations": len(violations),
            "violated_metrics": "|".join(v["metric"] for v in violations),
        })

        # Plot the first scenario
        if target_service == chaos_scenarios[0][0]:
            plot_steady_state_check(
                baseline_df, chaos_df, target_service, bounds,
                PLOTS_DIR / "steady_state_check.png",
            )

    # Save impact report
    impact_df = pd.DataFrame(impact_records)
    impact_df.to_csv(DATA_DIR / "chaos_impact_report.csv", index=False)
    print(f"\nSaved chaos impact report to {DATA_DIR / 'chaos_impact_report.csv'}")

    # Summary
    print("\n--- Summary ---")
    violated = impact_df[~impact_df["is_steady"]]
    print(f"  {len(violated)}/{len(impact_df)} chaos scenarios violated steady state")
    for _, row in violated.iterrows():
        print(f"    {row['target_service']} ({row['chaos_type']}): "
              f"{row['num_violations']} metric violations ({row['violated_metrics']})")

    print("\n-- Practice complete. --")


if __name__ == "__main__":
    main()
