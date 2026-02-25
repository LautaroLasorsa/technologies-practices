#!/usr/bin/env python3
"""Exercise 3: Correlation-Based Causal Inference for RCA.

This script uses Pearson correlation and temporal precedence (simplified
Granger causality) to infer causal relationships between services from
their metric time series -- without relying on the known dependency graph.

Concepts practiced:
  - Pairwise Pearson correlation between service metrics
  - Cross-correlation at multiple lags for temporal precedence
  - Constructing a "causal graph" from correlation + temporal signals
  - Comparing inferred causality with the true dependency graph
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
PLOTS_DIR = Path("plots")


# ---------------------------------------------------------------------------
# Data loading (scaffolded)
# ---------------------------------------------------------------------------


def load_graph() -> nx.DiGraph:
    """Load the service dependency graph from JSON."""
    data = json.loads((DATA_DIR / "service_graph.json").read_text())
    return nx.node_link_graph(data)


def load_metrics() -> pd.DataFrame:
    """Load time-series metrics."""
    return pd.read_csv(DATA_DIR / "metrics.csv")


def load_ground_truth() -> dict:
    """Load ground truth root causes."""
    return json.loads((DATA_DIR / "ground_truth.json").read_text())


def compute_anomaly_indicators(
    metrics_df: pd.DataFrame,
    anomaly_start: int,
    metric_name: str = "latency_ms",
) -> pd.DataFrame:
    """Convert raw metrics to binary anomaly indicators.

    For each service at each timestep, the indicator is 1 if the metric
    exceeds (normal_mean + 2 * normal_std), else 0. This produces a
    binary time series suitable for cross-correlation analysis.
    """
    normal = metrics_df[metrics_df["timestamp"] < anomaly_start]
    services = metrics_df["service"].unique()

    records = []
    for svc in services:
        svc_normal = normal[normal["service"] == svc][metric_name]
        threshold = svc_normal.mean() + 2 * svc_normal.std()

        svc_all = metrics_df[metrics_df["service"] == svc].sort_values("timestamp")
        for _, row in svc_all.iterrows():
            records.append({
                "timestamp": int(row["timestamp"]),
                "service": svc,
                "anomaly_indicator": 1.0 if row[metric_name] > threshold else 0.0,
                "metric_value": row[metric_name],
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Exercise 3a: Pairwise Pearson correlation
# ---------------------------------------------------------------------------


def compute_pairwise_correlation(
    metrics_df: pd.DataFrame,
    services: list[str],
    metric_name: str = "latency_ms",
) -> pd.DataFrame:
    """Compute Pearson correlation of a metric between all pairs of services.

    If two services are causally related (one's fault causes the other's
    degradation), their metric time series will be correlated. This function
    computes the full pairwise Pearson correlation matrix to identify
    co-anomalous service pairs.

    The key design decision is: should you correlate the RAW metric values
    (e.g., raw latency_ms) or the ANOMALY INDICATORS (binary 0/1)?

    Raw values: captures the continuous relationship ("when A's latency
    goes up, B's latency also goes up"). More information, but also more
    noise from normal fluctuations.

    Anomaly indicators: captures only co-occurrence of anomalous states
    ("when A is anomalous, B is also anomalous"). Cleaner signal, but
    loses magnitude information.

    For this exercise, use RAW metric values -- they provide richer signal
    and Pearson correlation handles continuous data well.

    Algorithm:
      1. Pivot the metrics DataFrame so that:
         - Rows = timestamps (index)
         - Columns = service names
         - Values = the chosen metric (e.g., latency_ms)
         Use: metrics_df.pivot_table(index='timestamp', columns='service',
              values=metric_name)
      2. Select only the columns for the requested services
      3. Compute the Pearson correlation matrix using pandas:
         pivoted[services].corr(method='pearson')
      4. Return the resulting DataFrame (services x services)

    Parameters:
      metrics_df: Full metrics DataFrame with columns:
                  [timestamp, service, latency_ms, error_rate, cpu_pct]
      services: List of service names to include in the correlation matrix
      metric_name: Which metric column to correlate (default 'latency_ms')

    Returns:
      pandas DataFrame of shape (len(services), len(services)) where
      entry [i, j] is the Pearson correlation coefficient between
      service i and service j's metric time series. Diagonal is 1.0.

    Example:
      >>> services = ["api-gateway", "order-service", "inventory-db"]
      >>> corr = compute_pairwise_correlation(metrics_df, services)
      >>> corr.loc["api-gateway", "inventory-db"]
      0.87  # high correlation = likely causal relationship

    Hints:
      - pd.DataFrame.pivot_table handles the reshaping
      - pd.DataFrame.corr() computes pairwise Pearson correlation
      - The result is symmetric: corr[A][B] == corr[B][A]
      - NaN values may appear if a service has constant metric values;
        handle with .fillna(0)
    """
    raise NotImplementedError("TODO(human): Implement compute_pairwise_correlation")


# ---------------------------------------------------------------------------
# Exercise 3b: Temporal precedence score (simplified Granger causality)
# ---------------------------------------------------------------------------


def temporal_precedence_score(
    metrics_df: pd.DataFrame,
    source: str,
    target: str,
    metric_name: str = "latency_ms",
    max_lag: int = 20,
) -> float:
    """Compute a simplified Granger-like temporal precedence score.

    Granger causality tests whether one time series helps predict another.
    A simplified version: check whether anomalies in the SOURCE service
    consistently PRECEDE anomalies in the TARGET service. If so, the
    source may be causing the target's anomaly.

    The approach uses cross-correlation of the two services' metric time
    series at various lags. If the peak cross-correlation occurs at a
    POSITIVE lag (meaning the source signal leads the target signal),
    the source temporally precedes the target.

    Algorithm:
      1. Extract the metric time series for source and target:
         - source_ts: array of metric values for source, sorted by timestamp
         - target_ts: array of metric values for target, sorted by timestamp
      2. Normalize both series (subtract mean, divide by std) to get
         standardized series. This makes the cross-correlation values
         comparable across service pairs.
         Handle zero std: if std is 0, return 0.0 (no variability = no signal)
      3. Compute cross-correlation at lags from -max_lag to +max_lag:
         - For each lag k in range(-max_lag, max_lag+1):
           - If k >= 0: correlate source_ts[:-k or None] with target_ts[k:]
           - If k < 0: correlate source_ts[-k:] with target_ts[:k or None]
           - cross_corr[k] = mean(source_shifted * target_shifted)
         - Alternatively, use np.correlate(source_ts, target_ts, mode='full')
           and extract the relevant lag range.
      4. Find the lag with the maximum cross-correlation:
         - best_lag = argmax of cross_corr
      5. Compute the temporal precedence score:
         - If best_lag > 0 (source leads target): score = cross_corr[best_lag]
         - If best_lag <= 0 (target leads source or simultaneous): score = 0.0
         - Clip to [0, 1] range
         The score represents: "how strongly does source's signal precede
         target's signal?"

    Parameters:
      metrics_df: Full metrics DataFrame
      source: Service name that may be the CAUSE
      target: Service name that may be the EFFECT
      metric_name: Which metric to use (default 'latency_ms')
      max_lag: Maximum number of timesteps to check for precedence.
               Should be at least as large as the expected propagation
               delay between services. Default 20 covers most cases.

    Returns:
      Float score in [0, 1]. Higher = stronger evidence that source's
      anomaly precedes target's anomaly.
      - 0.0 = no evidence of source preceding target
      - 1.0 = strong evidence that source leads target

    Example:
      >>> # If inventory-db anomaly precedes order-service anomaly:
      >>> score = temporal_precedence_score(df, "inventory-db", "order-service")
      >>> score
      0.73  # strong temporal precedence

      >>> # If order-service anomaly does NOT precede inventory-db:
      >>> score = temporal_precedence_score(df, "order-service", "inventory-db")
      >>> score
      0.0  # no precedence (wrong direction)

    Hints:
      - Extract time series: df[df['service']==svc].sort_values('timestamp')[metric_name].values
      - Normalize: (ts - ts.mean()) / ts.std()
      - np.correlate gives raw cross-correlation; for normalized, divide
        by len(ts) after normalizing both series
      - The lag index in np.correlate(a, b, 'full') output: index 0
        corresponds to lag -(len(b)-1), and index len(a)+len(b)-2
        corresponds to lag +(len(a)-1). The center is at len(b)-1 (lag=0).
      - Simpler approach: just loop over lag values and compute correlation
        manually. Less efficient but clearer.
    """
    raise NotImplementedError("TODO(human): Implement temporal_precedence_score")


# ---------------------------------------------------------------------------
# Causal graph construction (scaffolded)
# ---------------------------------------------------------------------------


def build_causal_graph(
    metrics_df: pd.DataFrame,
    services: list[str],
    correlation_threshold: float = 0.5,
    precedence_threshold: float = 0.1,
) -> nx.DiGraph:
    """Build a directed causal graph from correlation + temporal precedence.

    For each pair of services (A, B), if:
      1. Their Pearson correlation exceeds correlation_threshold, AND
      2. A's temporal precedence score over B exceeds precedence_threshold
    then add a directed edge A -> B (A causes B).

    This constructs a causal graph purely from metric data, without
    knowing the true dependency graph.
    """
    corr_matrix = compute_pairwise_correlation(metrics_df, services)
    causal_graph = nx.DiGraph()
    causal_graph.add_nodes_from(services)

    for source in services:
        for target in services:
            if source == target:
                continue

            correlation = corr_matrix.loc[source, target]
            if abs(correlation) < correlation_threshold:
                continue

            precedence = temporal_precedence_score(metrics_df, source, target)
            if precedence < precedence_threshold:
                continue

            causal_graph.add_edge(source, target,
                                  correlation=correlation,
                                  precedence=precedence,
                                  weight=correlation * precedence)

    return causal_graph


def compare_graphs(
    inferred: nx.DiGraph,
    true_graph: nx.DiGraph,
    services: list[str],
) -> dict[str, float]:
    """Compare inferred causal graph with true dependency graph."""
    true_edges = set(true_graph.edges()) & {(u, v) for u in services for v in services}
    inferred_edges = set(inferred.edges())

    true_positives = len(inferred_edges & true_edges)
    false_positives = len(inferred_edges - true_edges)
    false_negatives = len(true_edges - inferred_edges)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Visualization (scaffolded)
# ---------------------------------------------------------------------------


def visualize_correlation_matrix(
    corr_matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    """Visualize the pairwise correlation matrix as a heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.imshow(corr_matrix.values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(corr_matrix.index, fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson Correlation")
    ax.set_title("Pairwise Latency Correlation Between Services", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved correlation heatmap: {output_path}")


def visualize_causal_graph(
    causal_graph: nx.DiGraph,
    true_graph: nx.DiGraph,
    output_path: Path,
) -> None:
    """Visualize inferred causal graph, highlighting correct vs incorrect edges."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    pos = nx.spring_layout(causal_graph, seed=42, k=2.0, iterations=100)

    true_edges = set(true_graph.edges())

    # Classify edges
    correct_edges = [(u, v) for u, v in causal_graph.edges() if (u, v) in true_edges]
    incorrect_edges = [(u, v) for u, v in causal_graph.edges() if (u, v) not in true_edges]

    # Draw nodes
    nx.draw_networkx_nodes(causal_graph, pos, ax=ax, node_color="#4ECDC4",
                           node_size=700, edgecolors="#333333", linewidths=1.0)
    nx.draw_networkx_labels(causal_graph, pos, ax=ax, font_size=6, font_weight="bold")

    # Draw correct edges (green)
    if correct_edges:
        nx.draw_networkx_edges(causal_graph, pos, ax=ax, edgelist=correct_edges,
                               edge_color="#00AA00", arrows=True, arrowsize=15,
                               arrowstyle="-|>", width=2.5,
                               connectionstyle="arc3,rad=0.1")

    # Draw incorrect edges (red, dashed)
    if incorrect_edges:
        nx.draw_networkx_edges(causal_graph, pos, ax=ax, edgelist=incorrect_edges,
                               edge_color="#FF0000", arrows=True, arrowsize=15,
                               arrowstyle="-|>", width=1.5, style="dashed",
                               connectionstyle="arc3,rad=0.1")

    ax.set_title("Inferred Causal Graph (green = correct edge, red dashed = false positive)", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved causal graph: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  Exercise 3: Correlation-Based Causal Inference")
    print("=" * 60)

    # Load data
    print("\n[1/7] Loading graph and metrics...")
    true_graph = load_graph()
    metrics_df = load_metrics()
    ground_truth = load_ground_truth()
    anomaly_start = ground_truth["anomaly_start_timestep"]
    true_root_causes = set(ground_truth["root_causes"])

    # Get list of services involved in anomaly propagation
    propagation_services = [p["service"] for p in ground_truth["propagation_order"]]
    all_services = sorted(metrics_df["service"].unique())

    print(f"  Total services: {len(all_services)}")
    print(f"  Propagation services: {len(propagation_services)}")
    print(f"  True root causes: {true_root_causes}")

    # Compute pairwise correlation
    print("\n[2/7] Computing pairwise Pearson correlation...")
    corr_matrix = compute_pairwise_correlation(metrics_df, all_services, "latency_ms")

    print(f"\n  Correlation matrix shape: {corr_matrix.shape}")
    print(f"  Mean correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
    print(f"  Max off-diagonal: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max():.3f}")

    # Show top correlated pairs
    pairs = []
    for i, s1 in enumerate(all_services):
        for j, s2 in enumerate(all_services):
            if i < j:
                pairs.append((s1, s2, corr_matrix.loc[s1, s2]))
    pairs.sort(key=lambda x: -abs(x[2]))

    print(f"\n  Top 10 correlated pairs:")
    print(f"  {'Service A':<24} {'Service B':<24} {'Correlation':>12} {'True Edge?':>12}")
    print(f"  {'-'*24} {'-'*24} {'-'*12} {'-'*12}")
    for s1, s2, corr in pairs[:10]:
        has_edge = "YES" if true_graph.has_edge(s1, s2) or true_graph.has_edge(s2, s1) else "no"
        print(f"  {s1:<24} {s2:<24} {corr:>12.3f} {has_edge:>12}")

    # Temporal precedence analysis
    print("\n[3/7] Computing temporal precedence scores...")
    print(f"\n  Checking precedence for propagation-related pairs:")
    print(f"  {'Source':<24} {'Target':<24} {'Precedence':>12} {'True Direction?':>16}")
    print(f"  {'-'*24} {'-'*24} {'-'*12} {'-'*16}")

    for source in propagation_services[:6]:
        for target in propagation_services[:6]:
            if source == target:
                continue
            if not (true_graph.has_edge(source, target) or true_graph.has_edge(target, source)):
                continue
            score = temporal_precedence_score(metrics_df, source, target, "latency_ms", max_lag=20)
            if score > 0.05:
                is_true = "YES" if true_graph.has_edge(source, target) else "no"
                print(f"  {source:<24} {target:<24} {score:>12.3f} {is_true:>16}")

    # Build inferred causal graph
    print("\n[4/7] Building inferred causal graph...")
    causal_graph = build_causal_graph(
        metrics_df, propagation_services,
        correlation_threshold=0.5,
        precedence_threshold=0.1,
    )
    print(f"  Inferred graph: {causal_graph.number_of_nodes()} nodes, {causal_graph.number_of_edges()} edges")

    # Compare with true graph
    print("\n[5/7] Comparing inferred vs true dependency graph...")
    comparison = compare_graphs(causal_graph, true_graph, propagation_services)

    print(f"\n  True positives  (correctly inferred edges): {comparison['true_positives']}")
    print(f"  False positives (spurious edges):            {comparison['false_positives']}")
    print(f"  False negatives (missed true edges):         {comparison['false_negatives']}")
    print(f"  Precision: {comparison['precision']:.3f}")
    print(f"  Recall:    {comparison['recall']:.3f}")
    print(f"  F1 Score:  {comparison['f1']:.3f}")

    # Root cause identification from causal graph
    print("\n[6/7] Identifying root causes from causal graph...")
    # In the inferred causal graph, root causes are nodes with high out-degree
    # (they cause many other nodes) and low in-degree (few things cause them)
    if causal_graph.number_of_edges() > 0:
        cause_scores = {}
        for node in causal_graph.nodes:
            out_deg = causal_graph.out_degree(node)
            in_deg = causal_graph.in_degree(node)
            # Score: out-degree minus in-degree (net "causer" vs "victim")
            cause_scores[node] = out_deg - in_deg

        ranked_causes = sorted(cause_scores.items(), key=lambda x: -x[1])
        print(f"\n  {'Rank':<6} {'Service':<28} {'Out-deg':>8} {'In-deg':>8} {'Net Score':>10}")
        print(f"  {'-'*6} {'-'*28} {'-'*8} {'-'*8} {'-'*10}")
        for rank, (svc, score) in enumerate(ranked_causes[:10], 1):
            out_d = causal_graph.out_degree(svc)
            in_d = causal_graph.in_degree(svc)
            marker = " *** ROOT CAUSE" if svc in true_root_causes else ""
            print(f"  {rank:<6} {svc:<28} {out_d:>8} {in_d:>8} {score:>10}{marker}")

        top_inferred_cause = ranked_causes[0][0] if ranked_causes else None
        if top_inferred_cause in true_root_causes:
            print(f"\n  SUCCESS: Top inferred cause '{top_inferred_cause}' matches ground truth!")
        else:
            print(f"\n  MISS: Top inferred cause '{top_inferred_cause}', true root cause is {true_root_causes}")
    else:
        print("  No edges inferred -- try lowering thresholds")

    # Visualize
    print("\n[7/7] Visualizing results...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    visualize_correlation_matrix(corr_matrix, PLOTS_DIR / "correlation_matrix.png")
    if causal_graph.number_of_edges() > 0:
        visualize_causal_graph(causal_graph, true_graph, PLOTS_DIR / "causal_graph.png")

    print()


if __name__ == "__main__":
    main()
