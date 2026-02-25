#!/usr/bin/env python3
"""Build a synthetic microservice dependency graph with injected cascading failure.

Generates:
  - data/service_graph.json   -- Adjacency data with node/edge attributes
  - data/metrics.csv          -- Time-series metrics per service (latency, error_rate, cpu)
  - data/ground_truth.json    -- True root cause nodes and propagation order
  - plots/topology.png        -- Visualization of the dependency graph
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
NUM_TIMESTEPS = 200
NORMAL_PERIOD = 80  # First 80 timesteps are "normal"; anomaly starts after
DATA_DIR = Path("data")
PLOTS_DIR = Path("plots")

# Service definitions: (name, team, tier)
SERVICES: list[tuple[str, str, str]] = [
    ("api-gateway", "platform", "frontend"),
    ("web-frontend", "platform", "frontend"),
    ("mobile-bff", "platform", "frontend"),
    ("user-service", "identity", "backend"),
    ("auth-service", "identity", "backend"),
    ("order-service", "commerce", "backend"),
    ("payment-service", "commerce", "backend"),
    ("inventory-service", "commerce", "backend"),
    ("shipping-service", "logistics", "backend"),
    ("notification-service", "platform", "backend"),
    ("recommendation-engine", "ml", "backend"),
    ("search-service", "ml", "backend"),
    ("product-catalog", "commerce", "backend"),
    ("user-db", "identity", "data"),
    ("order-db", "commerce", "data"),
    ("inventory-db", "commerce", "data"),
    ("cache-redis", "platform", "data"),
    ("message-queue", "platform", "data"),
]

# Dependency edges: (caller, callee, avg_latency_ms, calls_per_second)
# Direction: caller -> callee means "caller depends on callee"
EDGES: list[tuple[str, str, float, float]] = [
    # Frontend -> Backend
    ("api-gateway", "user-service", 5.0, 500.0),
    ("api-gateway", "order-service", 8.0, 300.0),
    ("api-gateway", "search-service", 12.0, 200.0),
    ("api-gateway", "product-catalog", 6.0, 400.0),
    ("web-frontend", "api-gateway", 2.0, 600.0),
    ("mobile-bff", "api-gateway", 3.0, 400.0),
    ("mobile-bff", "recommendation-engine", 15.0, 100.0),
    # Backend -> Backend
    ("user-service", "auth-service", 4.0, 450.0),
    ("order-service", "payment-service", 20.0, 150.0),
    ("order-service", "inventory-service", 10.0, 200.0),
    ("order-service", "notification-service", 5.0, 100.0),
    ("payment-service", "notification-service", 3.0, 80.0),
    ("shipping-service", "inventory-service", 8.0, 50.0),
    ("shipping-service", "notification-service", 4.0, 50.0),
    ("recommendation-engine", "product-catalog", 10.0, 150.0),
    ("search-service", "product-catalog", 7.0, 250.0),
    # Backend -> Data
    ("user-service", "user-db", 3.0, 500.0),
    ("user-service", "cache-redis", 1.0, 800.0),
    ("auth-service", "user-db", 2.0, 400.0),
    ("auth-service", "cache-redis", 1.0, 600.0),
    ("order-service", "order-db", 4.0, 300.0),
    ("inventory-service", "inventory-db", 3.0, 250.0),
    ("product-catalog", "inventory-db", 5.0, 200.0),
    ("product-catalog", "cache-redis", 1.0, 500.0),
    ("notification-service", "message-queue", 2.0, 200.0),
]

# Ground truth: root cause nodes and the order in which anomaly propagates
ROOT_CAUSES = ["inventory-db"]
# Propagation order (approximate): each tuple is (node, delay_in_timesteps)
PROPAGATION_ORDER: list[tuple[str, int]] = [
    ("inventory-db", 0),
    ("inventory-service", 3),
    ("product-catalog", 5),
    ("order-service", 6),
    ("shipping-service", 7),
    ("search-service", 8),
    ("recommendation-engine", 10),
    ("api-gateway", 9),
    ("notification-service", 10),
    ("payment-service", 11),
    ("web-frontend", 12),
    ("mobile-bff", 13),
]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_service_graph() -> nx.DiGraph:
    """Build the service dependency DAG with node and edge attributes."""
    graph = nx.DiGraph()

    for name, team, tier in SERVICES:
        graph.add_node(name, team=team, tier=tier)

    for caller, callee, latency, freq in EDGES:
        graph.add_edge(caller, callee, avg_latency_ms=latency, calls_per_sec=freq)

    return graph


# ---------------------------------------------------------------------------
# Synthetic metric generation
# ---------------------------------------------------------------------------


def generate_normal_metrics(
    rng: np.random.Generator,
    num_timesteps: int,
    service_name: str,
    tier: str,
) -> dict[str, np.ndarray]:
    """Generate baseline (normal) metrics for a service."""
    # Base latency depends on tier
    base_latency = {"frontend": 50.0, "backend": 30.0, "data": 10.0}[tier]
    base_error_rate = 0.01
    base_cpu = {"frontend": 0.3, "backend": 0.4, "data": 0.5}[tier]

    # Add some per-service variation
    svc_hash = hash(service_name) % 100
    base_latency += svc_hash * 0.2
    base_cpu += (svc_hash % 20) * 0.005

    latency = base_latency + rng.normal(0, base_latency * 0.05, num_timesteps)
    error_rate = base_error_rate + rng.normal(0, 0.003, num_timesteps)
    error_rate = np.clip(error_rate, 0, 1)
    cpu = base_cpu + rng.normal(0, 0.02, num_timesteps)
    cpu = np.clip(cpu, 0, 1)

    return {"latency_ms": latency, "error_rate": error_rate, "cpu_pct": cpu}


def inject_anomaly(
    metrics: dict[str, np.ndarray],
    start_step: int,
    severity: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Inject an anomaly into metrics starting at start_step with given severity."""
    result = {k: v.copy() for k, v in metrics.items()}
    n = len(result["latency_ms"])

    if start_step >= n:
        return result

    anomaly_len = n - start_step

    # Latency: multiplicative increase with gradual ramp-up
    ramp = np.minimum(np.arange(anomaly_len) / 10.0, 1.0)  # ramp over 10 steps
    latency_multiplier = 1.0 + severity * 2.0 * ramp
    noise = rng.normal(0, severity * 5.0, anomaly_len)
    result["latency_ms"][start_step:] *= latency_multiplier
    result["latency_ms"][start_step:] += np.abs(noise)

    # Error rate: additive increase
    error_increase = severity * 0.15 * ramp + rng.normal(0, severity * 0.02, anomaly_len)
    result["error_rate"][start_step:] += np.abs(error_increase)
    result["error_rate"] = np.clip(result["error_rate"], 0, 1)

    # CPU: additive increase
    cpu_increase = severity * 0.25 * ramp + rng.normal(0, severity * 0.03, anomaly_len)
    result["cpu_pct"][start_step:] += np.abs(cpu_increase)
    result["cpu_pct"] = np.clip(result["cpu_pct"], 0, 1)

    return result


def generate_all_metrics(
    graph: nx.DiGraph,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate time-series metrics for all services with injected cascading anomaly."""
    propagation_map = dict(PROPAGATION_ORDER)
    records = []

    for node in graph.nodes:
        tier = graph.nodes[node]["tier"]
        metrics = generate_normal_metrics(rng, NUM_TIMESTEPS, node, tier)

        if node in propagation_map:
            delay = propagation_map[node]
            anomaly_start = NORMAL_PERIOD + delay

            # Severity decreases with distance from root cause
            if node in ROOT_CAUSES:
                severity = 1.0
            else:
                severity = max(0.15, 1.0 - delay * 0.07)

            metrics = inject_anomaly(metrics, anomaly_start, severity, rng)

        for t in range(NUM_TIMESTEPS):
            records.append({
                "timestamp": t,
                "service": node,
                "latency_ms": metrics["latency_ms"][t],
                "error_rate": metrics["error_rate"][t],
                "cpu_pct": metrics["cpu_pct"][t],
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_topology(graph: nx.DiGraph, output_path: Path) -> None:
    """Draw the service dependency graph colored by tier."""
    tier_colors = {"frontend": "#FF6B6B", "backend": "#4ECDC4", "data": "#45B7D1"}
    node_colors = [tier_colors.get(graph.nodes[n]["tier"], "#95A5A6") for n in graph.nodes]

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Use spring layout with fixed seed for reproducibility
    pos = nx.spring_layout(graph, seed=42, k=2.0, iterations=100)

    # Draw edges with arrows
    nx.draw_networkx_edges(
        graph, pos, ax=ax,
        edge_color="#CCCCCC", arrows=True,
        arrowsize=15, arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        width=1.5,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos, ax=ax,
        node_color=node_colors, node_size=800,
        edgecolors="#333333", linewidths=1.5,
    )

    # Draw labels
    nx.draw_networkx_labels(
        graph, pos, ax=ax,
        font_size=7, font_weight="bold",
    )

    # Highlight root cause nodes
    root_nodes = [n for n in graph.nodes if n in ROOT_CAUSES]
    if root_nodes:
        nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            nodelist=root_nodes,
            node_color="#FF0000", node_size=1000,
            edgecolors="#8B0000", linewidths=3.0,
        )

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#FF6B6B", edgecolor="#333", label="Frontend"),
        Patch(facecolor="#4ECDC4", edgecolor="#333", label="Backend"),
        Patch(facecolor="#45B7D1", edgecolor="#333", label="Data"),
        Patch(facecolor="#FF0000", edgecolor="#8B0000", label="Root Cause (ground truth)"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=9)

    ax.set_title("Service Dependency Graph (edges = 'caller depends on callee')", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved topology plot: {output_path}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_graph(graph: nx.DiGraph, path: Path) -> None:
    """Save graph as node-link JSON."""
    data = nx.node_link_data(graph)
    path.write_text(json.dumps(data, indent=2))
    print(f"  Saved graph: {path} ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")


def save_ground_truth(path: Path) -> None:
    """Save ground truth root causes and propagation order."""
    truth = {
        "root_causes": ROOT_CAUSES,
        "propagation_order": [
            {"service": svc, "delay_timesteps": delay}
            for svc, delay in PROPAGATION_ORDER
        ],
        "anomaly_start_timestep": NORMAL_PERIOD,
    }
    path.write_text(json.dumps(truth, indent=2))
    print(f"  Saved ground truth: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  Building Service Dependency Topology")
    print("=" * 60)

    rng = np.random.default_rng(SEED)

    # Create output directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build graph
    print("\n[1/4] Building service dependency graph...")
    graph = build_service_graph()
    print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    print(f"  Tiers: {set(nx.get_node_attributes(graph, 'tier').values())}")
    print(f"  Teams: {set(nx.get_node_attributes(graph, 'team').values())}")

    # Generate metrics
    print("\n[2/4] Generating synthetic metrics with cascading anomaly...")
    metrics_df = generate_all_metrics(graph, rng)
    metrics_path = DATA_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Saved metrics: {metrics_path} ({len(metrics_df)} rows)")
    print(f"  Root causes: {ROOT_CAUSES}")
    print(f"  Anomaly starts at timestep {NORMAL_PERIOD}")
    print(f"  Affected services: {len(PROPAGATION_ORDER)}/{graph.number_of_nodes()}")

    # Save graph and ground truth
    print("\n[3/4] Saving graph and ground truth...")
    save_graph(graph, DATA_DIR / "service_graph.json")
    save_ground_truth(DATA_DIR / "ground_truth.json")

    # Visualize
    print("\n[4/4] Visualizing topology...")
    visualize_topology(graph, PLOTS_DIR / "topology.png")

    # Summary statistics
    print("\n" + "-" * 60)
    print("  Summary")
    print("-" * 60)

    # Compute basic anomaly scores (mean latency in anomaly period vs normal)
    anomaly_period = metrics_df[metrics_df["timestamp"] >= NORMAL_PERIOD]
    normal_period = metrics_df[metrics_df["timestamp"] < NORMAL_PERIOD]

    print(f"\n  {'Service':<28} {'Normal Lat (ms)':>16} {'Anomaly Lat (ms)':>18} {'Ratio':>8}")
    print(f"  {'-'*28} {'-'*16} {'-'*18} {'-'*8}")

    for svc in sorted(graph.nodes):
        normal_lat = normal_period[normal_period["service"] == svc]["latency_ms"].mean()
        anomaly_lat = anomaly_period[anomaly_period["service"] == svc]["latency_ms"].mean()
        ratio = anomaly_lat / normal_lat if normal_lat > 0 else 0
        marker = " *** ROOT CAUSE" if svc in ROOT_CAUSES else ""
        print(f"  {svc:<28} {normal_lat:>16.1f} {anomaly_lat:>18.1f} {ratio:>8.2f}{marker}")

    print(f"\n  Output files:")
    print(f"    data/service_graph.json  -- Graph adjacency data")
    print(f"    data/metrics.csv         -- Time-series metrics ({NUM_TIMESTEPS} timesteps x {graph.number_of_nodes()} services)")
    print(f"    data/ground_truth.json   -- True root causes and propagation order")
    print(f"    plots/topology.png       -- Dependency graph visualization")
    print()


if __name__ == "__main__":
    main()
