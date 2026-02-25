"""Build a synthetic microservice dependency graph with realistic topology.

Generates a directed graph representing a microservice architecture with:
- 5 tiers: frontend, gateway, backend, data, external
- Realistic dependency edges with failure propagation probabilities
- Service metadata: replicas, circuit breakers, SLA criticality
- Baseline metrics for each service (100 time points of normal behavior)

Outputs:
- data/system_graph.json  -- node/edge data
- data/baseline_metrics.csv -- time series of normal operation metrics
- plots/system_topology.png -- visualization of the dependency graph
"""

import json
import random
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
PLOTS_DIR = Path(__file__).parent.parent / "plots"

SEED = 42


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_service_catalog() -> list[dict]:
    """Define the microservice catalog with tier assignments and metadata."""
    services = [
        # Frontend tier (3 nodes)
        {"name": "web-frontend", "tier": "frontend", "replicas": 3, "has_circuit_breaker": True, "sla_criticality": 4},
        {"name": "mobile-bff", "tier": "frontend", "replicas": 2, "has_circuit_breaker": True, "sla_criticality": 4},
        {"name": "admin-dashboard", "tier": "frontend", "replicas": 1, "has_circuit_breaker": False, "sla_criticality": 2},
        # API Gateway (1 node)
        {"name": "api-gateway", "tier": "gateway", "replicas": 3, "has_circuit_breaker": True, "sla_criticality": 5},
        # Backend services (7 nodes)
        {"name": "order-service", "tier": "backend", "replicas": 2, "has_circuit_breaker": True, "sla_criticality": 5},
        {"name": "payment-service", "tier": "backend", "replicas": 2, "has_circuit_breaker": True, "sla_criticality": 5},
        {"name": "inventory-service", "tier": "backend", "replicas": 2, "has_circuit_breaker": True, "sla_criticality": 4},
        {"name": "user-service", "tier": "backend", "replicas": 2, "has_circuit_breaker": True, "sla_criticality": 4},
        {"name": "notification-service", "tier": "backend", "replicas": 1, "has_circuit_breaker": False, "sla_criticality": 2},
        {"name": "recommendation-service", "tier": "backend", "replicas": 1, "has_circuit_breaker": False, "sla_criticality": 2},
        {"name": "search-service", "tier": "backend", "replicas": 2, "has_circuit_breaker": True, "sla_criticality": 3},
        # Data layer (4 nodes)
        {"name": "postgres-primary", "tier": "data", "replicas": 1, "has_circuit_breaker": False, "sla_criticality": 5},
        {"name": "redis-cache", "tier": "data", "replicas": 2, "has_circuit_breaker": False, "sla_criticality": 4},
        {"name": "elasticsearch", "tier": "data", "replicas": 2, "has_circuit_breaker": False, "sla_criticality": 3},
        {"name": "rabbitmq", "tier": "data", "replicas": 1, "has_circuit_breaker": False, "sla_criticality": 4},
        # External dependencies (3 nodes)
        {"name": "stripe-api", "tier": "external", "replicas": 1, "has_circuit_breaker": False, "sla_criticality": 5},
        {"name": "sendgrid-api", "tier": "external", "replicas": 1, "has_circuit_breaker": False, "sla_criticality": 2},
        {"name": "cdn-provider", "tier": "external", "replicas": 1, "has_circuit_breaker": False, "sla_criticality": 3},
    ]
    return services


def build_dependency_edges() -> list[dict]:
    """Define dependency edges between services.

    Edge direction: source DEPENDS ON target (source -> target means
    "source calls target"). Failure of target can propagate to source.
    """
    edges = [
        # Frontends depend on gateway
        {"source": "web-frontend", "target": "api-gateway", "call_freq": 500, "propagation_prob": 0.8},
        {"source": "mobile-bff", "target": "api-gateway", "call_freq": 300, "propagation_prob": 0.8},
        {"source": "admin-dashboard", "target": "api-gateway", "call_freq": 20, "propagation_prob": 0.7},
        # Frontends depend on CDN
        {"source": "web-frontend", "target": "cdn-provider", "call_freq": 1000, "propagation_prob": 0.3},
        {"source": "mobile-bff", "target": "cdn-provider", "call_freq": 200, "propagation_prob": 0.2},
        # Gateway routes to backend services
        {"source": "api-gateway", "target": "order-service", "call_freq": 100, "propagation_prob": 0.6},
        {"source": "api-gateway", "target": "user-service", "call_freq": 200, "propagation_prob": 0.7},
        {"source": "api-gateway", "target": "search-service", "call_freq": 150, "propagation_prob": 0.4},
        {"source": "api-gateway", "target": "recommendation-service", "call_freq": 80, "propagation_prob": 0.3},
        # Backend inter-service dependencies
        {"source": "order-service", "target": "payment-service", "call_freq": 80, "propagation_prob": 0.9},
        {"source": "order-service", "target": "inventory-service", "call_freq": 80, "propagation_prob": 0.8},
        {"source": "order-service", "target": "notification-service", "call_freq": 80, "propagation_prob": 0.2},
        {"source": "payment-service", "target": "stripe-api", "call_freq": 80, "propagation_prob": 0.9},
        {"source": "payment-service", "target": "user-service", "call_freq": 50, "propagation_prob": 0.5},
        {"source": "inventory-service", "target": "rabbitmq", "call_freq": 60, "propagation_prob": 0.7},
        {"source": "notification-service", "target": "sendgrid-api", "call_freq": 40, "propagation_prob": 0.8},
        {"source": "notification-service", "target": "rabbitmq", "call_freq": 40, "propagation_prob": 0.6},
        {"source": "recommendation-service", "target": "user-service", "call_freq": 30, "propagation_prob": 0.4},
        {"source": "recommendation-service", "target": "redis-cache", "call_freq": 100, "propagation_prob": 0.5},
        {"source": "search-service", "target": "elasticsearch", "call_freq": 150, "propagation_prob": 0.8},
        # Data layer dependencies
        {"source": "order-service", "target": "postgres-primary", "call_freq": 100, "propagation_prob": 0.9},
        {"source": "payment-service", "target": "postgres-primary", "call_freq": 80, "propagation_prob": 0.9},
        {"source": "user-service", "target": "postgres-primary", "call_freq": 120, "propagation_prob": 0.9},
        {"source": "user-service", "target": "redis-cache", "call_freq": 200, "propagation_prob": 0.4},
        {"source": "inventory-service", "target": "postgres-primary", "call_freq": 60, "propagation_prob": 0.8},
        {"source": "search-service", "target": "redis-cache", "call_freq": 50, "propagation_prob": 0.3},
    ]
    return edges


def build_graph(services: list[dict], edges: list[dict]) -> nx.DiGraph:
    """Construct a networkx DiGraph from service catalog and edges."""
    graph = nx.DiGraph()
    for svc in services:
        graph.add_node(
            svc["name"],
            tier=svc["tier"],
            replicas=svc["replicas"],
            has_circuit_breaker=svc["has_circuit_breaker"],
            sla_criticality=svc["sla_criticality"],
        )
    for edge in edges:
        graph.add_edge(
            edge["source"],
            edge["target"],
            call_frequency=edge["call_freq"],
            failure_propagation_probability=edge["propagation_prob"],
        )
    return graph


def save_graph(graph: nx.DiGraph, path: Path) -> None:
    """Serialize graph to JSON (node-link format)."""
    data = nx.node_link_data(graph)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved graph to {path} ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")


# ---------------------------------------------------------------------------
# Baseline metrics generation
# ---------------------------------------------------------------------------

def generate_baseline_metrics(graph: nx.DiGraph, num_points: int = 100) -> pd.DataFrame:
    """Generate synthetic baseline metrics for each service.

    Metrics per service:
    - latency_ms: mean response time (varies by tier)
    - error_rate: fraction of failed requests (low in normal operation)
    - throughput_rps: requests per second (based on incoming call frequency)
    - cpu_percent: CPU utilization
    """
    rng = np.random.default_rng(SEED)
    records = []

    tier_latency = {"frontend": 50, "gateway": 10, "backend": 30, "data": 5, "external": 80}
    tier_cpu_base = {"frontend": 30, "gateway": 40, "backend": 50, "data": 60, "external": 20}

    for node in graph.nodes():
        attrs = graph.nodes[node]
        tier = attrs["tier"]
        base_latency = tier_latency[tier]
        base_cpu = tier_cpu_base[tier]

        # Throughput is based on total incoming call frequency
        in_edges = graph.in_edges(node, data=True)
        base_throughput = sum(d.get("call_frequency", 10) for _, _, d in in_edges)
        if base_throughput == 0:
            base_throughput = 50  # external/leaf services

        for t in range(num_points):
            # Add realistic time-series noise: slight trend + gaussian noise
            time_factor = 1.0 + 0.1 * np.sin(2 * np.pi * t / num_points)  # diurnal pattern
            records.append({
                "timestamp": t,
                "service": node,
                "latency_ms": max(1.0, base_latency * time_factor + rng.normal(0, base_latency * 0.1)),
                "error_rate": max(0.0, min(1.0, 0.005 + rng.normal(0, 0.002))),
                "throughput_rps": max(1.0, base_throughput * time_factor + rng.normal(0, base_throughput * 0.05)),
                "cpu_percent": max(1.0, min(99.0, base_cpu * time_factor + rng.normal(0, base_cpu * 0.08))),
            })

    df = pd.DataFrame(records)
    return df


def save_baseline_metrics(df: pd.DataFrame, path: Path) -> None:
    """Save baseline metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved baseline metrics to {path} ({len(df)} rows, {df['service'].nunique()} services)")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

TIER_COLORS = {
    "frontend": "#4CAF50",
    "gateway": "#FF9800",
    "backend": "#2196F3",
    "data": "#9C27B0",
    "external": "#F44336",
}

TIER_Y_POSITIONS = {
    "frontend": 4,
    "gateway": 3,
    "backend": 2,
    "data": 1,
    "external": 0,
}


def visualize_graph(graph: nx.DiGraph, path: Path) -> None:
    """Visualize the microservice dependency graph with tier-based layout."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Position nodes in rows by tier
    tier_nodes: dict[str, list[str]] = {}
    for node in graph.nodes():
        tier = graph.nodes[node]["tier"]
        tier_nodes.setdefault(tier, []).append(node)

    pos = {}
    for tier, nodes in tier_nodes.items():
        y = TIER_Y_POSITIONS[tier]
        n = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            x = (i - (n - 1) / 2) * 2.5
            pos[node] = (x, y)

    fig, ax = plt.subplots(1, 1, figsize=(18, 10))

    # Draw edges
    nx.draw_networkx_edges(
        graph, pos, ax=ax,
        edge_color="#CCCCCC", arrows=True, arrowsize=12,
        connectionstyle="arc3,rad=0.1", alpha=0.6,
    )

    # Draw nodes colored by tier, sized by criticality
    for tier, color in TIER_COLORS.items():
        nodes_in_tier = [n for n in graph.nodes() if graph.nodes[n]["tier"] == tier]
        sizes = [200 + graph.nodes[n]["sla_criticality"] * 150 for n in nodes_in_tier]
        nx.draw_networkx_nodes(
            graph, pos, nodelist=nodes_in_tier, ax=ax,
            node_color=color, node_size=sizes, alpha=0.85,
        )

    # Labels
    nx.draw_networkx_labels(
        graph, pos, ax=ax,
        font_size=7, font_weight="bold",
        verticalalignment="center",
    )

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=t.capitalize()) for t, c in TIER_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9)

    ax.set_title("Microservice Dependency Graph", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved topology visualization to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("Building Synthetic Microservice System Model")
    print("=" * 60)

    # Build graph
    services = build_service_catalog()
    edges = build_dependency_edges()
    graph = build_graph(services, edges)

    print(f"\nSystem topology:")
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")
    for tier in TIER_COLORS:
        count = sum(1 for n in graph.nodes() if graph.nodes[n]["tier"] == tier)
        print(f"  {tier:>10s}: {count} services")

    # Save graph
    save_graph(graph, DATA_DIR / "system_graph.json")

    # Generate and save baseline metrics
    print("\nGenerating baseline metrics (100 time points per service)...")
    metrics_df = generate_baseline_metrics(graph)
    save_baseline_metrics(metrics_df, DATA_DIR / "baseline_metrics.csv")

    # Visualize
    print("\nRendering topology visualization...")
    visualize_graph(graph, PLOTS_DIR / "system_topology.png")

    print("\n-- Done. Run src/01_failure_simulation.py next. --")


if __name__ == "__main__":
    main()
