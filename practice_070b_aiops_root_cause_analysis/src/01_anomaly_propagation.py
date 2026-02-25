#!/usr/bin/env python3
"""Exercise 1: Anomaly Propagation Tracing through Dependency Edges.

This script traces anomaly propagation backward through the service dependency
graph to recover the "cascade story" -- which service infected which.

Concepts practiced:
  - Reverse graph traversal (BFS/DFS on predecessors)
  - Anomaly score computation (comparing normal vs anomaly periods)
  - Propagation chain scoring using temporal and structural signals
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


def compute_anomaly_scores(
    metrics_df: pd.DataFrame,
    anomaly_start: int,
) -> dict[str, float]:
    """Compute a per-service anomaly score based on latency deviation.

    For each service, compare mean latency in the anomaly period vs the normal
    period. The anomaly score is the z-score: (anomaly_mean - normal_mean) / normal_std.
    Services with higher z-scores have more abnormal latency.
    """
    normal = metrics_df[metrics_df["timestamp"] < anomaly_start]
    anomaly = metrics_df[metrics_df["timestamp"] >= anomaly_start]

    scores: dict[str, float] = {}
    for svc in metrics_df["service"].unique():
        normal_lat = normal[normal["service"] == svc]["latency_ms"]
        anomaly_lat = anomaly[anomaly["service"] == svc]["latency_ms"]

        normal_mean = normal_lat.mean()
        normal_std = normal_lat.std()
        anomaly_mean = anomaly_lat.mean()

        if normal_std > 0:
            scores[svc] = (anomaly_mean - normal_mean) / normal_std
        else:
            scores[svc] = 0.0

    return scores


# ---------------------------------------------------------------------------
# Exercise 1a: Trace anomaly propagation
# ---------------------------------------------------------------------------


def trace_anomaly_propagation(
    graph: nx.DiGraph,
    anomaly_scores: dict[str, float],
    threshold: float = 2.0,
) -> list[list[str]]:
    """Trace backward from anomalous leaf nodes to find propagation chains.

    Given the service dependency DAG and per-node anomaly scores, this function
    identifies propagation chains -- paths from symptomatic (downstream) services
    back through the dependency edges to the likely root cause (upstream).

    In the dependency graph, an edge A -> B means "A calls B" (A depends on B).
    When B becomes slow, A experiences increased latency because it waits for B.
    So anomaly propagation flows in the SAME direction as dependency edges:
    a fault in B causes symptoms in A.

    To trace BACKWARD to the root cause, you follow edges in the FORWARD
    direction (from caller to callee): start at a symptomatic frontend node
    and follow its outgoing edges toward the backend/data tier.

    Alternatively, think of it as: the root cause is a node whose PREDECESSORS
    (callers) are also anomalous -- the anomaly "bubbles up" from callees
    to callers.

    Algorithm:
      1. Identify anomalous nodes: services with anomaly_score >= threshold.
      2. Find "symptom" nodes: anomalous nodes that have NO anomalous successors
         in the dependency graph (i.e., the anomaly "ends" here -- these are
         leaf-like in the anomaly subgraph). Actually, you want anomalous nodes
         that have no anomalous PREDECESSORS -- meaning they are the "deepest"
         anomalous nodes in the dependency chain (closest to the data tier).
         Wait -- think carefully about the direction:
           - Edge A -> B means A calls B.
           - If B is faulty, A shows symptoms.
           - Predecessors of B are nodes that call B (upstream callers).
           - Successors of B are nodes B calls (downstream dependencies).
           - To find the root cause, start from anomalous nodes with
             NO anomalous SUCCESSORS (no further downstream anomalous deps).
             These are the "leaves" of the anomaly -- likely root causes.
         Actually, for generating chains, start from anomalous nodes with
         no anomalous PREDECESSORS (callers that are also anomalous) --
         these are the outermost symptom nodes. Then trace their successors
         (callees) to find who caused the symptoms.
      3. From each symptom start node, perform BFS/DFS following OUTGOING edges
         (toward callees), only visiting anomalous nodes.
      4. Each path from a symptom node to an anomalous node with no further
         anomalous successors is a propagation chain.
      5. Return all chains found, sorted by length (longest first).

    Parameters:
      graph: Service dependency DiGraph (edge A->B means A calls B)
      anomaly_scores: Dict mapping service_name -> anomaly z-score
      threshold: Minimum anomaly score to consider a node anomalous

    Returns:
      List of propagation chains. Each chain is a list of service names
      from the outermost symptom (caller) to the deepest root cause (callee).
      Example: ["web-frontend", "api-gateway", "order-service", "inventory-db"]
      This means: web-frontend showed symptoms because api-gateway was slow,
      which was slow because order-service was slow, which was slow because
      inventory-db (the root cause) was slow.

    Hints:
      - Use graph.successors(node) to get callees of a node
      - Use graph.predecessors(node) to get callers of a node
      - Consider using a visited set to avoid cycles (if any exist)
      - A chain should contain ONLY anomalous nodes
      - You can use iterative DFS with a stack: [(start_node, [start_node])]
    """
    raise NotImplementedError("TODO(human): Implement trace_anomaly_propagation")


# ---------------------------------------------------------------------------
# Exercise 1b: Score propagation likelihood
# ---------------------------------------------------------------------------


def score_propagation_likelihood(
    graph: nx.DiGraph,
    chain: list[str],
    metrics_df: pd.DataFrame,
    anomaly_start: int,
) -> float:
    """Score how likely a propagation chain represents the real cascade.

    Not all propagation chains are equally plausible. A chain is more likely
    to be the real cascading failure path if:

    1. TEMPORAL ORDERING: The anomaly in upstream nodes (closer to root cause,
       i.e., later in the chain) should appear BEFORE the anomaly in downstream
       nodes (symptoms, i.e., earlier in the chain). Check when each node's
       latency first exceeds its normal range. If the chain's temporal order
       is consistent (each successive node's anomaly start time is earlier or
       equal), the chain is plausible.

    2. EDGE TRAFFIC VOLUME: High-frequency edges (high calls_per_sec) propagate
       anomalies faster and with more impact. Sum the calls_per_sec along the
       chain's edges -- higher total traffic means more propagation force.

    3. ANOMALY MAGNITUDE: Chains where ALL nodes have high anomaly scores are
       more plausible than chains passing through barely-anomalous nodes.
       Compute the minimum anomaly score along the chain -- a chain is only
       as strong as its weakest link.

    Scoring formula (suggested):
      score = temporal_consistency_bonus * traffic_score * min_anomaly_score

    Where:
      - temporal_consistency_bonus: 2.0 if temporal order is correct along
        the entire chain (root cause anomaly starts first), 0.5 otherwise.
        To check: for each node in the chain, compute the first timestep
        where latency exceeds (normal_mean + 2 * normal_std). The last node
        in the chain (root cause) should have the earliest anomaly onset.
      - traffic_score: sum of calls_per_sec for edges along the chain,
        normalized by dividing by 1000 (to keep scale reasonable).
      - min_anomaly_score: minimum anomaly z-score among chain nodes.

    Parameters:
      graph: Service dependency DiGraph
      chain: List of service names from symptom to root cause
             e.g., ["web-frontend", "api-gateway", "inventory-db"]
      metrics_df: Full time-series metrics DataFrame
      anomaly_start: Timestep where anomaly injection begins

    Returns:
      Float score >= 0. Higher = more likely to be a real propagation chain.

    Hints:
      - For anomaly onset detection: for each service, find the first timestep
        >= anomaly_start where latency > (normal_mean + 2*normal_std)
      - Edge data: graph[u][v]['calls_per_sec']
      - If an edge doesn't exist between consecutive chain nodes, that's a
        problem -- the chain shouldn't include non-adjacent nodes
      - Handle edge cases: what if onset can't be detected for a node?
    """
    raise NotImplementedError("TODO(human): Implement score_propagation_likelihood")


# ---------------------------------------------------------------------------
# Visualization (scaffolded)
# ---------------------------------------------------------------------------


def visualize_propagation_chains(
    graph: nx.DiGraph,
    chains: list[list[str]],
    anomaly_scores: dict[str, float],
    threshold: float,
    output_path: Path,
) -> None:
    """Visualize the top propagation chains on the graph."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    pos = nx.spring_layout(graph, seed=42, k=2.0, iterations=100)

    # Color nodes by anomaly score
    max_score = max(anomaly_scores.values()) if anomaly_scores else 1.0
    node_colors = []
    for n in graph.nodes:
        score = anomaly_scores.get(n, 0)
        if score >= threshold:
            # Red intensity proportional to anomaly score
            intensity = min(score / max_score, 1.0)
            node_colors.append((1.0, 1.0 - intensity * 0.8, 1.0 - intensity * 0.8))
        else:
            node_colors.append("#D5D5D5")

    # Draw base graph
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#DDDDDD", arrows=True,
                           arrowsize=10, arrowstyle="-|>", width=1.0)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=700,
                           edgecolors="#333333", linewidths=1.0)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=6, font_weight="bold")

    # Highlight top 3 chains with different colors
    chain_colors = ["#FF0000", "#FF8C00", "#FFD700"]
    for idx, chain in enumerate(chains[:3]):
        color = chain_colors[idx] if idx < len(chain_colors) else "#FF69B4"
        edges_in_chain = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)
                          if graph.has_edge(chain[i], chain[i + 1])]
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=edges_in_chain,
                               edge_color=color, arrows=True, arrowsize=20,
                               arrowstyle="-|>", width=3.0,
                               connectionstyle="arc3,rad=0.1")
        # Mark chain endpoints
        if chain:
            nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=[chain[0]],
                                   node_color=color, node_size=900,
                                   edgecolors="#000", linewidths=2.0)
            nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=[chain[-1]],
                                   node_color=color, node_size=1100,
                                   edgecolors="#000", linewidths=3.0, node_shape="*")

    ax.set_title("Anomaly Propagation Chains (red = anomalous, stars = root cause candidates)", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved propagation plot: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  Exercise 1: Anomaly Propagation Tracing")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading graph and metrics...")
    graph = load_graph()
    metrics_df = load_metrics()
    ground_truth = load_ground_truth()
    anomaly_start = ground_truth["anomaly_start_timestep"]
    true_root_causes = set(ground_truth["root_causes"])

    print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"  Metrics: {len(metrics_df)} rows")
    print(f"  True root causes: {true_root_causes}")

    # Compute anomaly scores
    print("\n[2/5] Computing anomaly scores...")
    anomaly_scores = compute_anomaly_scores(metrics_df, anomaly_start)
    threshold = 2.0

    print(f"\n  {'Service':<28} {'Anomaly Score':>14} {'Anomalous?':>12}")
    print(f"  {'-'*28} {'-'*14} {'-'*12}")
    for svc, score in sorted(anomaly_scores.items(), key=lambda x: -x[1]):
        is_anomalous = "YES" if score >= threshold else "no"
        marker = " *** ROOT CAUSE" if svc in true_root_causes else ""
        print(f"  {svc:<28} {score:>14.2f} {is_anomalous:>12}{marker}")

    # Trace propagation chains
    print("\n[3/5] Tracing anomaly propagation chains...")
    chains = trace_anomaly_propagation(graph, anomaly_scores, threshold)

    print(f"\n  Found {len(chains)} propagation chains:")
    for i, chain in enumerate(chains[:10]):
        root = chain[-1] if chain else "?"
        in_truth = "MATCH" if root in true_root_causes else ""
        print(f"  Chain {i+1}: {' -> '.join(chain)}  [{in_truth}]")

    # Score chains
    print("\n[4/5] Scoring propagation chains...")
    scored = []
    for chain in chains:
        score = score_propagation_likelihood(graph, chain, metrics_df, anomaly_start)
        scored.append((chain, score))
    scored.sort(key=lambda x: -x[1])

    print(f"\n  {'Rank':<6} {'Score':>8} {'Chain (symptom -> root cause)'}")
    print(f"  {'-'*6} {'-'*8} {'-'*50}")
    for rank, (chain, score) in enumerate(scored[:10], 1):
        root = chain[-1] if chain else "?"
        match = " *** CORRECT" if root in true_root_causes else ""
        print(f"  {rank:<6} {score:>8.2f} {' -> '.join(chain)}{match}")

    # Evaluate: does the top-ranked chain point to the true root cause?
    if scored:
        top_root = scored[0][0][-1] if scored[0][0] else None
        if top_root in true_root_causes:
            print(f"\n  SUCCESS: Top chain correctly identifies '{top_root}' as root cause!")
        else:
            print(f"\n  MISS: Top chain points to '{top_root}', true root cause is {true_root_causes}")

    # Visualize
    print("\n[5/5] Visualizing propagation chains...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    visualize_propagation_chains(
        graph,
        [c for c, _ in scored[:3]],
        anomaly_scores,
        threshold,
        PLOTS_DIR / "propagation_chains.png",
    )

    print()


if __name__ == "__main__":
    main()
