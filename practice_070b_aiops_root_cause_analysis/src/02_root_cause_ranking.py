#!/usr/bin/env python3
"""Exercise 2: Root Cause Ranking with Graph Centrality Measures.

This script ranks candidate root causes using Personalized PageRank and
betweenness centrality -- the core algorithms used by MicroRCA, TraceRank,
and similar production RCA systems.

Concepts practiced:
  - Personalized PageRank with anomaly-biased personalization vector
  - Betweenness centrality on anomalous subgraph
  - Score fusion: combining multiple ranking signals
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
    """Compute per-service anomaly z-scores (same as Exercise 1)."""
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
# Exercise 2a: Personalized PageRank for RCA
# ---------------------------------------------------------------------------


def rank_by_pagerank(
    graph: nx.DiGraph,
    anomaly_scores: dict[str, float],
    alpha: float = 0.85,
    personalization_weight: float = 1.0,
) -> list[tuple[str, float]]:
    """Rank root cause candidates using Personalized PageRank.

    This implements the core algorithm from MicroRCA and TraceRank:
    run PageRank on the service dependency graph with a personalization
    vector biased toward anomalous nodes. The intuition is that a
    random walker who restarts at anomalous nodes will accumulate
    at the nodes that INFLUENCE (are depended upon by) many anomalous
    services -- i.e., the likely root causes.

    How Personalized PageRank works:
      Standard PageRank simulates a "random surfer" who follows edges
      and occasionally (with probability 1-alpha) restarts at a random
      node. Personalized PageRank changes WHERE the surfer restarts:
      instead of restarting uniformly at any node, they restart at
      nodes specified by the personalization vector, with probability
      proportional to the vector's values.

    For RCA:
      - Anomalous nodes get HIGH personalization weight (the surfer
        restarts at anomalous services frequently)
      - Normal nodes get ZERO or near-zero weight
      - The surfer follows dependency edges, so they naturally flow
        toward upstream services (callees)
      - After convergence, nodes with high PageRank are those that
        many anomalous downstream services depend on -- likely root causes

    Algorithm:
      1. Build the personalization vector:
         - For each node, if anomaly_score > 0, set personalization to
           anomaly_score * personalization_weight
         - For nodes with anomaly_score <= 0, set personalization to
           a small epsilon (e.g., 0.001) to avoid zero-weight nodes
           (PageRank requires all nodes to have some weight if they
           exist in the graph, otherwise you get numerical issues)
      2. Call nx.pagerank(graph, alpha=alpha, personalization=personalization)
         - alpha (damping factor): probability of following an edge vs
           restarting. Higher alpha = more exploration of the graph.
           Default 0.85 is standard. For RCA, 0.85-0.90 works well.
      3. Sort results by PageRank score descending
      4. Return list of (node, score) tuples

    Parameters:
      graph: Service dependency DiGraph (edge A->B means A calls B)
      anomaly_scores: Dict mapping service -> anomaly z-score
      alpha: Damping factor for PageRank (0.85 = 85% chance of following
             an edge, 15% chance of restarting at a personalization node)
      personalization_weight: Multiplier for anomaly scores in the
             personalization vector. Higher = stronger bias toward
             anomalous nodes as restart points.

    Returns:
      List of (service_name, pagerank_score) tuples, sorted by score
      descending. The top-ranked service is the most likely root cause.

    Example:
      >>> scores = {"svc-a": 5.0, "svc-b": 0.1, "svc-c": 8.0}
      >>> ranking = rank_by_pagerank(graph, scores, alpha=0.85)
      >>> ranking[0]  # top candidate
      ("svc-c-dependency", 0.23)

    Hints:
      - nx.pagerank(G, alpha=alpha, personalization=pers_dict) where
        pers_dict maps node -> weight (will be normalized internally)
      - Make sure ALL graph nodes appear in the personalization dict
      - Use max(score, epsilon) to avoid zero weights for normal nodes
      - The graph direction matters: PageRank flows along edges, so
        rank accumulates at nodes with many incoming edges (callees
        that many callers depend on)
    """
    raise NotImplementedError("TODO(human): Implement rank_by_pagerank")


# ---------------------------------------------------------------------------
# Exercise 2b: Betweenness centrality on anomalous subgraph
# ---------------------------------------------------------------------------


def rank_by_centrality(
    graph: nx.DiGraph,
    anomalous_nodes: set[str],
) -> list[tuple[str, float]]:
    """Rank candidates by betweenness centrality on the anomalous subgraph.

    Betweenness centrality measures how often a node appears on shortest
    paths between other nodes. In the context of RCA, a node with high
    betweenness among anomalous services is a "propagation hub" -- many
    anomaly paths flow through it, making it a likely root cause or
    critical intermediary.

    Algorithm:
      1. Build the anomalous subgraph:
         - Include all anomalous nodes
         - Include 1-hop neighbors of anomalous nodes (even if normal),
           because they provide structural context. Use
           graph.predecessors(n) and graph.successors(n) to find neighbors.
         - Extract the subgraph induced by this expanded node set:
           subgraph = graph.subgraph(expanded_nodes).copy()
      2. Compute betweenness centrality on the subgraph:
         - Use nx.betweenness_centrality(subgraph, weight='calls_per_sec')
         - Using calls_per_sec as weight means high-traffic paths are
           considered "shorter" (more important). Note: betweenness
           interprets weight as distance/cost, so you may want to use
           1/calls_per_sec as the weight, or simply omit weight for
           unweighted betweenness. Think about which makes more sense
           for RCA: do you want high-traffic edges to be "closer" or
           "farther"?
         - Actually, for RCA, high-traffic edges propagate anomalies
           more effectively, so you want them to have LOWER weight
           (closer), meaning use weight = 1/calls_per_sec.
      3. Filter results to only include anomalous nodes (not their
         normal neighbors)
      4. Sort by betweenness centrality descending
      5. Return list of (node, centrality_score) tuples

    Parameters:
      graph: Service dependency DiGraph with 'calls_per_sec' edge attribute
      anomalous_nodes: Set of service names that are anomalous

    Returns:
      List of (service_name, betweenness_centrality) tuples for anomalous
      nodes only, sorted by centrality descending. The top-ranked node is
      the most central propagation hub.

    Example:
      >>> anomalous = {"api-gateway", "order-service", "inventory-db"}
      >>> ranking = rank_by_centrality(graph, anomalous)
      >>> ranking[0]
      ("order-service", 0.45)  # sits on many shortest paths

    Hints:
      - To get 1-hop neighbors: set().union(*(set(graph.predecessors(n)) |
        set(graph.successors(n)) for n in anomalous_nodes))
      - graph.subgraph(nodes).copy() creates an independent subgraph
      - For weighted betweenness with inverse frequency:
        first set edge attribute 'inv_freq' = 1/calls_per_sec,
        then use weight='inv_freq'
      - If the subgraph is disconnected, betweenness still works
        (disconnected components just don't contribute to each other's
        centrality)
    """
    raise NotImplementedError("TODO(human): Implement rank_by_centrality")


# ---------------------------------------------------------------------------
# Exercise 2c: Combined ranking
# ---------------------------------------------------------------------------


def combined_ranking(
    pagerank_scores: list[tuple[str, float]],
    centrality_scores: list[tuple[str, float]],
    weights: tuple[float, float] = (0.6, 0.4),
) -> list[tuple[str, float]]:
    """Combine PageRank and centrality rankings into a final score.

    In practice, no single RCA signal is perfect:
      - PageRank captures "upstream influence on anomalous nodes"
      - Betweenness captures "structural bottleneck position"
    Combining them yields a more robust ranking than either alone.

    Algorithm:
      1. Normalize each set of scores to the [0, 1] range:
         - For each list of (node, score) tuples, find min and max scores
         - normalized_score = (score - min) / (max - min)
         - Handle the edge case where max == min (set all normalized to 1.0)
      2. Build a combined score for each node:
         - combined = weights[0] * normalized_pagerank + weights[1] * normalized_centrality
         - If a node appears in one ranking but not the other, use 0.0
           for the missing score
      3. Sort by combined score descending
      4. Return list of (node, combined_score) tuples

    Parameters:
      pagerank_scores: List of (service_name, pagerank_score), already sorted
      centrality_scores: List of (service_name, centrality_score), already sorted
      weights: Tuple (pagerank_weight, centrality_weight). Should sum to 1.0.
               Default (0.6, 0.4) gives more weight to PageRank because it
               directly incorporates anomaly magnitude, while centrality is
               purely structural.

    Returns:
      List of (service_name, combined_score) tuples, sorted by score descending.
      Scores are in [0, 1].

    Example:
      >>> pr = [("svc-a", 0.3), ("svc-b", 0.2), ("svc-c", 0.1)]
      >>> bc = [("svc-b", 0.5), ("svc-a", 0.3), ("svc-c", 0.1)]
      >>> combined = combined_ranking(pr, bc, weights=(0.6, 0.4))
      >>> combined[0]
      ("svc-a", 0.8)  # high in both rankings

    Hints:
      - Convert lists to dicts for easy lookup: dict(pagerank_scores)
      - Use set union of both dicts' keys to cover all nodes
      - min-max normalization: (x - min) / (max - min), with max != min guard
      - The weights tuple lets you experiment: try (0.5, 0.5) or (0.7, 0.3)
    """
    raise NotImplementedError("TODO(human): Implement combined_ranking")


# ---------------------------------------------------------------------------
# Evaluation (scaffolded)
# ---------------------------------------------------------------------------


def evaluate_ranking(
    ranking: list[tuple[str, float]],
    true_root_causes: set[str],
    top_k: int = 5,
) -> dict[str, float]:
    """Evaluate a ranking against ground truth root causes."""
    top_k_nodes = {node for node, _ in ranking[:top_k]}

    # Precision@k: what fraction of top-k are true root causes?
    precision = len(top_k_nodes & true_root_causes) / top_k if top_k > 0 else 0.0

    # Recall@k: what fraction of true root causes are in top-k?
    recall = len(top_k_nodes & true_root_causes) / len(true_root_causes) if true_root_causes else 0.0

    # Mean Reciprocal Rank: 1/rank of first true root cause
    mrr = 0.0
    for rank, (node, _) in enumerate(ranking, 1):
        if node in true_root_causes:
            mrr = 1.0 / rank
            break

    return {"precision_at_k": precision, "recall_at_k": recall, "mrr": mrr}


# ---------------------------------------------------------------------------
# Visualization (scaffolded)
# ---------------------------------------------------------------------------


def visualize_ranking(
    graph: nx.DiGraph,
    ranking: list[tuple[str, float]],
    true_root_causes: set[str],
    title: str,
    output_path: Path,
) -> None:
    """Visualize graph with nodes colored by ranking score."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    pos = nx.spring_layout(graph, seed=42, k=2.0, iterations=100)

    # Build score map
    score_map = dict(ranking)
    max_score = max(score_map.values()) if score_map else 1.0

    # Node colors: intensity by score
    node_colors = []
    for n in graph.nodes:
        score = score_map.get(n, 0.0)
        if score > 0 and max_score > 0:
            intensity = score / max_score
            # Orange-red gradient
            node_colors.append((1.0, 1.0 - intensity * 0.7, 0.3 - intensity * 0.3))
        else:
            node_colors.append("#D5D5D5")

    # Draw graph
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#DDDDDD", arrows=True,
                           arrowsize=10, arrowstyle="-|>", width=1.0)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=700,
                           edgecolors="#333333", linewidths=1.0)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=6, font_weight="bold")

    # Highlight true root causes
    true_rc_in_graph = [n for n in true_root_causes if n in graph.nodes]
    if true_rc_in_graph:
        nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=true_rc_in_graph,
                               node_size=1100, node_color="none",
                               edgecolors="#00AA00", linewidths=4.0)

    # Highlight top-3 ranked
    top3 = [n for n, _ in ranking[:3]]
    for i, node in enumerate(top3):
        if node in graph.nodes:
            nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=[node],
                                   node_size=900 - i * 50, node_color="none",
                                   edgecolors="#0000FF", linewidths=3.0 - i * 0.5)

    ax.set_title(f"{title}\n(blue border = top-3 ranked, green border = true root cause)", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ranking plot: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  Exercise 2: Root Cause Ranking with Graph Centrality")
    print("=" * 60)

    # Load data
    print("\n[1/7] Loading graph and metrics...")
    graph = load_graph()
    metrics_df = load_metrics()
    ground_truth = load_ground_truth()
    anomaly_start = ground_truth["anomaly_start_timestep"]
    true_root_causes = set(ground_truth["root_causes"])

    # Compute anomaly scores
    print("\n[2/7] Computing anomaly scores...")
    anomaly_scores = compute_anomaly_scores(metrics_df, anomaly_start)
    threshold = 2.0
    anomalous_nodes = {svc for svc, score in anomaly_scores.items() if score >= threshold}
    print(f"  Anomalous nodes ({len(anomalous_nodes)}): {sorted(anomalous_nodes)}")

    # PageRank ranking
    print("\n[3/7] Computing Personalized PageRank ranking...")
    pr_ranking = rank_by_pagerank(graph, anomaly_scores, alpha=0.85)

    print(f"\n  {'Rank':<6} {'Service':<28} {'PageRank Score':>15}")
    print(f"  {'-'*6} {'-'*28} {'-'*15}")
    for rank, (svc, score) in enumerate(pr_ranking[:10], 1):
        marker = " *** ROOT CAUSE" if svc in true_root_causes else ""
        print(f"  {rank:<6} {svc:<28} {score:>15.6f}{marker}")

    pr_eval = evaluate_ranking(pr_ranking, true_root_causes, top_k=3)
    print(f"\n  PageRank evaluation (top-3): P={pr_eval['precision_at_k']:.2f}, "
          f"R={pr_eval['recall_at_k']:.2f}, MRR={pr_eval['mrr']:.2f}")

    # Betweenness centrality ranking
    print("\n[4/7] Computing betweenness centrality ranking...")
    bc_ranking = rank_by_centrality(graph, anomalous_nodes)

    print(f"\n  {'Rank':<6} {'Service':<28} {'Betweenness':>15}")
    print(f"  {'-'*6} {'-'*28} {'-'*15}")
    for rank, (svc, score) in enumerate(bc_ranking[:10], 1):
        marker = " *** ROOT CAUSE" if svc in true_root_causes else ""
        print(f"  {rank:<6} {svc:<28} {score:>15.6f}{marker}")

    bc_eval = evaluate_ranking(bc_ranking, true_root_causes, top_k=3)
    print(f"\n  Betweenness evaluation (top-3): P={bc_eval['precision_at_k']:.2f}, "
          f"R={bc_eval['recall_at_k']:.2f}, MRR={bc_eval['mrr']:.2f}")

    # Combined ranking
    print("\n[5/7] Computing combined ranking...")
    final_ranking = combined_ranking(pr_ranking, bc_ranking, weights=(0.6, 0.4))

    print(f"\n  {'Rank':<6} {'Service':<28} {'Combined Score':>15}")
    print(f"  {'-'*6} {'-'*28} {'-'*15}")
    for rank, (svc, score) in enumerate(final_ranking[:10], 1):
        marker = " *** ROOT CAUSE" if svc in true_root_causes else ""
        print(f"  {rank:<6} {svc:<28} {score:>15.6f}{marker}")

    final_eval = evaluate_ranking(final_ranking, true_root_causes, top_k=3)
    print(f"\n  Combined evaluation (top-3): P={final_eval['precision_at_k']:.2f}, "
          f"R={final_eval['recall_at_k']:.2f}, MRR={final_eval['mrr']:.2f}")

    # Comparison summary
    print("\n[6/7] Method comparison:")
    print(f"\n  {'Method':<20} {'Precision@3':>12} {'Recall@3':>10} {'MRR':>8}")
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*8}")
    for name, ev in [("PageRank", pr_eval), ("Betweenness", bc_eval), ("Combined", final_eval)]:
        print(f"  {name:<20} {ev['precision_at_k']:>12.2f} {ev['recall_at_k']:>10.2f} {ev['mrr']:>8.2f}")

    # Visualize
    print("\n[7/7] Visualizing rankings...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    visualize_ranking(graph, pr_ranking, true_root_causes,
                      "Personalized PageRank Ranking", PLOTS_DIR / "ranking_pagerank.png")
    visualize_ranking(graph, final_ranking, true_root_causes,
                      "Combined Ranking (PageRank + Betweenness)", PLOTS_DIR / "ranking_combined.png")

    print()


if __name__ == "__main__":
    main()
