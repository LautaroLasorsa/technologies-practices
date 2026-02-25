"""Monte Carlo cascading failure simulation.

Simulates what happens when a service fails: its dependents may also fail,
depending on edge propagation probabilities and circuit breaker presence.
Running many simulations gives a distribution of blast radius outcomes.

Outputs:
- data/blast_radii.csv -- blast radius metrics per node
- plots/blast_radius_distribution.png -- ranked bar chart of mean blast radii
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
PLOTS_DIR = Path(__file__).parent.parent / "plots"

NUM_SIMULATIONS = 1000
SEED = 42


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_graph(path: Path) -> nx.DiGraph:
    """Load system graph from JSON."""
    with open(path) as f:
        data = json.load(f)
    return nx.node_link_graph(data)


# ---------------------------------------------------------------------------
# TODO(human): Monte Carlo failure simulation
# ---------------------------------------------------------------------------

def simulate_single_failure(
    graph: nx.DiGraph,
    failed_node: str,
    num_simulations: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, list[int], Counter]:
    """Simulate cascading failure starting from a single node.

    This is a Monte Carlo simulation: run it many times with different random
    draws to build a distribution of outcomes. Each simulation works like this:

    1. Mark `failed_node` as failed (set of failed nodes starts with just it).
    2. BFS propagation loop:
       - For each currently-failed node, look at its PREDECESSORS in the graph
         (nodes that DEPEND ON it -- i.e., edges pointing FROM predecessor TO
         the failed node). These are the services that call the failed service.
       - For each predecessor that is not yet failed, check if the failure
         propagates: draw from Bernoulli(p) where p is the edge's
         `failure_propagation_probability`.
       - If the predecessor has `has_circuit_breaker=True`, multiply p by 0.1
         (circuit breakers dramatically reduce propagation).
       - If the predecessor has `replicas > 1`, multiply p by (1/replicas)
         (redundancy reduces single-point failure probability).
       - Collect all newly failed nodes.
    3. Repeat step 2 until no new nodes fail in a round (fixed point reached).
    4. Record len(failed_nodes) - 1 as the "affected count" (excluding the
       originally failed node).

    Run the above for `num_simulations` iterations. Return:
    - mean_affected: average number of affected nodes (excluding the initial)
    - affected_counts: list of affected counts per simulation (for distribution)
    - failure_frequency: Counter mapping node_name -> number of simulations
      in which that node was affected (useful for identifying most vulnerable
      downstream services)

    Hints:
    - Use `graph.predecessors(node)` to find nodes that depend on a given node
    - Use `graph.edges[pred, node]` to access edge attributes
    - Use `graph.nodes[pred]` to access node attributes
    - Use a set for `failed_nodes` and a list/deque for the BFS frontier

    Args:
        graph: The microservice dependency graph.
        failed_node: Name of the node to initially fail.
        num_simulations: Number of Monte Carlo iterations.
        rng: NumPy random generator (for reproducibility).

    Returns:
        Tuple of (mean_affected, affected_counts, failure_frequency).
    """
    # TODO(human): Implement the Monte Carlo cascading failure simulation.
    # Start with the outer loop over num_simulations, then implement the
    # inner BFS propagation. Remember: predecessors of a failed node are
    # the services that CALL it -- those are the ones at risk.
    raise NotImplementedError("Implement simulate_single_failure")


def compute_blast_radius(
    graph: nx.DiGraph,
    failed_node: str,
    num_simulations: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compute comprehensive blast radius metrics for a single node failure.

    Uses simulate_single_failure() to run the Monte Carlo simulation, then
    computes summary statistics from the resulting distribution.

    Return a dict with keys:
    - "node": the failed node name
    - "mean_affected": mean number of nodes affected across simulations
    - "std_affected": standard deviation of affected counts
    - "max_affected": maximum cascade size observed
    - "p95_affected": 95th percentile of affected counts (use np.percentile)
    - "critical_services_affected": count of UNIQUE nodes with
      sla_criticality >= 4 that appeared in ANY simulation (use the
      failure_frequency counter from simulate_single_failure)
    - "most_affected_nodes": list of the top-3 most frequently affected node
      names (from the failure_frequency counter)

    This function is a thin wrapper around simulate_single_failure -- the
    heavy lifting is already done, you just need to aggregate.

    Args:
        graph: The microservice dependency graph.
        failed_node: Name of the node to initially fail.
        num_simulations: Number of Monte Carlo iterations.
        rng: NumPy random generator (for reproducibility).

    Returns:
        Dict of blast radius summary metrics.
    """
    # TODO(human): Call simulate_single_failure, then compute the summary
    # statistics. Use np.percentile for P95. For critical_services_affected,
    # check which nodes in the failure_frequency counter have
    # sla_criticality >= 4 in the graph.
    raise NotImplementedError("Implement compute_blast_radius")


# ---------------------------------------------------------------------------
# Orchestration (scaffolded)
# ---------------------------------------------------------------------------

def run_all_simulations(graph: nx.DiGraph) -> pd.DataFrame:
    """Run blast radius simulation for every node in the graph."""
    rng = np.random.default_rng(SEED)
    results = []

    print(f"\nRunning {NUM_SIMULATIONS} simulations per node ({graph.number_of_nodes()} nodes)...")
    for i, node in enumerate(sorted(graph.nodes())):
        metrics = compute_blast_radius(graph, node, NUM_SIMULATIONS, rng)
        results.append(metrics)
        tier = graph.nodes[node]["tier"]
        print(
            f"  [{i+1:2d}/{graph.number_of_nodes()}] {node:30s} "
            f"(tier={tier:10s}) -> mean={metrics['mean_affected']:.2f}, "
            f"P95={metrics['p95_affected']:.1f}, "
            f"critical={metrics['critical_services_affected']}"
        )

    df = pd.DataFrame(results)
    df = df.sort_values("mean_affected", ascending=False).reset_index(drop=True)
    return df


def save_results(df: pd.DataFrame, path: Path) -> None:
    """Save blast radius results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert list columns to strings for CSV
    df_save = df.copy()
    if "most_affected_nodes" in df_save.columns:
        df_save["most_affected_nodes"] = df_save["most_affected_nodes"].apply(
            lambda x: "|".join(x) if isinstance(x, list) else str(x)
        )
    df_save.to_csv(path, index=False)
    print(f"\nSaved blast radius results to {path}")


def plot_blast_radii(df: pd.DataFrame, graph: nx.DiGraph, path: Path) -> None:
    """Plot ranked blast radius bar chart."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    tier_colors = {
        "frontend": "#4CAF50",
        "gateway": "#FF9800",
        "backend": "#2196F3",
        "data": "#9C27B0",
        "external": "#F44336",
    }

    colors = [tier_colors.get(graph.nodes[n]["tier"], "#999999") for n in df["node"]]
    bars = ax.barh(range(len(df)), df["mean_affected"], color=colors, alpha=0.8)

    # Error bars for std
    ax.barh(
        range(len(df)), df["std_affected"],
        left=df["mean_affected"], color=colors, alpha=0.3,
    )

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["node"], fontsize=8)
    ax.set_xlabel("Mean Affected Nodes (+ 1 std)", fontsize=11)
    ax.set_title("Blast Radius by Service (Monte Carlo Simulation)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    # Legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=t.capitalize()) for t, c in tier_colors.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved blast radius plot to {path}")


def main() -> None:
    print("=" * 60)
    print("Monte Carlo Cascading Failure Simulation")
    print("=" * 60)

    graph_path = DATA_DIR / "system_graph.json"
    if not graph_path.exists():
        print(f"ERROR: {graph_path} not found. Run src/00_system_model.py first.")
        return

    graph = load_graph(graph_path)
    print(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Run simulations
    df = run_all_simulations(graph)

    # Display top-5 most impactful failures
    print("\n--- Top 5 Highest Blast Radius ---")
    for _, row in df.head(5).iterrows():
        print(
            f"  {row['node']:30s}  mean={row['mean_affected']:.2f}  "
            f"P95={row['p95_affected']:.1f}  max={row['max_affected']}  "
            f"critical={row['critical_services_affected']}  "
            f"cascade_to={row['most_affected_nodes']}"
        )

    # Save and plot
    save_results(df, DATA_DIR / "blast_radii.csv")
    plot_blast_radii(df, graph, PLOTS_DIR / "blast_radius_distribution.png")

    print("\n-- Done. Run src/02_blast_radius_model.py next. --")


if __name__ == "__main__":
    main()
