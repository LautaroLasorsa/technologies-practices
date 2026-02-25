"""Train a RandomForest model to predict blast radius from graph features.

Instead of running expensive Monte Carlo simulations every time the system
changes, we extract structural features from the graph (centrality measures,
topology metrics, service metadata) and train an ML model to predict blast
radius directly.

Outputs:
- data/node_features.csv -- extracted features per node
- models/blast_radius_model.pkl -- trained RandomForest model
- plots/predicted_vs_actual.png -- scatter plot of model accuracy
- plots/feature_importances.png -- bar chart of feature importances
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
PLOTS_DIR = Path(__file__).parent.parent / "plots"

SEED = 42

TIER_ENCODING = {
    "frontend": 0,
    "gateway": 1,
    "backend": 2,
    "data": 3,
    "external": 4,
}


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_graph(path: Path) -> nx.DiGraph:
    """Load system graph from JSON."""
    with open(path) as f:
        data = json.load(f)
    return nx.node_link_graph(data)


def load_blast_radii(path: Path) -> pd.DataFrame:
    """Load blast radius results from CSV."""
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# TODO(human): Feature extraction
# ---------------------------------------------------------------------------

def extract_node_features(graph: nx.DiGraph, node: str) -> dict:
    """Extract graph-structural and metadata features for a single node.

    Features to compute:

    Graph centrality features (use networkx algorithms):
    - in_degree: number of incoming edges (services that call this node)
    - out_degree: number of outgoing edges (services this node calls)
    - betweenness_centrality: fraction of all-pairs shortest paths that pass
      through this node. High betweenness = "bridge" node whose failure
      disconnects parts of the graph. Use nx.betweenness_centrality(graph).
    - pagerank: recursive importance score -- a node is important if important
      nodes point to it. Use nx.pagerank(graph).
    - closeness_centrality: inverse of the mean shortest-path distance to all
      reachable nodes. High closeness = failures propagate quickly from here.
      Use nx.closeness_centrality(graph).

    Topology features:
    - num_downstream_services: total number of nodes reachable from this node
      by following outgoing edges (transitive closure). Use
      nx.descendants(graph, node). This counts how many services could be
      indirectly affected if this node starts misbehaving.

    Edge-derived features:
    - avg_edge_propagation_prob: mean failure_propagation_probability across
      all INCOMING edges (edges from predecessors to this node). If no
      incoming edges, use 0.0.

    Service metadata features (from graph.nodes[node]):
    - has_circuit_breaker: 1 if True, 0 if False
    - replicas: number of replicas (integer)
    - sla_criticality: criticality score (1-5)
    - tier_encoded: integer encoding of the tier (use TIER_ENCODING dict)

    Return a dict with all feature names as keys and their values.

    Performance note: betweenness_centrality, pagerank, and closeness_centrality
    are computed over the ENTIRE graph. To avoid recomputing for every node,
    compute them once outside this function and pass them in, OR compute once
    and cache. For this exercise, computing inside is fine (small graph), but
    be aware that in production you'd precompute these.

    Args:
        graph: The microservice dependency graph.
        node: Name of the node to extract features for.

    Returns:
        Dict mapping feature names to values.
    """
    # TODO(human): Implement feature extraction. Start with the simple ones
    # (in_degree, out_degree, metadata) and then add the centrality measures.
    # For centrality functions, note that they return a dict mapping ALL nodes
    # to their centrality values -- index into it with [node].
    raise NotImplementedError("Implement extract_node_features")


# ---------------------------------------------------------------------------
# TODO(human): Model training
# ---------------------------------------------------------------------------

def train_blast_radius_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[RandomForestRegressor, dict]:
    """Train a RandomForestRegressor to predict blast radius from features.

    Steps:
    1. Define a parameter grid to search over:
       - n_estimators: [50, 100, 200]
       - max_depth: [3, 5, None]  (None = unlimited depth)
       Use nested loops or itertools.product -- no need for GridSearchCV,
       just manually try each combination with cross_val_score.

    2. For each (n_estimators, max_depth) combo:
       - Create a RandomForestRegressor(n_estimators=..., max_depth=...,
         random_state=SEED)
       - Evaluate using cross_val_score with cv=min(5, len(X)) folds and
         scoring="neg_mean_squared_error"
       - Record mean score

    3. Select the best hyperparameters (highest mean cross-val score).

    4. Train a final model on ALL data with the best hyperparameters.

    5. Print feature importances: for each feature, print its name and
       importance (model.feature_importances_), sorted descending.
       This reveals WHY certain nodes are high-risk.

    Return:
    - The trained model (RandomForestRegressor)
    - A dict with:
        "best_n_estimators": int,
        "best_max_depth": int or None,
        "best_cv_score": float (mean neg MSE, will be negative),
        "feature_importances": dict mapping feature_name -> importance

    Args:
        X: Feature matrix (DataFrame with named columns).
        y: Target vector (mean_blast_radius per node).

    Returns:
        Tuple of (trained_model, metrics_dict).
    """
    # TODO(human): Implement the hyperparameter search and model training.
    # Remember: cross_val_score with neg_mean_squared_error returns NEGATIVE
    # values (higher = better). Use min(5, len(X)) for cv to handle small
    # datasets gracefully.
    raise NotImplementedError("Implement train_blast_radius_model")


# ---------------------------------------------------------------------------
# Orchestration (scaffolded)
# ---------------------------------------------------------------------------

def extract_all_features(graph: nx.DiGraph) -> pd.DataFrame:
    """Extract features for every node in the graph."""
    records = []
    for node in sorted(graph.nodes()):
        features = extract_node_features(graph, node)
        features["node"] = node
        records.append(features)
    df = pd.DataFrame(records)
    # Move 'node' to first column
    cols = ["node"] + [c for c in df.columns if c != "node"]
    return df[cols]


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """Scatter plot of predicted vs actual blast radius."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.7, s=80, edgecolors="black", linewidth=0.5)

    # Perfect prediction line
    lims = [min(min(y_true), min(y_pred)) - 0.5, max(max(y_true), max(y_pred)) + 0.5]
    ax.plot(lims, lims, "r--", alpha=0.5, label="Perfect prediction")

    ax.set_xlabel("Actual Mean Blast Radius", fontsize=12)
    ax.set_ylabel("Predicted Mean Blast Radius", fontsize=12)
    ax.set_title("Blast Radius: Predicted vs Actual", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved predicted-vs-actual plot to {path}")


def plot_feature_importances(importances: dict, path: Path) -> None:
    """Bar chart of feature importances."""
    path.parent.mkdir(parents=True, exist_ok=True)

    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(names)), values, color="#2196F3", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title("RandomForest Feature Importances for Blast Radius", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved feature importances plot to {path}")


def main() -> None:
    print("=" * 60)
    print("Blast Radius Prediction Model")
    print("=" * 60)

    # Load data
    graph_path = DATA_DIR / "system_graph.json"
    blast_path = DATA_DIR / "blast_radii.csv"
    if not graph_path.exists() or not blast_path.exists():
        print("ERROR: Required data files not found.")
        print("  Run src/00_system_model.py and src/01_failure_simulation.py first.")
        return

    graph = load_graph(graph_path)
    blast_df = load_blast_radii(blast_path)
    print(f"Loaded graph ({graph.number_of_nodes()} nodes) and blast radii ({len(blast_df)} rows)")

    # Extract features
    print("\nExtracting node features...")
    features_df = extract_all_features(graph)
    features_df.to_csv(DATA_DIR / "node_features.csv", index=False)
    print(f"Saved features to {DATA_DIR / 'node_features.csv'}")
    print(f"Features: {[c for c in features_df.columns if c != 'node']}")

    # Merge features with blast radius targets
    merged = features_df.merge(blast_df[["node", "mean_affected"]], on="node")
    feature_cols = [c for c in features_df.columns if c != "node"]
    X = merged[feature_cols]
    y = merged["mean_affected"]

    print(f"\nTraining data: {len(X)} samples, {len(feature_cols)} features")

    # Train model
    print("\nTraining RandomForest model with cross-validation...")
    model, metrics = train_blast_radius_model(X, y)

    print(f"\nBest hyperparameters:")
    print(f"  n_estimators: {metrics['best_n_estimators']}")
    print(f"  max_depth: {metrics['best_max_depth']}")
    print(f"  CV score (neg MSE): {metrics['best_cv_score']:.4f}")

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "blast_radius_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved model to {model_path}")

    # Also save feature column names for later use
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    # Plot results
    y_pred = model.predict(X)
    plot_predicted_vs_actual(y.values, y_pred, PLOTS_DIR / "predicted_vs_actual.png")
    plot_feature_importances(metrics["feature_importances"], PLOTS_DIR / "feature_importances.png")

    print("\n-- Done. Run src/03_experiment_prioritizer.py next. --")


if __name__ == "__main__":
    main()
