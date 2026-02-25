"""Rank chaos experiments by predicted impact and model uncertainty.

Uses the trained RandomForest model to predict blast radius for each node,
and combines the prediction with model uncertainty (variance across trees)
and service criticality to produce a prioritized experiment list.

The core insight: we want to maximize LEARNING per experiment, not just
maximize damage. High-uncertainty experiments teach us the most.

Outputs:
- data/experiment_rankings.csv -- prioritized experiment list
- plots/experiment_scatter.png -- impact vs uncertainty scatter plot
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
PLOTS_DIR = Path(__file__).parent.parent / "plots"

# Mapping from tier to suggested failure types
FAILURE_TYPE_SUGGESTIONS = {
    "frontend": "latency injection (simulate slow rendering)",
    "gateway": "connection limit exhaustion",
    "backend": "process crash (kill -9)",
    "data": "read-only mode / connection pool exhaustion",
    "external": "timeout / DNS failure simulation",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_graph(path: Path) -> nx.DiGraph:
    """Load system graph from JSON."""
    with open(path) as f:
        data = json.load(f)
    return nx.node_link_graph(data)


def load_model(model_path: Path, columns_path: Path) -> tuple:
    """Load trained model and feature column names."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(columns_path) as f:
        feature_columns = json.load(f)
    return model, feature_columns


# ---------------------------------------------------------------------------
# TODO(human): Experiment scoring
# ---------------------------------------------------------------------------

def score_experiment(
    predicted_impact: float,
    uncertainty: float,
    criticality_weight: float,
) -> float:
    """Compute a priority score for a chaos experiment.

    The score balances three factors:
    1. predicted_impact: How many services we expect to be affected (blast
       radius). Higher = more impactful experiment.
    2. uncertainty: How confident the model is in its prediction. High
       uncertainty means we DON'T KNOW what will happen, so running the
       experiment has high LEARNING VALUE (exploration).
    3. criticality_weight: How critical the target service is. Failing a
       payment service teaches us more about real-world risk than failing
       a logging service.

    Formula:
        score = predicted_impact * (1 + uncertainty_bonus) * criticality_weight

    Where uncertainty_bonus normalizes the raw uncertainty into a [0, 1] range:
        uncertainty_bonus = uncertainty / (1 + uncertainty)

    This sigmoid-like normalization prevents extreme uncertainty values from
    dominating the score while still giving a meaningful boost. When
    uncertainty is 0, bonus is 0. When uncertainty is 1, bonus is 0.5.
    When uncertainty is very large, bonus approaches 1.

    This is analogous to the Upper Confidence Bound (UCB) strategy in
    multi-armed bandits: prefer actions with high expected reward OR high
    uncertainty. Over time, as we run experiments and reduce uncertainty,
    the prioritizer naturally shifts from exploration to exploitation.

    Args:
        predicted_impact: Model's predicted mean blast radius for this node.
        uncertainty: Standard deviation of predictions across RandomForest
            trees (high = model disagrees with itself = high learning value).
        criticality_weight: Multiplier based on the service's SLA criticality
            (e.g., criticality 5 -> weight 2.0, criticality 1 -> weight 0.5).

    Returns:
        Priority score (higher = should run this experiment sooner).
    """
    # TODO(human): Implement the scoring formula. Compute uncertainty_bonus
    # as uncertainty / (1 + uncertainty), then combine the three factors.
    # This is a short function but understanding WHY this formula works is
    # the key learning point.
    raise NotImplementedError("Implement score_experiment")


def rank_experiments(
    graph: nx.DiGraph,
    model,
    feature_columns: list[str],
    top_k: int = 10,
) -> list[dict]:
    """Rank chaos experiments by priority score.

    For each node in the graph:
    1. Extract features (reuse extract_node_features from 02_blast_radius_model)
    2. Predict blast radius using the trained model
    3. Compute uncertainty: get predictions from EACH individual tree in the
       RandomForest and compute the standard deviation across trees.
       - Access individual trees via model.estimators_
       - For each tree: tree.predict(X_row) gives one prediction
       - np.std([tree.predict(X_row)[0] for tree in model.estimators_])
       This measures how much the trees DISAGREE -- high disagreement means
       the model is uncertain about this node.
    4. Compute criticality_weight from sla_criticality:
       weight = sla_criticality / 3.0  (so criticality 3 -> weight 1.0,
       criticality 5 -> weight 1.67, criticality 1 -> weight 0.33)
    5. Score the experiment using score_experiment()
    6. Suggest a failure type based on the node's tier (use
       FAILURE_TYPE_SUGGESTIONS dict)

    Return the top_k experiments sorted by score (descending), each as a dict:
    {
        "node": str,
        "tier": str,
        "predicted_impact": float,
        "uncertainty": float,
        "criticality_weight": float,
        "priority_score": float,
        "suggested_failure_type": str,
    }

    Args:
        graph: The microservice dependency graph.
        model: Trained RandomForestRegressor.
        feature_columns: List of feature column names (in order).
        top_k: Number of top experiments to return.

    Returns:
        List of experiment dicts, sorted by priority_score descending.
    """
    # TODO(human): Implement the experiment ranking loop. The key new skill
    # here is extracting per-tree predictions to measure uncertainty. This
    # is a powerful technique: RandomForest's ensemble gives you a free
    # uncertainty estimate without any Bayesian machinery.
    #
    # Import extract_node_features from the previous module:
    #   from practice_074_ml_chaos_engineering.src.blast_radius_model import ...
    # Or simply copy the feature extraction logic inline.
    raise NotImplementedError("Implement rank_experiments")


# ---------------------------------------------------------------------------
# Visualization (scaffolded)
# ---------------------------------------------------------------------------

def plot_experiment_scatter(experiments: list[dict], path: Path) -> None:
    """Scatter plot: predicted impact vs uncertainty, colored by priority score."""
    path.parent.mkdir(parents=True, exist_ok=True)

    impacts = [e["predicted_impact"] for e in experiments]
    uncertainties = [e["uncertainty"] for e in experiments]
    scores = [e["priority_score"] for e in experiments]
    names = [e["node"] for e in experiments]

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        impacts, uncertainties,
        c=scores, cmap="YlOrRd", s=120, alpha=0.8,
        edgecolors="black", linewidth=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="Priority Score")

    # Label each point
    for i, name in enumerate(names):
        short_name = name.replace("-service", "").replace("-api", "")
        ax.annotate(
            short_name, (impacts[i], uncertainties[i]),
            textcoords="offset points", xytext=(5, 5),
            fontsize=7, alpha=0.8,
        )

    ax.set_xlabel("Predicted Impact (Mean Blast Radius)", fontsize=12)
    ax.set_ylabel("Model Uncertainty (Std Across Trees)", fontsize=12)
    ax.set_title(
        "Chaos Experiment Prioritization: Impact vs Uncertainty",
        fontsize=13, fontweight="bold",
    )

    # Add quadrant labels
    mid_x = np.median(impacts)
    mid_y = np.median(uncertainties)
    ax.axvline(mid_x, color="gray", linestyle="--", alpha=0.3)
    ax.axhline(mid_y, color="gray", linestyle="--", alpha=0.3)
    ax.text(max(impacts) * 0.95, max(uncertainties) * 0.95,
            "HIGH PRIORITY\n(high impact +\nhigh uncertainty)",
            ha="right", va="top", fontsize=8, color="red", alpha=0.6)
    ax.text(min(impacts) + 0.1, min(uncertainties) + 0.01,
            "LOW PRIORITY\n(low impact +\nlow uncertainty)",
            ha="left", va="bottom", fontsize=8, color="green", alpha=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved experiment scatter plot to {path}")


def main() -> None:
    print("=" * 60)
    print("Chaos Experiment Prioritizer")
    print("=" * 60)

    # Load graph
    graph_path = DATA_DIR / "system_graph.json"
    if not graph_path.exists():
        print("ERROR: system_graph.json not found. Run previous scripts first.")
        return

    graph = load_graph(graph_path)

    # Load model
    model_path = MODELS_DIR / "blast_radius_model.pkl"
    columns_path = MODELS_DIR / "feature_columns.json"
    if not model_path.exists() or not columns_path.exists():
        print("ERROR: Model not found. Run src/02_blast_radius_model.py first.")
        return

    model, feature_columns = load_model(model_path, columns_path)
    print(f"Loaded model with {len(model.estimators_)} trees and {len(feature_columns)} features")

    # Rank experiments
    print("\nRanking chaos experiments...")
    all_experiments = rank_experiments(graph, model, feature_columns, top_k=len(graph.nodes()))

    # Display top-10 prioritized experiments
    print("\n--- Prioritized Chaos Experiments ---")
    print(f"{'Rank':>4}  {'Service':30s}  {'Tier':10s}  {'Impact':>8}  {'Uncert':>8}  {'Score':>8}  Suggested Failure")
    print("-" * 120)
    for i, exp in enumerate(all_experiments[:10]):
        print(
            f"{i+1:4d}  {exp['node']:30s}  {exp['tier']:10s}  "
            f"{exp['predicted_impact']:8.3f}  {exp['uncertainty']:8.3f}  "
            f"{exp['priority_score']:8.3f}  {exp['suggested_failure_type']}"
        )

    # Save results
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rankings_df = pd.DataFrame(all_experiments)
    rankings_df.to_csv(DATA_DIR / "experiment_rankings.csv", index=False)
    print(f"\nSaved rankings to {DATA_DIR / 'experiment_rankings.csv'}")

    # Plot
    plot_experiment_scatter(all_experiments, PLOTS_DIR / "experiment_scatter.png")

    print("\n-- Done. Run src/04_steady_state.py next. --")


if __name__ == "__main__":
    main()
