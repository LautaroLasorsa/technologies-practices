#!/usr/bin/env python3
"""Exercise 5-7: Cluster log templates and visualize with UMAP.

Pipeline:
  1. Load template embeddings from data/template_embeddings.npy
  2. Cluster with KMeans (user picks optimal K via silhouette scoring)
  3. Cluster with HDBSCAN (automatic K, noise detection)
  4. Reduce to 2D with UMAP and create scatter plots for both methods
  5. Compare cluster counts and quality metrics
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
PLOTS_DIR = Path(__file__).parent.parent / "plots"


# ---------------------------------------------------------------------------
# TODO(human): Exercise 5 -- KMeans clustering with silhouette scoring
# ---------------------------------------------------------------------------
def cluster_kmeans(
    embeddings: np.ndarray,
    k_range: tuple[int, int] = (5, 20),
) -> tuple[np.ndarray, int, float]:
    """Cluster template embeddings with KMeans, selecting optimal K via silhouette score.

    TODO(human): Implement KMeans clustering with automatic K selection.

    Steps:
      1. Import KMeans from sklearn.cluster and silhouette_score from sklearn.metrics.
      2. For each k in range(k_range[0], k_range[1] + 1):
         a. Fit KMeans(n_clusters=k, random_state=42, n_init=10) on the embeddings.
         b. Compute silhouette_score(embeddings, kmeans.labels_).
            Silhouette score measures how similar each point is to its own cluster
            vs the nearest neighboring cluster. Range: [-1, 1], higher = better.
         c. Print: f"  k={k:>2}: silhouette={score:.4f}"
         d. Track the best k (highest silhouette score).
      3. Re-fit KMeans with the best k.
      4. Return (labels, best_k, best_score):
         - labels: np.ndarray of cluster assignments, shape (n_templates,)
         - best_k: int, the optimal number of clusters
         - best_score: float, the silhouette score at best_k

    Note: If n_templates is very small (< k_range[0]), silhouette_score may fail.
    Guard against this by checking len(embeddings) > k_range[0].

    Why KMeans for logs? KMeans assigns every template to exactly one cluster,
    which is useful when you want exhaustive categorization (every log type gets
    a category). But it forces you to guess K upfront -- the silhouette heuristic
    helps, but isn't perfect.

    Args:
        embeddings: np.ndarray of shape (n_templates, embedding_dim).
        k_range: Tuple (min_k, max_k) to search over.

    Returns:
        Tuple of (labels, best_k, best_score).
    """
    raise NotImplementedError("TODO(human): Implement KMeans clustering")


# ---------------------------------------------------------------------------
# TODO(human): Exercise 6 -- HDBSCAN clustering
# ---------------------------------------------------------------------------
def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
) -> tuple[np.ndarray, int, int]:
    """Cluster template embeddings with HDBSCAN (density-based, automatic K).

    TODO(human): Implement HDBSCAN clustering on the template embeddings.

    Steps:
      1. Import HDBSCAN from sklearn.cluster (scikit-learn >= 1.3 includes it)
         or from hdbscan if using the standalone package.
      2. Create and fit the clusterer:
         clusterer = HDBSCAN(
             min_cluster_size=min_cluster_size,
             metric="euclidean",
             store_centers="centroid",   # optional, for analysis
         )
         clusterer.fit(embeddings)
      3. Extract labels: clusterer.labels_
         Note: label -1 means "noise" (point doesn't belong to any dense cluster).
         This is a key advantage over KMeans -- HDBSCAN acknowledges that some
         templates may be genuinely unique/anomalous rather than forcing them
         into a cluster.
      4. Count: n_clusters = number of unique labels excluding -1
                n_noise   = count of labels == -1
      5. Print summary:
         f"  HDBSCAN found {n_clusters} clusters, {n_noise} noise points"
      6. Return (labels, n_clusters, n_noise).

    What does min_cluster_size mean? It's the minimum number of templates that
    must be close together (in embedding space) to form a cluster. Smaller values
    (2-3) find more clusters including small ones. Larger values (5-10) require
    denser, bigger groups and classify more points as noise.

    For log templates (typically 40-80 unique templates), min_cluster_size=3 is
    a good starting point. Experiment: try 2 (more clusters) and 5 (fewer, larger).

    Args:
        embeddings: np.ndarray of shape (n_templates, embedding_dim).
        min_cluster_size: Minimum templates to form a cluster.

    Returns:
        Tuple of (labels, n_clusters, n_noise).
    """
    raise NotImplementedError("TODO(human): Implement HDBSCAN clustering")


# ---------------------------------------------------------------------------
# TODO(human): Exercise 7 -- UMAP visualization
# ---------------------------------------------------------------------------
def visualize_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    templates: list[str],
    method_name: str,
    save_path: Path,
) -> None:
    """Reduce embeddings to 2D with UMAP and create a scatter plot colored by cluster.

    TODO(human): Implement UMAP dimensionality reduction and scatter plot visualization.

    Steps:
      1. Import umap (the package is umap-learn, imported as: import umap).
      2. Reduce to 2D:
         reducer = umap.UMAP(
             n_components=2,
             n_neighbors=15,     # balance local vs global structure
             min_dist=0.05,      # tight clusters for visual clarity
             metric="cosine",    # appropriate for normalized text embeddings
             random_state=42,
         )
         coords_2d = reducer.fit_transform(embeddings)
         coords_2d has shape (n_templates, 2).

         UMAP parameters explained:
           - n_neighbors=15: each point considers 15 nearest neighbors when
             building the graph. Lower = more local structure, higher = more global.
           - min_dist=0.05: points can pack very tightly in 2D. Good for seeing
             distinct clusters. Increase to 0.3+ for more spread-out layouts.
           - metric="cosine": matches how we measure similarity for text embeddings.

      3. Import matplotlib.pyplot as plt.
      4. Create a scatter plot:
         fig, ax = plt.subplots(figsize=(12, 8))
         scatter = ax.scatter(
             coords_2d[:, 0], coords_2d[:, 1],
             c=labels, cmap="tab20", s=60, alpha=0.8, edgecolors="white", linewidths=0.5
         )
         For noise points (label == -1), you may want to color them gray or mark
         them with a different marker (e.g., 'x').
      5. Add a colorbar or legend showing cluster IDs.
      6. Set title: f"Log Template Clusters ({method_name})"
         Add axis labels: "UMAP Dimension 1", "UMAP Dimension 2".
      7. Optionally annotate a few points with their (truncated) template text.
      8. Save: fig.savefig(save_path, dpi=150, bbox_inches="tight")
         plt.close(fig)

    Args:
        embeddings: np.ndarray of shape (n_templates, embedding_dim).
        labels: np.ndarray of cluster labels (from KMeans or HDBSCAN).
        templates: List of template strings (for optional annotation).
        method_name: "KMeans" or "HDBSCAN" (used in title and filename).
        save_path: Path to save the plot image.
    """
    raise NotImplementedError("TODO(human): Implement UMAP visualization")


# ---------------------------------------------------------------------------
# Scaffolded: main pipeline
# ---------------------------------------------------------------------------
def print_cluster_contents(
    labels: np.ndarray,
    templates: list[str],
    method_name: str,
) -> None:
    """Print templates grouped by cluster for inspection."""
    cluster_map: dict[int, list[str]] = {}
    for label, template in zip(labels, templates):
        cluster_map.setdefault(int(label), []).append(template)

    print(f"\n{'='*80}")
    print(f"CLUSTER CONTENTS ({method_name})")
    print(f"{'='*80}")

    for cluster_id in sorted(cluster_map.keys()):
        members = cluster_map[cluster_id]
        label = f"Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        print(f"\n  [{label}] ({len(members)} templates):")
        for t in members[:5]:
            print(f"    - {t[:90]}")
        if len(members) > 5:
            print(f"    ... and {len(members)-5} more")


def main() -> None:
    """Run the clustering pipeline."""
    embeddings_path = DATA_DIR / "template_embeddings.npy"
    template_order_path = DATA_DIR / "template_order.csv"

    if not embeddings_path.exists():
        print("ERROR: data/template_embeddings.npy not found. Run 02_embeddings.py first.")
        return

    # Load data
    print("Loading embeddings and template names...")
    embeddings = np.load(embeddings_path)
    templates = pd.read_csv(template_order_path)["template"].tolist()
    print(f"  {len(templates)} templates, embedding dim = {embeddings.shape[1]}")

    # Ensure plots directory exists
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- KMeans ---
    print(f"\n{'='*80}")
    print("KMEANS CLUSTERING")
    print(f"{'='*80}")
    print("\nSearching for optimal K via silhouette score...")
    km_labels, best_k, best_score = cluster_kmeans(embeddings)
    print(f"\n  Best K = {best_k}, silhouette = {best_score:.4f}")
    print_cluster_contents(km_labels, templates, "KMeans")

    # --- HDBSCAN ---
    print(f"\n{'='*80}")
    print("HDBSCAN CLUSTERING")
    print(f"{'='*80}")
    hdb_labels, n_clusters, n_noise = cluster_hdbscan(embeddings)
    print(f"\n  Clusters: {n_clusters}, Noise points: {n_noise}")
    print_cluster_contents(hdb_labels, templates, "HDBSCAN")

    # --- Comparison ---
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"  KMeans:  {best_k} clusters, silhouette = {best_score:.4f}")

    # Compute silhouette for HDBSCAN (excluding noise points)
    from sklearn.metrics import silhouette_score

    non_noise_mask = hdb_labels >= 0
    if non_noise_mask.sum() > 1 and len(set(hdb_labels[non_noise_mask])) > 1:
        hdb_silhouette = silhouette_score(
            embeddings[non_noise_mask], hdb_labels[non_noise_mask]
        )
        print(f"  HDBSCAN: {n_clusters} clusters + {n_noise} noise, silhouette = {hdb_silhouette:.4f} (excl. noise)")
    else:
        print(f"  HDBSCAN: {n_clusters} clusters + {n_noise} noise (silhouette not computable)")

    # --- Visualizations ---
    print(f"\nGenerating UMAP visualizations...")
    km_plot_path = PLOTS_DIR / "clusters_kmeans.png"
    hdb_plot_path = PLOTS_DIR / "clusters_hdbscan.png"

    visualize_clusters(embeddings, km_labels, templates, "KMeans", km_plot_path)
    print(f"  Saved KMeans plot to {km_plot_path}")

    visualize_clusters(embeddings, hdb_labels, templates, "HDBSCAN", hdb_plot_path)
    print(f"  Saved HDBSCAN plot to {hdb_plot_path}")

    # Save cluster assignments for anomaly detection step
    cluster_assignments_path = DATA_DIR / "cluster_assignments.csv"
    assignments_df = pd.DataFrame({
        "template": templates,
        "kmeans_cluster": km_labels,
        "hdbscan_cluster": hdb_labels,
    })
    assignments_df.to_csv(cluster_assignments_path, index=False)
    print(f"\nSaved cluster assignments to {cluster_assignments_path}")


if __name__ == "__main__":
    main()
