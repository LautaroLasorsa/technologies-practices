#!/usr/bin/env python3
"""Exercise 3-4: Generate semantic embeddings for log templates.

Pipeline:
  1. Load unique templates from data/unique_templates.csv
  2. Encode templates with sentence-transformers (all-MiniLM-L6-v2)
  3. Compute pairwise cosine similarity between template embeddings
  4. Save embeddings for clustering (next step)

The embeddings capture semantic meaning: templates about timeouts cluster
near each other in vector space even if they use different words.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# TODO(human): Exercise 3 -- Embed templates with sentence-transformers
# ---------------------------------------------------------------------------
def embed_templates(
    templates: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, np.ndarray]:
    """Load a sentence-transformer model and encode templates into dense vectors.

    TODO(human): Use sentence-transformers to create embeddings for log templates.

    Steps:
      1. Import SentenceTransformer from sentence_transformers.
      2. Load the model: model = SentenceTransformer(model_name)
         The first run downloads the model (~22 MB). Subsequent runs use cache.
      3. Encode all templates at once:
         embeddings = model.encode(templates, show_progress_bar=True)
         This returns a numpy array of shape (n_templates, 384).
      4. Build and return a dict mapping template_string -> embedding_vector.

    Design consideration: We embed the *templates* (with <IP>, <NUM> placeholders),
    NOT the raw log lines. Why?
      - Templates isolate the semantic event type from instance-specific variables.
      - "Connection to <IP>:<NUM> timed out" captures the meaning "timeout" without
        noise from specific IPs and ports that vary per occurrence.
      - Embedding 50-100 unique templates is orders of magnitude faster than
        embedding 10,000 raw lines, and produces cleaner semantic clusters.

    Why all-MiniLM-L6-v2?
      - 384-dim embeddings (compact, fast cosine similarity)
      - Only 22 MB (runs on CPU in seconds, no GPU needed)
      - Trained on 1B sentence pairs -- excellent for short texts like log templates
      - Good balance of speed and quality for this use case

    Args:
        templates: List of unique template strings (from Drain3 parsing).
        model_name: Sentence-transformer model name (default: "all-MiniLM-L6-v2").

    Returns:
        Dict mapping each template string to its numpy embedding vector (shape: (384,)).
    """
    raise NotImplementedError("TODO(human): Embed templates with sentence-transformers")


# ---------------------------------------------------------------------------
# TODO(human): Exercise 4 -- Compute cosine similarity matrix
# ---------------------------------------------------------------------------
def compute_similarity_matrix(
    embeddings_dict: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute pairwise cosine similarity between all template embeddings.

    TODO(human): Build a similarity matrix for visual inspection of template relationships.

    Steps:
      1. Extract the template names and embedding vectors from the dict.
         templates = list(embeddings_dict.keys())
         vectors = np.array([embeddings_dict[t] for t in templates])
      2. Import cosine_similarity from sklearn.metrics.pairwise.
      3. Compute: sim_matrix = cosine_similarity(vectors)
         This returns an (n, n) numpy array where sim_matrix[i][j] is the
         cosine similarity between template i and template j. Range: [-1, 1],
         where 1 = identical direction, 0 = orthogonal, -1 = opposite.
      4. Wrap in a pandas DataFrame with template names as both index and columns:
         pd.DataFrame(sim_matrix, index=templates, columns=templates)
      5. Return the DataFrame.

    This matrix reveals which templates the model considers semantically related.
    For example, "Connection to <IP>:<NUM> timed out" and "Socket read timeout
    from <IP>" should have high similarity (~0.7-0.9), confirming the embeddings
    capture "timeout" semantics despite different wording.

    Args:
        embeddings_dict: Dict mapping template string to embedding vector.

    Returns:
        pd.DataFrame of shape (n_templates, n_templates) with cosine similarities.
    """
    raise NotImplementedError("TODO(human): Compute cosine similarity matrix")


# ---------------------------------------------------------------------------
# Scaffolded: main pipeline
# ---------------------------------------------------------------------------
def find_most_similar_pairs(
    sim_df: pd.DataFrame,
    top_k: int = 5,
) -> list[tuple[str, str, float]]:
    """Find the top-k most similar template pairs (excluding self-similarity)."""
    pairs = []
    templates = sim_df.index.tolist()

    for i in range(len(templates)):
        for j in range(i + 1, len(templates)):
            pairs.append((templates[i], templates[j], sim_df.iloc[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def main() -> None:
    """Run the embedding pipeline."""
    templates_path = DATA_DIR / "unique_templates.csv"
    if not templates_path.exists():
        print("ERROR: data/unique_templates.csv not found. Run 01_drain_parsing.py first.")
        return

    # Load unique templates
    print(f"Loading templates from {templates_path}...")
    templates_df = pd.read_csv(templates_path)
    templates = templates_df["template"].tolist()
    print(f"  Loaded {len(templates)} unique templates")

    # Embed templates
    print(f"\nEmbedding {len(templates)} templates with sentence-transformers...")
    embeddings_dict = embed_templates(templates)
    print(f"  Embedding dimension: {next(iter(embeddings_dict.values())).shape}")

    # Compute similarity matrix
    print("\nComputing pairwise cosine similarity...")
    sim_df = compute_similarity_matrix(embeddings_dict)

    # Find most similar pairs
    top_pairs = find_most_similar_pairs(sim_df, top_k=5)
    print(f"\n{'='*80}")
    print(f"TOP 5 MOST SIMILAR TEMPLATE PAIRS")
    print(f"{'='*80}")
    for i, (t1, t2, score) in enumerate(top_pairs, 1):
        print(f"\n  Pair {i} (similarity: {score:.4f}):")
        print(f"    A: {t1[:90]}")
        print(f"    B: {t2[:90]}")

    # Save embeddings as numpy arrays for clustering step
    # Save as: template_list (ordered) and embedding_matrix
    ordered_templates = list(embeddings_dict.keys())
    embedding_matrix = np.array([embeddings_dict[t] for t in ordered_templates])

    embeddings_path = DATA_DIR / "template_embeddings.npy"
    np.save(embeddings_path, embedding_matrix)
    print(f"\nSaved embedding matrix {embedding_matrix.shape} to {embeddings_path}")

    # Also save the template order (so we know which row is which template)
    template_order_path = DATA_DIR / "template_order.csv"
    pd.DataFrame({"template": ordered_templates}).to_csv(template_order_path, index=False)
    print(f"Saved template order to {template_order_path}")

    # Save similarity matrix for reference
    sim_path = DATA_DIR / "similarity_matrix.csv"
    sim_df.to_csv(sim_path)
    print(f"Saved similarity matrix to {sim_path}")


if __name__ == "__main__":
    main()
