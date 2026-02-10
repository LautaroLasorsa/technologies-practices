"""
Phase 3: Similarity search and filtered search.

YOU IMPLEMENT the core functions marked with TODO(human).

Concepts:
  - **Nearest-neighbor search**: given a query vector, find the k closest points
    by the collection's distance metric (Cosine here). Returns ScoredPoint objects
    with id, score, payload, and optionally the vector.
  - **Filtered search**: combine vector similarity with payload conditions.
    Qdrant's Filter uses must/should/must_not clauses, each containing
    FieldCondition objects (MatchValue for exact match, Range for numeric ranges).
  - **Scroll**: paginated retrieval of points (no vector search involved).
    Useful for bulk reads, exports, or iterating the full collection.
  - **Score threshold**: discard results below a minimum similarity score.

Docs:
  - client.query_points(): https://python-client.qdrant.tech/qdrant_client.qdrant_client#QdrantClient.query_points
  - client.scroll(): https://python-client.qdrant.tech/qdrant_client.qdrant_client#QdrantClient.scroll
  - models.Filter: https://qdrant.tech/documentation/concepts/filtering/

Run: python app/search.py
"""

import json
from pathlib import Path

import numpy as np
from qdrant_client import models

from client import create_client
from config import COLLECTION_NAME, VECTOR_DIMENSION


def make_query_vector(seed: int = 123) -> list[float]:
    """Generate a deterministic random unit query vector."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(VECTOR_DIMENSION).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def print_results(results: list, label: str) -> None:
    """Pretty-print search results."""
    print(f"\n  --- {label} ({len(results)} results) ---")
    for r in results:
        payload = r.payload
        print(f"    [{r.id:>4}] score={r.score:.4f}  "
              f"cat={payload['category']:<10} year={payload['year']}  "
              f"words={payload['word_count']:>5}  {payload['title'][:60]}")


def basic_search(client, query_vector: list[float], top_k: int = 5) -> list:
    """Find the top-k most similar articles to the query vector.

    TODO(human): Implement this function.

    Steps:
      1. Call client.query_points() with:
         - collection_name=COLLECTION_NAME
         - query=query_vector
         - limit=top_k
         - with_payload=True     (include metadata in results)
      2. Return the list of ScoredPoint from the response's .points attribute

    The returned ScoredPoint objects have:
      - .id: point ID
      - .score: similarity score (higher = more similar for Cosine)
      - .payload: dict of metadata
      - .vector: the stored vector (only if with_vectors=True)

    Docs: https://python-client.qdrant.tech/qdrant_client.qdrant_client#QdrantClient.query_points
    """
    # TODO(human): implement basic similarity search
    raise NotImplementedError("Implement basic_search()")


def filtered_search(
    client,
    query_vector: list[float],
    category: str,
    min_year: int,
    top_k: int = 5,
) -> list:
    """Search for similar articles filtered by category AND minimum year.

    TODO(human): Implement this function.

    Steps:
      1. Build a models.Filter with must=[ ... ] containing:
         - models.FieldCondition(key="category", match=models.MatchValue(value=category))
         - models.FieldCondition(key="year", range=models.Range(gte=min_year))
      2. Call client.query_points() with:
         - collection_name, query, limit, with_payload (same as basic_search)
         - query_filter=<the filter you built>
      3. Return the .points from the response

    How filtering works in Qdrant:
      Qdrant doesn't do post-filtering (search first, then discard). Instead,
      it uses "pre-filtering" -- the HNSW graph has extra edges for indexed
      payload fields, so it navigates only through nodes matching the filter.
      This is why we created payload indices in setup.py.

    Docs: https://qdrant.tech/documentation/concepts/filtering/
    """
    # TODO(human): implement filtered similarity search
    raise NotImplementedError("Implement filtered_search()")


def scroll_all(client, category: str, page_size: int = 20) -> list:
    """Scroll through all articles in a given category (no vector search).

    TODO(human): Implement this function.

    Steps:
      1. Build a models.Filter for category match (same FieldCondition as above)
      2. Initialize offset = None (Qdrant uses opaque scroll tokens, not page numbers)
      3. Loop:
         a. Call client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=<filter>,
                limit=page_size,
                offset=offset,
                with_payload=True,
            )
            This returns a tuple: (points, next_offset)
         b. Collect the points
         c. If next_offset is None, stop (no more pages)
         d. Otherwise, set offset = next_offset and continue
      4. Return all collected points

    Scroll vs Search:
      - search() requires a query vector and returns scored results
      - scroll() is for bulk retrieval with optional filters, no ranking
      Use scroll when you need to iterate/export, not find similar items.

    Docs: https://python-client.qdrant.tech/qdrant_client.qdrant_client#QdrantClient.scroll
    """
    # TODO(human): implement paginated scroll
    raise NotImplementedError("Implement scroll_all()")


def search_with_score_threshold(
    client,
    query_vector: list[float],
    min_score: float,
    top_k: int = 20,
) -> list:
    """Search with a minimum score threshold -- only return highly similar results.

    TODO(human): Implement this function.

    Steps:
      1. Call client.query_points() with:
         - collection_name, query, limit=top_k, with_payload=True
         - score_threshold=min_score
      2. Return the .points from the response

    Why thresholds matter:
      Without a threshold, you always get top_k results even if they're all
      very dissimilar (e.g., score=0.02). In production RAG pipelines, you'd
      set a threshold to avoid retrieving irrelevant context for the LLM.

    Docs: https://python-client.qdrant.tech/qdrant_client.qdrant_client#QdrantClient.query_points
    """
    # TODO(human): implement search with score threshold
    raise NotImplementedError("Implement search_with_score_threshold()")


def main() -> None:
    print("=== Phase 3: Similarity search & filtering ===\n")

    client = create_client()
    query_vec = make_query_vector()
    print(f"  Query vector (first 4 dims): {query_vec[:4]}")

    # --- Basic search ---
    print("\n[1/4] Basic similarity search (top 5)...")
    results = basic_search(client, query_vec, top_k=5)
    print_results(results, "Basic Search")

    # --- Filtered search ---
    print("\n[2/4] Filtered search (category='AI', year >= 2023, top 5)...")
    results = filtered_search(client, query_vec, category="AI", min_year=2023, top_k=5)
    print_results(results, "Filtered Search (AI, 2023+)")

    # --- Scroll ---
    print("\n[3/4] Scrolling all 'Security' articles...")
    all_security = scroll_all(client, category="Security")
    print(f"  Retrieved {len(all_security)} 'Security' articles via scroll")
    if all_security:
        first = all_security[0].payload
        print(f"  First: [{all_security[0].id}] {first['title'][:60]}")

    # --- Score threshold ---
    print("\n[4/4] Search with score threshold (min_score=0.3)...")
    results = search_with_score_threshold(client, query_vec, min_score=0.3, top_k=20)
    print_results(results, "Threshold Search (score >= 0.3)")

    print("\nSearch phase complete. Next: run 'python app/benchmark.py'")


if __name__ == "__main__":
    main()
