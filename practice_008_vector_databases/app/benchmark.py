"""
Phase 4: Benchmarking distance metrics and HNSW parameters.

YOU IMPLEMENT the core functions marked with TODO(human).

Concepts:
  - **Distance metrics**: Cosine, Dot Product, and Euclidean produce different
    similarity scores but may yield identical rankings for L2-normalized vectors.
    This benchmark measures latency differences (the computation cost varies).
  - **HNSW tuning**: the `m` parameter controls graph connectivity (more edges =
    better recall, more memory). `ef_construct` controls build-time quality
    (higher = slower indexing, better graph). The search-time `ef` can also be
    tuned via search params.
  - **Recall**: fraction of true nearest neighbors found by ANN vs brute-force.
    We approximate this by comparing HNSW results against a high-ef search.

Docs:
  - Distance metrics: https://qdrant.tech/documentation/concepts/collections/
  - HNSW config: https://qdrant.tech/documentation/concepts/indexing/

Run: python app/benchmark.py
"""

import json
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from client import create_client
from config import VECTOR_DIMENSION

ARTICLES_PATH = Path(__file__).parent / "articles.json"

# Benchmark collection names (temporary, deleted after benchmarking)
BENCH_PREFIX = "bench_"


def load_articles(path: Path) -> list[dict]:
    """Load articles from the JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def make_query_vectors(n: int = 50, seed: int = 999) -> list[list[float]]:
    """Generate n deterministic random unit query vectors for benchmarking."""
    rng = np.random.default_rng(seed)
    vectors = []
    for _ in range(n):
        vec = rng.standard_normal(VECTOR_DIMENSION).astype(np.float32)
        vec /= np.linalg.norm(vec)
        vectors.append(vec.tolist())
    return vectors


def create_bench_collection(
    client: QdrantClient,
    name: str,
    distance: models.Distance,
    m: int = 16,
    ef_construct: int = 100,
) -> None:
    """Create a temporary benchmark collection with given params."""
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=VECTOR_DIMENSION,
            distance=distance,
        ),
        hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
    )


def ingest_into_collection(client: QdrantClient, name: str, articles: list[dict]) -> None:
    """Bulk-ingest articles into the given collection."""
    points = [
        models.PointStruct(
            id=a["id"],
            vector=a["vector"],
            payload={"category": a["category"], "year": a["year"]},
        )
        for a in articles
    ]
    # Upsert in one batch (small dataset, fits in memory)
    client.upsert(collection_name=name, points=points)


def measure_search_latency(
    client: QdrantClient,
    collection_name: str,
    query_vectors: list[list[float]],
    top_k: int = 10,
) -> float:
    """Run all query vectors against the collection and return mean latency in ms."""
    latencies = []
    for qv in query_vectors:
        start = time.perf_counter()
        client.query_points(
            collection_name=collection_name,
            query=qv,
            limit=top_k,
            with_payload=False,
        )
        latencies.append((time.perf_counter() - start) * 1000)
    return sum(latencies) / len(latencies)


# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches how distance metric choice affects search performance.
# Understanding the computational cost differences between Cosine, Dot Product, and Euclidean
# is essential for designing systems where latency matters—milliseconds add up at scale.

def benchmark_distance_metrics(client: QdrantClient, articles: list[dict]) -> None:
    """Compare search latency across Cosine, Dot Product, and Euclidean.

    TODO(human): Implement this function.

    Steps:
      1. Define a dict mapping metric names to models.Distance values:
           {"Cosine": models.Distance.COSINE,
            "Dot":    models.Distance.DOT,
            "Euclid": models.Distance.EUCLID}
      2. Generate query vectors using make_query_vectors(n=50)
      3. For each metric:
         a. Create a bench collection: create_bench_collection(client, name, distance)
            where name = f"{BENCH_PREFIX}{metric_name.lower()}"
         b. Ingest articles: ingest_into_collection(client, name, articles)
         c. Measure latency: measure_search_latency(client, name, query_vectors)
         d. Print the result: metric name and mean latency
         e. Clean up: client.delete_collection(name)
      4. Print a summary table

    Hint: for L2-normalized vectors (ours are), Cosine and Dot should give
    nearly identical results. Euclidean may differ slightly in latency.

    Docs: https://qdrant.tech/documentation/concepts/collections/
    """
    print("\n  Comparing distance metrics (50 queries each)...")
    # TODO(human): implement distance metric benchmark
    raise NotImplementedError("Implement benchmark_distance_metrics()")


# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches HNSW parameter tuning, a critical production skill for vector databases.
# Understanding the recall/latency/memory tradeoffs controlled by m and ef_construct enables
# you to optimize vector search for your specific requirements (speed vs accuracy).

def benchmark_hnsw_params(client: QdrantClient, articles: list[dict]) -> None:
    """Compare build time and search latency across HNSW configurations.

    TODO(human): Implement this function.

    Steps:
      1. Define configurations to test:
           configs = [
               {"m": 8,  "ef_construct": 64},
               {"m": 16, "ef_construct": 100},   # default
               {"m": 32, "ef_construct": 200},
               {"m": 48, "ef_construct": 300},
           ]
      2. Generate query vectors using make_query_vectors(n=50)
      3. For each config:
         a. Name = f"{BENCH_PREFIX}hnsw_m{m}_ef{ef_construct}"
         b. Time the collection creation + ingestion together (this is "build time"):
              start = time.perf_counter()
              create_bench_collection(client, name, models.Distance.COSINE, m, ef_construct)
              ingest_into_collection(client, name, articles)
              build_time = time.perf_counter() - start
         c. Measure search latency: measure_search_latency(client, name, query_vectors)
         d. Print: m, ef_construct, build_time, search_latency
         e. Clean up: client.delete_collection(name)
      4. Print a summary table

    What to observe:
      - Higher m/ef_construct => slower build, potentially faster/better search
      - With only 500 points, differences may be small. In production (millions
        of points), these parameters matter enormously.
      - m affects memory: each point stores ~m*2 edges in the graph.

    Docs: https://qdrant.tech/documentation/concepts/indexing/
    """
    print("\n  Comparing HNSW configurations (50 queries each)...")
    # TODO(human): implement HNSW parameter benchmark
    raise NotImplementedError("Implement benchmark_hnsw_params()")


def main() -> None:
    print("=== Phase 4: Benchmarking ===\n")

    client = create_client()
    articles = load_articles(ARTICLES_PATH)
    print(f"  Loaded {len(articles)} articles for benchmarking")

    print("\n[1/2] Distance metric comparison...")
    benchmark_distance_metrics(client, articles)

    print("\n[2/2] HNSW parameter comparison...")
    benchmark_hnsw_params(client, articles)

    print("\nBenchmarking complete!")
    print("Reflect: How do these numbers change with 10x more data? With 256-dim vectors?")


if __name__ == "__main__":
    main()
