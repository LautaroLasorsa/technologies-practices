"""
Sample data generator -- FULLY IMPLEMENTED.

Generates synthetic "technical articles" with:
  - id: unique integer
  - title: generated article title
  - category: one of CATEGORIES
  - year: publication year (2018-2025)
  - word_count: random article length
  - vector: 128-dim random unit vector (simulates an embedding)

The vectors are L2-normalized so Cosine and Dot Product give identical rankings
(useful to verify in the benchmarking phase).

Run: python app/generate_data.py
"""

import json
from pathlib import Path

import numpy as np

from config import CATEGORIES, VECTOR_DIMENSION, YEAR_MAX, YEAR_MIN

NUM_ARTICLES = 500
OUTPUT_PATH = Path(__file__).parent / "articles.json"

# Title fragments for generating plausible article names
ADJECTIVES = [
    "Scalable", "Efficient", "Distributed", "Real-Time", "Fault-Tolerant",
    "Adaptive", "Lightweight", "Secure", "High-Performance", "Resilient",
    "Concurrent", "Declarative", "Streaming", "Federated", "Incremental",
]

NOUNS = [
    "Search Engine", "Pipeline", "Framework", "Architecture", "Protocol",
    "Index", "Scheduler", "Optimizer", "Cache", "Allocator",
    "Runtime", "Compiler", "Monitor", "Gateway", "Orchestrator",
]


def generate_title(rng: np.random.Generator, article_id: int) -> str:
    """Generate a plausible technical article title."""
    adj = rng.choice(ADJECTIVES)
    noun = rng.choice(NOUNS)
    return f"{adj} {noun} for Modern {rng.choice(CATEGORIES)} Systems (#{article_id})"


def generate_unit_vector(rng: np.random.Generator, dim: int) -> list[float]:
    """Generate a random unit vector (L2-normalized)."""
    vec = rng.standard_normal(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def generate_article(rng: np.random.Generator, article_id: int) -> dict:
    """Generate a single synthetic article with metadata and embedding."""
    return {
        "id": article_id,
        "title": generate_title(rng, article_id),
        "category": rng.choice(CATEGORIES),
        "year": int(rng.integers(YEAR_MIN, YEAR_MAX + 1)),
        "word_count": int(rng.integers(500, 15_001)),
        "vector": generate_unit_vector(rng, VECTOR_DIMENSION),
    }


def generate_all_articles(num_articles: int, seed: int = 42) -> list[dict]:
    """Generate the full corpus of synthetic articles."""
    rng = np.random.default_rng(seed)
    return [generate_article(rng, i) for i in range(num_articles)]


def save_articles(articles: list[dict], path: Path) -> None:
    """Persist articles to a JSON file."""
    path.write_text(json.dumps(articles, indent=2), encoding="utf-8")
    print(f"  Saved {len(articles)} articles to {path}")


def print_sample(articles: list[dict], n: int = 3) -> None:
    """Print a few sample articles (without the full vector)."""
    print(f"\n  Sample articles (first {n}):")
    for article in articles[:n]:
        vec_preview = article["vector"][:4]
        print(f"    [{article['id']}] {article['title']}")
        print(f"        category={article['category']}, year={article['year']}, "
              f"words={article['word_count']}, vec={vec_preview}...")


def main() -> None:
    print("=== Generating synthetic article data ===\n")

    articles = generate_all_articles(NUM_ARTICLES)
    save_articles(articles, OUTPUT_PATH)
    print_sample(articles)

    # Summary stats
    categories = {}
    for a in articles:
        categories[a["category"]] = categories.get(a["category"], 0) + 1
    print(f"\n  Category distribution: {dict(sorted(categories.items()))}")
    print(f"\nData ready. Next: run 'python app/ingest.py'")


if __name__ == "__main__":
    main()
