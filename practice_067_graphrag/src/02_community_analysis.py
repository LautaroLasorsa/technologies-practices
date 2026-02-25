"""Exercise 2: Analyze Community Structure and Reports.

GraphRAG's indexing pipeline applies the Leiden algorithm to cluster
densely-connected entities into communities at multiple hierarchical levels.
For each community, an LLM generates a summary report describing the key
entities, relationships, and themes within that cluster.

This hierarchical structure is what powers Global Search: instead of
searching individual documents, global queries are answered by doing
map-reduce over these pre-computed community summaries.

This exercise explores the community hierarchy and the quality of
the generated reports.
"""

from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def load_communities() -> pd.DataFrame:
    """Load the communities table from Parquet."""
    path = OUTPUT_DIR / "communities.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"communities.parquet not found at {path}. "
            "Did you run `graphrag index --root .`?"
        )
    return pd.read_parquet(path)


def load_community_reports() -> pd.DataFrame:
    """Load the community reports table from Parquet."""
    path = OUTPUT_DIR / "community_reports.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"community_reports.parquet not found at {path}. "
            "Did you run `graphrag index --root .`?"
        )
    return pd.read_parquet(path)


def load_entities() -> pd.DataFrame:
    """Load entities to cross-reference with communities."""
    path = OUTPUT_DIR / "entities.parquet"
    return pd.read_parquet(path)


# TODO(human): Implement analyze_communities
#
# This exercise teaches you how the Leiden algorithm organizes the knowledge
# graph into a hierarchy and how community reports enable global search.
#
# 1. Analyze the community hierarchy:
#    - Print how many communities exist at each hierarchical level.
#      Communities have a `level` column (0 = most granular, higher = broader).
#    - For each level, compute the average number of entities per community.
#      Communities have a `relationship_ids` or `entity_ids` column (check
#      which columns are available — print communities.columns first).
#
# 2. Display community report quality:
#    - Community reports have a `rank` (or `rating`) column indicating
#      the LLM's self-assessed quality/importance score.
#    - Sort reports by this score (descending) and display the top 3:
#      print their title, level, rank/rating, and the first 300 characters
#      of their `summary` or `content` field.
#
# 3. Cross-reference communities with entities:
#    - Pick the highest-rated community report.
#    - If the communities table has entity references (e.g., `entity_ids`),
#      list the entity names belonging to that community.
#    - This shows you what the Leiden algorithm grouped together — entities
#      in the same community are densely connected in the knowledge graph.
#
# Function signature:
#   def analyze_communities(
#       communities: pd.DataFrame,
#       reports: pd.DataFrame,
#       entities: pd.DataFrame,
#   ) -> None
#
# Hints:
#   - Start by printing communities.columns and reports.columns to see
#     what fields are available (they vary slightly between GraphRAG versions)
#   - groupby("level").size() for community count per level
#   - Use .iloc[0] to get the top-rated report
#   - Community reports may have a `findings` column with structured JSON
#     — if so, that's more informative than raw `content`


def main() -> None:
    print("=" * 60)
    print("Exercise 2: Community Structure Analysis")
    print("=" * 60)

    print("\nLoading communities...")
    communities = load_communities()
    print(f"  Loaded {len(communities)} communities")

    print("\nLoading community reports...")
    reports = load_community_reports()
    print(f"  Loaded {len(reports)} community reports")

    print("\nLoading entities for cross-reference...")
    entities = load_entities()
    print(f"  Loaded {len(entities)} entities")

    print("\n" + "-" * 60)
    print("Analyzing community structure...")
    print("-" * 60)

    analyze_communities(communities, reports, entities)


if __name__ == "__main__":
    main()
