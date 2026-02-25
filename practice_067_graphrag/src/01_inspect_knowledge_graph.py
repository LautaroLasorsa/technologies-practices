"""Exercise 1: Inspect the Knowledge Graph produced by GraphRAG indexing.

After running `graphrag index --root .`, the pipeline produces Parquet files
in the `output/` directory containing entities, relationships, text units,
documents, communities, and community reports.

This exercise loads the core graph artifacts (entities + relationships),
builds a NetworkX graph, and computes structural metrics to understand
what the LLM extracted from our corpus.
"""

from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def load_entities() -> pd.DataFrame:
    """Load the entities table from Parquet."""
    path = OUTPUT_DIR / "entities.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"entities.parquet not found at {path}. "
            "Did you run `graphrag index --root .`?"
        )
    return pd.read_parquet(path)


def load_relationships() -> pd.DataFrame:
    """Load the relationships table from Parquet."""
    path = OUTPUT_DIR / "relationships.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"relationships.parquet not found at {path}. "
            "Did you run `graphrag index --root .`?"
        )
    return pd.read_parquet(path)


# TODO(human): Implement build_and_analyze_graph
#
# This is the core exercise. You will:
#
# 1. Build a NetworkX DiGraph from the entities and relationships DataFrames.
#    - Each entity row becomes a node. Use the entity's `title` as the node ID.
#      Store the entity's `type` and `description` as node attributes.
#    - Each relationship row becomes a directed edge from `source` to `target`.
#      Store the relationship's `description` and `combined_degree` as edge
#      attributes.
#
# 2. Print basic graph statistics:
#    - Total number of nodes and edges
#    - Entity type distribution (how many persons, organizations, locations, etc.)
#
# 3. Compute degree centrality using `nx.degree_centrality(G)` and print the
#    top 10 most connected entities with their type and centrality score.
#    Degree centrality measures how connected an entity is — entities with
#    high centrality are "hubs" in the knowledge graph that connect many
#    other entities together.
#
# 4. Print the top 10 relationships sorted by `combined_degree` (descending).
#    combined_degree is the sum of the source and target entity degrees —
#    it indicates relationships between important/well-connected entities.
#
# Function signature:
#   def build_and_analyze_graph(entities: pd.DataFrame, relationships: pd.DataFrame) -> None
#
# Hints:
#   - import networkx as nx
#   - nx.DiGraph() creates a directed graph
#   - G.add_node(id, **attrs) and G.add_edge(src, tgt, **attrs)
#   - entities DataFrame has columns: title, type, description, id, ...
#   - relationships DataFrame has columns: source, target, description,
#     combined_degree, id, ...
#   - Use pandas value_counts() for type distribution
#   - sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]


def main() -> None:
    print("=" * 60)
    print("Exercise 1: Knowledge Graph Inspection")
    print("=" * 60)

    print("\nLoading entities...")
    entities = load_entities()
    print(f"  Loaded {len(entities)} entities")
    print(f"  Columns: {list(entities.columns)}")

    print("\nLoading relationships...")
    relationships = load_relationships()
    print(f"  Loaded {len(relationships)} relationships")
    print(f"  Columns: {list(relationships.columns)}")

    print("\n" + "-" * 60)
    print("Building and analyzing knowledge graph...")
    print("-" * 60)

    build_and_analyze_graph(entities, relationships)


if __name__ == "__main__":
    main()
