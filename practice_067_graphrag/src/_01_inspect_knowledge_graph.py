"""Exercise 01: Inspect the Knowledge Graph produced by GraphRAG indexing.

Implement the two core graph operations:
  - Build a NetworkX DiGraph from entity/relationship DataFrames.
  - Rank entities by degree centrality to find the graph's "hubs".

The Parquet loaders, type distribution, and top-relationship ranking
are scaffolded.  The goal is to see what the LLM-based extractor
actually produced and which entities dominate the graph.

Run (after `graphrag index --root .`):
    uv run python -m src._01_inspect_knowledge_graph
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


# -- Parquet loaders (scaffolded) -----------------------------------------


def _load_parquet(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{name}.parquet not found at {path}. "
            "Did you run `graphrag index --root .`?"
        )
    return pd.read_parquet(path)


def load_entities() -> pd.DataFrame:
    return _load_parquet("entities")


def load_relationships() -> pd.DataFrame:
    return _load_parquet("relationships")


# -- TODO 1 ---------------------------------------------------------------


def build_graph(entities: pd.DataFrame, relationships: pd.DataFrame) -> nx.DiGraph:
    """Build a directed NetworkX graph from the entity/relationship tables.

    Each entity row -> a node keyed by `title`, with `type` and `description`
    stored as node attributes.  Each relationship row -> a directed edge
    `source` -> `target` with `description` and `combined_degree` as
    edge attributes.  Use `G.add_node(..., **attrs)` / `G.add_edge(...)`.
    """
    # TODO(human): create an nx.DiGraph, iterate entities to add nodes,
    # iterate relationships to add edges, and return the graph.
    raise NotImplementedError("Implement build_graph()")


# -- TODO 2 ---------------------------------------------------------------


def top_by_centrality(G: nx.DiGraph, n: int = 10) -> list[tuple[str, float]]:
    """Return the top-*n* nodes by degree centrality (descending).

    Degree centrality = fraction of nodes a node is connected to.  High
    centrality identifies "hub" entities that tie the graph together.
    Use `nx.degree_centrality(G)` and sort by score.
    """
    # TODO(human): compute centrality and return the top n (title, score) pairs.
    raise NotImplementedError("Implement top_by_centrality()")


# -- Presentation (scaffolded) --------------------------------------------


def _print_type_distribution(entities: pd.DataFrame) -> None:
    print("\nEntity type distribution:")
    for etype, count in entities["type"].value_counts().items():
        print(f"  {etype:20} {count:>4}")


def _print_top_hubs(G: nx.DiGraph, entities: pd.DataFrame, n: int = 10) -> None:
    print(f"\nTop {n} hubs by degree centrality:")
    type_by_title = dict(zip(entities["title"], entities["type"]))
    for title, score in top_by_centrality(G, n=n):
        etype = type_by_title.get(title, "?")
        print(f"  {title:40} [{etype:15}] {score:.3f}")


def _print_top_relationships(relationships: pd.DataFrame, n: int = 10) -> None:
    print(f"\nTop {n} relationships by combined_degree:")
    ranked = relationships.sort_values("combined_degree", ascending=False).head(n)
    for _, row in ranked.iterrows():
        desc = str(row.get("description", ""))[:80]
        print(f"  {row['source']:25} -> {row['target']:25}  "
              f"(deg={row['combined_degree']})  {desc}")


def _print_graph_stats(G: nx.DiGraph) -> None:
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


# -- Main orchestrator ----------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Exercise 1: Knowledge Graph Inspection")
    print("=" * 60)

    print("\nLoading entities...")
    entities = load_entities()
    print(f"  {len(entities)} entities  columns={list(entities.columns)}")

    print("\nLoading relationships...")
    relationships = load_relationships()
    print(f"  {len(relationships)} relationships  columns={list(relationships.columns)}")

    print("\n" + "-" * 60)
    G = build_graph(entities, relationships)
    _print_graph_stats(G)
    _print_type_distribution(entities)
    _print_top_hubs(G, entities)
    _print_top_relationships(relationships)


if __name__ == "__main__":
    main()
