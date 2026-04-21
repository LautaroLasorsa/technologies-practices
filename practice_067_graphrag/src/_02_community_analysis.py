"""Exercise 02: Community Structure Analysis.

Implement the two community-focused operations:
  - `level_stats`: summarise the Leiden hierarchy (# communities per
    level and average size in entities).
  - `entities_in_community`: list the entity titles inside a single
    community row.

Parquet loaders and top-report display are scaffolded.  The goal is to
see how Leiden slices the knowledge graph into nested communities and
how the LLM-generated reports summarise each cluster.

Run (after `graphrag index --root .`):
    uv run python -m src._02_community_analysis
"""

from __future__ import annotations

from pathlib import Path

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


def load_communities() -> pd.DataFrame:
    return _load_parquet("communities")


def load_community_reports() -> pd.DataFrame:
    return _load_parquet("community_reports")


def load_entities() -> pd.DataFrame:
    return _load_parquet("entities")


# -- TODO 1 ---------------------------------------------------------------


def level_stats(communities: pd.DataFrame) -> dict[int, tuple[int, float]]:
    """Return `{level: (n_communities, avg_entities_per_community)}`.

    `communities` has a `level` column (0 = finest, larger = broader) and an
    `entity_ids` column whose values are lists.  Group by level; for each
    group return (count of rows, mean length of the entity_ids lists).
    """
    # TODO(human): group communities by level, count rows, and average
    # len(entity_ids); return one (count, avg) tuple per level.
    raise NotImplementedError("Implement level_stats()")


# -- TODO 2 ---------------------------------------------------------------


def entities_in_community(community_row: pd.Series, entities: pd.DataFrame) -> list[str]:
    """Return the entity titles whose `id` appears in this community.

    `community_row["entity_ids"]` is a list of entity IDs.  Look them up
    in the `entities` DataFrame (matching on its `id` column) and return
    the resulting `title` values as a list.
    """
    # TODO(human): filter the entities DataFrame by community_row["entity_ids"]
    # and return the matching titles.
    raise NotImplementedError("Implement entities_in_community()")


# -- Presentation (scaffolded) --------------------------------------------


def _print_level_stats(communities: pd.DataFrame) -> None:
    print("\nCommunity hierarchy:")
    print(f"  {'level':>6}  {'count':>6}  {'avg entities':>14}")
    for level, (count, avg) in sorted(level_stats(communities).items()):
        print(f"  {level:>6}  {count:>6}  {avg:>14.1f}")


def _print_top_reports(reports: pd.DataFrame, n: int = 3) -> None:
    rank_col = "rank" if "rank" in reports.columns else "rating"
    text_col = "summary" if "summary" in reports.columns else "content"
    top = reports.sort_values(rank_col, ascending=False).head(n)
    print(f"\nTop {n} community reports (by {rank_col}):")
    for _, row in top.iterrows():
        print(f"\n  [level={row['level']}  {rank_col}={row[rank_col]}] {row.get('title', '?')}")
        print(f"  {str(row[text_col])[:300]}")


def _print_top_community_members(
    communities: pd.DataFrame, reports: pd.DataFrame, entities: pd.DataFrame,
) -> None:
    rank_col = "rank" if "rank" in reports.columns else "rating"
    top_report = reports.sort_values(rank_col, ascending=False).iloc[0]
    # `community` in reports matches the `community` column in communities.
    community_col = "community" if "community" in communities.columns else "id"
    match_value = top_report.get("community", top_report.get("id"))
    matching = communities[communities[community_col] == match_value]
    if matching.empty:
        print("\n(Could not cross-reference top community.)")
        return
    members = entities_in_community(matching.iloc[0], entities)
    print(f"\nEntities in top community '{top_report.get('title', '?')}':")
    for title in members:
        print(f"  - {title}")


# -- Main orchestrator ----------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Exercise 2: Community Structure Analysis")
    print("=" * 60)

    communities = load_communities()
    reports = load_community_reports()
    entities = load_entities()

    print(f"\nLoaded: {len(communities)} communities, "
          f"{len(reports)} reports, {len(entities)} entities")
    print(f"  community columns: {list(communities.columns)}")
    print(f"  report columns   : {list(reports.columns)}")

    print("\n" + "-" * 60)
    _print_level_stats(communities)
    _print_top_reports(reports)
    _print_top_community_members(communities, reports, entities)


if __name__ == "__main__":
    main()
