"""Phase 1 — d-separation via the moralized ancestral graph.

D-separation is the graphical criterion that tells you which conditional
independencies a DAG implies — "is X independent of Y given Z, according to
this causal model?" It's the machinery underneath every identification
strategy in this practice: the backdoor criterion (Phase 2) is d-separation
in a graph with the treatment's outgoing edges removed, and DoWhy's
`identify_effect` (Phase 5) is doing this same check internally, just with a
more elaborate search over candidate adjustment sets.

Run on its own to see a few worked d-separation checks on the confounder,
mediator, and collider graphs:
    uv run python -m src._01_dsep
"""
from __future__ import annotations

from .datasets import GRAPHS, Graph


def _parents(graph: Graph) -> dict[str, list[str]]:
    """Invert `graph`'s child-lists into a parent lookup — every node that
    appears anywhere (as a key or as someone's child) gets an entry."""
    parents: dict[str, list[str]] = {n: [] for n in graph}
    for node, children in graph.items():
        for child in children:
            parents.setdefault(child, []).append(node)
    return parents


def _ancestors(graph: Graph, parents: dict[str, list[str]], nodes: set[str]) -> set[str]:
    """All ancestors of `nodes` (inclusive) — plain BFS walking `parents`."""
    seen = set(nodes)
    frontier = list(nodes)
    while frontier:
        node = frontier.pop()
        for p in parents.get(node, []):
            if p not in seen:
                seen.add(p)
                frontier.append(p)
    return seen


# TODO(human) — d-separation via the moralized ancestral graph
# ---------------------------------------------------------------------------
# Goal: implement the standard d-separation test (Lauritzen et al., 1990):
# X and Y are d-separated by Z in a DAG iff Z separates X from Y in the
# *moralized ancestral graph* over {X, Y} u Z. This is the algorithmic
# route real tools use instead of enumerating every path and classifying
# each collider/chain/fork by hand — it turns a graph-theoretic reasoning
# problem into a plain undirected-graph reachability check.
#
# Why this matters: every identification strategy in this curriculum reduces
# to "does conditioning on this set block the paths that would otherwise
# bias my estimate?" — d-separation is the precise, checkable version of
# that question, and it's what the backdoor criterion (Phase 2) and DoWhy's
# `identify_effect` (Phase 5) both compute under the hood.
#
# Steps:
#   1. Build the *ancestral graph*: the subgraph induced by nodes({x, y} u z)
#      and all of their ancestors (use the scaffolded `_ancestors` helper —
#      it already walks the parent lookup for you).
#   2. *Moralize* it: for every node in the ancestral graph, connect every
#      pair of its parents with an undirected edge ("marrying" co-parents),
#      then drop all edge directions. A node's parents come from the
#      original `graph`, restricted to the ancestral node set.
#   3. In the resulting undirected moral graph, remove every node in `z`,
#      then check whether `x` can still reach `y` (BFS/DFS over the
#      remaining undirected adjacency). If it *can't* reach `y`, `z`
#      d-separates them.
#
# Return True exactly when `x` and `y` are d-separated by `z` (i.e. `y` is
# NOT reachable from `x` in the moral graph with `z` removed).
# ---------------------------------------------------------------------------
def is_d_separated(graph: Graph, x: str, y: str, z: set[str]) -> bool:
    """Check whether `x` and `y` are d-separated by the node set `z` in `graph`.

    `graph` maps each node to its list of children (directed edges). Returns
    `True` iff every path between `x` and `y` is blocked by `z`.
    """
    raise NotImplementedError("TODO(human): implement d-separation via the moralized ancestral graph")


def main() -> None:
    checks = [
        ("confounder", "D", "Y", set(), False, "no adjustment: backdoor path via Z is open"),
        ("confounder", "D", "Y", {"Z"}, True, "adjusting for Z blocks the backdoor path"),
        ("mediator", "D", "Y", set(), False, "no adjustment: D -> Y is a direct dependence"),
        ("collider", "D", "Y", set(), True, "no adjustment: D and Y share no open path"),
        ("collider", "D", "Y", {"C"}, False, "conditioning on collider C OPENS a path"),
    ]
    for scenario, x, y, z, expected, note in checks:
        graph = GRAPHS[scenario]
        try:
            result = is_d_separated(graph, x, y, z)
            mark = "OK" if result == expected else "MISMATCH"
            print(f"[{scenario:11s}] d_sep({x},{y} | {sorted(z)}) = {result!s:5s} (expected {expected!s:5s}) {mark} — {note}")
        except NotImplementedError as e:
            print(f"(skipped — {e})")
            break


if __name__ == "__main__":
    main()
