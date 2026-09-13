"""Phase 2 — the backdoor criterion.

A "backdoor path" from treatment to outcome is any path that starts with an
arrow *into* the treatment (D <- ...) — it represents confounding, not the
causal effect itself. The backdoor criterion (Pearl, 1993) gives a purely
graphical test for when adjusting for a set Z removes all such bias: Z must
block every backdoor path, and Z must not include any descendant of the
treatment (adjusting for a mediator or a downstream effect of D throws away
part of the very effect you're trying to measure).

Run on its own to check a few candidate adjustment sets on the confounder
and mediator graphs:
    uv run python -m src._02_backdoor
"""
from __future__ import annotations

from ._01_dsep import is_d_separated
from .datasets import GRAPHS, Graph


def _descendants(graph: Graph, node: str) -> set[str]:
    """All descendants of `node` (exclusive) — plain BFS over `graph`'s
    child-lists."""
    seen: set[str] = set()
    frontier = list(graph.get(node, []))
    while frontier:
        child = frontier.pop()
        if child not in seen:
            seen.add(child)
            frontier.extend(graph.get(child, []))
    return seen


def _remove_outgoing_edges(graph: Graph, node: str) -> Graph:
    """A copy of `graph` with every edge *out of* `node` deleted — this is
    the "surgery" that turns "is this a backdoor path" into a plain
    d-separation question (Phase 1)."""
    return {n: ([] if n == node else list(children)) for n, children in graph.items()}


# TODO(human) — the backdoor criterion
# ---------------------------------------------------------------------------
# Goal: implement Pearl's backdoor criterion as a two-part check, reusing
# Phase 1's `is_d_separated` rather than re-deriving path-blocking logic.
#
# Why this matters: this is the graphical test that tells you an adjustment
# set is *safe* — it's the formal version of "control for confounders, not
# for mediators or colliders" that the rest of this practice makes concrete
# with simulated bias. DoWhy's `identify_effect` (Phase 5) is, underneath its
# API, searching for a set that passes exactly this test.
#
# The criterion, precisely: `z` is a valid backdoor adjustment set for
# (treatment, outcome) iff both hold:
#   1. No node in `z` is a descendant of `treatment` (use the scaffolded
#      `_descendants` helper) — adjusting for a mediator or any downstream
#      effect of the treatment blocks part of the real causal path, not just
#      confounding.
#   2. `z` blocks every *backdoor* path from `treatment` to `outcome`. You
#      don't need to enumerate paths by hand: remove every edge *out of*
#      `treatment` (use the scaffolded `_remove_outgoing_edges` helper —
#      this deletes exactly the paths that start with the causal effect
#      itself, leaving only backdoor paths reachable), then check
#      `is_d_separated` between `treatment` and `outcome` given `z` in that
#      modified graph.
# ---------------------------------------------------------------------------
def is_valid_backdoor_set(graph: Graph, treatment: str, outcome: str, z: set[str]) -> bool:
    """Check whether `z` satisfies the backdoor criterion for `treatment` -> `outcome`.

    Returns `True` iff `z` contains no descendant of `treatment` AND `z`
    d-separates `treatment` from `outcome` once `treatment`'s outgoing edges
    are removed.
    """
    raise NotImplementedError("TODO(human): implement the backdoor criterion")


def main() -> None:
    checks = [
        ("confounder", "D", "Y", set(), False, "empty set leaves the Z-confounding path open"),
        ("confounder", "D", "Y", {"Z"}, True, "Z blocks the only backdoor path"),
        ("mediator", "D", "Y", {"M"}, False, "M is a descendant of D — invalid regardless of paths"),
        ("collider", "D", "Y", set(), True, "no backdoor path exists at all"),
        ("collider", "D", "Y", {"C"}, False, "C is a collider, not a confounder — invalid"),
    ]
    for scenario, treatment, outcome, z, expected, note in checks:
        graph = GRAPHS[scenario]
        try:
            result = is_valid_backdoor_set(graph, treatment, outcome, z)
            mark = "OK" if result == expected else "MISMATCH"
            print(f"[{scenario:11s}] backdoor({treatment},{outcome} | {sorted(z)}) = {result!s:5s} (expected {expected!s:5s}) {mark} — {note}")
        except NotImplementedError as e:
            print(f"(skipped — {e})")
            break


if __name__ == "__main__":
    main()
