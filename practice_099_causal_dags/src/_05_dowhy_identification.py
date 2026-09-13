"""Phase 5 — cross-checking with DoWhy's identification API.

Phases 1-2 implemented d-separation and the backdoor criterion from
scratch. This phase hands the *same* graphs to `DoWhy` and asks it to
identify the effect on its own — DoWhy searches over candidate adjustment
sets using exactly the backdoor (and, where applicable, frontdoor)
criterion Phase 2 implemented, so its answer is a check on your own
reasoning, not a replacement for it. Fully scaffolded: calling a library's
public API isn't the taught technique, the graphical reasoning underneath
it is (Phases 1-2).

Run on its own to print DoWhy's identified estimand for each scenario:
    uv run python -m src._05_dowhy_identification
"""
from __future__ import annotations

from dowhy import CausalModel

from .datasets import GRAPHS, Graph, load_dataset


def _to_gml(graph: Graph) -> str:
    """Convert this practice's plain adjacency-list `Graph` into the GML
    string DoWhy's `CausalModel(graph=...)` expects."""
    nodes = {n for n in graph} | {c for children in graph.values() for c in children}
    node_lines = "\n".join(f'node [ id "{n}" label "{n}" ]' for n in nodes)
    edge_lines = "\n".join(
        f'edge [ source "{parent}" target "{child}" ]' for parent, children in graph.items() for child in children
    )
    return f"graph [ directed 1 {node_lines} {edge_lines} ]"


def identify_effect_dowhy(scenario: str, n: int = 1000, seed: int = 0) -> str:
    """Build a DoWhy `CausalModel` for `scenario` and return its identified estimand as text.

    `scenario` must be a key in `src.datasets.GRAPHS` (only nodes present in
    the loaded dataframe are used — latent nodes like the mbias scenario's
    U1/U2 are graph-only and excluded from DoWhy's `data=`).
    """
    data = load_dataset(scenario, n=n, seed=seed)
    graph = GRAPHS[scenario]
    model = CausalModel(
        data=data.df,
        treatment=data.treatment,
        outcome=data.outcome,
        graph=_to_gml(graph),
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    return str(estimand)


def main() -> None:
    for scenario in ("confounder", "mediator", "collider"):
        print(f"=== {scenario} ===")
        print(identify_effect_dowhy(scenario))
        print()


if __name__ == "__main__":
    main()
