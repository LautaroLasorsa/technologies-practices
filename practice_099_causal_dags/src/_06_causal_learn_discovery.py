"""Phase 6 — structure discovery with causal-learn.

Every earlier phase *assumed* the DAG was known and reasoned forward from
it (which sets identify the effect). This phase runs the other direction:
given only observational data (no graph), can a constraint-based discovery
algorithm recover the skeleton we already know is true? The PC algorithm
(Spirtes & Glymour, 1991) tests conditional independencies between every
pair of variables and removes edges whose implied independence holds in the
data, converging on an equivalence class of DAGs consistent with those
tests. Fully scaffolded: running a library's discovery algorithm isn't the
taught technique — recognizing *why* it can't fully distinguish the
confounder/mediator/collider shapes from data alone is the point (all three
three-node structures encode the same conditional-independence pattern up
to orientation, which is exactly why a known graph, not just data, is
needed for identification).

Run on its own to see the learned skeleton for each scenario:
    uv run python -m src._06_causal_learn_discovery
"""
from __future__ import annotations

from causallearn.search.ConstraintBased.PC import pc

from .datasets import load_dataset


def discover_skeleton(scenario: str, n: int = 2000, seed: int = 0, alpha: float = 0.05) -> str:
    """Run the PC algorithm on `scenario`'s simulated data and return a
    human-readable summary of the learned graph's edges."""
    data = load_dataset(scenario, n=n, seed=seed)
    columns = list(data.df.columns)
    result = pc(data.df.to_numpy(), alpha=alpha, indep_test="fisherz")
    lines = []
    for i, ci in enumerate(columns):
        for j, cj in enumerate(columns):
            if i < j and result.G.graph[i, j] != 0:
                lines.append(f"{ci} -- {cj}")
    return "\n".join(lines) if lines else "(no edges recovered at this alpha)"


def main() -> None:
    for scenario in ("confounder", "mediator", "collider"):
        print(f"=== {scenario}: PC-algorithm skeleton ===")
        print(discover_skeleton(scenario))
        print()
    print(
        "Note: confounder, mediator, and (unconditional) collider skeletons\n"
        "look identical from data alone — d-separation constraints don't\n"
        "distinguish them without extra assumptions (e.g. no unmeasured\n"
        "confounding, or a known temporal order). Structure discovery narrows\n"
        "the search; it doesn't replace domain knowledge of the true DAG."
    )


if __name__ == "__main__":
    main()
