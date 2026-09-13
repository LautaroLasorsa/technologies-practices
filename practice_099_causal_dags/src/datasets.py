"""Synthetic causal-DAG datasets with a known ground-truth effect.

Every scenario here is a small structural causal model (SCM) with a *known*
true effect of `D` (treatment) on `Y` (outcome), so every phase can compare
an identification strategy's estimate against the ground truth instead of
against nothing. Each scenario also ships its true DAG as a plain adjacency
list (`Graph`) — the same representation the d-separation/backdoor checkers
in `_01_dsep.py`/`_02_backdoor.py` operate on, and what `_05_dowhy_identification.py`
and the notebook's DAG-drawing cells consume for layout. A fixed seed
(`numpy.random.default_rng`) makes every run reproducible. No TODO here:
data generation is infrastructure, not the taught technique — see
`_01_dsep.py` through `_04_mbias_selection.py` for the actual exercises.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

Graph = dict[str, list[str]]  # node -> list of children (directed edges)

# True effect of D on Y shared by the scenarios where it applies directly
# (confounder, collider-via-nothing, mbias). The mediator scenario instead
# decomposes into a direct + indirect path — see MEDIATOR_DIRECT/MEDIATOR_PATH.
TRUE_ATE = 2.0

# Mediator scenario: D -> M -> Y (indirect path) plus D -> Y (direct path).
# Total effect = MEDIATOR_DIRECT + MEDIATOR_D_TO_M * MEDIATOR_M_TO_Y.
MEDIATOR_DIRECT = 2.0
MEDIATOR_D_TO_M = 0.6
MEDIATOR_M_TO_Y = 0.9

# Fixed 2-D layout for drawing each scenario's DAG with xy.pyplot (scatter +
# annotate) — see src/plotting.py::draw_dag and this practice's CLAUDE.md
# "DAG drawing" note for why a hand-picked layout is used instead of an
# automatic graph-layout algorithm.
LAYOUTS: dict[str, dict[str, tuple[float, float]]] = {
    "confounder": {"Z": (0.5, 1.0), "D": (0.0, 0.0), "Y": (1.0, 0.0)},
    "mediator": {"D": (0.0, 0.0), "M": (0.5, 0.6), "Y": (1.0, 0.0)},
    "collider": {"D": (0.0, 0.0), "Y": (1.0, 0.0), "C": (0.5, -0.6)},
    "mbias": {
        "U1": (0.0, 1.0),
        "U2": (1.0, 1.0),
        "D": (0.0, 0.0),
        "M": (0.5, 0.6),
        "Y": (1.0, 0.0),
    },
}

GRAPHS: dict[str, Graph] = {
    "confounder": {"Z": ["D", "Y"], "D": ["Y"], "Y": []},
    "mediator": {"D": ["M", "Y"], "M": ["Y"], "Y": []},
    "collider": {"D": ["C"], "Y": ["C"], "C": []},
    "mbias": {"U1": ["D", "M"], "U2": ["M", "Y"], "D": ["Y"], "M": [], "Y": []},
}


@dataclass
class CausalData:
    """One simulated causal-DAG dataset."""

    df: pd.DataFrame  # observed columns only — latent nodes (e.g. U1/U2) are never exposed
    graph: Graph  # the true DAG, including latent nodes, for the identification exercises
    treatment: str = "D"
    outcome: str = "Y"
    true_effect: float = TRUE_ATE
    true_effect_label: str = "true ATE"


def load_dataset(scenario: str, n: int = 500, seed: int = 0) -> CausalData:
    """Generate a synthetic causal dataset for the given DAG scenario.

    scenario:
      - "confounder": Z confounds D and Y (Z -> D, Z -> Y, D -> Y). The
        backdoor path D <- Z -> Y biases the naive D~Y association; adjusting
        for Z removes it.
      - "mediator": D -> M -> Y plus a direct D -> Y edge. Adjusting for M
        blocks the indirect path, recovering the *direct* effect only —
        correct if that's what you want, wrong if you wanted the *total*
        effect.
      - "collider": D and Y have NO causal effect on each other; both cause
        C (D -> C <- Y). D and Y are marginally independent, but
        conditioning on their common effect C induces a spurious
        association — bias created from nothing.
      - "mbias": two independent latent causes U1 (of D and M) and U2 (of M
        and Y), plus a true D -> Y edge. M is a collider between U1 and U2.
        Not conditioning on M leaves the D-Y association unbiased; adding M
        as a "control" opens the collider path U1 -> M <- U2 and biases it —
        a correct model made worse by an extra covariate.
    """
    rng = np.random.default_rng(seed)

    if scenario == "confounder":
        z = rng.normal(size=n)
        d = 1.5 * z + rng.normal(size=n)
        y = TRUE_ATE * d + 1.0 * z + rng.normal(size=n)
        df = pd.DataFrame({"Z": z, "D": d, "Y": y})
        return CausalData(df=df, graph=GRAPHS["confounder"], true_effect=TRUE_ATE)

    if scenario == "mediator":
        d = rng.normal(size=n)
        m = MEDIATOR_D_TO_M * d + rng.normal(size=n)
        y = MEDIATOR_DIRECT * d + MEDIATOR_M_TO_Y * m + rng.normal(size=n)
        df = pd.DataFrame({"D": d, "M": m, "Y": y})
        total = MEDIATOR_DIRECT + MEDIATOR_D_TO_M * MEDIATOR_M_TO_Y
        return CausalData(df=df, graph=GRAPHS["mediator"], true_effect=total, true_effect_label="true TOTAL effect")

    if scenario == "collider":
        d = rng.normal(size=n)
        y = rng.normal(size=n)  # no causal link to D at all
        c = 1.2 * d + 1.2 * y + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"D": d, "Y": y, "C": c})
        return CausalData(df=df, graph=GRAPHS["collider"], true_effect=0.0)

    if scenario == "mbias":
        u1 = rng.normal(size=n)
        u2 = rng.normal(size=n)
        d = 1.2 * u1 + rng.normal(size=n)
        m = 1.0 * u1 + 1.0 * u2 + rng.normal(size=n)
        y = TRUE_ATE * d + 1.2 * u2 + rng.normal(size=n)
        df = pd.DataFrame({"D": d, "M": m, "Y": y})  # U1/U2 stay latent — never exposed
        return CausalData(df=df, graph=GRAPHS["mbias"], true_effect=TRUE_ATE)

    raise ValueError(f"Unknown scenario: {scenario!r}")
