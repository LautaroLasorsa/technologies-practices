"""RDD datasets: synthetic (known ground truth) and the real Lee (2008) data.

Three synthetic scenarios share one data-generating process (DGP) with a
*known* true jump `TRUE_TAU`, so every phase can compare an estimator's
output against ground truth instead of against nothing — the same
"simulate with a known effect, show the estimator recovers it" pattern used
throughout this curriculum. The real Lee (2008) dataset (vendored under
`data/lee2008.csv`, see this practice's CLAUDE.md for provenance) has no
known ground truth — there, the check is against a published estimate and
against `rdrobust`, not against a DGP. No TODO here: data generation and
loading are infrastructure, not the taught technique.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

# True jump in E[Y|X] at the cutoff, shared by every synthetic scenario.
# "sharp" and "manipulated" realize it directly (D jumps 0->1 at the
# cutoff); "fuzzy" realizes it only through an imperfect first stage, so
# the *reduced-form* jump in Y is smaller than TRUE_TAU (see Phase 5).
TRUE_TAU = 4.0


@dataclass
class RDDData:
    """One RDD dataset: a running variable, an outcome, and observed treatment."""

    running: np.ndarray  # (n,) running/forcing variable, centered so the cutoff is 0
    outcome: np.ndarray  # (n,) outcome Y
    treatment: np.ndarray  # (n,) observed treatment D in {0, 1}; deterministic under "sharp", not under "fuzzy"
    cutoff: float = 0.0
    covariate: np.ndarray | None = None  # pre-treatment covariate, for the balance/placebo check (Phase 4)
    tau_true: float | None = None  # known DGP jump; None for the real Lee (2008) data

    def as_frame(self) -> pd.DataFrame:
        """Tidy pandas view, mainly for notebook `.head()` display."""
        cols = {"running": self.running, "outcome": self.outcome, "treatment": self.treatment}
        if self.covariate is not None:
            cols["covariate"] = self.covariate
        return pd.DataFrame(cols)


def _smooth_trend(x: np.ndarray) -> np.ndarray:
    """The (nonlinear, but locally smooth) baseline relationship shared by
    every synthetic scenario. A global polynomial fit over the *whole*
    support would need a high order to track this curvature; a local
    linear fit restricted to a small window either side of the cutoff
    only ever sees an approximately linear slice of it — which is exactly
    why local linear regression is the recommended RDD estimator (Phase 1)
    and fitting one global high-order polynomial is not (Gelman & Imbens,
    2019). The quadratic term also gives the trend nonzero curvature at
    the cutoff, which Phase 3's bandwidth selector needs to estimate."""
    return 5.0 * x + 3.0 * x**2 - 2.0 * x**3


def load_dataset(scenario: str = "sharp", n: int = 2000, cutoff: float = 0.0, seed: int = 0) -> RDDData:
    """Generate a synthetic RDD dataset for the given scenario.

    scenario:
      - "sharp": treatment is a deterministic function of the running
        variable, `D = 1{X >= cutoff}` — the textbook sharp RDD design.
      - "fuzzy": crossing the cutoff only *shifts the probability* of
        treatment (an imperfect first stage), so `D` is stochastic and the
        jump in `E[Y|X]` at the cutoff understates `TRUE_TAU` — recovering
        it needs the Wald ratio (Phase 5), not a direct sharp-RDD read.
      - "manipulated": like "sharp", but the running variable's *density*
        has been given a discontinuity at the cutoff (extra mass sorted
        just to the right) — a synthetic case where the McCrary test
        (Phase 4) should reject continuity, in contrast to Lee (2008)'s
        real data where it should not.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)

    if scenario == "manipulated":
        near = (x >= cutoff - 0.15) & (x < cutoff)
        flip = near & (rng.random(n) < 0.7)
        x = np.where(flip, rng.uniform(cutoff, cutoff + 0.05, size=n), x)

    noise = rng.normal(scale=2.0, size=n)
    covariate = 0.8 * x + rng.normal(scale=1.0, size=n)

    if scenario in ("sharp", "manipulated"):
        treatment = (x >= cutoff).astype(float)
    elif scenario == "fuzzy":
        prop_score = np.clip(0.15 + 0.55 * (x >= cutoff) + 0.10 * x, 0.0, 1.0)
        treatment = rng.binomial(1, prop_score).astype(float)
    else:
        raise ValueError(f"Unknown scenario: {scenario!r}")

    outcome = _smooth_trend(x) + TRUE_TAU * treatment + noise
    return RDDData(running=x, outcome=outcome, treatment=treatment, cutoff=cutoff, covariate=covariate, tau_true=TRUE_TAU)


def load_lee2008() -> RDDData:
    """Load the real Lee (2008) US House elections dataset.

    `running` is the Democratic vote-share margin in election t-1
    (positive = Democrat won, so the cutoff is exactly 0); `outcome` is
    the Democratic vote share in election t; `treatment` is "Democratic
    incumbent party going into election t", `1{running >= 0}` — a sharp
    design by construction (a party cannot half-win a seat). There is no
    `tau_true`: this is real data, and the published incumbency-advantage
    estimate (Lee 2008; Imbens & Kalyanaraman 2012) is roughly 0.07-0.09
    (7-9 percentage points), not a value baked into a DGP. See this
    practice's CLAUDE.md for the dataset's provenance.
    """
    df = pd.read_csv(DATA_DIR / "lee2008.csv")
    treatment = (df["x"].to_numpy() >= 0.0).astype(float)
    return RDDData(
        running=df["x"].to_numpy(),
        outcome=df["y"].to_numpy(),
        treatment=treatment,
        cutoff=0.0,
        covariate=None,
        tau_true=None,
    )
