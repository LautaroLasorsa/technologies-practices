"""Phase 2 — Covariate balance via standardized mean differences.

An estimated propensity score is only useful if it actually balances the
covariates. This phase implements the standard metric used to check that —
before touching any matching or weighting estimator.

Run on its own to see raw (pre-adjustment) balance on the observational
LaLonde comparison — expect large imbalances, since NSW and CPS-1 come from
very different populations:
    uv run python -m src._02_balance
"""
from __future__ import annotations

import numpy as np

from .datasets import COVARIATES, load_lalonde


# TODO(human) — standardized mean difference (SMD)
# ---------------------------------------------------------------------------
# Goal: implement the standard covariate-balance metric used to judge
# whether an adjustment method (matching, weighting) has made the treated
# and control groups comparable on observables.
#
# Why this matters: raw mean differences aren't comparable across
# covariates with different scales (age in years vs. re74 in dollars) — the
# standardized mean difference divides by a pooled standard deviation so
# every covariate's imbalance lands on the same unitless scale. The Love
# plot (Phase 6) plots exactly this quantity before and after adjustment;
# the usual rule of thumb is |SMD| < 0.1 counts as "balanced" (Austin,
# 2011).
#
# Formula (Rosenbaum & Rubin, 1985; Austin, 2011):
#   SMD = (mean(x | treat=1) - mean(x | treat=0))
#         / sqrt((var(x | treat=1) + var(x | treat=0)) / 2)
# using the *unweighted* variance of each raw group as the denominator even
# when the means themselves are computed on a weighted/matched sample — so
# SMDs before and after adjustment stay comparable to the same yardstick.
#
# `weights`, when given, reweights each group's *mean* only (the
# denominator variances stay unweighted, per the note above): use
# np.average(x, weights=w) for the weighted group mean.
# ---------------------------------------------------------------------------
def standardized_mean_diff(
    x: np.ndarray, treat: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Standardized mean difference of covariate `x` between treat=1/0.

    `weights`, if given, is a (n,) array of per-unit weights (e.g. IPW
    weights) used only for the group means; the denominator always uses
    the raw (unweighted) pooled standard deviation.
    """
    raise NotImplementedError("TODO(human): implement the standardized mean difference")


def balance_table(
    X: np.ndarray, treat: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """SMD for every covariate column — the numbers behind the Love plot."""
    return np.array(
        [standardized_mean_diff(X[:, j], treat, weights) for j in range(X.shape[1])]
    )


def main() -> None:
    data = load_lalonde()
    obs = data.observational
    X = obs[COVARIATES].to_numpy()
    treat = obs["treat"].to_numpy()
    try:
        smd = balance_table(X, treat)
        for name, val in zip(COVARIATES, smd):
            print(f"{name:10s} SMD = {val:+.3f}")
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
