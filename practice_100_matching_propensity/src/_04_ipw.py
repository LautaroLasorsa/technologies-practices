"""Phase 4 — Inverse probability weighting (IPW).

Instead of discarding poorly matched units (Phase 3), IPW reweights every
unit by the inverse of its probability of receiving the treatment arm it
actually got, so the reweighted sample looks, in expectation, like the full
population had been randomized.

Run on its own to see the IPW ATE on the observational LaLonde comparison:
    uv run python -m src._04_ipw
"""
from __future__ import annotations

import numpy as np

from ._01_propensity_score import fit_propensity_score
from .datasets import COVARIATES, OUTCOME, load_lalonde


# TODO(human) — inverse probability weighted ATE
# ---------------------------------------------------------------------------
# Goal: implement the Hajek (self-normalized) inverse-probability-weighted
# estimator of the average treatment effect.
#
# Why this matters: a treated unit that looked unlikely to be treated
# (small e(X)) is "surprising" and gets up-weighted; a control unit that
# looked likely to be treated gets up-weighted too. This reweighting makes
# the reweighted sample look, in expectation, like the full population had
# been randomized (Horvitz & Thompson, 1952; Rosenbaum, 1987). The Hajek
# version normalizes by the sum of weights within each arm (not by n),
# which is less variable in finite samples than the raw Horvitz-Thompson
# estimator and guarantees each arm's weights sum to 1.
#
# Steps:
#   1. Per-unit weight: w_i = treat_i / e_i + (1 - treat_i) / (1 - e_i).
#   2. Hajek-normalized group means:
#      mu1 = sum(treat_i * w_i * y_i) / sum(treat_i * w_i)
#      mu0 = sum((1 - treat_i) * w_i * y_i) / sum((1 - treat_i) * w_i)
#   3. ATE = mu1 - mu0.
# ---------------------------------------------------------------------------
def ipw_ate(y: np.ndarray, treat: np.ndarray, propensity: np.ndarray) -> float:
    """Hajek-normalized inverse-probability-weighted ATE.

    `propensity` is e(X) for every unit (as returned by
    `fit_propensity_score`). Returns a single scalar ATE estimate.
    """
    raise NotImplementedError("TODO(human): implement the IPW estimator")


def main() -> None:
    data = load_lalonde()
    obs = data.observational
    X = obs[COVARIATES].to_numpy()
    treat = obs["treat"].to_numpy()
    y = obs[OUTCOME].to_numpy()
    try:
        fit = fit_propensity_score(X, treat)
        tau = ipw_ate(y, treat, fit.scores)
        print(f"IPW ATE estimate: {tau:,.0f}  (experimental ATE = {data.tau_experimental:,.0f})")
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
