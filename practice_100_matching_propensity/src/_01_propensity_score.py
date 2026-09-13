"""Phase 1 — Propensity score estimation.

The propensity score e(X) = P(T=1 | X) is the probability a unit received
treatment, given its observed covariates. Every estimator in this practice
(matching, IPW, AIPW) is built on top of e(X) estimated here first.

Run on its own to see the fitted propensity scores' range on the
observational LaLonde comparison:
    uv run python -m src._01_propensity_score
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .datasets import COVARIATES, load_lalonde


@dataclass
class PropensityFit:
    """Fitted propensity scores plus the model that produced them."""

    scores: np.ndarray  # (n,) P(treat=1 | X) for every unit
    model: LogisticRegression


# TODO(human) — propensity score model
# ---------------------------------------------------------------------------
# Goal: estimate e(X) = P(T=1 | X) with a logistic regression on the
# observed covariates, and return the fitted probability for every unit
# (treated and control alike).
#
# Why this matters: Rosenbaum & Rubin (1983) showed the propensity score is
# a *balancing score* — conditioning on e(X) alone balances the entire
# covariate vector X between treatment arms, exactly like randomization
# would have. That's what makes matching/weighting *on e(X)* a workable
# substitute for matching on all of X directly, which becomes infeasible
# once X has more than a couple of continuous covariates (the curse of
# dimensionality).
#
# Steps:
#   1. Standardize the covariates with the already-imported StandardScaler
#      — logistic regression's log-loss surface is easier to optimize on a
#      comparable scale, and this dataset mixes ages (~20-55) with incomes
#      (up to ~$25,000).
#   2. Fit sklearn's LogisticRegression on the scaled covariates and treat.
#   3. Return predict_proba(...)[:, 1] — P(treat=1 | X) — for every row.
#
# Scores near 0 or 1 signal poor overlap (visible in Phase 6's density
# plot); nothing here needs to guard against that, just return the raw
# probabilities.
# ---------------------------------------------------------------------------
def fit_propensity_score(X: np.ndarray, treat: np.ndarray) -> PropensityFit:
    """Fit a logistic-regression propensity score model.

    `X` is the (n, p) covariate matrix, `treat` the (n,) 0/1 assignment.
    Returns the fitted propensity score for every row plus the model.
    """
    raise NotImplementedError("TODO(human): implement the propensity score model")


def main() -> None:
    data = load_lalonde()
    obs = data.observational
    X = obs[COVARIATES].to_numpy()
    treat = obs["treat"].to_numpy()
    try:
        fit = fit_propensity_score(X, treat)
        print(
            f"Propensity scores: min={fit.scores.min():.3f} "
            f"max={fit.scores.max():.3f} mean={fit.scores.mean():.3f}"
        )
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
