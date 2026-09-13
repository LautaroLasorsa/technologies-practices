"""Phase 5 — Doubly robust (AIPW) estimation.

Phase 4's pure IPW estimator is only as good as the propensity model; a
pure outcome-regression estimator is only as good as the outcome model.
Augmented IPW (AIPW) hedges between the two: it stays consistent if
*either* model is correctly specified, not both (Robins, Rotnitzky & Zhao,
1994) — the "doubly robust" property.

Run on its own to see the AIPW ATE on the observational LaLonde comparison:
    uv run python -m src._05_aipw
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from ._01_propensity_score import fit_propensity_score
from .datasets import COVARIATES, OUTCOME, load_lalonde


def fit_outcome_regressions(
    X: np.ndarray, treat: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit separate linear outcome models on the treated/control arms and
    predict *both* potential outcomes for every unit. Plumbing, not the
    taught technique — AIPW's contribution is how these predictions get
    combined with the propensity weights below."""
    mu1_model = LinearRegression().fit(X[treat == 1], y[treat == 1])
    mu0_model = LinearRegression().fit(X[treat == 0], y[treat == 0])
    return mu1_model.predict(X), mu0_model.predict(X)


# TODO(human) — augmented inverse probability weighting (AIPW / doubly robust)
# ---------------------------------------------------------------------------
# Goal: implement the AIPW estimator, combining the outcome-regression
# predictions (mu1, mu0, already fit above) with an IPW-weighted residual
# correction term.
#
# Why this matters: AIPW starts from the outcome-model prediction and
# *augments* it with a propensity-weighted correction built from each
# unit's actual outcome, so a misspecified outcome model gets corrected by
# the propensity weights (or a misspecified propensity model gets
# corrected by the outcome model) — whichever one is right is enough.
#
# Per-unit AIPW pseudo-outcome (Robins et al., 1994; Glynn & Quinn, 2010,
# eq. 6):
#   psi_i = (mu1_i - mu0_i)
#           + treat_i * (y_i - mu1_i) / e_i
#           - (1 - treat_i) * (y_i - mu0_i) / (1 - e_i)
# The AIPW estimate is mean(psi_i) over all n units.
# ---------------------------------------------------------------------------
def aipw_ate(
    y: np.ndarray, treat: np.ndarray, propensity: np.ndarray, mu1: np.ndarray, mu0: np.ndarray
) -> float:
    """Augmented IPW (doubly robust) ATE estimate.

    `mu1`/`mu0` are outcome-model predictions for every unit under
    treatment/control (from `fit_outcome_regressions`); `propensity` is
    e(X). Returns a single scalar ATE estimate.
    """
    raise NotImplementedError("TODO(human): implement the AIPW estimator")


def main() -> None:
    data = load_lalonde()
    obs = data.observational
    X = obs[COVARIATES].to_numpy()
    treat = obs["treat"].to_numpy()
    y = obs[OUTCOME].to_numpy()
    try:
        fit = fit_propensity_score(X, treat)
        mu1, mu0 = fit_outcome_regressions(X, treat, y)
        tau = aipw_ate(y, treat, fit.scores, mu1, mu0)
        print(f"AIPW ATE estimate: {tau:,.0f}  (experimental ATE = {data.tau_experimental:,.0f})")
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
