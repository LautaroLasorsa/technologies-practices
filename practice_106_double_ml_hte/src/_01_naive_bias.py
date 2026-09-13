"""Phase 1 -- The naive ML plug-in estimator, and why it's biased.

It is tempting to fit one flexible ML model of Y on X, treat the residual
as "the part of Y not explained by covariates", and attribute it to the
treatment D. This phase implements exactly that naive estimator so its
bias is visible (Phase 3), not asserted.

Run on its own to see the naive estimator's bias against the known ATE:
    uv run python -m src._01_naive_bias
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .datasets import load_dataset


def default_model_y():
    """Flexible nuisance model for E[Y|X] -- a small gradient-boosted
    regressor. Deliberately regularized (shallow trees, few estimators) so
    it behaves like a realistic applied-ML nuisance model, not an oracle."""
    return GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=0)


# TODO(human) -- naive plug-in estimator (ignores D when estimating g0(X))
# ---------------------------------------------------------------------------
# Goal: implement the "naive" ML-based estimator of the partially linear
# model  Y = theta*D + g0(X) + U  that appears as the biased baseline in
# Chernozhukov et al. (2018), "Double/Debiased Machine Learning for
# Treatment and Causal Parameters".
#
# Why this matters: it is tempting to fit a single flexible ML model
# g_hat(X) ~= E[Y|X] (ignoring D entirely), then attribute whatever is left
# over in the residual Y - g_hat(X) to the treatment D. This looks
# reasonable, but the *moment condition* it solves,
#   E[(Y - g0(X) - theta*D) * D] = 0,
# is NOT Neyman orthogonal in g0: differentiate it with respect to g0 in
# the direction (g_hat - g0) and the derivative does not vanish, because D
# is correlated with X (there is confounding/selection into treatment) and
# so is g_hat's regularization error. A first-order error in g_hat
# therefore translates into a first-order (non-vanishing, non-shrinking-
# with-n) error in theta_hat. This is "regularization bias": the very act
# of regularizing g_hat (needed for any flexible ML model to generalize)
# introduces just enough smoothing error, correlated with D through X, to
# bias theta_hat -- and it does not go away with more data the way
# ordinary sampling noise does.
#
# Steps:
#   1. Fit model_y on the FULL sample: model_y.fit(X, Y).
#   2. g_hat = model_y.predict(X) (in-sample -- no cross-fitting here,
#      that is a separate problem tackled in Phase 2).
#   3. resid = Y - g_hat.
#   4. Solve the *un-orthogonalized* moment for theta: regress resid on D
#      with an intercept, i.e. theta_hat = Cov(D, resid) / Var(D) (the
#      simple-regression closed form) -- D itself is used raw, it is never
#      residualized against X.
# ---------------------------------------------------------------------------
def naive_plugin_theta(X: np.ndarray, D: np.ndarray, Y: np.ndarray, model_y=None) -> float:
    """Naive plug-in estimate of the constant treatment effect theta.

    Fits `model_y` to predict Y from X alone (D is never residualized),
    then regresses the residual on D. Returns the scalar theta_hat.
    """
    raise NotImplementedError("TODO(human): implement the naive plug-in estimator")


def main() -> None:
    data = load_dataset(n=1000, seed=0)
    print(f"True ATE: {data.ate_true:.3f}\n")
    try:
        theta_hat = naive_plugin_theta(data.X, data.D, data.Y, default_model_y())
        print(f"Naive plug-in theta_hat: {theta_hat:.3f}  (bias = {theta_hat - data.ate_true:+.3f})")
    except NotImplementedError as e:
        print(f"(skipped -- {e})")


if __name__ == "__main__":
    main()
