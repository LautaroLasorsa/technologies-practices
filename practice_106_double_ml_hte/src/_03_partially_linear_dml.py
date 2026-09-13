"""Phase 3 -- The partially linear DML moment, and the headline comparison.

This phase assembles the three estimators the whole practice compares:
naive (Phase 1), orthogonal-but-not-cross-fit, and full DML (orthogonal +
Phase 2's cross-fitting). Same data, same true ATE, three answers -- the
Monte Carlo loop below makes the bias each one carries visible rather than
asserted.

Run on its own to see all three estimators against the known ATE:
    uv run python -m src._03_partially_linear_dml
"""
from __future__ import annotations

import numpy as np

from ._01_naive_bias import default_model_y, naive_plugin_theta
from ._02_cross_fitting import cross_fit_residuals, default_model_d
from .datasets import load_dataset


# TODO(human) -- the Neyman-orthogonal partially-linear moment
# ---------------------------------------------------------------------------
# Goal: implement Robinson's (1988) residual-on-residual regression -- the
# partially linear model's Neyman-orthogonal moment condition,
#   E[(Y - g0(X) - theta*D) * (D - m0(X))] = 0,
# solved in its residualized form
#   theta_hat = Cov(D_resid, Y_resid) / Var(D_resid).
#
# Why this matters: unlike Phase 1's naive moment (regress the Y-residual
# on raw D), this one residualizes BOTH sides against X first. That single
# change is what makes the moment first-order insensitive to nuisance
# estimation error (Neyman orthogonality): perturb g0 or m0 slightly in the
# direction of their ML estimation error, and the derivative of this moment
# with respect to that perturbation is exactly zero at the truth, whereas
# Phase 1's naive moment has no such property. Concretely, once Y_resid and
# D_resid are already residualized (by Phase 2's cross-fitting -- or, with
# the bias this phase's comparison exposes, by same-sample nuisances), this
# step is just Frisch-Waugh-Lovell: theta_hat is the slope of a bivariate
# OLS of Y_resid on D_resid, no intercept needed since both residuals are
# already demeaned by construction.
#
# Steps:
#   1. theta_hat = sum(D_resid * Y_resid) / sum(D_resid ** 2).
# ---------------------------------------------------------------------------
def dml_theta(Y_resid: np.ndarray, D_resid: np.ndarray) -> float:
    """Residual-on-residual OLS slope -- the partially linear DML estimate of theta.

    Both inputs must already be residualized against X (by cross-fitting
    or otherwise). Returns the scalar theta_hat.
    """
    raise NotImplementedError("TODO(human): implement the residual-on-residual moment")


def dml_without_crossfit(X: np.ndarray, D: np.ndarray, Y: np.ndarray) -> float:
    """Orthogonal moment, but nuisances fit on the FULL sample (no splitting).

    Isolates the own-observation-overfitting failure mode Phase 2 exists to
    fix: same moment as `full_dml_theta`, same nuisance model classes, only
    the sample-splitting is removed.
    """
    model_y, model_d = default_model_y(), default_model_d()
    model_y.fit(X, Y)
    model_d.fit(X, D)
    Y_resid = Y - model_y.predict(X)
    D_resid = D - model_d.predict_proba(X)[:, 1]
    return dml_theta(Y_resid, D_resid)


def full_dml_theta(X: np.ndarray, D: np.ndarray, Y: np.ndarray, n_folds: int = 5, seed: int = 0) -> float:
    """Full DML: cross-fitted nuisances (Phase 2) + orthogonal moment (this phase)."""
    Y_resid, D_resid = cross_fit_residuals(X, D, Y, n_folds=n_folds, seed=seed)
    return dml_theta(Y_resid, D_resid)


def bias_monte_carlo(n_reps: int = 200, n: int = 500, seed: int = 0) -> dict[str, np.ndarray]:
    """Monte Carlo: redraw the dataset `n_reps` times, run all three
    estimators on each draw, and return each method's array of
    (theta_hat - true_ate) biases. Fully scaffolded -- the estimators being
    compared are the teaching content, not this loop."""
    rng = np.random.default_rng(seed)
    biases: dict[str, list[float]] = {"naive": [], "dml_no_crossfit": [], "full_dml": []}
    for _ in range(n_reps):
        data = load_dataset(n=n, seed=int(rng.integers(0, 2**31 - 1)))
        biases["naive"].append(naive_plugin_theta(data.X, data.D, data.Y, default_model_y()) - data.ate_true)
        biases["dml_no_crossfit"].append(dml_without_crossfit(data.X, data.D, data.Y) - data.ate_true)
        biases["full_dml"].append(full_dml_theta(data.X, data.D, data.Y) - data.ate_true)
    return {k: np.array(v) for k, v in biases.items()}


def main() -> None:
    data = load_dataset(n=1000, seed=0)
    print(f"True ATE: {data.ate_true:.3f}\n")
    try:
        print(f"Naive:              {naive_plugin_theta(data.X, data.D, data.Y, default_model_y()):.3f}")
        print(f"DML, no cross-fit:  {dml_without_crossfit(data.X, data.D, data.Y):.3f}")
        print(f"Full DML:           {full_dml_theta(data.X, data.D, data.Y):.3f}")
    except NotImplementedError as e:
        print(f"(skipped -- {e})")


if __name__ == "__main__":
    main()
