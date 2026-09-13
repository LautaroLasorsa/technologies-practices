"""Phase 2 — Simplex-constrained synthetic-control weights.

This is the core of the method: choosing donor weights `w` so the weighted
donor average tracks California's *pre-treatment* outcome path as closely
as possible, then applying those same (fixed) weights to the donors'
post-treatment path to get the counterfactual "what would California have
looked like without Prop 99." Everything else in this practice (the gap,
the placebo test) is built on top of this one optimization.

Run on its own to fit the weights and cross-check against `pysyncon` when
it's importable:
    uv run python -m src._02_synthetic_weights
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from ._01_fit_loss import rmspe

# `trust-constr`'s quasi-Newton Hessian update warns "delta_grad == 0.0"
# whenever a step doesn't change the (analytically constant) gradient of
# this quadratic objective — harmless here, but noisy across ~38 placebo
# refits (Phase 4), so it's silenced once at import time rather than left
# for every learner to rediscover.
warnings.filterwarnings("ignore", message="delta_grad == 0.0.*")
from .datasets import SyntheticControlData, TREATMENT_YEAR, load_dataset


@dataclass
class SyntheticControlFit:
    """Result of fitting one synthetic control to a `SyntheticControlData`."""

    weights: np.ndarray      # (J,) donor weights; non-negative, sums to 1
    y_synthetic: np.ndarray  # (T,) synthetic path, ALL years (pre + post)
    pre_rmspe: float         # pre-treatment fit quality (lower is better)


# TODO(human) — simplex-constrained weight optimization
# ---------------------------------------------------------------------------
# Goal: solve for the donor weights that best reproduce California's
# pre-treatment outcome path, as a convex combination of donors.
#
# Why this matters — the OR framing: this is a quadratic program over the
# probability simplex,
#
#     minimize_w   || y1_pre - Y0_pre @ w ||^2
#     subject to   w_j >= 0  for all j,   sum_j w_j = 1
#
# where `y1_pre` is the treated unit's pre-treatment outcome vector and
# `Y0_pre` stacks every donor's pre-treatment outcome as columns. The
# simplex constraint is not a modeling nicety — it is what makes the
# result interpretable and (crucially) what stops the fit from
# *extrapolating*: a convex combination of donors can never predict a value
# outside the range the donors themselves span at each time point, unlike
# an unconstrained or negative-weight OLS fit, which can. The published
# method (Abadie, Diamond & Hainmueller 2010) nests this exact problem
# (there called the "W" problem) inside an outer problem that also chooses
# how much weight each *predictor* gets (the "V" problem) — see this
# practice's CLAUDE.md, "What this practice simplifies," for why that outer
# loop is intentionally skipped here.
#
# Steps:
#   1. Use `scipy.optimize.minimize` with `method="trust-constr"` — donor
#      pools are routinely larger than the pre-treatment window (here: 38
#      donors, 19 pre-treatment years), which makes `Y0_pre.T @ Y0_pre`
#      rank-deficient. `"SLSQP"`'s quasi-Newton subproblem is unreliable on
#      that flat a loss surface (it can report false convergence right at
#      the starting point); `"trust-constr"` handles the bound + linear
#      constraints directly and converges reliably here.
#   2. Objective: sum of squared residuals, `np.sum((y1_pre - Y0_pre @ w)**2)`.
#   3. `bounds=Bounds(0, 1)` (enforces `w_j >= 0`; the upper bound of 1 is
#      not load-bearing given the equality constraint below, but keeps the
#      optimizer's search region bounded).
#   4. `constraints=LinearConstraint(np.ones((1, J)), 1, 1)` — a single
#      linear constraint pinning `sum(w)` between 1 and 1, i.e. `== 1`.
#   5. Initial guess: equal weights, `np.full(J, 1 / J)` — a point already
#      inside the feasible region.
#   6. Return `result.x` (the fitted weight vector), not the raw
#      `OptimizeResult`.
# ---------------------------------------------------------------------------
def solve_synthetic_weights(Y0_pre: np.ndarray, y1_pre: np.ndarray) -> np.ndarray:
    """Fit simplex-constrained donor weights against a pre-treatment window.

    `Y0_pre` is `(T0, J)` (donors as columns), `y1_pre` is `(T0,)`. Returns
    the `(J,)` weight vector: non-negative, summing to 1 (up to solver
    tolerance).
    """
    raise NotImplementedError("TODO(human): implement the simplex-constrained QP")


def fit_synthetic_control(data: SyntheticControlData) -> SyntheticControlFit:
    """Fit weights on the pre-treatment window, then apply them to the
    donors' full (pre+post) panel to get the synthetic counterfactual path.
    Fully scaffolded — this is orchestration around `solve_synthetic_weights`,
    not the taught technique."""
    Y0_pre = data.Y_donors[data.pre_mask]
    y1_pre = data.y_treated[data.pre_mask]
    w = solve_synthetic_weights(Y0_pre, y1_pre)
    y_synthetic = data.Y_donors @ w
    pre_fit = rmspe(y1_pre, y_synthetic[data.pre_mask])
    return SyntheticControlFit(weights=w, y_synthetic=y_synthetic, pre_rmspe=pre_fit)


def compare_to_pysyncon(data: SyntheticControlData) -> None:
    """Cross-check against `pysyncon`'s `Synth`, configured to match the
    same objective our hand-rolled QP solves: every pre-treatment year of
    the outcome, weighted equally (`custom_V=` a vector of ones), with no
    aggregated covariates. `pysyncon` additionally rescales each row by its
    cross-sectional standard deviation before optimizing (a stabilization
    step our version skips), so an exact weight-for-weight match isn't
    expected — comparable *pre-treatment RMSPE* is the meaningful check.
    Skips gracefully if `pysyncon` isn't importable (see this practice's
    CLAUDE.md for the Windows-install caveat)."""
    try:
        import pandas as pd
        from pysyncon import Synth
    except ImportError as e:
        print(f"(skipped — pysyncon not importable: {e})")
        return

    pre_years = data.years[data.pre_mask]
    X0 = pd.DataFrame(data.Y_donors[data.pre_mask], index=pre_years, columns=data.donor_names)
    X1 = pd.Series(data.y_treated[data.pre_mask], index=pre_years, name="California")
    custom_V = np.ones(len(pre_years))

    synth = Synth()
    synth.fit(X0=X0, X1=X1, Z0=X0, Z1=X1, custom_V=custom_V)

    ours = fit_synthetic_control(data)
    theirs_synth = data.Y_donors @ synth.W
    theirs_pre_rmspe = rmspe(data.y_treated[data.pre_mask], theirs_synth[data.pre_mask])
    print(f"Our pre-treatment RMSPE:      {ours.pre_rmspe:.3f}")
    print(f"pysyncon pre-treatment RMSPE: {theirs_pre_rmspe:.3f}")
    print(f"Max |weight diff| (informational, not expected to be ~0): "
          f"{np.max(np.abs(ours.weights - synth.W)):.3f}")


def main() -> None:
    data = load_dataset()
    try:
        fit = fit_synthetic_control(data)
    except NotImplementedError as e:
        print(f"(skipped — {e})")
        return
    top = sorted(zip(data.donor_names, fit.weights), key=lambda kv: -kv[1])[:5]
    print(f"Pre-treatment RMSPE: {fit.pre_rmspe:.3f}")
    print("Top 5 donor weights:")
    for name, w in top:
        print(f"  {name:16s} {w:.3f}")
    print(f"weights sum to: {fit.weights.sum():.6f}  (should be ~1.0)")
    print(f"Treatment year: {TREATMENT_YEAR}\n")
    compare_to_pysyncon(data)


if __name__ == "__main__":
    main()
