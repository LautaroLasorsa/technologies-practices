"""Phase 3 — Frisch-Waugh-Lovell: why demeaning works.

Phases 1-2 asserted that demeaning gives the same beta as including a full
set of unit (and time) dummy variables directly ("LSDV"), without proof.
The Frisch-Waugh-Lovell theorem (Frisch & Waugh, 1933; generalized by
Lovell, 1963) is *why* that's true, and it's a general statement about
multiple regression, not a fixed-effects-specific trick: in a regression
of y on [X_focal, W], the coefficient on X_focal equals the coefficient
from (1) regressing y on W and keeping the residuals, (2) regressing each
column of X_focal on W and keeping the residuals, and (3) regressing the
residuals from (1) on the residuals from (2). "Partialling out" W from
both y and X_focal, then regressing what's left, recovers exactly the
coefficient X_focal would have gotten if W had been included directly.

Demeaning *is* an application of this theorem with W set to a full set of
unit dummy columns: subtracting the unit mean is algebraically identical
to partialling out those dummies (the projection onto a dummy-variable
subspace is exactly the group-mean projection). This phase implements the
generic partialling-out routine and checks that using unit dummies as W
reproduces Phase 1's within-estimator beta, without ever demeaning
anything.

Run on its own to see the FWL route match Phase 1:
    uv run python -m src._03_fwl_demonstration
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._01_within_transform import fit_within, ols_lstsq
from .datasets import SMALL_PANEL, generate_panel


def build_unit_dummies(unit_id: np.ndarray) -> np.ndarray:
    """One-hot dummy matrix for `unit_id`, one column per unique unit.

    Fully scaffolded — building dummies is bookkeeping, not the theorem
    this phase is teaching. `pandas.get_dummies` is deliberately not used
    so the shape (n, n_units) is unambiguous for the caller.
    """
    units = np.unique(unit_id)
    return (unit_id[:, None] == units[None, :]).astype(float)


# TODO(human) — generic Frisch-Waugh-Lovell partialling-out
# ---------------------------------------------------------------------------
# Goal: implement the general FWL partialling-out routine: given y, a
# "focal" regressor matrix X_focal (the coefficients you actually want),
# and a "control" matrix W (everything else in the model, e.g. dummy
# columns), recover the same beta on X_focal that a single regression of
# y on [X_focal, W] would produce — without ever running that combined
# regression.
#
# Why this matters: this is the *general* theorem the within transformation
# is a special case of. Proving it here (with W = unit dummies) is what
# turns "demeaning happens to give the right answer" into "demeaning gives
# the right answer *because* it's projecting out the same subspace that
# including dummies directly would project out."
#
# Steps:
#   1. Regress y on W (`ols_lstsq(W, y)`), keep the residual: y_perp.
#   2. For each column of X_focal, regress it on W and keep the residual
#      (do this for all columns at once: `ols_lstsq` works column-by-column
#      if you loop, or you can regress the whole X_focal matrix on W in
#      one call and take `X_focal - W @ beta_hat` as the residual matrix —
#      either is fine). Call the result X_focal_perp.
#   3. Regress y_perp on X_focal_perp (`ols_lstsq(X_focal_perp, y_perp)`);
#      its beta is the FWL answer.
#   4. Return that beta.
# ---------------------------------------------------------------------------
def fwl_partial_out(y: np.ndarray, X_focal: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Recover X_focal's coefficient by partialling out W from both y and X_focal.

    Returns the (k,) beta on `X_focal` — identical to what a single
    regression of `y` on `[X_focal, W]` would give X_focal's columns.
    """
    raise NotImplementedError("TODO(human): implement the FWL partialling-out routine")


def demonstrate_equivalence(df: pd.DataFrame) -> None:
    """Compare the FWL route (W = unit dummies) against Phase 1's within estimator."""
    y = df["y"].to_numpy()
    X_focal = df[["x1", "x2"]].to_numpy()
    W = build_unit_dummies(df["unit_id"].to_numpy())

    fwl_beta = fwl_partial_out(y, X_focal, W)
    within_beta = fit_within(df, "y", ["x1", "x2"], "unit_id").beta

    print(f"beta (FWL, W = unit dummies): {fwl_beta}")
    print(f"beta (within, Phase 1):       {within_beta}")
    assert np.allclose(fwl_beta, within_beta, atol=1e-6), "FWL should exactly match the within estimator"


def main() -> None:
    panel = generate_panel(**SMALL_PANEL, seed=0)
    print(f"True beta: {panel.beta_true}\n")
    try:
        demonstrate_equivalence(panel.df)
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
