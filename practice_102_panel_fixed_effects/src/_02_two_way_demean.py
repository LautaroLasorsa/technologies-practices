"""Phase 2 — Two-way fixed effects: demeaning by unit *and* time.

Phase 1's one-way within estimator removes `alpha_i` (the unit effect),
but this practice's DGP also has `gamma_t` (a time effect) that trends
upward together with `x2`'s rollout. One-way FE leaves that trend in the
error term, still correlated with `x2` — so it's still biased for beta2.
Two-way FE removes both: subtract each unit's mean, each period's mean,
and add back the grand mean (otherwise you'd subtract the grand mean
twice). For a *balanced* panel (every unit observed in every period, true
of this practice's synthetic data), that closed-form double-demeaning
formula is exact; an *unbalanced* panel needs iterative alternating
projections instead (Guimaraes & Portugal, 2010) — pyfixest's/
`reghdfe`'s real workhorse, which is why the practice's speed benchmark
(Phase 5) uses pyfixest rather than hand-rolled iteration once panels get
large or unbalanced.

Run on its own to see the two-way estimator matched against pyfixest and
linearmodels.PanelOLS:
    uv run python -m src._02_two_way_demean
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._01_within_transform import OLSResult, ols_lstsq
from .datasets import SMALL_PANEL, generate_panel


# TODO(human) — two-way (unit + time) demeaning
# ---------------------------------------------------------------------------
# Goal: implement the closed-form two-way within transformation for a
# *balanced* panel:
#   v_tilde_it = v_it - mean_i(v) - mean_t(v) + mean(v)
# applied to y and to every column of X, where mean_i is the unit's own
# mean, mean_t is the period's own mean across units, and mean() is the
# grand mean over all observations.
#
# Why this matters: subtracting only the unit mean (Phase 1) leaves any
# common time-varying shock in the residual; subtracting only the time
# mean would leave unit heterogeneity in. Subtracting both handles each,
# but subtracts the grand mean twice in the process (it's included in
# both mean_i and mean_t) — adding it back once is what makes the formula
# exact rather than an approximation. This only works because the panel
# is balanced; see this module's docstring for the unbalanced case.
#
# Steps:
#   1. Compute mean_i(v): group by unit_id, transform("mean"), for y and
#      each column of X.
#   2. Compute mean_t(v): group by time_id, transform("mean"), same.
#   3. Compute mean(v): the plain overall mean.
#   4. v_tilde = v - mean_i(v) - mean_t(v) + mean(v).
#   5. Return (y_tilde, X_tilde) as numpy arrays.
# ---------------------------------------------------------------------------
def two_way_demean(
    y: np.ndarray, X: np.ndarray, unit_id: np.ndarray, time_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract unit and time means (adding back the grand mean once).

    `unit_id` and `time_id` are (n,) arrays of labels, one per row.
    Returns `(y_tilde, X_tilde)`, the doubly-demeaned arrays. Requires a
    balanced panel (every unit observed in every period).
    """
    raise NotImplementedError("TODO(human): implement the two-way (unit + time) demeaning transformation")


def fit_two_way(
    df: pd.DataFrame, y_col: str, x_cols: list[str], unit_col: str, time_col: str
) -> OLSResult:
    """Demean by `unit_col` and `time_col`, then fit OLS on the result."""
    y = df[y_col].to_numpy()
    X = df[x_cols].to_numpy()
    unit_id = df[unit_col].to_numpy()
    time_id = df[time_col].to_numpy()
    y_tilde, X_tilde = two_way_demean(y, X, unit_id, time_id)
    return ols_lstsq(X_tilde, y_tilde)


def validate_against_libraries(df: pd.DataFrame) -> None:
    """Fit the two-way estimator, pyfixest, and linearmodels.PanelOLS on the
    same panel; assert all three agree on beta to numerical precision."""
    import pyfixest as pf
    from linearmodels.panel import PanelOLS

    ours = fit_two_way(df, "y", ["x1", "x2"], "unit_id", "time_id")

    fixest_fit = pf.feols("y ~ x1 + x2 | unit_id + time_id", data=df)
    fixest_beta = fixest_fit.coef().to_numpy()

    panel_df = df.set_index(["unit_id", "time_id"])
    lm_fit = PanelOLS(
        panel_df["y"], panel_df[["x1", "x2"]], entity_effects=True, time_effects=True
    ).fit()
    lm_beta = lm_fit.params.to_numpy()

    print(f"beta (ours):         {ours.beta}")
    print(f"beta (pyfixest):     {fixest_beta}")
    print(f"beta (linearmodels): {lm_beta}")
    assert np.allclose(ours.beta, fixest_beta, atol=1e-6), "two-way vs pyfixest mismatch"
    assert np.allclose(ours.beta, lm_beta, atol=1e-6), "two-way vs linearmodels mismatch"


def main() -> None:
    panel = generate_panel(**SMALL_PANEL, seed=0)
    print(f"True beta: {panel.beta_true}\n")
    try:
        validate_against_libraries(panel.df)
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
