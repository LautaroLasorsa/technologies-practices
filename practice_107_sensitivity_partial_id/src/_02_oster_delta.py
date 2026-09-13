"""Phase 2 — Oster's delta and the R-max bound.

Rosenbaum bounds (Phase 1) ask "how much hidden bias would break this
result." Oster's (2019) approach asks a complementary question using
information already in hand: *how much does the coefficient move* when
you go from a "short" regression (no controls) to a "long" one (with the
observed controls `x1`, `x2`)? If including observable controls barely
moves the coefficient, and those controls only explain a modest share of
the outcome's variance, Oster's method lets you bound how much an
*unobservable* confounder -- assumed no more predictive than the
observables, up to a chosen ceiling `R_max` -- could still be biasing the
estimate.

Run on its own to see delta computed for this practice's confounded design:
    uv run python -m src._02_oster_delta
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm

from .datasets import load_dataset


@dataclass
class OsterInputs:
    """The four numbers Oster's delta is computed from."""

    beta_short: float  # coefficient on treatment, no controls
    beta_long: float  # coefficient on treatment, with controls
    r2_short: float  # R^2 of the short regression
    r2_long: float  # R^2 of the long regression


def fit_short_and_long(data) -> OsterInputs:
    """Fit `y ~ t` (short) and `y ~ t + x1 + x2` (long) via OLS.

    Fully scaffolded -- running two OLS regressions is not the teaching
    point; what you do with their four summary numbers is.
    """
    df = data.df
    y = df["y"].to_numpy()
    t = df["t"].to_numpy()

    x_short = sm.add_constant(t)
    fit_short = sm.OLS(y, x_short).fit()

    x_long = sm.add_constant(df[["t", "x1", "x2"]].to_numpy())
    fit_long = sm.OLS(y, x_long).fit()

    return OsterInputs(
        beta_short=fit_short.params[1],
        beta_long=fit_long.params[1],
        r2_short=fit_short.rsquared,
        r2_long=fit_long.rsquared,
    )


# TODO(human) — Oster's delta and the bias-adjusted coefficient
# ---------------------------------------------------------------------------
# Goal: implement Oster's (2019) `delta` statistic, plus the bias-adjusted
# coefficient it implies at `delta = 1`.
#
# Why this matters: coefficient stability under added controls is a common
# informal check ("the effect barely moved when I added controls, so it's
# probably not confounded") -- Oster formalizes it. The intuition: if
# going from no controls to the observed controls moved `beta` by a little
# while moving R^2 by a lot, then an *unobservable* confounder with
# similar explanatory power to the observables would have to move `beta`
# proportionally *less* to fully explain away the effect -- `delta` is
# exactly the multiple of "unobservable selection relative to observable
# selection" needed to drive the bias-adjusted coefficient to zero.
#
# Formula (Oster, 2019, "Unobservable Selection and Coefficient Stability",
# eq. 3, solved for delta such that beta*(delta) = 0):
#
#   delta = beta_long * (r_max - r2_long)
#           / ((beta_short - beta_long) * (r2_long - r2_short))
#
# where `r_max` is the researcher-chosen ceiling on how much of the
# outcome's variance *all* regressors (observed + unobservable) could
# jointly explain (Oster suggests `r_max = min(1.3 * r2_long, 1.0)` as a
# common default -- also implement that helper below).
#
# The bias-adjusted coefficient at a hypothesized `delta` (Oster eq. 2):
#
#   beta_star(delta) = beta_long
#       - delta * (beta_short - beta_long) * (r_max - r2_long) / (r2_long - r2_short)
#
# Guard against `r2_long - r2_short` being ~0 (undefined delta; controls
# added no explanatory power) by returning `np.inf` with the observed sign
# of `(beta_short - beta_long)` in that case.
# ---------------------------------------------------------------------------
def oster_delta(inputs: OsterInputs, r_max: float | None = None) -> tuple[float, float]:
    """Oster's delta, and the bias-adjusted coefficient at `delta = 1`.

    `r_max` defaults to `min(1.3 * r2_long, 1.0)` (Oster's common default)
    when not given. Returns `(delta, beta_star_at_delta_1)`.
    """
    raise NotImplementedError("TODO(human): implement Oster's delta and beta_star(delta)")


def main() -> None:
    data = load_dataset(n=1000, confound_strength=1.2, seed=0)
    inputs = fit_short_and_long(data)
    print(f"beta_short={inputs.beta_short:.3f}  R2_short={inputs.r2_short:.4f}")
    print(f"beta_long={inputs.beta_long:.3f}  R2_long={inputs.r2_long:.4f}")
    try:
        delta, beta_star = oster_delta(inputs)
        print(f"delta={delta:.3f}  beta_star(delta=1)={beta_star:.3f}  (true ATE: {data.ate_true})")
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
