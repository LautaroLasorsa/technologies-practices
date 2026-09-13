"""Phase 2 — Neyman's variance estimator and its conservatism.

Neyman (1923) derived the variance of the difference-in-means estimator
under complete randomization directly from the design — no distributional
assumptions about `Y(0)`/`Y(1)` are needed, only that treatment assignment
is a random draw of a fixed number of units. The result,
`Var(tau_hat) = Var(Y(1))/n1 + Var(Y(0))/n0`, is estimated by plugging in
the two groups' *sample* variances. The catch: the true design-based
variance also has a `-Var(tau_i)/n` term (from the individual treatment
effects' heterogeneity) that Neyman's estimator cannot identify — because
`tau_i` is never jointly observed for any unit, that term can't be
estimated and is simply dropped. Dropping a nonnegative term makes the
estimator (weakly) **conservative**: it never *understates* the true
sampling variance, so confidence intervals built from it are always
valid, if occasionally wider than necessary.

Run on its own to see the Neyman standard error next to the true ATE:
    uv run python -m src._02_neyman_variance
"""
from __future__ import annotations

import numpy as np

from ._01_diff_in_means import difference_in_means
from .datasets import load_dataset


# TODO(human) — Neyman variance estimator for the diff-in-means ATE
# ---------------------------------------------------------------------------
# Goal: implement Neyman's (1923) design-based variance estimator for the
# difference-in-means estimator under complete randomization:
#   Var_hat(tau_hat) = s1^2 / n1 + s0^2 / n0
# where `s1^2`/`s0^2` are the *sample* variances (ddof=1) of the observed
# outcome within the treated/control groups, and `n1`/`n0` are the group
# sizes.
#
# Why this matters: this formula looks identical to the two-sample t-test
# variance from intro statistics, but the derivation is different — it
# comes from the randomization design itself (which units happened to be
# drawn into the treated group), not from an assumed outcome distribution.
# It is also provably **conservative**: the true finite-population
# variance has an extra `-Var(tau_i)/n` term that can't be estimated
# because `tau_i` is never observed for any single unit (Phase 1's
# science-table caveat, made quantitative). Dropping that nonnegative term
# means this estimator's expectation is >= the true variance — safe, but
# not tight when treatment effects are very heterogeneous.
#
# Steps:
#   1. Split `y` into treated/control groups by `treatment`.
#   2. Compute each group's sample variance with `ddof=1` and divide by its
#      group size.
#   3. Return the sum of the two terms.
# ---------------------------------------------------------------------------
def neyman_variance(y: np.ndarray, treatment: np.ndarray) -> float:
    """Neyman's conservative variance estimate for the diff-in-means ATE.

    Returns a single float; `sqrt(...)` gives the standard error used for
    a 95% CI as `tau_hat +/- 1.96 * se`.
    """
    raise NotImplementedError("TODO(human): implement Neyman's variance estimator")


def main() -> None:
    data = load_dataset(n=500, tau=2.0, seed=0)
    ate_hat = difference_in_means(data.y_observed, data.treatment)
    try:
        var_hat = neyman_variance(data.y_observed, data.treatment)
        se_hat = np.sqrt(var_hat)
        lo, hi = ate_hat - 1.96 * se_hat, ate_hat + 1.96 * se_hat
        print(f"True ATE:            {data.tau_true:.4f}")
        print(f"Estimate:            {ate_hat:.4f}")
        print(f"Neyman SE:           {se_hat:.4f}")
        print(f"95% CI:              [{lo:.4f}, {hi:.4f}]")
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
