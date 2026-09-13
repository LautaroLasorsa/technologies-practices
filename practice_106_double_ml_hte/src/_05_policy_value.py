"""Phase 5 -- From CATE to policy: treatment assignment and its value.

A CATE estimate only matters if it changes a decision. This phase turns
tau_hat(x) into a budget-constrained treatment-assignment POLICY (treat
whoever benefits most, subject to a budget) and evaluates that policy's
value on the observed data via inverse-propensity weighting -- without
ever needing to re-run the DGP under the new policy.

Run on its own to see the policy value across a grid of budgets:
    uv run python -m src._05_policy_value
"""
from __future__ import annotations

import numpy as np

from ._04_meta_learner import t_learner_cate
from .datasets import DMLData, load_dataset


# TODO(human) -- policy value under a budget-constrained treatment rule
# ---------------------------------------------------------------------------
# Goal: turn a CATE estimate into a treatment-assignment POLICY (treat the
# `budget` fraction of units with the highest tau_hat) and evaluate that
# policy's VALUE on the observed data using inverse-propensity weighting
# (IPW) -- without needing a randomized experiment on the new policy.
#
# Why this matters: the natural decision rule here is "treat whoever
# benefits most, subject to a budget" -- but you cannot just re-run the DGP
# under the new policy to see how well it does. The IPW policy-value
# estimator answers this from the SAME observational data the CATE was
# estimated on:
#   V(pi) = (1/n) * sum_i  1[D_i == pi(X_i)] / p_i(X_i)  *  Y_i
# where pi(X_i) in {0,1} is the policy's decision for unit i, and
# p_i(X_i) = e(X_i) if pi(X_i)==1 else (1 - e(X_i)) is the probability of
# observing the treatment status the policy would have assigned. Only
# units whose observed treatment matches what the policy would have chosen
# contribute (weighted up by how rare that match was) -- the same
# importance-weighting idea used in IPW/AIPW average-effect estimators,
# applied here to a whole decision rule instead of a single contrast.
#
# Steps:
#   1. Rank units by tau_hat, descending; the policy treats the top
#      `budget` fraction (pi=1 for those, pi=0 otherwise). Use
#      `np.argsort(-tau_hat)` and treat the first
#      `int(np.ceil(budget * len(tau_hat)))` ranked units.
#   2. p_i = propensity where pi==1, (1 - propensity) where pi==0.
#   3. matches = (D == pi).astype(float).
#   4. V_hat = np.mean(matches / p_i * Y).
# ---------------------------------------------------------------------------
def policy_value(tau_hat: np.ndarray, Y: np.ndarray, D: np.ndarray, propensity: np.ndarray, budget: float) -> float:
    """IPW estimate of the value of "treat the top `budget` fraction by tau_hat".

    `budget` is a fraction in [0, 1]. Returns the scalar policy value
    (higher is better -- an IPW estimate of E[Y] the population would
    realize under this policy).
    """
    raise NotImplementedError("TODO(human): implement the budget-constrained policy value")


def policy_value_curve(
    tau_hat: np.ndarray, data: DMLData, budgets: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Policy value across a grid of budgets from 0 (treat no one) to 1
    (treat everyone). Fully scaffolded -- the value estimator itself
    (`policy_value`, above) is the teaching content, not this sweep."""
    if budgets is None:
        budgets = np.linspace(0.0, 1.0, 11)
    values = np.array([policy_value(tau_hat, data.Y, data.D, data.propensity_true, b) for b in budgets])
    return budgets, values


def main() -> None:
    data = load_dataset(n=1000, seed=0)
    try:
        tau_hat = t_learner_cate(data.X, data.D, data.Y)
    except NotImplementedError as e:
        print(f"(skipped -- Phase 4 not implemented yet -- {e})")
        return
    try:
        budgets, values = policy_value_curve(tau_hat, data)
        for b, v in zip(budgets, values):
            print(f"budget={b:.1f}  V_hat={v:.3f}")
    except NotImplementedError as e:
        print(f"(skipped -- {e})")


if __name__ == "__main__":
    main()
