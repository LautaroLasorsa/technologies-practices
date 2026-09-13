"""Phase 1 — Rosenbaum bounds for a matched design.

A matched-pairs analysis (Wilcoxon signed-rank test on treated-minus-control
differences) reports a p-value *assuming* matched pairs differ in treatment
status only by chance, given the covariates used to match them. That
assumption is untestable from the data itself: two units matched to look
identical on `x1`, `x2` could still differ systematically on an unobserved
confounder `u` (see `src/datasets.py`). Rosenbaum's (2002) sensitivity
bounds ask a different, answerable question: *how large would the odds of
differential treatment assignment from hidden bias (Gamma) have to be*
before the matched-pair result stopped being statistically significant?

Run on its own to see the critical Gamma for this practice's matched design:
    uv run python -m src._01_rosenbaum_bounds
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import rankdata

from .datasets import build_matched_pairs, load_dataset


# TODO(human) — Rosenbaum's Gamma-sensitivity p-value bound
# ---------------------------------------------------------------------------
# Goal: implement the upper bound on the one-sided Wilcoxon signed-rank
# p-value for a matched-pairs design, under a hidden bias of strength
# `gamma` (Rosenbaum, 2002, "Observational Studies", ch. 4).
#
# Why this matters: a plain Wilcoxon signed-rank test on the matched-pair
# differences assumes each pair's treated/control assignment was as good as
# random (odds ratio 1) given the matched covariates. Gamma relaxes that:
# it allows a hidden confounder to make one member of a pair up to `gamma`
# times more likely to have received treatment than the other, purely by
# chance in who was assigned which unit. At `gamma = 1` this collapses back
# to the ordinary Wilcoxon test; the point of this function is the
# worst-case p-value *at a given gamma > 1*, which grows as gamma grows.
#
# Procedure:
#   1. Drop zero differences (`diffs == 0`); they carry no information.
#   2. Rank the *absolute* differences 1..m (`rankdata` handles ties).
#      Let `w_plus` be the sum of ranks for pairs with a positive
#      difference -- this is the observed Wilcoxon signed-rank statistic.
#   3. Under a hidden bias of `gamma`, the *least favorable* configuration
#      (the one Rosenbaum shows maximizes the upper-bound p-value) assigns
#      every pair the same worst-case probability
#      `p_plus = gamma / (1 + gamma)` of the positive-signed member being
#      the one that could have been treated by chance.
#   4. Under that worst case, `w_plus` has (approximately, for the normal
#      approximation to the signed-rank null) mean
#      `E = p_plus * sum(ranks)` and variance
#      `V = p_plus * (1 - p_plus) * sum(ranks ** 2)`.
#   5. z = (w_plus - E - 0.5) / sqrt(V)   (the -0.5 is a continuity
#      correction, standard for this normal approximation).
#   6. p_value = 1 - stats.norm.cdf(z)   (one-sided: testing whether the
#      matched design shows treated > control).
# ---------------------------------------------------------------------------
def rosenbaum_pvalue_bound(diffs: np.ndarray, gamma: float) -> float:
    """Upper bound on the one-sided Wilcoxon signed-rank p-value at hidden-bias strength `gamma`.

    `diffs` are the matched-pair (treated - control) outcome differences.
    Returns a p-value in `[0, 1]`; larger `gamma` values only ever push the
    bound *up* (harder to reject the null of no effect).
    """
    raise NotImplementedError("TODO(human): implement Rosenbaum's Gamma-sensitivity p-value bound")


def find_critical_gamma(diffs: np.ndarray, alpha: float = 0.05, gamma_grid: np.ndarray | None = None) -> float:
    """Smallest `gamma` on `gamma_grid` at which the p-value bound crosses `alpha`.

    Fully scaffolded: this is just a search loop over `rosenbaum_pvalue_bound`,
    not a new idea -- the estimator above is the teaching point.
    """
    if gamma_grid is None:
        gamma_grid = np.arange(1.0, 5.01, 0.05)
    for gamma in gamma_grid:
        if rosenbaum_pvalue_bound(diffs, gamma) > alpha:
            return float(gamma)
    return float(gamma_grid[-1])


def main() -> None:
    data = load_dataset(n=1000, confound_strength=1.2, seed=0)
    diffs = build_matched_pairs(data)
    print(f"Matched pairs: {len(diffs)}, mean diff: {diffs.mean():.3f} (true ATE: {data.ate_true})")
    try:
        for gamma in (1.0, 1.5, 2.0, 3.0):
            p = rosenbaum_pvalue_bound(diffs, gamma)
            print(f"Gamma={gamma:.1f}  upper-bound p-value={p:.4f}")
        gamma_star = find_critical_gamma(diffs)
        print(f"Critical Gamma (p crosses 0.05): {gamma_star:.2f}")
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
