"""Phase 3 — Fisher's sharp null and randomization inference.

Fisher's (1935) approach to inference asks a different question than
Neyman's: instead of estimating a variance formula, assume the **sharp
null** `H0: Y_i(1) = Y_i(0)` for *every* unit `i` (not just on average —
literally no unit's outcome would have changed under the other
treatment). Under that null, the observed outcome doesn't depend on which
units happened to be assigned to treatment, so the treatment labels could
have been shuffled among the same fixed outcomes in any of `C(n, n1)` ways
and produced an equally likely dataset. Repeatedly reshuffling the labels,
recomputing the same test statistic each time, and comparing the
*actually observed* statistic to that reshuffled distribution gives an
exact p-value with no distributional assumptions whatsoever — the
"randomization inference" or "permutation test" approach.

Run on its own to see the observed statistic against its null distribution:
    uv run python -m src._03_randomization_inference
"""
from __future__ import annotations

import numpy as np

from ._01_diff_in_means import difference_in_means
from .datasets import load_dataset


# TODO(human) — randomization (Fisher exact) test
# ---------------------------------------------------------------------------
# Goal: implement the randomization-inference loop under Fisher's sharp
# null. Reassign the treatment label at random (preserving the same
# number of treated units as the real design — complete randomization),
# recompute the diff-in-means statistic on the *same, fixed* outcome
# vector each time, and collect the resulting null distribution.
#
# Why this matters: this is a fundamentally different justification for
# inference than Phase 2's Neyman formula — it needs no variance formula
# and no asymptotic approximation, only the physical fact that the actual
# treatment assignment was one random draw among many equally-likely
# ones. It is also the direct computational cousin of what a real
# experiment's assignment mechanism *did*: Phase 1's `difference_in_means`
# is reused unchanged as the test statistic here.
#
# Steps:
#   1. Compute the observed statistic: `difference_in_means(y, treatment)`.
#   2. Repeat `n_perm` times: draw a fresh random permutation of the
#      `treatment` vector with `rng.permutation(treatment)` (this reshuffles
#      the *existing* 0/1 labels among units — it does not change how many
#      are treated) and recompute `difference_in_means(y, permuted)`.
#   3. Collect the `n_perm` reshuffled statistics into a numpy array.
#   4. Two-sided p-value: the fraction of reshuffled statistics at least as
#      extreme (in absolute value) as the observed one:
#      `mean(abs(null_stats) >= abs(observed_stat))`.
#   5. Return `(observed_stat, null_stats, p_value)`.
# ---------------------------------------------------------------------------

def randomization_test(
    y: np.ndarray, treatment: np.ndarray, n_perm: int, rng: np.random.Generator
) -> tuple[float, np.ndarray, float]:
    """Fisher's sharp-null randomization test via label reshuffling.

    Returns `(observed_stat, null_stats, p_value)` where `null_stats` has
    shape `(n_perm,)`.
    """

    observed_stat=difference_in_means(y,treatment)
    null_stats = []
    for _ in range(n_perm):
        null_treatment = rng.permutation(treatment)
        null_stats.append(difference_in_means(y,null_treatment))
    
    null_stats = np.array(null_stats)
    p = np.average(abs(null_stats)>=abs(observed_stat))
    return (observed_stat, null_stats, p)


def main() -> None:
    data = load_dataset(n=500, tau=2.0, seed=0)
    rng = np.random.default_rng(1)
    try:
        observed, null_stats, p_value = randomization_test(
            data.y_observed, data.treatment, n_perm=2000, rng=rng
        )
        print(f"Observed statistic:  {observed:.4f}")
        print(f"Null mean/std:       {null_stats.mean():.4f} / {null_stats.std():.4f}")
        print(f"Two-sided p-value:   {p_value:.4f}")
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
