"""Phase 4 — Placebo (permutation) inference.

A synthetic-control estimate has no textbook standard error — there's one
treated unit, so there's no sampling distribution to invoke. The field's
answer (Abadie, Diamond & Hainmueller 2010) is a placebo/permutation test:
re-run the *exact same* method on every donor in turn, pretending each one
was treated when it wasn't, and see how unusual California's own
post/pre-treatment fit ratio looks against that donor-generated null
distribution. If California's ratio isn't unusual compared to the placebo
donors', the "effect" might just be estimation noise a state could produce
even with no intervention at all.

Run on its own for the full placebo test:
    uv run python -m src._04_placebo_inference
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._01_fit_loss import rmspe
from ._02_synthetic_weights import fit_synthetic_control, solve_synthetic_weights
from .datasets import SyntheticControlData, load_dataset


@dataclass
class PlaceboResult:
    """Full placebo distribution plus the true unit's rank within it."""

    donor_names: list[str]  # (J,) matches rows of `gaps`/`ratios`
    gaps: np.ndarray        # (J, T) each donor's own gap path, treated as if it were California
    ratios: np.ndarray      # (J,) each donor's post/pre RMSPE ratio
    true_gap: np.ndarray    # (T,) California's real gap path
    true_ratio: float       # California's own post/pre RMSPE ratio
    p_value: float


def _placebo_fit(data: SyntheticControlData, treated_col: int) -> tuple[np.ndarray, float]:
    """Refit the synthetic control with donor `treated_col` as the
    placebo-treated unit and every OTHER donor as its donor pool (the real
    California is excluded from every placebo donor pool — it really was
    treated, so it can't stand in as an untreated comparison state). Fully
    scaffolded — reuses Phase 1/2's `rmspe`/`solve_synthetic_weights`
    directly, the same functions used for the real treated unit."""
    y1 = data.Y_donors[:, treated_col]
    other_cols = [j for j in range(data.Y_donors.shape[1]) if j != treated_col]
    Y0 = data.Y_donors[:, other_cols]

    w = solve_synthetic_weights(Y0[data.pre_mask], y1[data.pre_mask])
    y_synth = Y0 @ w
    gap = y1 - y_synth
    pre = rmspe(y1[data.pre_mask], y_synth[data.pre_mask])
    post = rmspe(y1[data.post_mask], y_synth[data.post_mask])
    ratio = post / pre if pre > 0 else np.inf
    return gap, ratio


# TODO(human) — permutation p-value
# ---------------------------------------------------------------------------
# Goal: turn a list of placebo post/pre RMSPE ratios plus the true treated
# unit's own ratio into a single p-value.
#
# Why this matters: the post/pre RMSPE ratio is the standard one-number
# summary for this test (Abadie, Diamond & Hainmueller 2010) — it rewards a
# large post-treatment gap (a real effect) while *penalizing* a donor that
# never fit well pre-treatment in the first place (a large ratio purely
# from a bad pre-fit is not evidence of an effect). The p-value is exact
# and distribution-free by construction: it's a rank, not a normal-theory
# calculation, which is exactly right here since there's no asymptotic
# justification for one treated unit.
#
# Steps:
#   1. Concatenate `true_ratio` with every value in `placebo_ratios` into
#      one array of "all units, including the true one."
#   2. Count how many of those values are `>= true_ratio` (this count
#      includes the true unit itself — it always counts at least once).
#   3. Divide by the total number of units (true + placebos) to get the
#      p-value: the fraction of units at least as extreme as the true one.
#      A p-value of, e.g., 0.05 means only 1 in 20 units (including
#      California) produced as large a post/pre ratio — evidence the
#      effect is unusual, not an artifact of the method itself.
# ---------------------------------------------------------------------------
def permutation_p_value(true_ratio: float, placebo_ratios: np.ndarray) -> float:
    """Rank-based permutation p-value for one treated unit vs. its placebos.

    Returns the fraction of units (the true one plus every placebo) whose
    post/pre RMSPE ratio is `>= true_ratio` — the smaller this is, the more
    unusual the true unit's effect looks against the donor-generated null.
    """
    raise NotImplementedError("TODO(human): implement the permutation p-value")


def run_placebo_test(data: SyntheticControlData) -> PlaceboResult:
    """Fit the real synthetic control, then re-run the same method with
    every donor as the placebo-treated unit, and rank the true ratio
    against the resulting null distribution. Fully scaffolded — the loop
    is orchestration around already-implemented pieces; only the ranking
    itself (`permutation_p_value`) is a TODO."""
    fit = fit_synthetic_control(data)
    true_gap = data.y_treated - fit.y_synthetic
    true_pre = rmspe(data.y_treated[data.pre_mask], fit.y_synthetic[data.pre_mask])
    true_post = rmspe(data.y_treated[data.post_mask], fit.y_synthetic[data.post_mask])
    true_ratio = true_post / true_pre if true_pre > 0 else np.inf

    gaps, ratios = [], []
    for j in range(data.Y_donors.shape[1]):
        gap, ratio = _placebo_fit(data, j)
        gaps.append(gap)
        ratios.append(ratio)
    ratios = np.array(ratios)

    p_value = permutation_p_value(true_ratio, ratios)
    return PlaceboResult(
        donor_names=data.donor_names,
        gaps=np.array(gaps),
        ratios=ratios,
        true_gap=true_gap,
        true_ratio=true_ratio,
        p_value=p_value,
    )


def main() -> None:
    data = load_dataset()
    try:
        result = run_placebo_test(data)
    except NotImplementedError as e:
        print(f"(skipped — {e})")
        return
    rank = int(round(result.p_value * (len(result.ratios) + 1)))
    print(f"California post/pre RMSPE ratio: {result.true_ratio:.2f}")
    print(f"Rank among {len(result.ratios) + 1} units (1 = most extreme): {rank}")
    print(f"Permutation p-value: {result.p_value:.3f}")


if __name__ == "__main__":
    main()
