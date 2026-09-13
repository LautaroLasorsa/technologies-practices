"""Phase 3 — Nearest-neighbor / caliper matching on the propensity score.

Matching directly on e(X) approximates what a randomized experiment does
within a narrow propensity-score stratum: each treated unit is paired with
the control unit(s) that looked most like it *before* treatment, and the
treatment effect is read off their outcome difference.

Run on its own to see the matched ATT on the observational LaLonde
comparison:
    uv run python -m src._03_matching
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._01_propensity_score import fit_propensity_score
from .datasets import COVARIATES, OUTCOME, load_lalonde


@dataclass
class MatchResult:
    """Result of 1:1 nearest-neighbor caliper matching."""

    matched_control_idx: np.ndarray  # (n_treated,) index into ps_control/y_control, or -1 if unmatched
    att: float  # average treatment effect on the treated, over matched pairs only


# TODO(human) — nearest-neighbor caliper matching
# ---------------------------------------------------------------------------
# Goal: implement 1:1 nearest-neighbor matching on the propensity score,
# with a caliper that discards matches too far apart to trust.
#
# Why this matters: the caliper (Rosenbaum & Rubin, 1985 recommend
# 0.2 * sd(logit(e(X)))) prevents pairing a treated unit with a control
# whose propensity score is nowhere close, which would otherwise silently
# reintroduce the bias this whole practice is trying to fix.
#
# Steps:
#   1. Caliper width: c = 0.2 * std(logit(ps)), where logit(p) = log(p /
#      (1 - p)), computed over *all* units (treated and control pooled).
#   2. For every treated unit i (independently, matching *with*
#      replacement — the observational comparison group here vastly
#      outnumbers the treated group, so this keeps the exercise simple
#      without meaningfully changing the result): find the control index j
#      minimizing |ps_treated[i] - ps_control[j]| (vectorized broadcasting,
#      no inner loop needed).
#   3. Keep the match only if |ps_treated[i] - ps_control[j]| <= c;
#      otherwise record -1 for that treated unit and exclude it from the
#      ATT.
#   4. ATT = mean over matched pairs of (y_treated[i] - y_control[j]).
# ---------------------------------------------------------------------------
def nearest_neighbor_match(
    ps_treated: np.ndarray, ps_control: np.ndarray, y_treated: np.ndarray, y_control: np.ndarray
) -> MatchResult:
    """1:1 nearest-neighbor caliper matching on the propensity score.

    Matches with replacement; treated units outside the caliper are
    excluded from the ATT. Returns the matched control indices (-1 where
    unmatched) and the resulting ATT.
    """
    raise NotImplementedError("TODO(human): implement nearest-neighbor caliper matching")


def main() -> None:
    data = load_lalonde()
    obs = data.observational
    X = obs[COVARIATES].to_numpy()
    treat = obs["treat"].to_numpy()
    y = obs[OUTCOME].to_numpy()
    try:
        fit = fit_propensity_score(X, treat)
    except NotImplementedError as e:
        print(f"(skipped — Phase 1 not implemented yet — {e})")
        return
    ps_t, ps_c = fit.scores[treat == 1], fit.scores[treat == 0]
    y_t, y_c = y[treat == 1], y[treat == 0]
    try:
        result = nearest_neighbor_match(ps_t, ps_c, y_t, y_c)
        n_matched = int((result.matched_control_idx >= 0).sum())
        print(f"Matched {n_matched}/{len(ps_t)} treated units; ATT = {result.att:,.0f}")
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
