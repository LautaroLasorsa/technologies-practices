"""Phase 3 — E-values.

Rosenbaum bounds (Phase 1) and Oster's delta (Phase 2) both answer "how
much hidden bias would it take" in units specific to their own method (an
odds ratio on treatment assignment; a multiple of observable selection).
Ding & VanderWeele's (2016) E-value puts the same question in one common,
interpretable unit: the minimum strength of association -- on the
risk-ratio scale, and required on *both* the confounder-to-treatment and
confounder-to-outcome links simultaneously -- that an unmeasured
confounder would need in order to fully explain away an observed
association. It requires nothing about the DGP except a risk ratio and
(optionally) a confidence-interval limit.

Run on its own to see the E-values for this practice's confounded design:
    uv run python -m src._03_e_value
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .datasets import binarize_outcome, load_dataset


def risk_ratio_with_ci(y1: np.ndarray, y0: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Risk ratio between two binary-outcome groups, with a normal-approximation CI on log(RR).

    Fully scaffolded -- this is standard 2x2-table arithmetic, not the
    taught technique (the E-value formula is).
    """
    p1, p0 = y1.mean(), y0.mean()
    rr = p1 / p0
    se_log_rr = np.sqrt((1 - p1) / (len(y1) * p1) + (1 - p0) / (len(y0) * p0))
    z = stats.norm.ppf(1 - alpha / 2)
    log_rr = np.log(rr)
    ci_lo = np.exp(log_rr - z * se_log_rr)
    ci_hi = np.exp(log_rr + z * se_log_rr)
    return rr, ci_lo, ci_hi


# TODO(human) — the E-value formula
# ---------------------------------------------------------------------------
# Goal: implement the Ding & VanderWeele (2016) E-value for a risk ratio.
#
# Why this matters: an unmeasured confounder needs to be associated with
# *both* treatment and outcome to bias an estimate -- and by an amount that
# multiplies, not adds. The E-value is the answer to "if a confounder had
# risk-ratio strength E with treatment, and risk-ratio strength E with the
# outcome (the *same* E on both links, the worst case for a fixed total
# "budget" of confounding), what is the smallest E that could shift the
# observed risk ratio all the way down to 1 (no effect)?" A large E-value
# means only an implausibly strong confounder could explain the result away;
# a small one (close to 1) means a fairly weak, plausible confounder could.
#
# Formula (Ding & VanderWeele, 2016, "Sensitivity Analysis Without
# Assumptions", eq. 2), for a risk ratio `rr >= 1`:
#
#   E(rr) = rr + sqrt(rr * (rr - 1))
#
# For `rr < 1` (a protective association), first invert it:
#   rr' = 1 / rr, then apply the same formula to `rr'`.
#
# This one formula is applied twice by the caller below: once to the point
# estimate, and once to whichever confidence-interval limit is closer to
# the null (1.0) -- that second E-value is always <= the first, and
# answers "how much confounding would it take for the interval to include
# no effect," a strictly easier bar to clear than explaining away the
# point estimate.
# ---------------------------------------------------------------------------
def e_value(rr: float) -> float:
    """E-value for a risk ratio (Ding & VanderWeele, 2016).

    Handles `rr < 1` by inverting first. Returns a value `>= 1`; `1.0`
    means no confounding at all would be needed.
    """
    raise NotImplementedError("TODO(human): implement the Ding & VanderWeele E-value formula")


def main() -> None:
    data = load_dataset(n=1000, confound_strength=1.2, seed=0)
    y1, y0 = binarize_outcome(data, threshold=65.0)
    rr, ci_lo, ci_hi = risk_ratio_with_ci(y1, y0)
    ci_limit = ci_lo if abs(np.log(ci_lo)) < abs(np.log(ci_hi)) else ci_hi
    print(f"Risk ratio: {rr:.3f}  95% CI: ({ci_lo:.3f}, {ci_hi:.3f})")
    try:
        e_point = e_value(rr)
        e_ci = e_value(ci_limit)
        print(f"E-value (point estimate): {e_point:.3f}")
        print(f"E-value (CI limit closer to null): {e_ci:.3f}")
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
