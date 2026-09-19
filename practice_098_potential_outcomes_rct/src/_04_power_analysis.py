"""Phase 4 — Power analysis and the minimum detectable effect (MDE).

Before running an experiment, the practical question is never "what's the
variance formula" — it's "how many units do I need to reliably detect an
effect of the size I actually care about." Power analysis answers that by
inverting the same two-sample comparison Phases 1-2 already built: given a
per-arm sample size `n`, an assumed outcome variance `sigma^2`, and a
significance level `alpha`, what's the smallest true effect `MDE` that a
test would detect with probability `power` (conventionally 80%)? Both
directions of the same normal-approximation formula matter in practice —
MDE given `n` (this is the number stakeholders ask for before an
experiment launches), and achieved power given a candidate effect size
(used to check "is my planned sample size even worth running").

Run on its own to see the MDE at a few sample sizes:
    uv run python -m src._04_power_analysis
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .datasets import load_dataset


# TODO(human) — minimum detectable effect and achieved power
# ---------------------------------------------------------------------------
# Goal: implement the standard normal-approximation power formulas for a
# two-sample comparison with `n` units per arm and common per-unit
# variance `sigma2`.
#
# Why this matters: this is the formula behind every "how many users do I
# need" experiment-design conversation — it turns Phase 2's variance
# formula around to answer a design question instead of an inference
# question. The two functions are two directions of the same algebra:
# `minimum_detectable_effect` solves for the smallest detectable effect
# given a sample size and a target power; `power_for_effect` solves for
# the achieved power given a candidate effect and sample size. Keep them
# side by side since the notebook uses both to build the power curve.
#
# Formulas (two-sided test, equal allocation, `z_a = norm.ppf(1 - alpha/2)`,
# `z_b = norm.ppf(power)`):
#   MDE(n)          = (z_a + z_b) * sqrt(2 * sigma2 / n)
#   power(effect, n) = norm.cdf(effect / sqrt(2 * sigma2 / n) - z_a)
# (the `2 * sigma2 / n` term is the variance of a diff-in-means with equal
# group sizes `n` per arm and common variance `sigma2` in each arm — the
# per-arm-equal-variance special case of Phase 2's Neyman formula.)
# ---------------------------------------------------------------------------
def minimum_detectable_effect(n: int, sigma2: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true effect detectable with the given power, per-arm `n`.

    `sigma2` is the assumed common per-arm outcome variance. Returns a
    single float in the same units as the outcome.
    """
    return (norm.ppf(1-alpha/2) + norm.ppf(power)) * np.sqrt(2 * sigma2 / n)

def power_for_effect(effect: float, n: int, sigma2: float, alpha: float = 0.05) -> float:
    """Achieved statistical power for a given true effect and per-arm `n`.

    Returns a probability in `[0, 1]`.
    """
    z_a = norm.ppf(1-alpha/2)
    d = abs(effect/np.sqrt(2*sigma2/n))
    return norm.cdf(d-z_a) + norm.cdf(-d-z_a)


def main() -> None:
    data = load_dataset(n=500, tau=2.0, seed=0)
    sigma2 = float(np.var(data.y_observed, ddof=1))
    for n in (50, 100, 250, 500, 1000):
        try:
            mde = minimum_detectable_effect(n, sigma2)
            print(f"n={n:5d}  MDE={mde:.4f}")
        except NotImplementedError as e:
            print(f"(skipped — {e})")
            break
    try:
        power = power_for_effect(data.tau_true, n=250, sigma2=sigma2)
        print(f"\nPower to detect true ATE ({data.tau_true}) at n=250 per arm: {power:.3f}")
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
