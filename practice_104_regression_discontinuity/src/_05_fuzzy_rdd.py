"""Phase 5 — Fuzzy RDD: the Wald ratio.

Sharp RDD assumes treatment is a *deterministic* function of crossing the
cutoff. That's often false: crossing a test-score cutoff makes a scholarship
more likely, not certain (some students who miss it still get one through
an appeal; some who clear it never claim it). When the cutoff only *shifts
the probability* of treatment, comparing raw outcome means at the cutoff no
longer estimates the effect of treatment — it estimates the effect of being
*eligible* for it (an intent-to-treat effect), diluted by the fraction who
don't comply. Fuzzy RDD recovers the effect of treatment itself via a Wald
ratio, exactly analogous to instrumental variables: eligibility (crossing
the cutoff) is used as an instrument for actual treatment take-up.

Run on its own to see the Wald estimate recover the true effect on the
synthetic fuzzy scenario:
    uv run python -m src._05_fuzzy_rdd
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._01_local_linear_rdd import local_linear_rdd
from .datasets import load_dataset


@dataclass
class FuzzyRDDFit:
    """Result of a fuzzy RDD Wald-ratio estimate."""

    tau_hat: float  # the Wald ratio: reduced_form / first_stage
    reduced_form: float  # jump in E[Y|X] at the cutoff (intent-to-treat effect)
    first_stage: float  # jump in E[D|X] at the cutoff (compliance rate at the cutoff)


# TODO(human) — fuzzy RDD Wald ratio
# ---------------------------------------------------------------------------
# Goal: estimate the fuzzy RDD treatment effect as the ratio of two sharp-
# RDD-style jumps: the jump in the *outcome* at the cutoff (the "reduced
# form"), divided by the jump in the *treatment take-up rate* at the cutoff
# (the "first stage").
#
# Why this matters: when treatment take-up isn't deterministic, the
# reduced-form jump in Y only tells you the effect of *eligibility* — it's
# automatically shrunk by whatever fraction of eligible units actually take
# up treatment. Dividing by the first-stage jump rescales it back to the
# effect of *treatment itself*, exactly the way two-stage least squares
# rescales an intent-to-treat effect by the compliance rate in a randomized
# trial with imperfect compliance. This is the RD analogue of a
# Wald/IV estimator, and it's why fuzzy RDD identification additionally
# requires the first-stage jump to be nonzero (a "relevant instrument") —
# a vanishing first stage makes the ratio explode.
#
# Steps:
#   1. Call `local_linear_rdd(running, outcome, cutoff, bandwidth)` for the
#      reduced form; take its `tau_hat`.
#   2. Call `local_linear_rdd(running, treatment, cutoff, bandwidth)` for
#      the first stage (yes — reuse the SAME function, with `treatment` in
#      place of `outcome`); take its `tau_hat`.
#   3. `tau_hat = reduced_form / first_stage`.
# ---------------------------------------------------------------------------
def fuzzy_rdd_wald(
    running: np.ndarray, outcome: np.ndarray, treatment: np.ndarray, cutoff: float, bandwidth: float
) -> FuzzyRDDFit:
    """Fuzzy RDD Wald-ratio estimate of the effect of treatment take-up.

    `treatment` is the *observed* (possibly non-deterministic) 0/1
    treatment indicator. Returns a `FuzzyRDDFit` with the Wald estimate
    and the two jumps it's built from.
    """
    raise NotImplementedError("TODO(human): implement the fuzzy RDD Wald ratio")


def main() -> None:
    data = load_dataset("fuzzy", n=3000, seed=0)
    print(f"True tau: {data.tau_true}\n")
    try:
        fit = fuzzy_rdd_wald(data.running, data.outcome, data.treatment, data.cutoff, bandwidth=0.4)
        print(f"first stage (jump in P(D=1|X)):  {fit.first_stage:.3f}")
        print(f"reduced form (jump in E[Y|X]):    {fit.reduced_form:.3f}")
        print(f"Wald estimate (tau_hat):          {fit.tau_hat:.3f}")
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
