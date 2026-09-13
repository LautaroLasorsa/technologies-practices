"""Phase 1 — Sharp RDD: the local linear estimator.

An RDD estimate is a comparison of two *extrapolations to the cutoff* — one
fit using only observations just below it, one using only observations just
above it — not a comparison of the raw group means. This phase implements
that estimator directly: a kernel-weighted local linear regression fit
separately on each side of the cutoff, restricted to a bandwidth-wide
window around it. The gap between the two extrapolated intercepts *is* the
sharp RDD estimate.

Run on its own to see the from-scratch estimator matched against
`rdrobust`'s "Conventional" (uncorrected) local linear estimate at the same
bandwidth and kernel:
    uv run python -m src._01_local_linear_rdd
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datasets import load_dataset


@dataclass
class LocalLinearFit:
    """Result of a two-sided local linear RDD fit."""

    tau_hat: float  # intercept_right - intercept_left, the RDD estimate
    intercept_left: float
    slope_left: float
    intercept_right: float
    slope_right: float
    n_left: int
    n_right: int

    def predict_left(self, x_grid: np.ndarray) -> np.ndarray:
        return self.intercept_left + self.slope_left * x_grid

    def predict_right(self, x_grid: np.ndarray) -> np.ndarray:
        return self.intercept_right + self.slope_right * x_grid


def _triangular_kernel(u: np.ndarray) -> np.ndarray:
    """Triangular kernel: weight 1 at the cutoff, falling linearly to 0 at
    the bandwidth edge. The standard default kernel for RDD (it downweights
    observations far from the cutoff instead of a boxcar's all-or-nothing
    cutoff), and what `rdrobust` uses by default."""
    return np.clip(1.0 - np.abs(u), 0.0, None)


# TODO(human) — local linear RDD estimator with a kernel weight
# ---------------------------------------------------------------------------
# Goal: implement the sharp RDD estimator as two separate kernel-weighted
# local linear regressions, one on each side of the cutoff, and return the
# gap between their intercepts at the cutoff.
#
# Why this matters: naive RDD implementations fit ONE regression with a
# treatment dummy over the whole sample, or (worse) a single global
# high-order polynomial — both are known bad practice (Gelman & Imbens,
# 2019, "Why High-Order Polynomials Should Not Be Used in Regression
# Discontinuity Designs"): a global polynomial lets curvature far from the
# cutoff distort the extrapolation right at it, and a single dummy-variable
# regression forces both sides to share a slope. Fitting two independent
# *local* linear regressions, weighted so nearby points matter more than
# distant ones, avoids both problems — this is exactly what `rdrobust`'s
# "Conventional" (p=1) estimate computes.
#
# Steps:
#   1. Restrict to observations within the bandwidth: |running - cutoff| <= bandwidth.
#   2. Compute kernel weights `w = _triangular_kernel((running - cutoff) / bandwidth)`
#      for the restricted sample.
#   3. Split into left (`running < cutoff`) and right (`running >= cutoff`) subsets.
#   4. For each side, fit a weighted least squares regression of `outcome`
#      on `[1, running - cutoff]` (center the running variable on the
#      cutoff so the intercept IS the extrapolated value AT the cutoff).
#      Weighted least squares via plain `np.linalg.lstsq` is a reweighting
#      trick: scale both the design matrix's rows and `outcome` by
#      `sqrt(w)` before calling `lstsq` — this is algebraically identical
#      to solving the weighted normal equations.
#   5. `tau_hat = intercept_right - intercept_left`.
#
# Do not fit one pooled regression with a treatment dummy — the two sides
# must be estimated independently.
# ---------------------------------------------------------------------------
def local_linear_rdd(
    running: np.ndarray,
    outcome: np.ndarray,
    cutoff: float,
    bandwidth: float,
    kernel: str = "triangular",
) -> LocalLinearFit:
    """Two-sided kernel-weighted local linear RDD estimate.

    Only `kernel="triangular"` needs to be supported. Returns a
    `LocalLinearFit` whose `tau_hat` is the estimated jump at `cutoff`, and
    whose `predict_left`/`predict_right` evaluate each side's fitted line
    on an arbitrary grid (used for plotting).
    """
    raise NotImplementedError("TODO(human): implement the kernel-weighted local linear RDD estimator")


def compare_to_rdrobust(running: np.ndarray, outcome: np.ndarray, cutoff: float, bandwidth: float) -> None:
    """Fit both the from-scratch estimator and `rdrobust`'s "Conventional"
    local linear estimate at the same bandwidth and kernel; assert they
    agree closely."""
    from rdrobust import rdrobust

    fit = local_linear_rdd(running, outcome, cutoff, bandwidth)
    result = rdrobust(outcome, running, c=cutoff, p=1, kernel="triangular", h=bandwidth)
    rdrobust_tau = float(result.coef.values[0][0])
    diff = abs(fit.tau_hat - rdrobust_tau)
    print(f"tau_hat (ours):      {fit.tau_hat:.6f}")
    print(f"tau_hat (rdrobust):  {rdrobust_tau:.6f}")
    print(f"|diff|:              {diff:.2e}")
    assert diff < 1e-6, "local_linear_rdd should match rdrobust's Conventional estimate at the same h"


def main() -> None:
    data = load_dataset("sharp", n=2000, seed=0)
    print(f"True tau: {data.tau_true}\n")
    try:
        compare_to_rdrobust(data.running, data.outcome, data.cutoff, bandwidth=0.3)
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
