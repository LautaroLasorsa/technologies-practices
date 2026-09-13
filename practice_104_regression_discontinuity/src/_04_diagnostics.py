"""Phase 4 — Manipulation and balance checks (fully scaffolded, no TODO).

Sharp/fuzzy RDD identification rests on one untestable assumption: units
cannot precisely manipulate the running variable to land on their preferred
side of the cutoff. Two checks make that assumption *falsifiable* even
though it can't be directly proven:

  1. The McCrary (2008) density test — if units are sorting themselves
     across the cutoff (candidates who "just barely" lose a close election
     concede or contest it, students who "just barely" fail a test retake
     it), the running variable's *density* should show a jump at the
     cutoff, even though nothing else does.
  2. Covariate balance placebo checks — a pre-treatment covariate (one the
     treatment cannot possibly affect) should show NO jump at the cutoff.
     A jump means either manipulation or that the "cutoff" is coinciding
     with some other discontinuous policy change.

Both are implemented here as fully worked functions — the point of this
phase is running and interpreting them, not re-deriving the estimators
(Phase 1's `local_linear_rdd` already IS the balance-check estimator; the
density test reuses the same "local linear extrapolation to the cutoff,
from each side" idea applied to a histogram instead of to Y).

Run on its own to compare the density test on the real (unmanipulated) Lee
(2008) data against the synthetic "manipulated" scenario, and to run a
covariate balance check:
    uv run python -m src._04_diagnostics
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._01_local_linear_rdd import LocalLinearFit, local_linear_rdd
from .datasets import load_dataset, load_lee2008


@dataclass
class DensityTestResult:
    """Result of the (simplified) McCrary-style density discontinuity test."""

    bin_centers: np.ndarray
    bin_density: np.ndarray
    jump: float  # density_right(cutoff) - density_left(cutoff)
    z_stat: float
    p_value: float
    fit_left: tuple[np.ndarray, np.ndarray]
    fit_right: tuple[np.ndarray, np.ndarray]


def _normal_sf_two_sided(z: float) -> float:
    """Two-sided p-value for a standard normal z-stat, via `math.erf`
    (avoids a `scipy` dependency for one CDF lookup)."""
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def _weighted_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """WLS-fit `y ~ 1 + x` with weights `w`; return `(intercept, slope,
    intercept_se)`. Shared by both sides of the density test below."""
    design = np.column_stack([np.ones_like(x), x])
    sqrt_w = np.sqrt(w)
    wd, wy = design * sqrt_w[:, None], y * sqrt_w
    coefs, *_ = np.linalg.lstsq(wd, wy, rcond=None)
    resid = wy - wd @ coefs
    n, k = design.shape
    sigma2 = np.sum(resid**2) / max(n - k, 1)
    cov = sigma2 * np.linalg.inv(wd.T @ wd)
    return float(coefs[0]), float(coefs[1]), float(np.sqrt(cov[0, 0]))


def mccrary_density_test(running: np.ndarray, cutoff: float, bin_width: float = 0.05) -> DensityTestResult:
    """Simplified McCrary (2008)-style density discontinuity test.

    Bins the running variable, treats bin counts / (n * bin_width) as a
    density estimate, and fits a count-weighted local linear regression of
    that density on the bin center separately either side of the cutoff —
    the same "extrapolate to the cutoff from each side" idea as Phase 1,
    applied to a histogram instead of to an outcome. This is a simplified
    version of McCrary's original local-likelihood procedure (no adaptive
    bandwidth selection, no boundary-kernel correction) — it recovers the
    same qualitative answer (does the running variable's density jump at
    the cutoff?) without reproducing the published algorithm bit-for-bit.
    """
    n = len(running)
    right_edges = np.arange(cutoff, running.max() + bin_width, bin_width)
    left_edges = np.arange(cutoff, running.min() - bin_width, -bin_width)[::-1]

    centers, densities, counts = [], [], []
    for edges in (left_edges, right_edges):
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (running >= lo) & (running < hi)
            count = int(mask.sum())
            if count == 0:
                continue
            centers.append((lo + hi) / 2.0)
            counts.append(count)
            densities.append(count / (n * bin_width))
    centers, densities, counts = np.array(centers), np.array(densities), np.array(counts)

    left_mask, right_mask = centers < cutoff, centers >= cutoff
    left_int, left_slope, left_se = _weighted_fit(centers[left_mask] - cutoff, densities[left_mask], counts[left_mask])
    right_int, right_slope, right_se = _weighted_fit(centers[right_mask] - cutoff, densities[right_mask], counts[right_mask])

    jump = right_int - left_int
    z_stat = jump / math.sqrt(left_se**2 + right_se**2)
    p_value = _normal_sf_two_sided(z_stat)

    grid_left = np.linspace(centers[left_mask].min(), cutoff, 50) - cutoff
    grid_right = np.linspace(cutoff, centers[right_mask].max(), 50) - cutoff
    return DensityTestResult(
        bin_centers=centers,
        bin_density=densities,
        jump=jump,
        z_stat=z_stat,
        p_value=p_value,
        fit_left=(grid_left + cutoff, left_int + left_slope * grid_left),
        fit_right=(grid_right + cutoff, right_int + right_slope * grid_right),
    )


def covariate_balance_check(
    running: np.ndarray, covariate: np.ndarray, cutoff: float, bandwidth: float
) -> LocalLinearFit:
    """Placebo check: run the Phase 1 estimator with a pre-treatment
    covariate as the "outcome". A well-identified RDD should show no jump
    (`tau_hat` close to 0) — covariates measured before the cutoff cannot
    be caused by crossing it."""
    return local_linear_rdd(running, covariate, cutoff, bandwidth)


def main() -> None:
    lee = load_lee2008()
    manipulated = load_dataset("manipulated", n=2000, seed=0)

    for label, data in (("Lee (2008), real data", lee), ("synthetic, manipulated", manipulated)):
        result = mccrary_density_test(data.running, data.cutoff)
        verdict = "REJECTS continuity" if result.p_value < 0.05 else "fails to reject continuity"
        print(f"{label:24s} jump={result.jump:+.3f}  z={result.z_stat:+.2f}  p={result.p_value:.4f}  ({verdict})")

    print()
    sharp = load_dataset("sharp", n=2000, seed=0)
    try:
        balance = covariate_balance_check(sharp.running, sharp.covariate, sharp.cutoff, bandwidth=0.3)
        print(f"Covariate balance check (should be ~0): tau_hat = {balance.tau_hat:+.3f}")
    except NotImplementedError as e:
        print(f"(skipped — Phase 1 not implemented yet — {e})")


if __name__ == "__main__":
    main()
