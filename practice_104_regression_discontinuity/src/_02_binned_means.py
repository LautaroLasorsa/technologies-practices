"""Phase 2 — Binned means for the canonical RDD plot.

A raw scatter of thousands of points hides the discontinuity in a cloud of
noise. The standard fix (used by every RDD paper's Figure 1, and by
`rdrobust`'s own `rdplot`) is to collapse the running variable into evenly
spaced bins and plot the *mean outcome per bin* instead — smoothing out
sampling noise while leaving a real jump at the cutoff clearly visible.

Run on its own to print the binned means either side of the cutoff:
    uv run python -m src._02_binned_means
"""
from __future__ import annotations

import numpy as np

from .datasets import load_dataset


# TODO(human) — evenly spaced binned means
# ---------------------------------------------------------------------------
# Goal: partition the running variable into evenly spaced bins of width
# `bin_width`, anchored so that no bin straddles the cutoff (bin edges are
# placed at cutoff, cutoff +/- bin_width, cutoff +/- 2*bin_width, ...), and
# compute the mean outcome within each non-empty bin.
#
# Why this matters: this is exactly the "binscatter" idea RDD plots rely
# on — it's a visualization tool, not an estimator (Phase 1's local linear
# fit is the estimate; this is only what makes the jump visible to the
# eye). Anchoring bins at the cutoff is the one detail that actually
# matters: if a bin straddled the cutoff, its mean would blend treated and
# untreated observations and the plot would visually erase the very
# discontinuity you're trying to show.
#
# Steps:
#   1. Build right-side bin edges: cutoff, cutoff + bin_width, cutoff + 2*bin_width, ...
#      out to running.max(); left-side edges the mirror image out to running.min().
#   2. Assign each observation to a bin via `np.digitize` (or equivalent) —
#      separately for the left and right subsets, so bin 0 never crosses
#      the cutoff.
#   3. For each non-empty bin, compute the bin center (midpoint of its
#      edges) and the mean of `outcome` within it.
#   4. Concatenate left and right bins, sorted by bin center ascending.
# ---------------------------------------------------------------------------
def binned_means(
    running: np.ndarray, outcome: np.ndarray, cutoff: float, bin_width: float
) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spaced, cutoff-anchored binned means of `outcome` over `running`.

    Returns `(bin_centers, bin_means)`, both 1-D and sorted by
    `bin_centers` ascending — directly usable as the "binned means" series
    in `plotting.rdd_scatter_plot`.
    """
    raise NotImplementedError("TODO(human): implement cutoff-anchored binned means")


def main() -> None:
    data = load_dataset("sharp", n=2000, seed=0)
    try:
        centers, means = binned_means(data.running, data.outcome, data.cutoff, bin_width=0.1)
        for c, m in zip(centers, means):
            print(f"bin center {c:+.3f}: mean outcome {m:.3f}")
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
