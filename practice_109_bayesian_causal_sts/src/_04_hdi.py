"""Phase 4 — the highest density interval (HDI).

A credible interval is not unique: any interval covering `prob` mass of the
posterior qualifies. The *equal-tailed* interval (the `[2.5%, 97.5%]`
quantiles) is the common default, but for a skewed or multimodal posterior
it can exclude high-density points near the edges while including
low-density ones near the center. The HDI instead is the *narrowest*
interval containing `prob` mass — every point inside it has higher density
than every point outside it. `xy` has no built-in HDI (or KDE) primitive
(see this practice's CLAUDE.md, "Plotting" section), so this phase
implements the standard sorted-samples algorithm directly — the same one
`arviz.hdi` uses internally.

Run on its own to compute the HDI of a known Normal sample and sanity-check
it against `arviz.hdi`:
    uv run python -m src._04_hdi
"""
from __future__ import annotations

import numpy as np


# TODO(human) — highest density interval via the sorted-window algorithm
# ---------------------------------------------------------------------------
# Goal: implement the HDI of a 1-D sample of posterior draws, without
# calling `arviz.hdi` or any other library HDI helper.
#
# Why this matters: for a symmetric posterior the HDI and the equal-tailed
# interval coincide, but real posteriors (effect sizes bounded by a sign
# constraint, variance parameters, anything skewed) often are not
# symmetric — and the HDI is what every "94% credible interval" claim in
# this practice's plots and summaries actually means.
#
# Steps:
#   1. Sort the samples ascending: `sorted_samples = np.sort(samples)`.
#   2. Compute the number of samples the interval must span:
#      `n = len(samples)`, `interval_size = int(np.ceil(prob * n))`.
#   3. For every valid starting index `i` (from `0` to `n - interval_size`),
#      the candidate interval is `[sorted_samples[i], sorted_samples[i + interval_size - 1]]`,
#      with width `sorted_samples[i + interval_size - 1] - sorted_samples[i]`.
#      Compute all candidate widths at once with a vectorized slice
#      (`sorted_samples[interval_size - 1:] - sorted_samples[:n - interval_size + 1]`).
#   4. Take the narrowest one: `best = np.argmin(widths)`.
#   5. Return `(sorted_samples[best], sorted_samples[best + interval_size - 1])`.
# ---------------------------------------------------------------------------
def hdi(samples: np.ndarray, prob: float = 0.94) -> tuple[float, float]:
    """Highest density interval of `samples` at mass `prob`.

    Returns `(lo, hi)`, the narrowest interval covering `prob` fraction of
    the (1-D) sample — computed by exhaustive search over the sorted
    sample, the same approach `arviz.hdi` uses.
    """
    raise NotImplementedError("TODO(human): implement the sorted-window HDI algorithm")


def main() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=5.0, scale=2.0, size=20_000)
    try:
        lo, hi = hdi(samples, prob=0.94)
    except NotImplementedError as e:
        print(f"(skipped — {e})")
        return
    print(f"ours:   94% HDI = ({lo:.3f}, {hi:.3f})")
    try:
        import arviz as az

        az_lo, az_hi = az.hdi(samples, hdi_prob=0.94)
        print(f"arviz:  94% HDI = ({az_lo:.3f}, {az_hi:.3f})")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
