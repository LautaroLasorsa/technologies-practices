"""Phase 3 — Bandwidth selection and the bias-variance tradeoff.

Every local linear RDD estimate depends on a bandwidth choice, and that
choice is a bias-variance tradeoff made explicit: a wide bandwidth pulls in
more data (lower variance) but also more curvature the local *linear* fit
can't capture (higher bias); a narrow bandwidth does the reverse. Imbens &
Kalyanaraman (2012) and Calonico, Cattaneo & Titiunik (2014) formalize this
into data-driven, MSE-optimal bandwidth selectors — this phase implements a
simplified plug-in rule-of-thumb (ROT) in that spirit (see the TODO below
and this practice's CLAUDE.md for what's simplified and why).

Run on its own to see the selected bandwidth next to `rdrobust`'s own
(fully MSE-optimal, not simplified) bandwidth choice:
    uv run python -m src._03_bandwidth_selection
"""
from __future__ import annotations

import numpy as np

from .datasets import load_dataset


# TODO(human) — simplified rule-of-thumb (ROT) plug-in bandwidth
# ---------------------------------------------------------------------------
# Goal: implement a plug-in bandwidth selector that estimates how much
# curvature and how much noise are in the data, then trades them off via
# the standard nonparametric bandwidth scaling h ~ n^(-1/5).
#
# Why this matters: the "right" bandwidth isn't a constant you can look up
# — it depends on the data. A wide bandwidth uses more observations (lower
# variance in tau_hat) but also averages in more curvature that a local
# *linear* fit can't represent (higher bias, since the true relationship
# isn't linear that far from the cutoff). A plug-in selector estimates
# both ingredients directly: fit a flexible (quartic) polynomial once to
# measure the curvature m''(cutoff) and the residual variance sigma^2,
# then plugs them into the bandwidth formula that balances bias^2 against
# variance. This is a simplified version of the real IK/CCT pilot stage —
# the full algorithms add boundary-kernel corrections and iterate the
# pilot bandwidth itself; that extra machinery is described in this
# practice's CLAUDE.md but not implemented here. Cross-check the result
# against the bandwidth-sensitivity plot (Phase 6) and against `rdrobust`'s
# own optimal bandwidth — expect the same order of magnitude, not an exact
# match.
#
# Steps:
#   1. Center the running variable: u = running - cutoff.
#   2. Fit ONE pooled quartic (`u**0 .. u**4`) to `outcome` via
#      `np.linalg.lstsq` — this pilot fit only needs to capture curvature,
#      not be the RDD estimator itself.
#   3. Residual variance: sigma2 = sum(resid**2) / (n - 5).
#   4. Curvature at the cutoff: the pilot quartic's 2nd derivative at u=0,
#      which is `2 * coefficient_of(u**2)`.
#   5. Plug into h = C * (sigma2 * (u.max() - u.min()) / (n * curvature**2)) ** (1/5),
#      with C = 3.4 (a standard triangular-kernel ROT constant). Guard
#      against a near-zero curvature estimate (e.g. floor |curvature| at
#      1e-8) so the formula never divides by zero.
# ---------------------------------------------------------------------------
def rule_of_thumb_bandwidth(running: np.ndarray, outcome: np.ndarray, cutoff: float) -> float:
    """Simplified plug-in (ROT) bandwidth for local linear RDD.

    Returns a single positive float bandwidth, directly usable as the
    `bandwidth` argument to `local_linear_rdd`.
    """
    raise NotImplementedError("TODO(human): implement the plug-in ROT bandwidth selector")


def main() -> None:
    data = load_dataset("sharp", n=2000, seed=0)
    try:
        h = rule_of_thumb_bandwidth(data.running, data.outcome, data.cutoff)
        print(f"ROT bandwidth (ours):      {h:.4f}")
    except NotImplementedError as e:
        print(f"(skipped — {e})")
        return

    try:
        from rdrobust import rdrobust

        result = rdrobust(data.outcome, data.running, c=data.cutoff, p=1, kernel="triangular")
        h_left = float(result.bws.values[0][0])
        h_right = float(result.bws.values[0][1])
        print(f"rdrobust optimal bandwidth: left={h_left:.4f}  right={h_right:.4f}")
    except Exception as e:  # pragma: no cover - informational comparison only
        print(f"(rdrobust comparison unavailable — {e})")


if __name__ == "__main__":
    main()
