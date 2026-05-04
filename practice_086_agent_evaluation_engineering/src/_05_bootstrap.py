"""Phase 5 — Paired bootstrap CI for system A vs system B.

Why paired?  When two systems are scored on *the same* prompts, their
per-prompt scores are *correlated* — easy prompts inflate both, hard
ones depress both.  Unpaired sampling treats those as independent and
overstates the variance.  Paired bootstrap on per-prompt deltas
(Du et al., 2025 — https://arxiv.org/abs/2511.19794) gives a
defensible confidence interval for "system A beats B" on this eval set.

Procedure (percentile method):

    deltas = [score_A_i - score_B_i for i in tasks]
    for b in range(B):
        idx = sample_with_replacement(range(N), N)
        boot_means[b] = mean(deltas[idx])
    lo, hi = percentile(boot_means, [2.5, 97.5])

If the resulting CI excludes 0, A's improvement over B is significant
at the chosen level.

Run on its own to compute a paired-bootstrap CI on a fake score matrix:
    uv run python -m src._05_bootstrap
"""

from __future__ import annotations

import numpy as np

from .models import BootstrapCI, TaskScore


# ---------------------------------------------------------------------------
# TODO(human) — Paired bootstrap resampling
# ---------------------------------------------------------------------------
# Goal: implement the paired bootstrap on per-task score deltas and
# return a percentile confidence interval.
#
# Inputs:
#   - ``deltas``   : 1-D numpy array of per-task deltas (A - B), shape (N,)
#   - ``n_boot``   : number of bootstrap resamples (typical: 10000)
#   - ``confidence``: e.g. 0.95
#   - ``rng``      : a numpy Generator for reproducibility
#
# Output: a tuple ``(lo, hi, boot_means)`` where
#   - lo, hi  : the lower/upper percentile bounds at the given confidence
#   - boot_means : np.ndarray of shape (n_boot,) with the resampled means
#                   (returned so the caller can plot the distribution if
#                   they want — don't bother computing it twice)
#
# Procedure (~6–10 lines):
#   1. n = len(deltas)
#   2. Draw an index array of shape (n_boot, n) with rng.integers(0, n, ...)
#      — this samples task indices WITH REPLACEMENT, which is the heart
#      of the bootstrap.
#   3. Compute per-resample means with deltas[idx].mean(axis=1).
#   4. Convert ``confidence`` to a (lo_pct, hi_pct) pair, e.g. for 0.95
#      use (2.5, 97.5).
#   5. Use np.percentile on boot_means to get lo and hi.
#   6. Return (float(lo), float(hi), boot_means).
#
# DO NOT use scipy.stats.bootstrap here — implementing the resample
# loop by hand is exactly the point.  scipy is in the deps for the
# verify-setup smoke test, not as a shortcut for this exercise.
# ---------------------------------------------------------------------------
def paired_bootstrap_percentile(
    deltas: np.ndarray,
    n_boot: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float, np.ndarray]:
    """Paired bootstrap percentile CI on a 1-D array of per-prompt deltas."""
    raise NotImplementedError("TODO(human): implement paired_bootstrap_percentile")


# ---------------------------------------------------------------------------
# Convenience wrapper (fully scaffolded)
# ---------------------------------------------------------------------------


def paired_bootstrap_ci(
    scores_A: list[TaskScore],
    scores_B: list[TaskScore],
    *,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Paired bootstrap CI on mean(pass@1_A - pass@1_B) across tasks.

    Pairs by ``task_id`` (raises if mismatched) and uses each system's
    per-task ``pass_at_1`` as the metric — keeping it scalar makes the
    CI directly comparable to the headline pass^k delta.
    """
    by_id_B = {ts.task_id: ts for ts in scores_B}
    if set(t.task_id for t in scores_A) != set(by_id_B):
        raise ValueError("Mismatched task_ids between systems A and B")

    deltas = np.array(
        [ts.pass_at_1 - by_id_B[ts.task_id].pass_at_1 for ts in scores_A],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    lo, hi, _ = paired_bootstrap_percentile(deltas, n_boot=n_boot, confidence=confidence, rng=rng)
    significant = (lo > 0.0) or (hi < 0.0)
    return BootstrapCI(
        delta=float(deltas.mean()),
        lo=lo,
        hi=hi,
        confidence=confidence,
        n_bootstrap=n_boot,
        significant=significant,
    )


# -- Sanity demo (scaffolded) -----------------------------------------------


def main() -> None:
    from ._04_pass_at_k import build_task_scores

    # Synthetic: A is uniformly slightly better than B on every task.
    rng = np.random.default_rng(42)
    A_raw = {f"T{i:02d}": rng.binomial(1, 0.85, size=3).tolist() for i in range(20)}
    B_raw = {f"T{i:02d}": rng.binomial(1, 0.65, size=3).tolist() for i in range(20)}
    A = build_task_scores("A", A_raw)
    B = build_task_scores("B", B_raw)

    print("Computing paired-bootstrap 95% CI on (pass@1_A - pass@1_B) ...")
    try:
        ci = paired_bootstrap_ci(A, B, n_boot=5000, confidence=0.95, seed=1)
        print(
            f"  delta = {ci.delta:+.3f}   "
            f"95% CI = [{ci.lo:+.3f}, {ci.hi:+.3f}]   "
            f"significant = {ci.significant}"
        )
    except NotImplementedError as e:
        print(f"  (skipped — {e})")


if __name__ == "__main__":
    main()
