"""Phase 3 — three structures, one estimator: confounder, mediator, collider.

The same three-node shape (X - middle - Y) produces three completely
different rules for whether to condition on the middle node, depending on
which way the arrows point:
  - confounder (Z -> D, Z -> Y):   condition on it — it's the only way to
    remove the backdoor bias.
  - mediator (D -> M -> Y):        don't, if you want the *total* effect —
    conditioning on M blocks part of the causal path itself.
  - collider (D -> C <- Y):        never — conditioning on a common effect
    of D and Y manufactures an association where none exists.

This phase implements one generic regression-adjustment estimator and
applies it to all three DGPs from `src/datasets.py` with both the correct
and the incorrect adjustment set, so the divergence from the known true
effect is visible directly instead of asserted.

Run on its own to see all three structures, adjusted both ways:
    uv run python -m src._03_structures
"""
from __future__ import annotations

from .datasets import load_dataset


# TODO(human) — regression-adjustment effect estimator
# ---------------------------------------------------------------------------
# Goal: implement the one estimator this whole practice reuses — the OLS
# coefficient on `treatment` from regressing `outcome` on `treatment` plus
# whatever covariates are in `adjust_for`. This is literally what "control
# for X" means in applied work: add X as a regressor and read off the
# treatment's coefficient.
#
# Why this matters: the *only* thing that changes between "confounder:
# adjust" (correct), "mediator: adjust" (wrong, if you want the total
# effect), and "collider: adjust" (badly wrong) is which variable goes into
# `adjust_for` — the estimator itself never changes. Seeing one function
# produce the right answer in one case and a biased answer in another,
# with nothing but its argument different, is the point of this phase.
#
# Steps:
#   1. Build the design matrix: an intercept column of ones, the `treatment`
#      column, and one column per name in `adjust_for` (in that order), all
#      pulled from `df`.
#   2. Solve the OLS normal equations for the coefficient vector — e.g.
#      `np.linalg.lstsq(X, y, rcond=None)[0]` (this phase is about the
#      *adjustment set*, not re-deriving OLS mechanics — Practice 097 already
#      covers the QR route in depth).
#   3. Return the coefficient on the `treatment` column specifically (index
#      1, right after the intercept).
# ---------------------------------------------------------------------------
def estimate_effect_by_adjustment(df, treatment: str, outcome: str, adjust_for: list[str]) -> float:
    """OLS estimate of `treatment`'s effect on `outcome`, adjusting for `adjust_for`.

    Fits `outcome ~ intercept + treatment + adjust_for` by least squares and
    returns the coefficient on `treatment` only.
    """
    raise NotImplementedError("TODO(human): implement the regression-adjustment estimator")


def main() -> None:
    # (scenario, "correct" adjustment set, "wrong" adjustment set) — "correct"
    # always means "recovers the stated true_effect"; "wrong" means adjusting
    # for the one variable that structurally shouldn't be there.
    scenarios = [
        ("confounder", ["Z"], []),
        ("mediator", [], ["M"]),
        ("collider", [], ["C"]),
    ]
    for name, correct_set, wrong_set in scenarios:
        data = load_dataset(name, n=1000, seed=0)
        print(f"[{name}] {data.true_effect_label} = {data.true_effect:.3f}")
        try:
            est_correct = estimate_effect_by_adjustment(data.df, data.treatment, data.outcome, correct_set)
            est_wrong = estimate_effect_by_adjustment(data.df, data.treatment, data.outcome, wrong_set)
        except NotImplementedError as e:
            print(f"(skipped — {e})")
            break
        print(f"  adjust_for={correct_set!s:8s} -> estimate = {est_correct:.3f}  (correct)")
        print(f"  adjust_for={wrong_set!s:8s} -> estimate = {est_wrong:.3f}  (wrong)\n")


if __name__ == "__main__":
    main()
