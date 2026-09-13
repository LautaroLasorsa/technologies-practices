"""Phase 2 -- Cross-fitting: sample splitting to remove own-observation overfitting.

Neyman orthogonality (Phase 3) makes the moment condition insensitive to
*bias* in the nuisance estimates -- but it says nothing about *own-
observation overfitting*: if a nuisance model is fit on the same data whose
residual later enters the moment, a flexible model that has partially
memorized that data produces artificially small residuals correlated with
the observation's own outcome and treatment. Cross-fitting (Chernozhukov et
al., 2018) is plain sample splitting: fit nuisances on one part of the
data, predict on the other, so no observation ever "grades its own homework".

Run on its own to see the cross-fitted residuals:
    uv run python -m src._02_cross_fitting
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold

from .datasets import load_dataset


def default_model_y():
    return GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=0)


def default_model_d():
    """Flexible propensity model for E[D|X] -- a classifier, since D is binary."""
    return GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=0)


# TODO(human) -- the cross-fitting fold loop
# ---------------------------------------------------------------------------
# Goal: implement K-fold cross-fitting: for every fold, fit BOTH nuisance
# models (model_y for E[Y|X], model_d for the propensity E[D|X]) on the
# OTHER K-1 folds, then predict on the held-out fold. Every observation's
# residual is therefore built from a prediction that never saw that
# observation during nuisance training.
#
# Why this matters: this is sample splitting, and nothing more exotic --
# but it is the specific device that breaks the correlation between a
# nuisance model's own-observation overfitting and that same observation's
# contribution to the moment condition. Orthogonality (Phase 3) handles
# first-order *bias* from imperfect nuisances; cross-fitting handles the
# *dependence* between the nuisance-fitting sample and the estimation
# sample. Both are needed together -- dropping either one reintroduces
# bias, which is exactly what Phase 3's three-way comparison shows.
#
# Steps:
#   1. Build a `KFold(n_splits=n_folds, shuffle=True, random_state=seed)`.
#   2. For each (train_idx, test_idx) split: fit a FRESH model_y and
#      model_d (via `model_y_factory()`/`model_d_factory()` -- never reuse
#      a fitted model across folds) on X[train_idx] (Y[train_idx],
#      D[train_idx]); predict on X[test_idx]. Use
#      `model_d.predict_proba(X[test_idx])[:, 1]` for the propensity
#      (D is binary).
#   3. Write Y_resid[test_idx] = Y[test_idx] - y_pred and
#      D_resid[test_idx] = D[test_idx] - d_pred_proba into pre-allocated
#      output arrays (every index is written exactly once, across folds).
# ---------------------------------------------------------------------------
def cross_fit_residuals(
    X: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
    model_y_factory=default_model_y,
    model_d_factory=default_model_d,
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold residuals Y - E_hat[Y|X] and D - E_hat[D|X], via K-fold cross-fitting.

    Returns `(Y_resid, D_resid)`, each shape (n,) -- every entry computed
    from a model that never trained on that observation.
    """
    raise NotImplementedError("TODO(human): implement the cross-fitting fold loop")


def main() -> None:
    data = load_dataset(n=1000, seed=0)
    try:
        Y_resid, D_resid = cross_fit_residuals(data.X, data.D, data.Y)
        print(f"Y_resid std: {Y_resid.std():.3f}   D_resid std: {D_resid.std():.3f}")
    except NotImplementedError as e:
        print(f"(skipped -- {e})")


if __name__ == "__main__":
    main()
