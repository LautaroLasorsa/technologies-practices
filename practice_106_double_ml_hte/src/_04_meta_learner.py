"""Phase 4 -- Heterogeneous treatment effects: the T-learner.

Phases 1-3 estimate a single constant theta -- the average treatment
effect. This phase drops the constant-effect assumption and estimates the
CATE, tau(x) = E[Y(1) - Y(0) | X=x], using the simplest meta-learner: fit
one outcome model per treatment arm and difference their predictions.

Run on its own to compare the T-learner's CATE estimate to the known tau(x):
    uv run python -m src._04_meta_learner
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .datasets import load_dataset


def default_outcome_model():
    return RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0)


# TODO(human) -- the T-learner: two separately-fit outcome models
# ---------------------------------------------------------------------------
# Goal: implement the T-learner (Kunzel et al., 2019, "Metalearners for
# estimating heterogeneous treatment effects using machine learning"), the
# simplest meta-learner for CATE(x) = E[Y(1) - Y(0) | X=x].
#
# Why this matters: the partially linear model (Phases 1-3) is blind to
# heterogeneity by construction -- it estimates one theta for everyone. The
# T-learner instead fits two SEPARATE outcome models, one per treatment
# arm, and reads the CATE off as the difference between their predictions
# at the same x. No orthogonalization or cross-fitting is needed here (each
# model only ever sees its own arm's data, so there is no shared-sample
# overfitting between arms) -- but it also means the T-learner has no
# bias-correction machinery in common with Phases 1-3, which is exactly why
# more sophisticated meta-learners (X-learner, DR-learner) exist for cases
# where one arm is much smaller than the other or confounding is severe.
#
# Steps:
#   1. Fit model_1 on the treated subset: X[D==1], Y[D==1].
#   2. Fit model_0 on the control subset: X[D==0], Y[D==0].
#   3. tau_hat(x) = model_1.predict(X) - model_0.predict(X), evaluated on
#      the FULL X (both models must predict on every row, not just their
#      own arm).
# ---------------------------------------------------------------------------
def t_learner_cate(X: np.ndarray, D: np.ndarray, Y: np.ndarray, model_factory=default_outcome_model) -> np.ndarray:
    """T-learner CATE estimate: mu1_hat(x) - mu0_hat(x) for every row of X.

    `model_factory` is called twice (once per arm) so the two models never
    share fitted state. Returns a (n,) array of tau_hat(x_i).
    """
    raise NotImplementedError("TODO(human): implement the T-learner")


def main() -> None:
    data = load_dataset(n=1000, seed=0)
    try:
        tau_hat = t_learner_cate(data.X, data.D, data.Y)
        mae = np.mean(np.abs(tau_hat - data.tau_true))
        print(f"T-learner CATE MAE vs. true tau(x): {mae:.3f}")
    except NotImplementedError as e:
        print(f"(skipped -- {e})")


if __name__ == "__main__":
    main()
