"""Synthetic confounded-treatment dataset with a known true ATE.

Two orthogonal identification problems live in the same data-generating
process (DGP), so every phase draws from one self-contained source:

1. **Hidden confounding** — treatment assignment depends on an unobserved
   confounder `u` in addition to the observed covariates `x1`, `x2`. An
   analyst who only ever sees `x1`, `x2`, `t`, `y` (columns of `df`) cannot
   fully adjust for selection into treatment, no matter how good their
   matching or regression controls look on paper. This is what Phases 1-2
   (Rosenbaum bounds, Oster's delta) and Phase 3 (E-values) interrogate.
2. **Non-ignorable attrition** — a second, independent corruption: whether
   the outcome is *observed at all* (`response`) depends on the realized
   outcome itself, not just on `t` or the covariates. This is what Phase 4
   (Manski / Lee bounds) interrogates.

`u` is returned separately from `df` (never merge it back in) — it exists
here purely so the notebook can show what "the thing every method in this
practice is trying to bound the influence of" actually looks like. No TODO
here: data generation is infrastructure, not the taught technique.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

# True average treatment effect baked into the DGP. Every phase's estimate
# is judged against this, and against how tightly each method's bounds
# manage to trap it.
TRUE_ATE = 2.0

# The outcome is a bounded score (e.g. a 0-100 test/performance score) --
# Manski's worst-case bounds (Phase 4) need a *known* finite range to plug
# in for unobserved values, and this is that range.
Y_MIN, Y_MAX = 0.0, 100.0


@dataclass
class SensitivityData:
    """One simulated sensitivity-analysis dataset.

    `df` holds everything an analyst could actually observe: `x1`, `x2`
    (covariates), `t` (treatment), `y` (outcome, always observed --
    used by Phases 1-3), `y_obs` (outcome after non-ignorable attrition,
    used by Phase 4 only), and `response` (1 if `y_obs` is not missing).
    `u` is the hidden confounder -- keep it out of every estimator call;
    it is here only for the notebook to *show*, never for it to *use*.
    """

    df: pd.DataFrame
    u: np.ndarray
    ate_true: float = TRUE_ATE
    y_min: float = Y_MIN
    y_max: float = Y_MAX


def load_dataset(n: int = 1000, confound_strength: float = 1.2, seed: int = 0) -> SensitivityData:
    """Generate the confounded-treatment / non-ignorable-attrition dataset.

    `confound_strength` scales how strongly the hidden confounder `u` pulls
    on both treatment assignment and the outcome -- turning it up makes
    every naive (covariate-only) estimate more biased, which is exactly
    the bias every method in this practice is trying to put a number on.
    """
    rng = np.random.default_rng(seed)

    x1 = rng.normal(size=n)
    x2 = rng.integers(0, 2, size=n).astype(float)
    u = rng.normal(size=n)  # hidden confounder -- never observed by the analyst

    # Selection into treatment depends on x1, x2 *and* u: a covariate-only
    # analyst can match/regress on x1, x2 all day and still miss this.
    logit_p = -0.2 + 0.8 * x1 + 0.5 * x2 + confound_strength * u
    p_treat = 1.0 / (1.0 + np.exp(-logit_p))
    t = rng.binomial(1, p_treat)

    noise = rng.normal(scale=8.0, size=n)
    y = 50.0 + TRUE_ATE * t + 4.0 * x1 + 3.0 * x2 + 6.0 * confound_strength * u + noise
    y = np.clip(y, Y_MIN, Y_MAX)

    # Non-ignorable attrition: whether y is *observed* depends on the
    # realized outcome itself (e.g. low performers are more likely to drop
    # out of the study), on top of a treatment-arm difference in response
    # rates -- this is exactly the case point identification cannot patch
    # over, no matter how good the covariates are.
    logit_r = 1.3 + 0.02 * (y - 50.0) - 0.3 * t
    p_response = 1.0 / (1.0 + np.exp(-logit_r))
    response = rng.binomial(1, p_response)
    y_obs = np.where(response == 1, y, np.nan)

    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "t": t,
            "y": y,
            "y_obs": y_obs,
            "response": response,
        }
    )
    return SensitivityData(df=df, u=u)


def build_matched_pairs(data: SensitivityData) -> np.ndarray:
    """Greedy 1:1 nearest-neighbor matching on the *observed*-covariate propensity score.

    Matching uses only `x1`, `x2` (a logistic propensity model) -- never
    `u`. That is deliberate: it is exactly why Rosenbaum bounds (Phase 1)
    are needed. A matched design can look perfectly balanced on everything
    the analyst can see and still hide a systematic imbalance in `u`
    between matched treated/control units.

    Returns the array of treated-minus-control `y` differences, one per
    matched pair.
    """
    df = data.df
    design = sm.add_constant(df[["x1", "x2"]].to_numpy())
    prop_score = sm.Logit(df["t"].to_numpy(), design).fit(disp=0).predict(design)

    treated_idx = df.index[df["t"] == 1].to_numpy()
    control_idx = df.index[df["t"] == 0].to_numpy()
    control_ps = prop_score[control_idx]
    used = np.zeros(len(control_idx), dtype=bool)

    diffs = []
    for i in treated_idx:
        dist = np.abs(control_ps - prop_score[i])
        dist[used] = np.inf
        j = int(np.argmin(dist))
        if not np.isfinite(dist[j]):
            continue
        used[j] = True
        diffs.append(df.loc[i, "y"] - df.loc[control_idx[j], "y"])
    return np.array(diffs)


def binarize_outcome(data: SensitivityData, threshold: float = 65.0) -> tuple[np.ndarray, np.ndarray]:
    """Threshold `y` into a binary "success" indicator, split by treatment arm.

    E-values (Phase 3) are defined on the risk-ratio scale, which needs a
    binary outcome -- this reuses the same underlying `y` rather than
    introducing a second, unrelated DGP. Returns `(y_bin_treated, y_bin_control)`.
    """
    df = data.df
    y_bin = (df["y"].to_numpy() >= threshold).astype(int)
    return y_bin[df["t"].to_numpy() == 1], y_bin[df["t"].to_numpy() == 0]
