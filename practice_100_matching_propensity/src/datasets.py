"""LaLonde/Dehejia-Wahba datasets, via the `causaldata` package.

This practice's centerpiece is the LaLonde (1986) replication: a randomized
job-training experiment (the National Supported Work, NSW, demonstration)
gives a *known-good* experimental ATE, and an observational comparison
group drawn from a totally different population (the Current Population
Survey, CPS) is used to show how badly a naive "just compare means" estimate
fails outside a randomized experiment — and how much matching/IPW/AIPW can
recover.

Both pieces are real data shipped by `causaldata` (verified against the
installed package, not assumed):
  - `causaldata.nsw_mixtape` — the Dehejia & Wahba (1999/2002) experimental
    NSW subsample: 185 treated + 260 control, *both arms randomized*, with
    1978 earnings (`re78`) as the outcome. `treat==1` vs `treat==0` here
    recovers the textbook ATE of ~$1,794.
  - `causaldata.cps_mixtape` — the CPS-1 non-experimental comparison group
    (15,992 units, all `treat==0`) used by LaLonde (1986) and Dehejia-Wahba
    to show that observational comparisons of this kind badly fail to
    recover the experimental benchmark.

Note: the installed `causaldata` 0.1.x does **not** ship a PSID comparison
group (only `nsw_mixtape` and `cps_mixtape` exist among its ~34 datasets) —
this practice therefore uses CPS-1 as its one observational comparison
group instead of the PSID/CPS pair the classic LaLonde papers report side
by side. See this practice's CLAUDE.md Notes for the verification detail.

No TODO here: data loading is infrastructure, not the taught technique.
"""
from __future__ import annotations

from dataclasses import dataclass

import causaldata
import pandas as pd

COVARIATES = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]
OUTCOME = "re78"


@dataclass
class LaLondeData:
    """The LaLonde replication setup: one experimental benchmark, one
    observational comparison that fails to recover it."""

    experimental: pd.DataFrame  # NSW treated + NSW experimental control (both randomized)
    observational: pd.DataFrame  # NSW treated + CPS-1 non-experimental comparison group
    tau_experimental: float  # known-good ATE, computed from the randomized experiment alone


def load_lalonde() -> LaLondeData:
    """Load the NSW experimental benchmark and the NSW-vs-CPS1 observational
    comparison, both real `causaldata` datasets (not synthetic)."""
    nsw = causaldata.nsw_mixtape.load_pandas().data.reset_index(drop=True)
    cps = causaldata.cps_mixtape.load_pandas().data.reset_index(drop=True)

    nsw_treated = nsw.loc[nsw["treat"] == 1].reset_index(drop=True)
    observational = pd.concat([nsw_treated, cps], ignore_index=True)

    tau_experimental = (
        nsw.loc[nsw["treat"] == 1, OUTCOME].mean() - nsw.loc[nsw["treat"] == 0, OUTCOME].mean()
    )

    return LaLondeData(
        experimental=nsw, observational=observational, tau_experimental=float(tau_experimental)
    )
