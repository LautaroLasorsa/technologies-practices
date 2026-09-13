"""California Proposition 99 state panel data.

Real data, not synthetic: California's Proposition 99 (passed Nov. 1988)
raised the state cigarette tax by 25 cents/pack and funded a tobacco-control
program. This is the classic Abadie, Diamond & Hainmueller (2010) dataset —
annual per-capita cigarette sales (in packs) for California (the treated
unit) and 38 other US states (the donor pool). The donor pool already
excludes states that ran a large tobacco-control program of their own during
the sample window (see this practice's CLAUDE.md, "Donor-pool selection" —
that exclusion is a modeling decision made *before* any weight is fit, not
something the optimizer can undo). Vendored under `data/prop99.csv`; see
CLAUDE.md for provenance. No TODO here: data loading is infrastructure, not
the taught technique.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "data" / "prop99.csv"

TREATED_UNIT = "California"
# Proposition 99 passed in November 1988; 1988 is therefore the last
# unaffected (pre-treatment) year and 1989 is the first treated year.
TREATMENT_YEAR = 1988


@dataclass
class SyntheticControlData:
    """The Prop 99 panel, reshaped from long (state, year, cigsale) rows
    into a treated vector and a donor-pool matrix indexed by year."""

    years: np.ndarray        # (T,) all years, pre + post, ascending
    y_treated: np.ndarray    # (T,) California per-capita cigarette sales
    Y_donors: np.ndarray     # (T, J) donor-pool per-capita cigarette sales
    donor_names: list[str]   # (J,) donor state names, matches Y_donors columns
    pre_mask: np.ndarray     # (T,) bool, True for year <= TREATMENT_YEAR
    post_mask: np.ndarray    # (T,) bool, True for year > TREATMENT_YEAR


def load_dataset() -> SyntheticControlData:
    """Load `data/prop99.csv` and reshape it into treated/donor arrays.

    Only the outcome (`cigsale`) is used — this practice's synthetic
    control matches on the treated unit's entire pre-treatment outcome
    path rather than on aggregated covariates (see CLAUDE.md, "What this
    practice simplifies"); `lnincome`/`beer`/`retprice`/`age15to24` are left
    in the CSV for reference but unused here.
    """
    df = pd.read_csv(DATA_PATH)
    pivot = df.pivot(index="year", columns="state", values="cigsale").sort_index()
    years = pivot.index.to_numpy()
    y_treated = pivot[TREATED_UNIT].to_numpy()
    donor_names = [c for c in pivot.columns if c != TREATED_UNIT]
    Y_donors = pivot[donor_names].to_numpy()
    pre_mask = years <= TREATMENT_YEAR
    post_mask = ~pre_mask
    return SyntheticControlData(
        years=years,
        y_treated=y_treated,
        Y_donors=Y_donors,
        donor_names=donor_names,
        pre_mask=pre_mask,
        post_mask=post_mask,
    )
