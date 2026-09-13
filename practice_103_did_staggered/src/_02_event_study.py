"""Phase 2 -- the event-study specification, and testing pre-trends.

A single 2x2 DiD (Phase 1) only ever compares *one* pre period against
*one* post period. The event-study specification generalises that to
every period in the panel at once, by replacing the single `treat*post`
interaction with one dummy per relative-time bin (time since treatment,
"event time"). The coefficients on the *pre*-period dummies are the
standard pre-trends test: under the parallel-trends assumption they should
be indistinguishable from zero, because nothing "different" should be
happening to the treated group before it is actually treated.

Run on its own to print the fitted event-study coefficients for one
treated cohort vs. the never-treated group:
    uv run python -m src._02_event_study
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyfixest as pf

from .datasets import load_event_study_subset, load_staggered_panel

REF_PERIOD = -1  # the omitted (baseline) relative-time bin


def fit_event_study_model(df: pd.DataFrame) -> pf.Feols:
    """Fit `y ~ i(rel_time, ref=-1) | unit + time` -- one dummy per
    relative-time bin, unit and time fixed effects, clustered by unit.

    Never-treated (control) units are assigned `rel_time = REF_PERIOD` so
    they act as the omitted baseline throughout -- they never pick up an
    event-time dummy, but still identify the time fixed effects. Fully
    scaffolded: fitting the regression is standard mechanics, not the
    concept this phase teaches (extracting/reading its coefficients is).
    """
    df = df.copy()
    df["rel_time_filled"] = df["rel_time"].fillna(REF_PERIOD).astype(int)
    return pf.feols(f"y ~ i(rel_time_filled, ref={REF_PERIOD}) | unit + time", data=df, vcov={"CRV1": "unit"})


# TODO(human) -- extract a tidy event-study table from the fitted model
# ---------------------------------------------------------------------------
# Goal: turn `model.coef()` / `model.se()` / `model.confint()` (each indexed
# by pyfixest's generated names like `"rel_time_filled::-3"`) into a tidy,
# sorted-by-event-time DataFrame ready for `plotting.event_study_plot`.
#
# Why this matters: a fitted model's raw output is indexed by string
# coefficient names, not by the event-time integers a plot or a pre-trends
# check needs. This bookkeeping step -- turning "one row per fitted
# dummy" into "one row per relative time, in order, with the reference
# period spliced back in at effect=0" -- is what every event-study plot in
# applied econometrics is built from, whether you write it by hand (here)
# or a package (`pyfixest`'s own `.iplot()`, `differences`) does it for
# you.
#
# Steps:
#   1. From `model.coef().index`, parse out the integer relative-time value
#      after the "::" in each name (e.g. `"rel_time_filled::-3"` -> `-3`).
#   2. Build a DataFrame with columns `rel_time`, `estimate`, `se`, `ci_lo`,
#      `ci_hi`, one row per fitted dummy, using `model.coef()`, `model.se()`,
#      and `model.confint()` (whose columns are the 2.5%/97.5% bounds).
#   3. Append one more row for the omitted reference period itself
#      (`rel_time=REF_PERIOD`, `estimate=0.0`, `se=0.0`, `ci_lo=0.0`,
#      `ci_hi=0.0`) -- it has no fitted coefficient, but belongs on the plot.
#   4. Sort by `rel_time` and reset the index before returning.
# ---------------------------------------------------------------------------
def event_study_coefficients(model: pf.Feols) -> pd.DataFrame:
    """Tidy event-study table: one row per relative-time bin, sorted, with
    the omitted reference period included at `estimate=0`.

    Returns columns `rel_time`, `estimate`, `se`, `ci_lo`, `ci_hi`.
    """
    raise NotImplementedError("TODO(human): extract the tidy event-study coefficient table")


def main() -> None:
    panel = load_staggered_panel(seed=0)
    df = load_event_study_subset(panel, cohort="mid", control="never")
    model = fit_event_study_model(df)
    try:
        table = event_study_coefficients(model)
        print(table.to_string(index=False))
        pre = table[table["rel_time"] < 0]
        print(f"\nMax |pre-trend coefficient|: {np.abs(pre['estimate']).max():.3f} (should be small)")
    except NotImplementedError as e:
        print(f"(skipped -- {e})")


if __name__ == "__main__":
    main()
