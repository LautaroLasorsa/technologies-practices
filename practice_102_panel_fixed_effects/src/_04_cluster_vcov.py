"""Phase 4 — Clustering at the level of treatment assignment.

Two-way FE (Phase 2) fixes beta_hat's *bias*; it says nothing about
whether the standard errors built on top of it are trustworthy. In this
practice's DGP, `x2` (the "treatment") is assigned at the cluster-period
level — every unit in a cluster shares the same x2 in a given period —
and the error term carries a matching cluster-period shock. That's
exactly the setup where ordinary (or even heteroskedasticity-robust but
non-clustered) standard errors are wrong: many units share the same
regressor value and the same shock, so they don't carry as much
independent information as their raw count suggests. The fix is to
cluster at the level the treatment actually varies at (here,
`cluster_id`) — sum each cluster's score contributions *before* squaring
them, so within-cluster correlation is absorbed instead of ignored (Liang
& Zeger, 1986; Bertrand, Duflo & Mullainathan, 2004, make this exact
point for difference-in-differences designs). The coverage simulation
below builds on the two-way estimator specifically (not the one-way
estimator of Phase 1), which this practice's DGP makes biased for
`beta_x2` — a biased point estimate would make "does the CI cover the
truth" measure the wrong thing entirely.

Run on its own to see cluster-robust vs. naive standard errors on one
sample, then a Monte Carlo coverage comparison:
    uv run python -m src._04_cluster_vcov
"""
from __future__ import annotations

import numpy as np

from ._01_within_transform import ols_lstsq
from ._02_two_way_demean import fit_two_way, two_way_demean
from .datasets import SMALL_PANEL, TRUE_BETA, generate_panel


def naive_vcov(X: np.ndarray, resid: np.ndarray, XtX_inv: np.ndarray) -> np.ndarray:
    """Classical (homoskedastic, independent-errors) covariance matrix —
    the formula that ignores clustering entirely. Not a TODO: it's one
    line, and the point of this phase is the *contrast* with the
    cluster-robust estimator below, not re-deriving this formula (see
    practice 097 for that)."""
    n, k = X.shape
    sigma_hat_sq = float(np.sum(resid**2) / (n - k))
    return sigma_hat_sq * XtX_inv


# TODO(human) — cluster-robust (CR1) sandwich covariance estimator
# ---------------------------------------------------------------------------
# Goal: implement the CR1 cluster-robust covariance estimator (Liang &
# Zeger, 1986; surveyed in Cameron & Miller, 2015, "A Practitioner's Guide
# to Cluster-Robust Inference", eq. 8), clustering by `cluster_id` — the
# level at which this practice's treatment variable `x2` actually varies.
#
# Why this matters: `naive_vcov` above treats every one of the n rows as
# an independent piece of information. It isn't — rows sharing a
# cluster-period share both their x2 value and a common shock. The fix is
# to build the sandwich's "meat" from *cluster-level* summed scores rather
# than per-observation ones, so within-cluster correlation is absorbed
# rather than ignored.
#
# Procedure (Cameron & Miller, 2015, eq. 8):
#   1. For each cluster g, compute the (k,) score sum
#      s_g = sum_{i in g} X_i * resid_i (X_i is regressor row i).
#   2. Meat matrix M = sum_g outer(s_g, s_g).
#   3. Small-sample correction factor
#      c = (G / (G - 1)) * ((n - 1) / (n - k))
#      where G = number of clusters, n = observations, k = regressors.
#   4. Var(beta_hat) = c * XtX_inv @ M @ XtX_inv.
# ---------------------------------------------------------------------------
def cluster_robust_vcov(
    X: np.ndarray, resid: np.ndarray, XtX_inv: np.ndarray, cluster_id: np.ndarray
) -> np.ndarray:
    """CR1 cluster-robust covariance matrix, with the standard small-sample correction.

    `cluster_id` is a (n,) array of cluster labels, one per observation.
    Returns the (k, k) covariance matrix.
    """
    raise NotImplementedError("TODO(human): implement the cluster-robust (CR1) covariance estimator")


def _one_rep_covers_truth(seed: int, alpha: float = 0.05) -> tuple[bool, bool]:
    """One Monte Carlo replication: fit the two-way estimator on a fresh
    panel draw, build a 95% CI for beta_x2 with both the naive and the
    cluster-robust variance, and report whether each CI covers the true
    beta_x2. Fully scaffolded — the estimators being compared are the
    point of this phase, not the simulation loop around them."""
    from scipy import stats

    panel = generate_panel(**SMALL_PANEL, seed=seed)
    df = panel.df
    cluster_id = df["cluster_id"].to_numpy()

    y_tilde, X_tilde = two_way_demean(
        df["y"].to_numpy(), df[["x1", "x2"]].to_numpy(), df["unit_id"].to_numpy(), df["time_id"].to_numpy()
    )
    fit = ols_lstsq(X_tilde, y_tilde)

    # The naive CI uses the usual large-sample normal critical value. The
    # clustered CI uses a t(G-1) critical value instead of the normal one —
    # the standard small-number-of-clusters correction (Cameron & Miller,
    # 2015, sec. 4): with only a handful of clusters, the cluster-robust
    # variance estimate is itself noisy, and a t distribution with few
    # degrees of freedom has fatter tails that account for that.
    n_clusters = len(np.unique(cluster_id))
    z = stats.norm.ppf(1 - alpha / 2)
    t_g = stats.t.ppf(1 - alpha / 2, df=n_clusters - 1)
    beta_x2_true = TRUE_BETA[1]
    beta_x2_hat = fit.beta[1]

    naive_se = np.sqrt(naive_vcov(X_tilde, fit.resid, fit.XtX_inv)[1, 1])
    cluster_se = np.sqrt(cluster_robust_vcov(X_tilde, fit.resid, fit.XtX_inv, cluster_id)[1, 1])

    naive_covers = abs(beta_x2_hat - beta_x2_true) <= z * naive_se
    cluster_covers = abs(beta_x2_hat - beta_x2_true) <= t_g * cluster_se
    return naive_covers, cluster_covers


def simulate_cluster_coverage(n_reps: int = 200, alpha: float = 0.05) -> dict[str, float]:
    """Monte Carlo: over `n_reps` fresh panel draws, track how often each
    method's nominal `1 - alpha` CI for beta_x2 actually covers the truth.

    Returns `{"naive": rate, "clustered": rate}` — under correct clustering
    the rate should sit near `1 - alpha`; ignoring clustering should sit
    well below it, since x2's cluster-period assignment plus the
    cluster-period error shock are exactly what naive SEs can't see.
    """
    naive_hits = 0
    cluster_hits = 0
    for seed in range(n_reps):
        naive_covers, cluster_covers = _one_rep_covers_truth(seed=1000 + seed, alpha=alpha)
        naive_hits += int(naive_covers)
        cluster_hits += int(cluster_covers)
    return {"naive": naive_hits / n_reps, "clustered": cluster_hits / n_reps}


def main() -> None:
    panel = generate_panel(**SMALL_PANEL, seed=0)
    df = panel.df
    try:
        fit = fit_two_way(df, "y", ["x1", "x2"], "unit_id", "time_id")
        _, X_tilde = two_way_demean(
            df["y"].to_numpy(), df[["x1", "x2"]].to_numpy(), df["unit_id"].to_numpy(), df["time_id"].to_numpy()
        )
    except NotImplementedError as e:
        print(f"(skipped — Phase 2 not implemented yet — {e})")
        return

    naive_se = np.sqrt(np.diag(naive_vcov(X_tilde, fit.resid, fit.XtX_inv)))
    print(f"Naive SE:    {naive_se}")
    try:
        cluster_se = np.sqrt(
            np.diag(cluster_robust_vcov(X_tilde, fit.resid, fit.XtX_inv, df["cluster_id"].to_numpy()))
        )
        print(f"Cluster SE:  {cluster_se}")
    except NotImplementedError as e:
        print(f"(skipped cluster SE — {e})")
        return

    print("\nRunning coverage simulation (this takes a few seconds)...")
    coverage = simulate_cluster_coverage(n_reps=200)
    print(f"Coverage — naive:     {coverage['naive']:.2%}")
    print(f"Coverage — clustered: {coverage['clustered']:.2%}")


if __name__ == "__main__":
    main()
