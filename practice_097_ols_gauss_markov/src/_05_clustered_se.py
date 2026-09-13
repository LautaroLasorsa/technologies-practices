"""Phase 5 — Cluster-robust standard errors.

HC0-HC3 (Phase 4) fix heteroskedasticity but still assume errors are
*independent* across observations. That's false whenever data comes in
natural groups sharing a common shock — students within a school, repeated
observations per firm, regions within a country. This phase implements the
cluster-robust ("CR1") sandwich estimator that relaxes exactly that part of
assumption A4, the standard fix in panel and grouped cross-sectional data.

Run on its own to see cluster-robust SEs on the clustered scenario:
    uv run python -m src._05_clustered_se
"""
from __future__ import annotations

import numpy as np

from ._01_ols_qr import ols_via_qr
from .datasets import load_dataset


# TODO(human) — cluster-robust (CR1) sandwich covariance estimator
# ---------------------------------------------------------------------------
# Goal: implement the CR1 cluster-robust covariance estimator (Liang &
# Zeger, 1986; surveyed in Cameron & Miller, 2015), which allows errors to
# be correlated *within* a cluster as long as clusters are independent of
# each other.
#
# Why this matters: if observations are grouped and share a within-group
# shock (the "clustered" scenario in datasets.py), even HC3 is still
# wrong — the sandwich's "meat" must be built from *cluster-level* summed
# scores, not per-observation ones, or standard errors are badly
# understated (the classic "everyone thinks their result is significant"
# failure mode of ignoring clustering).
#
# Procedure (Cameron & Miller, 2015, "A Practitioner's Guide to
# Cluster-Robust Inference", eq. 8):
#   1. For each cluster g, compute the (k,) score sum
#      s_g = sum_{i in g} X_i * u_i (X_i is regressor row i, u_i its
#      residual).
#   2. Meat matrix M = sum_g outer(s_g, s_g).
#   3. Small-sample correction factor
#      c = (G / (G - 1)) * ((n - 1) / (n - k))
#      where G = number of clusters, n = observations, k = regressors.
#   4. Var(beta_hat) = c * (X'X)^-1 @ M @ (X'X)^-1.
# ---------------------------------------------------------------------------
def cluster_robust_vcov(
    X: np.ndarray, resid: np.ndarray, XtX_inv: np.ndarray, cluster_id: np.ndarray
) -> np.ndarray:
    """CR1 cluster-robust covariance matrix, with the standard small-sample correction.

    `cluster_id` is a (n,) array of cluster labels, one per observation.
    Returns the (k, k) covariance matrix.
    """
    G = set(cluster_id)
    Sgs = [sum(X[i] * resid[i] for i in range(X.shape[0]) if cluster_id[i]==g) for g in G]
    M = sum([np.outer(s_g,s_g) for s_g in Sgs]) # eq = Sgs.T @ Sgs
    c = (len(G)/(len(G)-1)) * ((X.shape[0]-1)/(X.shape[0]-X.shape[1]))
    return c * XtX_inv @ M @ XtX_inv
    


def main() -> None:
    data = load_dataset("clustered", n=300, seed=0)
    try:
        fit = ols_via_qr(data.X, data.y)
    except NotImplementedError as e:
        print(f"(skipped — Phase 1 not implemented yet — {e})")
        return
    try:
        vcov = cluster_robust_vcov(data.X, fit.resid, fit.XtX_inv, data.cluster_id)
        print(f"Cluster-robust SE: {np.sqrt(np.diag(vcov))}")
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
