# Practice 104 — Regression Discontinuity Design

Sharp and fuzzy RDD from scratch: a kernel-weighted local linear estimator, a plug-in bandwidth selector, and a Wald-ratio fuzzy estimator, validated against `rdrobust` on synthetic data with a known jump, then applied to the real Lee (2008) US House elections dataset. Manipulation (McCrary density test) and covariate-balance checks make the design's identifying assumption falsifiable. See [`CLAUDE.md`](./CLAUDE.md) for full instructions, theoretical context, and the command table.
