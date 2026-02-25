"""Evaluation module.

Contains functions for running workloads and generating comparison reports
between the native PostgreSQL optimizer and learned hint selection.

Used by: 04_evaluation.py
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from src.shared import PLOTS_DIR


# ---------------------------------------------------------------------------
# TODO(human): Implement run_workload
# ---------------------------------------------------------------------------

# TODO(human): Implement run_workload(
#     conn,
#     queries: list[str],
#     hint_config: dict[str, str] | None = None,
#     n_repeats: int = 3,
# ) -> list[tuple[str, float]]
#
# Execute a list of queries against PostgreSQL and measure actual wall-clock
# execution time for each. This is the ground-truth measurement we use to
# compare the native optimizer against learned hint selection.
#
# Steps:
#   1. If hint_config is provided (not None and not empty), apply the hints
#      via SET commands before executing queries:
#        for key, val in hint_config.items():
#            cursor.execute(f"SET {key} = {val}")
#
#   2. For each query, execute it n_repeats times and take the MEDIAN
#      latency. We use median (not mean) because it's robust to outliers
#      from cache warming or background activity.
#
#      Measure wall-clock time using time.perf_counter():
#        start = time.perf_counter()
#        cursor.execute(query)
#        cursor.fetchall()  # IMPORTANT: must consume results!
#        elapsed_ms = (time.perf_counter() - start) * 1000
#
#      Why fetchall()? PostgreSQL's iterator model means execute() only
#      starts the plan -- rows are fetched lazily. Without fetchall(), you'd
#      only measure planning time, not execution time.
#
#   3. After all queries are done, RESET ALL to restore default settings.
#
#   4. Return a list of (query, median_latency_ms) tuples.
#
# Important considerations:
#   - The first execution of each query may be slower due to cache warming.
#     Using n_repeats=3 and taking the median handles this.
#   - Wrap individual query execution in try/except -- some hint configs
#     may cause queries to become very slow (especially disabling all
#     join types). Consider a timeout mechanism if needed.
#   - Print progress since the full workload can take a few minutes.
def run_workload(
    conn,
    queries: list[str],
    hint_config: dict[str, str] | None = None,
    n_repeats: int = 3,
) -> list[tuple[str, float]]:
    raise NotImplementedError("Implement run_workload — see TODO above")


# ---------------------------------------------------------------------------
# TODO(human): Implement generate_comparison_report
# ---------------------------------------------------------------------------

# TODO(human): Implement generate_comparison_report(
#     native_results: list[tuple[str, float]],
#     learned_results: list[tuple[str, float]],
# ) -> dict
#
# Compare the native optimizer results against learned hint selection
# results and generate a comprehensive report with summary statistics
# and visualizations.
#
# Statistics to compute:
#   - mean_native_ms, mean_learned_ms: average latency across all queries
#   - median_native_ms, median_learned_ms: median latency
#   - p99_native_ms, p99_learned_ms: 99th percentile latency (tail)
#   - total_native_ms, total_learned_ms: total workload time
#   - overall_speedup: total_native / total_learned
#   - per_query_speedup: list of (query, native_ms, learned_ms, speedup)
#   - num_improved: queries where learned is >5% faster
#   - num_regressed: queries where learned is >5% slower
#   - num_neutral: remaining queries
#
# Visualizations to generate (save to PLOTS_DIR):
#   1. Bar chart: per-query latency comparison (native vs learned side by
#      side). Use query index as x-axis labels. Color bars differently
#      for native (blue) and learned (green).
#
#   2. Speedup histogram: distribution of per-query speedup ratios.
#      Add a vertical line at 1.0 (no change). Queries left of 1.0
#      are regressions, right of 1.0 are improvements.
#
# Return a summary dict with all the statistics above.
#
# Why these specific metrics?
#   - Mean latency shows overall improvement
#   - Median is more robust to outliers
#   - P99 captures tail latency -- often the most impactful metric in
#     production (one slow query can block an entire request)
#   - Per-query speedup shows WHERE the model helps and WHERE it hurts
#   - The speedup histogram reveals the distribution -- is the model
#     consistently helpful or only on a few queries?
def generate_comparison_report(
    native_results: list[tuple[str, float]],
    learned_results: list[tuple[str, float]],
) -> dict:
    raise NotImplementedError("Implement generate_comparison_report — see TODO above")
