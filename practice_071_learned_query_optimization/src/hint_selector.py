"""Hint selection module.

Contains the Bao-style hint configuration generation and selection logic.
These functions implement the core decision loop: generate alternative plans
by toggling optimizer parameters, predict latency for each, select the best.

Used by: 03_hint_selection.py, 04_evaluation.py
"""

import numpy as np
import torch

from src.shared import MODELS_DIR, get_explain_json
from src.explain_features import parse_plan_node, flatten_plan_tree
from src.cost_model import LatencyPredictor


# ---------------------------------------------------------------------------
# TODO(human): Implement generate_hint_configs
# ---------------------------------------------------------------------------

# TODO(human): Implement generate_hint_configs() -> list[dict[str, str]]
#
# Generate a list of PostgreSQL optimizer hint configurations. Each config
# is a dict mapping SET parameter names to values. The "default" config
# (empty dict) lets PostgreSQL use its normal optimizer with all operators
# enabled.
#
# PostgreSQL's optimizer can be steered by disabling specific operators:
#   - enable_hashjoin: controls hash join usage (good for large equi-joins)
#   - enable_mergejoin: controls merge join (good for pre-sorted data)
#   - enable_nestloop: controls nested loop join (good for small inner tables)
#   - enable_seqscan: controls sequential scans (disabling forces index use)
#   - enable_indexscan: controls index scans
#   - enable_bitmapscan: controls bitmap index scans
#   - enable_sort: controls explicit sort operations
#   - enable_hashagg: controls hash aggregation vs sorted aggregation
#
# Strategy for choosing configurations:
#   Don't enumerate all 2^8 = 256 combinations -- most are redundant or
#   produce identical plans. Instead, select ~8-16 MEANINGFUL configs:
#
#   - Default (all enabled) -- the baseline
#   - Disable each join type individually (3 configs: no-hash, no-merge,
#     no-nestloop) -- forces the optimizer to use alternative join strategies
#   - Disable pairs of join types (3 configs) -- forces specific join type
#   - Disable seqscan -- forces index usage (catches cases where the
#     optimizer underestimates index benefit)
#   - Disable indexscan + bitmapscan -- forces sequential scans (catches
#     cases where index overhead isn't worth it)
#   - Disable hashagg -- forces sort-based aggregation
#   - A few combined configs targeting common pathological cases
#
# Return: list of dicts, e.g.:
#   [
#     {},  # default -- all operators enabled
#     {"enable_hashjoin": "off"},
#     {"enable_mergejoin": "off"},
#     {"enable_nestloop": "off"},
#     ...
#   ]
#
# Why not ALL combinations? Because:
#   1. Most queries only use 2-3 operator types, so disabling unused ones
#      produces identical plans
#   2. More configs = more EXPLAIN calls per query = slower hint selection
#   3. The Bao paper found that ~48 "arm" configurations covered the useful
#      space -- we use fewer since this is a learning exercise
def generate_hint_configs() -> list[dict[str, str]]:
    raise NotImplementedError("Implement generate_hint_configs — see TODO above")


# ---------------------------------------------------------------------------
# TODO(human): Implement select_best_hints
# ---------------------------------------------------------------------------

# TODO(human): Implement select_best_hints(
#     model: LatencyPredictor,
#     query: str,
#     configs: list[dict[str, str]],
#     conn,
#     feature_mean: np.ndarray,
#     feature_std: np.ndarray,
# ) -> tuple[dict[str, str], float, list[tuple[dict[str, str], float]]]
#
# For a given query, evaluate all hint configurations and select the one
# with the lowest PREDICTED latency. This is the core Bao decision loop.
#
# Steps for each hint configuration:
#   1. Apply the hint config via SET commands:
#        for key, val in config.items():
#            cursor.execute(f"SET {key} = {val}")
#      If config is empty (default), skip this step.
#
#   2. Run EXPLAIN (FORMAT JSON) -- WITHOUT ANALYZE -- to get the optimizer's
#      plan under this configuration. We use plain EXPLAIN (not ANALYZE)
#      because we don't want to actually execute the query for each config.
#      We only need the plan structure and cost estimates.
#
#   3. Parse the plan using parse_plan_node() and featurize using
#      flatten_plan_tree().
#
#   4. Normalize the features using the provided feature_mean and
#      feature_std (from training).
#
#   5. Predict latency using the model:
#        model.eval()
#        with torch.no_grad():
#            pred_log = model(features_tensor).item()
#        pred_ms = np.expm1(pred_log)  # inverse log1p
#
#   6. After evaluating ALL configs, RESET ALL settings to restore defaults.
#
#   7. Return a tuple of:
#      - best_config: the dict with lowest predicted latency
#      - best_pred_ms: the predicted latency for the best config
#      - all_results: list of (config, predicted_ms) for analysis
#
# Error handling:
#   Some hint configs may cause the optimizer to produce plans it normally
#   wouldn't, and in rare cases EXPLAIN might fail. Wrap in try/except
#   and skip failed configurations.
#
# Why EXPLAIN without ANALYZE?
#   At query time in a real Bao deployment, we want to select hints BEFORE
#   executing the query. Running ANALYZE would execute the query N times
#   (once per config) -- defeating the purpose. We only need the plan
#   structure, not actual execution times.
def select_best_hints(
    model: LatencyPredictor,
    query: str,
    configs: list[dict[str, str]],
    conn,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> tuple[dict[str, str], float, list[tuple[dict[str, str], float]]]:
    raise NotImplementedError("Implement select_best_hints — see TODO above")


# ---------------------------------------------------------------------------
# Model loading helper (scaffolded)
# ---------------------------------------------------------------------------

def load_model() -> tuple[LatencyPredictor, np.ndarray, np.ndarray]:
    """Load trained model and normalization statistics."""
    model_path = MODELS_DIR / "latency_predictor.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run Exercise 2 first: uv run python src/02_cost_model.py"
        )

    checkpoint = torch.load(model_path, weights_only=False)
    model = LatencyPredictor(checkpoint["n_features"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["feature_mean"], checkpoint["feature_std"]
