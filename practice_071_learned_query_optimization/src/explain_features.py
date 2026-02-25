"""Plan parsing and featurization module.

Contains the core functions for converting PostgreSQL EXPLAIN JSON output
into feature vectors suitable for ML training. These functions are the
foundation of the learned query optimizer -- all downstream components
(model training, hint selection, evaluation) depend on consistent plan
featurization.

Used by: 01_explain_parser.py, 03_hint_selection.py, 04_evaluation.py
"""

import numpy as np

from src.shared import NODE_TYPE_TO_IDX, NODE_TYPES


# ---------------------------------------------------------------------------
# TODO(human): Implement parse_plan_node
# ---------------------------------------------------------------------------

# TODO(human): Implement parse_plan_node(node_json: dict) -> dict
#
# Recursively parse a single plan node from PostgreSQL's EXPLAIN (FORMAT JSON)
# output. Each node in the JSON tree has keys like "Node Type", "Total Cost",
# "Plan Rows", "Plan Width", and optionally "Plans" (an array of child nodes).
#
# Your function should extract a feature dictionary for this node containing:
#   - "node_type": str  -- The operator name (e.g., "Hash Join", "Seq Scan")
#   - "node_type_idx": int  -- Index in NODE_TYPES list (-1 if unknown type)
#   - "estimated_rows": float  -- "Plan Rows" value (estimated output cardinality)
#   - "estimated_cost": float  -- "Total Cost" value (optimizer's cost estimate)
#   - "startup_cost": float  -- "Startup Cost" (cost before first row is emitted)
#   - "width": int  -- "Plan Width" (estimated average row width in bytes)
#   - "actual_time": float or None  -- "Actual Total Time" if ANALYZE was used
#   - "actual_rows": float or None  -- "Actual Rows" if ANALYZE was used
#   - "children": list[dict]  -- Recursively parsed child nodes from "Plans"
#
# Key considerations:
#   - The "Plans" key may not exist (leaf nodes like Seq Scan have no children)
#   - "Actual Total Time" and "Actual Rows" only exist when EXPLAIN ANALYZE
#     was used (not plain EXPLAIN). Use .get() with None as default.
#   - Some node types (like "Bitmap Heap Scan") have a child "Bitmap Index Scan"
#     that appears in the "Plans" array -- handle recursion for all children.
#   - The NODE_TYPE_TO_IDX dict maps known types to integer indices for later
#     one-hot encoding. Unknown types should map to -1.
#
# Example input (simplified):
#   {
#     "Node Type": "Hash Join",
#     "Total Cost": 1234.56,
#     "Plan Rows": 500,
#     "Plan Width": 120,
#     "Startup Cost": 100.0,
#     "Plans": [
#       {"Node Type": "Seq Scan", "Total Cost": 50.0, ...},
#       {"Node Type": "Hash", "Plans": [{"Node Type": "Seq Scan", ...}], ...}
#     ]
#   }
def parse_plan_node(node_json: dict) -> dict:
    raise NotImplementedError("Implement parse_plan_node — see TODO above")


# ---------------------------------------------------------------------------
# TODO(human): Implement flatten_plan_tree
# ---------------------------------------------------------------------------

# TODO(human): Implement flatten_plan_tree(plan_root: dict) -> np.ndarray
#
# Convert the recursive plan tree (output of parse_plan_node) into a fixed-size
# feature vector suitable for feeding into a neural network. Since plans have
# variable tree shapes and depths, we need to aggregate features into a
# fixed-length representation.
#
# Recommended feature vector components (in order):
#   1. Node type counts (len(NODE_TYPES) values):
#      Count how many times each node type appears in the tree.
#      e.g., [2 Seq Scans, 0 Index Scans, 1 Hash Join, ...]
#      This tells the model WHAT operators the plan uses.
#
#   2. Cost aggregates (3 values):
#      - total_cost: sum of estimated_cost across all nodes
#      - max_cost: maximum single-node estimated_cost
#      - total_startup_cost: sum of startup_cost across all nodes
#      These capture the optimizer's view of plan expense.
#
#   3. Row estimate aggregates (3 values):
#      - total_rows: sum of estimated_rows across all nodes
#      - max_rows: maximum estimated_rows at any single node
#      - min_rows: minimum estimated_rows (leaf selectivity signal)
#      These capture data volume flowing through the plan.
#
#   4. Structural features (3 values):
#      - tree_depth: maximum depth of the plan tree
#      - num_nodes: total number of operator nodes
#      - total_width: sum of width across all nodes
#      These capture plan shape and row width.
#
# Total vector size: len(NODE_TYPES) + 3 + 3 + 3 = len(NODE_TYPES) + 9
#
# Implementation hint: write a recursive helper that walks the tree and
# accumulates these statistics. For depth, track the current level and
# take the maximum. For counts, use a dict or numpy array.
#
# Why this approach? Flat featurization loses structural info (which child
# is left vs. right in a join) but is simple and surprisingly effective for
# plan comparison. The Bao paper uses tree-CNN to preserve structure, but
# flat features work well as a starting point.
def flatten_plan_tree(plan_root: dict) -> np.ndarray:
    raise NotImplementedError("Implement flatten_plan_tree — see TODO above")
