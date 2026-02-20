"""Pipeline stage 4: Evaluate -- compute metrics on the test set.

This script loads the trained model and the featurized test data,
computes classification metrics, and writes them to metrics/scores.json.

DVC tracks:
  - deps: models/model.pkl, data/prepared/test_featurized.csv, src/04_evaluate.py
  - metrics: metrics/scores.json (with cache: false)

The `cache: false` flag on metrics tells DVC NOT to cache this file in
the DVC store -- instead, it's committed directly to Git. This makes
sense because metrics files are tiny (a few bytes of JSON) and you want
them visible in Git diffs and in `dvc metrics diff`.

After running `dvc repro`, you can use:
  - `dvc metrics show`     -- display current metrics
  - `dvc metrics diff`     -- compare metrics between commits/branches
  - `dvc params diff`      -- compare params between commits/branches
These commands are the heart of DVC experiment tracking.
"""

import json
from pathlib import Path

import pandas as pd
import yaml


PRACTICE_ROOT = Path(__file__).resolve().parent.parent


def load_params() -> dict:
    """Load parameters from params.yaml at the practice root."""
    params_path = PRACTICE_ROOT / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)


# TODO(human): Implement the evaluate() function
#
# This is the fourth and final stage of your DVC pipeline. It evaluates
# the trained model and writes metrics that DVC can track and compare.
#
# What to implement:
#
# 1. Load the trained model from models/model.pkl:
#      import joblib
#      model = joblib.load(PRACTICE_ROOT / "models" / "model.pkl")
#
# 2. Read test_featurized.csv from data/prepared/ using pandas.
#
# 3. Separate features from target:
#      X_test = test_df.drop(columns=["churned"])
#      y_test = test_df["churned"]
#
# 4. Make predictions:
#      y_pred = model.predict(X_test)
#      y_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1
#
# 5. Compute metrics:
#      from sklearn.metrics import (
#          accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
#      )
#      metrics = {
#          "accuracy": round(accuracy_score(y_test, y_pred), 4),
#          "precision": round(precision_score(y_test, y_pred), 4),
#          "recall": round(recall_score(y_test, y_pred), 4),
#          "f1": round(f1_score(y_test, y_pred), 4),
#          "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
#      }
#
# 6. Write metrics to JSON:
#      metrics_dir = PRACTICE_ROOT / "metrics"
#      metrics_dir.mkdir(parents=True, exist_ok=True)
#      with open(metrics_dir / "scores.json", "w") as f:
#          json.dump(metrics, f, indent=2)
#
# 7. Print metrics in a readable format:
#      for name, value in metrics.items():
#          print(f"  {name:>12}: {value:.4f}")
#
# Why this matters for DVC:
#   Metrics files declared in dvc.yaml with `cache: false` are committed
#   to Git (not the DVC remote). This means:
#     a) `git diff` shows metric changes between commits
#     b) `dvc metrics diff` formats the comparison nicely
#     c) `dvc metrics diff HEAD~1` compares current vs previous commit
#     d) `dvc metrics diff --md` outputs a markdown table (great for PRs)
#
#   This is the DVC experiment tracking workflow:
#     1. Change params.yaml (e.g., train.C = 10.0)
#     2. Run `dvc repro`     -- re-runs affected stages
#     3. Run `dvc metrics diff` -- see how metrics changed
#     4. If better: `git add . && git commit`
#     5. If worse: `git checkout .` to discard changes
#
#   The params.yaml + metrics/scores.json pair is the core of DVC
#   experiment management. Params define what you changed, metrics
#   define the result. Both are in Git, creating a full audit trail.
#
# Expected output file (metrics/scores.json):
#   {
#     "accuracy": 0.7350,
#     "precision": 0.6842,
#     "recall": 0.6234,
#     "f1": 0.6524,
#     "roc_auc": 0.8012
#   }


def evaluate() -> None:
    raise NotImplementedError(
        "TODO(human): Implement the evaluate() function. "
        "Load model, predict on test set, write metrics/scores.json."
    )


def main() -> None:
    print("=" * 60)
    print("Stage: evaluate")
    print("=" * 60)
    evaluate()


if __name__ == "__main__":
    main()
