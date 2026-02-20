"""Pipeline stage 2: Featurize -- normalize and transform features.

This script reads the train/test splits produced by the prepare stage,
applies feature scaling (StandardScaler), and writes the transformed
data back to data/prepared/ (overwriting the originals with scaled
versions, plus saving the feature names for reference).

DVC tracks:
  - deps: data/prepared/train.csv, data/prepared/test.csv, src/02_featurize.py
  - params: featurize.features (list of column names to use as features)
  - outs: data/prepared/train_featurized.csv, data/prepared/test_featurized.csv

Key learning: the featurize stage depends on the OUTPUTS of prepare.
This dependency chain (prepare -> featurize -> train -> evaluate) forms
the DVC pipeline DAG. When you change a param in the prepare stage,
DVC knows it must re-run prepare, then featurize (because its inputs
changed), then train, then evaluate -- the full downstream cascade.
"""

from pathlib import Path

import pandas as pd
import yaml


PRACTICE_ROOT = Path(__file__).resolve().parent.parent


def load_params() -> dict:
    """Load parameters from params.yaml at the practice root."""
    params_path = PRACTICE_ROOT / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)


# TODO(human): Implement the featurize() function
#
# This is the second stage of your DVC pipeline. It transforms raw
# features into a format suitable for model training.
#
# What to implement:
#
# 1. Load parameters from params.yaml. You need:
#      - featurize.features  (list of str, e.g. ["age", "tenure_months",
#        "monthly_charge", "support_calls", "usage_hours"])
#
# 2. Read train.csv and test.csv from data/prepared/ using pandas.
#
# 3. Separate features (X) from target (y):
#      feature_cols = params["featurize"]["features"]
#      X_train = train_df[feature_cols]
#      y_train = train_df["churned"]
#      X_test  = test_df[feature_cols]
#      y_test  = test_df["churned"]
#
# 4. Fit a StandardScaler on the TRAINING data, then transform BOTH
#    train and test data with it:
#      from sklearn.preprocessing import StandardScaler
#      scaler = StandardScaler()
#      X_train_scaled = pd.DataFrame(
#          scaler.fit_transform(X_train),
#          columns=feature_cols
#      )
#      X_test_scaled = pd.DataFrame(
#          scaler.transform(X_test),   # <-- transform only, NOT fit_transform!
#          columns=feature_cols
#      )
#
#    IMPORTANT: fit_transform on train, transform (only) on test.
#    This prevents data leakage -- the scaler learns statistics only
#    from training data. This is a fundamental ML best practice, and
#    DVC's pipeline structure naturally enforces it by separating stages.
#
# 5. Recombine features with target:
#      train_out = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
#      test_out  = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)
#
# 6. Write outputs:
#      data/prepared/train_featurized.csv  (index=False)
#      data/prepared/test_featurized.csv   (index=False)
#
# 7. Print a summary showing:
#    - Which features were selected
#    - Mean and std of each feature in the training set (should be ~0 and ~1
#      after scaling)
#    - Number of rows in each output file
#
# Why this matters for DVC:
#   The featurize stage depends on outputs of prepare AND on its own params.
#   If you change featurize.features in params.yaml (e.g., remove a column),
#   DVC will re-run featurize, train, and evaluate -- but NOT prepare
#   (because prepare's deps/params didn't change). This selective re-execution
#   is the core value of DVC pipelines over ad-hoc scripts.


def featurize() -> None:
    raise NotImplementedError(
        "TODO(human): Implement the featurize() function. "
        "Read train/test CSVs, scale features, write featurized outputs."
    )


def main() -> None:
    print("=" * 60)
    print("Stage: featurize")
    print("=" * 60)
    featurize()


if __name__ == "__main__":
    main()
