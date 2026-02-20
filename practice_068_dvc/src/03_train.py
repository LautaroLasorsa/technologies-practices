"""Pipeline stage 3: Train -- fit a logistic regression model.

This script reads the featurized train data, trains a LogisticRegression
model, and saves the trained model to models/model.pkl using joblib.

DVC tracks:
  - deps: data/prepared/train_featurized.csv, src/03_train.py
  - params: train.C, train.max_iter, train.solver
  - outs: models/model.pkl

The model hyperparameters come from params.yaml. When you change a
hyperparameter (e.g., increase train.C from 1.0 to 10.0), DVC detects
that the params dependency changed and re-runs this stage + evaluate.
The prepare and featurize stages are NOT re-run because their
deps/params are unchanged. This is smart caching in action.
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


# TODO(human): Implement the train() function
#
# This is the third stage of your DVC pipeline. It trains a model and
# saves it to disk so the evaluate stage can load and test it.
#
# What to implement:
#
# 1. Load parameters from params.yaml. You need:
#      - train.C         (float, e.g. 1.0 -- regularization strength)
#      - train.max_iter  (int, e.g. 100 -- max iterations for solver)
#      - train.solver    (str, e.g. "lbfgs" -- optimization algorithm)
#
# 2. Read train_featurized.csv from data/prepared/ using pandas.
#
# 3. Separate features from target:
#      X_train = train_df.drop(columns=["churned"])
#      y_train = train_df["churned"]
#
# 4. Create and fit a LogisticRegression model:
#      from sklearn.linear_model import LogisticRegression
#      model = LogisticRegression(
#          C=params["train"]["C"],
#          max_iter=params["train"]["max_iter"],
#          solver=params["train"]["solver"],
#          random_state=42
#      )
#      model.fit(X_train, y_train)
#
# 5. Create the models/ directory and save the model:
#      import joblib
#      models_dir = PRACTICE_ROOT / "models"
#      models_dir.mkdir(parents=True, exist_ok=True)
#      joblib.dump(model, models_dir / "model.pkl")
#
# 6. Print training summary:
#    - Hyperparameters used (C, max_iter, solver)
#    - Training accuracy: model.score(X_train, y_train)
#    - Number of iterations: model.n_iter_ (how many iterations the solver took)
#    - Model coefficients: model.coef_ (shows which features matter most)
#
# Why this matters for DVC:
#   The model artifact (model.pkl) is tracked as an output in dvc.yaml.
#   DVC hashes this binary file and stores it in the DVC cache. When you
#   switch git branches (different hyperparams -> different model), `dvc
#   checkout` restores the correct model.pkl from cache without retraining.
#   This is how DVC enables "time travel" for ML artifacts.
#
# Hint on model coefficients:
#   After training, `model.coef_[0]` gives the weight for each feature.
#   Zip it with the feature column names to see which features the model
#   relies on most. Positive coefficients increase churn probability,
#   negative ones decrease it.


def train() -> None:
    raise NotImplementedError(
        "TODO(human): Implement the train() function. "
        "Read featurized data, train LogisticRegression, save model.pkl."
    )


def main() -> None:
    print("=" * 60)
    print("Stage: train")
    print("=" * 60)
    train()


if __name__ == "__main__":
    main()
