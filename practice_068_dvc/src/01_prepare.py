"""Pipeline stage 1: Prepare -- split raw data into train/test sets.

This script is invoked by DVC as a pipeline stage defined in dvc.yaml.
It reads the raw CSV, splits it into train and test sets, and writes
them to data/prepared/.

DVC tracks:
  - deps: data/raw/customers.csv, src/01_prepare.py
  - params: prepare.test_ratio, prepare.seed
  - outs: data/prepared/

When DVC runs this stage via `dvc repro`, it checks whether any deps
or params have changed since the last run. If nothing changed, the
stage is skipped (smart caching). If you change `prepare.test_ratio`
in params.yaml, only this stage and its downstream consumers re-run.
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


# TODO(human): Implement the prepare() function
#
# This is the first stage of your DVC pipeline. DVC will call this script
# as a shell command (`python src/01_prepare.py`) defined in dvc.yaml.
# Your job is to read the raw data, split it, and write the output files.
#
# What to implement:
#
# 1. Load parameters from params.yaml using the load_params() helper above.
#    The params you need are under the "prepare" key:
#      - prepare.test_ratio  (float, e.g. 0.2 -- fraction for test set)
#      - prepare.seed        (int, e.g. 42 -- random seed for reproducibility)
#
# 2. Read the raw CSV from data/raw/customers.csv using pandas.
#
# 3. Split the DataFrame into train and test sets.
#    Use sklearn.model_selection.train_test_split:
#      from sklearn.model_selection import train_test_split
#      train_df, test_df = train_test_split(
#          df, test_size=params["prepare"]["test_ratio"],
#          random_state=params["prepare"]["seed"],
#          stratify=df["churned"]   # <-- preserve class balance
#      )
#    Stratification ensures both sets have similar churn rates.
#
# 4. Create the output directory: data/prepared/
#    Use os.makedirs(..., exist_ok=True) or Path.mkdir(parents=True, exist_ok=True)
#
# 5. Write train_df and test_df to CSV:
#      data/prepared/train.csv   (index=False)
#      data/prepared/test.csv    (index=False)
#
# 6. Print a summary (use ASCII only, no Unicode box-drawing):
#    - Number of rows in train and test
#    - Class distribution in each set
#
# Why this matters for DVC:
#   This script has NO knowledge of DVC -- it's a plain Python script that
#   reads inputs and writes outputs. The DVC magic happens in dvc.yaml,
#   where you declare the deps (inputs), params, and outs (outputs).
#   DVC uses these declarations to decide when to re-run the stage.
#   This clean separation (script knows nothing about DVC, dvc.yaml
#   declares the contract) is a key DVC design principle.
#
# Expected file structure after running:
#   data/
#     raw/
#       customers.csv        (input, tracked by DVC via .dvc file)
#     prepared/
#       train.csv            (output, tracked by DVC via dvc.yaml)
#       test.csv             (output, tracked by DVC via dvc.yaml)


def prepare() -> None:
    raise NotImplementedError(
        "TODO(human): Implement the prepare() function. "
        "Read raw CSV, split into train/test, write to data/prepared/."
    )


def main() -> None:
    print("=" * 60)
    print("Stage: prepare")
    print("=" * 60)
    prepare()


if __name__ == "__main__":
    main()
