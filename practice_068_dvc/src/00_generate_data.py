"""Phase 1 helper: Generate a synthetic classification dataset.

This script creates a CSV file that simulates a real-world ML dataset
(e.g., predicting customer churn). The data is intentionally simple --
the focus of this practice is DVC, not the ML model.

The generated dataset has:
  - 1000 rows (enough to demonstrate versioning, small enough to be fast)
  - 5 numeric features (age, tenure_months, monthly_charge, support_calls, usage_hours)
  - 1 binary target column (churned: 0 or 1)

Run: uv run python src/00_generate_data.py
Output: data/raw/customers.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


PRACTICE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PRACTICE_ROOT / "data" / "raw"


def generate_dataset(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic customer churn dataset.

    The target (churned) is correlated with features so a simple model
    can learn a pattern -- this makes the metrics meaningful when we
    compare experiments with different hyperparameters.
    """
    rng = np.random.RandomState(seed)

    age = rng.randint(18, 70, size=n_samples).astype(float)
    tenure_months = rng.randint(1, 72, size=n_samples).astype(float)
    monthly_charge = rng.uniform(20.0, 120.0, size=n_samples)
    support_calls = rng.poisson(lam=2.0, size=n_samples).astype(float)
    usage_hours = rng.uniform(0.5, 50.0, size=n_samples)

    # Target: higher churn when high charges, many support calls, low tenure
    logit = (
        -2.0
        + 0.03 * monthly_charge
        + 0.4 * support_calls
        - 0.04 * tenure_months
        - 0.01 * usage_hours
        + rng.normal(0, 0.5, size=n_samples)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churned = (rng.uniform(size=n_samples) < prob).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "tenure_months": tenure_months,
            "monthly_charge": monthly_charge,
            "support_calls": support_calls,
            "usage_hours": usage_hours,
            "churned": churned,
        }
    )
    return df


def main() -> None:
    print("=" * 60)
    print("Phase 1: Generate Synthetic Dataset")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    df = generate_dataset(n_samples=1000, seed=42)
    out_path = DATA_DIR / "customers.csv"
    df.to_csv(out_path, index=False)

    print(f"\n  Created {out_path.relative_to(PRACTICE_ROOT)}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Target distribution:")
    print(f"    churned=0: {(df['churned'] == 0).sum()}")
    print(f"    churned=1: {(df['churned'] == 1).sum()}")
    print(f"\n  File size: {out_path.stat().st_size:,} bytes")
    print("\n  Next step: track this file with DVC (see CLAUDE.md Phase 1)")


if __name__ == "__main__":
    main()
