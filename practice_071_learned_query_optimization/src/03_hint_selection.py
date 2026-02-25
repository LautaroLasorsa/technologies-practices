"""Exercise 3: Bao-Style Hint Selection.

This exercise implements the core Bao strategy:
1. Define a set of PostgreSQL optimizer hint configurations
2. For each query, get the plan under each configuration
3. Use the trained model to predict latency for each plan
4. Select the configuration with the lowest predicted latency

The key insight: we don't replace the optimizer -- we just ask it to produce
plans under different constraints, then use our model to pick the best one.

TODO(human) functions are in src/hint_selector.py:
  - generate_hint_configs()
  - select_best_hints()
"""

import numpy as np
import pandas as pd
import psycopg2

from src.shared import DB_CONFIG, DATA_DIR, TEMPLATE_QUERIES, get_explain_json
from src.hint_selector import (
    generate_hint_configs,
    load_model,
    select_best_hints,
)


# ---------------------------------------------------------------------------
# Main (scaffolded)
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Exercise 3: Bao-Style Hint Selection")
    print("=" * 60)

    # Load trained model
    print("\nLoading trained model...")
    model, feature_mean, feature_std = load_model()
    print("  Model loaded.")

    # Generate hint configurations
    print("\nGenerating hint configurations...")
    configs = generate_hint_configs()
    print(f"  Generated {len(configs)} configurations:")
    for i, cfg in enumerate(configs):
        desc = "default (all enabled)" if not cfg else ", ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"    [{i}] {desc}")

    # Connect to database
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True

    # Test hint selection on a subset of queries
    test_queries = TEMPLATE_QUERIES[:10]  # first 10 queries

    print(f"\nRunning hint selection on {len(test_queries)} queries...")
    print("-" * 60)

    results = []
    for qi, query in enumerate(test_queries):
        print(f"\nQuery {qi}: {query[:70]}...")

        best_config, best_pred, all_preds = select_best_hints(
            model, query, configs, conn, feature_mean, feature_std,
        )

        # Get actual latency with best hints for comparison
        with conn.cursor() as cur:
            for key, val in best_config.items():
                cur.execute(f"SET {key} = {val}")

        actual_plan = get_explain_json(conn, query, analyze=True)
        actual_latency = actual_plan.get("Actual Total Time", 0.0)

        with conn.cursor() as cur:
            cur.execute("RESET ALL")

        # Get default latency
        default_plan = get_explain_json(conn, query, analyze=True)
        default_latency = default_plan.get("Actual Total Time", 0.0)

        best_desc = "default" if not best_config else ", ".join(
            f"{k}={v}" for k, v in best_config.items()
        )
        speedup = default_latency / max(actual_latency, 0.001)

        print(f"  Best config: {best_desc}")
        print(f"  Predicted: {best_pred:.2f} ms")
        print(f"  Actual (with hints): {actual_latency:.2f} ms")
        print(f"  Default (no hints): {default_latency:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")

        results.append({
            "query_idx": qi,
            "query": query[:80],
            "best_config": best_desc,
            "predicted_ms": best_pred,
            "actual_with_hints_ms": actual_latency,
            "default_ms": default_latency,
            "speedup": speedup,
        })

    # Summary
    print("\n" + "=" * 60)
    print("Hint Selection Summary")
    print("=" * 60)

    speedups = [r["speedup"] for r in results]
    improved = sum(1 for s in speedups if s > 1.05)
    regressed = sum(1 for s in speedups if s < 0.95)
    neutral = len(speedups) - improved - regressed

    print(f"\n  Queries improved (>5% faster): {improved}/{len(results)}")
    print(f"  Queries regressed (>5% slower): {regressed}/{len(results)}")
    print(f"  Queries neutral: {neutral}/{len(results)}")
    print(f"  Mean speedup: {np.mean(speedups):.2f}x")
    print(f"  Median speedup: {np.median(speedups):.2f}x")

    # Save results
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(DATA_DIR / "hint_selection_results.csv", index=False)
    print(f"\n  Results saved to {DATA_DIR / 'hint_selection_results.csv'}")

    conn.close()
    print("\nExercise 3 complete!")


if __name__ == "__main__":
    main()
