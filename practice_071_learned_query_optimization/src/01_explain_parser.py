"""Exercise 1: Parse EXPLAIN JSON into Feature Vectors.

PostgreSQL's EXPLAIN (ANALYZE, FORMAT JSON) returns a tree-structured JSON
representation of the query execution plan. Each node in the tree is an
operator (Seq Scan, Hash Join, Sort, etc.) with cost estimates and actual
execution statistics.

This exercise teaches you to:
1. Extract structured features from each plan node
2. Aggregate those features into a fixed-size vector suitable for ML

The feature vectors produced here become the training data for Exercise 2.

TODO(human) functions are in src/explain_features.py:
  - parse_plan_node()
  - flatten_plan_tree()
"""

import json

import numpy as np
import pandas as pd
import psycopg2

from src.shared import DATA_DIR, DB_CONFIG, TEMPLATE_QUERIES, get_explain_json
from src.explain_features import parse_plan_node, flatten_plan_tree


# ---------------------------------------------------------------------------
# Data collection (scaffolded)
# ---------------------------------------------------------------------------

def collect_plan_data(
    conn,
    queries: list[str],
    n_repeats: int = 3,
) -> list[dict]:
    """Collect plan features and actual latencies for a set of queries.

    Each query is executed n_repeats times with EXPLAIN ANALYZE to get
    stable latency measurements. The median latency is used as the target.
    """
    records = []
    total = len(queries) * n_repeats
    done = 0

    for qi, query in enumerate(queries):
        latencies = []
        plan_json = None

        for rep in range(n_repeats):
            done += 1
            print(f"\r  Collecting [{done}/{total}] query {qi+1}/{len(queries)}, "
                  f"repeat {rep+1}/{n_repeats}...", end="", flush=True)

            plan_json = get_explain_json(conn, query, analyze=True)
            # Actual Total Time at root = total query time in ms
            latency_ms = plan_json.get("Actual Total Time", 0.0)
            latencies.append(latency_ms)

        # Parse the last plan (structure is the same across repeats)
        parsed = parse_plan_node(plan_json)
        features = flatten_plan_tree(parsed)
        median_latency = float(np.median(latencies))

        records.append({
            "query_idx": qi,
            "query": query,
            "latency_ms": median_latency,
            "features": features,
            "plan_json": json.dumps(plan_json),
        })

    print()  # newline after progress
    return records


def save_plan_data(records: list[dict]) -> None:
    """Save collected plan data to CSV files for Exercise 2."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Feature matrix
    feature_matrix = np.vstack([r["features"] for r in records])
    n_features = feature_matrix.shape[1]
    feature_cols = [f"feat_{i}" for i in range(n_features)]
    df_features = pd.DataFrame(feature_matrix, columns=feature_cols)
    df_features["query_idx"] = [r["query_idx"] for r in records]
    df_features["latency_ms"] = [r["latency_ms"] for r in records]
    df_features["query"] = [r["query"] for r in records]
    df_features.to_csv(DATA_DIR / "plan_features.csv", index=False)

    # Raw plans (for inspection)
    df_plans = pd.DataFrame([
        {"query_idx": r["query_idx"], "query": r["query"],
         "latency_ms": r["latency_ms"], "plan_json": r["plan_json"]}
        for r in records
    ])
    df_plans.to_csv(DATA_DIR / "raw_plans.csv", index=False)

    print(f"\n  Saved {len(records)} plan records to {DATA_DIR}/")
    print(f"  Feature vector size: {n_features}")
    print(f"  Files: plan_features.csv, raw_plans.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Exercise 1: Parse EXPLAIN JSON into Feature Vectors")
    print("=" * 60)

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True

    # Warm up the database cache
    print("\nWarming up database cache...")
    with conn.cursor() as cur:
        for table in ["customers", "products", "orders", "order_items", "reviews"]:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
    print("  Done.")

    # Demonstrate EXPLAIN JSON structure
    print("\nExample EXPLAIN (FORMAT JSON) output:")
    print("-" * 40)
    sample_plan = get_explain_json(conn, TEMPLATE_QUERIES[0], analyze=True)
    print(json.dumps(sample_plan, indent=2)[:1000])
    print("  ... (truncated)")

    # Collect plan data for all template queries
    print(f"\nCollecting plans for {len(TEMPLATE_QUERIES)} template queries "
          f"(3 repeats each)...")
    records = collect_plan_data(conn, TEMPLATE_QUERIES, n_repeats=3)

    # Show summary
    print("\nLatency summary:")
    print("-" * 40)
    for r in sorted(records, key=lambda x: x["latency_ms"], reverse=True):
        print(f"  Query {r['query_idx']:2d}: {r['latency_ms']:10.2f} ms  "
              f"{r['query'][:60]}...")

    # Save data
    save_plan_data(records)

    # Also collect plans with different hint configurations for Exercise 3
    # This gives us more diverse training data (same query, different plans)
    print("\nCollecting plans with varied optimizer hints for training diversity...")
    hint_configs = [
        {"enable_hashjoin": "off"},
        {"enable_mergejoin": "off"},
        {"enable_nestloop": "off"},
        {"enable_seqscan": "off"},
        {"enable_hashjoin": "off", "enable_mergejoin": "off"},
        {"enable_indexscan": "off", "enable_bitmapscan": "off"},
    ]

    augmented_records = list(records)  # start with default plans
    for hi, hints in enumerate(hint_configs):
        hint_desc = ", ".join(f"{k}={v}" for k, v in hints.items())
        print(f"\n  Hint config {hi+1}/{len(hint_configs)}: {hint_desc}")

        # Apply hints
        with conn.cursor() as cur:
            for key, val in hints.items():
                cur.execute(f"SET {key} = {val}")

        hint_records = collect_plan_data(conn, TEMPLATE_QUERIES, n_repeats=2)
        for r in hint_records:
            r["hint_config"] = hint_desc
        augmented_records.extend(hint_records)

        # Reset hints
        with conn.cursor() as cur:
            cur.execute("RESET ALL")

    # Save augmented dataset
    print(f"\nTotal training samples: {len(augmented_records)} "
          f"({len(records)} default + {len(augmented_records) - len(records)} hinted)")
    save_plan_data(augmented_records)

    conn.close()
    print("\nExercise 1 complete!")


if __name__ == "__main__":
    main()
