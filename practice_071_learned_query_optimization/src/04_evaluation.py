"""Exercise 4: Compare Learned Hints vs Native Optimizer.

The ultimate test: run a workload of queries through both the native
PostgreSQL optimizer (default settings) and the learned hint selector,
measuring actual wall-clock execution time.

This exercise quantifies whether the learned optimizer actually improves
query performance, and on which query types it helps vs hurts.

TODO(human) functions are in src/evaluator.py:
  - run_workload()
  - generate_comparison_report()
"""

from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

from src.shared import DB_CONFIG, DATA_DIR, TEMPLATE_QUERIES, get_explain_json
from src.hint_selector import generate_hint_configs, load_model, select_best_hints
from src.evaluator import run_workload, generate_comparison_report

# ---------------------------------------------------------------------------
# Test workload (extends template queries with variations)
# ---------------------------------------------------------------------------

EVAL_QUERIES = TEMPLATE_QUERIES + [
    # Additional queries for evaluation diversity
    "SELECT c.name, COUNT(DISTINCT oi.product_id) AS unique_products FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id WHERE c.is_premium = true GROUP BY c.customer_id, c.name ORDER BY unique_products DESC LIMIT 15",
    "SELECT p.category, COUNT(*) AS num_orders, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS net_revenue FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.category ORDER BY net_revenue DESC",
    "SELECT shipping_state, status, COUNT(*) AS cnt FROM orders WHERE order_date BETWEEN '2022-06-01' AND '2022-12-31' GROUP BY shipping_state, status ORDER BY shipping_state, cnt DESC",
    "SELECT c.city, AVG(o.total_amount) AS avg_order, COUNT(*) AS num_orders FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.city HAVING COUNT(*) > 50 ORDER BY avg_order DESC",
    "SELECT p.name, p.price, COUNT(r.review_id) AS num_reviews, AVG(r.rating) AS avg_rating FROM products p LEFT JOIN reviews r ON p.product_id = r.product_id WHERE p.category = 'Electronics' GROUP BY p.product_id, p.name, p.price HAVING COUNT(r.review_id) > 0 ORDER BY avg_rating DESC, num_reviews DESC LIMIT 20",
]


# ---------------------------------------------------------------------------
# Main (scaffolded)
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Exercise 4: Evaluate Learned vs Native Optimizer")
    print("=" * 60)

    # Load model
    print("\nLoading trained model...")
    model, feature_mean, feature_std = load_model()
    configs = generate_hint_configs()
    print(f"  Model loaded. {len(configs)} hint configurations.")

    # Connect
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True

    # Warm up caches
    print("\nWarming up database caches...")
    with conn.cursor() as cur:
        for table in ["customers", "products", "orders", "order_items", "reviews"]:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
    print("  Done.")

    # Run native optimizer workload
    print(f"\nRunning NATIVE optimizer on {len(EVAL_QUERIES)} queries...")
    print("-" * 60)
    native_results = run_workload(conn, EVAL_QUERIES, hint_config=None, n_repeats=3)

    # For learned optimizer: select best hints per query, then run
    print(f"\nSelecting learned hints for {len(EVAL_QUERIES)} queries...")
    print("-" * 60)
    learned_configs = {}
    for qi, query in enumerate(EVAL_QUERIES):
        print(f"\r  Selecting hints for query {qi+1}/{len(EVAL_QUERIES)}...",
              end="", flush=True)
        best_config, best_pred, _ = select_best_hints(
            model, query, configs, conn, feature_mean, feature_std,
        )
        learned_configs[qi] = best_config
    print()

    print(f"\nRunning LEARNED optimizer on {len(EVAL_QUERIES)} queries...")
    print("-" * 60)

    # Run each query with its selected hints
    learned_results = []
    for qi, query in enumerate(EVAL_QUERIES):
        config = learned_configs[qi]
        result = run_workload(conn, [query], hint_config=config, n_repeats=3)
        learned_results.extend(result)

    # Generate comparison report
    print("\n" + "=" * 60)
    print("Comparison Report")
    print("=" * 60)

    report = generate_comparison_report(native_results, learned_results)

    print(f"\n  Total workload time:")
    print(f"    Native:  {report['total_native_ms']:.1f} ms")
    print(f"    Learned: {report['total_learned_ms']:.1f} ms")
    print(f"    Overall speedup: {report['overall_speedup']:.2f}x")

    print(f"\n  Latency statistics:")
    print(f"    Mean:   native={report['mean_native_ms']:.2f} ms, "
          f"learned={report['mean_learned_ms']:.2f} ms")
    print(f"    Median: native={report['median_native_ms']:.2f} ms, "
          f"learned={report['median_learned_ms']:.2f} ms")
    print(f"    P99:    native={report['p99_native_ms']:.2f} ms, "
          f"learned={report['p99_learned_ms']:.2f} ms")

    print(f"\n  Query breakdown:")
    print(f"    Improved (>5% faster): {report['num_improved']}/{len(EVAL_QUERIES)}")
    print(f"    Regressed (>5% slower): {report['num_regressed']}/{len(EVAL_QUERIES)}")
    print(f"    Neutral: {report['num_neutral']}/{len(EVAL_QUERIES)}")

    # Print per-query details
    print("\n  Per-query results:")
    print(f"  {'Idx':>4s}  {'Native (ms)':>12s}  {'Learned (ms)':>12s}  "
          f"{'Speedup':>8s}  Query")
    print("  " + "-" * 80)
    for pq in report.get("per_query_speedup", []):
        marker = "+" if pq["speedup"] > 1.05 else ("-" if pq["speedup"] < 0.95 else " ")
        print(f"  {pq['idx']:4d}  {pq['native_ms']:12.2f}  {pq['learned_ms']:12.2f}  "
              f"{pq['speedup']:7.2f}x {marker} {pq['query'][:50]}")

    # Save report
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report.get("per_query_speedup", [])).to_csv(
        DATA_DIR / "evaluation_report.csv", index=False,
    )
    print(f"\n  Report saved to {DATA_DIR / 'evaluation_report.csv'}")

    conn.close()
    print("\nExercise 4 complete!")
    print("\nKey takeaway: the learned optimizer works best on queries where")
    print("PostgreSQL's cardinality estimates are poor (multi-join, correlated")
    print("filters). For simple queries, the native optimizer is already optimal.")


if __name__ == "__main__":
    main()
