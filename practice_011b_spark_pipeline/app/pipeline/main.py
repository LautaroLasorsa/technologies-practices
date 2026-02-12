"""Retail Analytics ETL Pipeline — Orchestrator.

Run with: uv run python -m pipeline.main

This is the top-level orchestrator that ties together all pipeline stages.
It follows the Stepdown Rule: high-level flow first, details in each module.
"""

from pipeline.analytics import (
    customer_lifetime_value_ranking,
    daily_revenue_with_running_total,
    day_over_day_change,
    top_products_per_category,
)
from pipeline.clean import clean_sales
from pipeline.enrich import enrich_sales
from pipeline.extract import read_products, read_sales
from pipeline.load import write_partitioned_parquet
from pipeline.session import create_spark_session
from pipeline.sql_queries import (
    above_average_products,
    customer_segments,
    monthly_revenue_by_category,
    register_views,
)


# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches Spark DataFrame caching and physical plan inspection.
# Understanding when to cache (reused DataFrames) and how to read explain() output is essential
# for optimizing Spark jobs—caching prevents redundant computation, explain() reveals bottlenecks.

def run_pipeline() -> None:
    """Execute the full ETL pipeline.

    TODO(human): Complete steps marked with TODO below.
    The structure is already laid out — you just need to uncomment/implement
    the caching, explain plan, and unpersist parts.
    """
    spark = create_spark_session()
    print("=" * 60)
    print("  Retail Analytics ETL Pipeline")
    print("=" * 60)

    # ── Phase 2: Extract & Clean ────────────────────────────────
    print("\n[1/6] Extracting raw data...")
    raw_sales = read_sales(spark)
    products = read_products(spark)
    print(f"  Raw sales rows: {raw_sales.count()}")
    print(f"  Products: {products.count()}")

    print("\n[2/6] Cleaning sales data...")
    cleaned = clean_sales(raw_sales)
    print(f"  Cleaned sales rows: {cleaned.count()}")

    # TODO(human): Cache the cleaned DataFrame — it's used in enrich AND later
    # in analytics. Caching avoids re-reading and re-cleaning the CSV twice.
    #   cleaned.cache()
    #   cleaned.count()  # Force materialization (cache is lazy until an action)
    #   print("  Cleaned DataFrame cached.")

    # ── Phase 3: Enrich ─────────────────────────────────────────
    print("\n[3/6] Enriching with product catalog (broadcast join)...")
    enriched = enrich_sales(cleaned, products)
    print(f"  Enriched rows: {enriched.count()}")
    enriched.printSchema()

    # TODO(human): Print the physical plan to see the broadcast join.
    # This helps you verify that Spark actually used BroadcastHashJoin.
    #   print("\n  Explain plan (enriched):")
    #   enriched.explain(True)

    # ── Phase 4: Window Functions ───────────────────────────────
    print("\n[4/6] Computing analytics (window functions)...")

    print("\n  -- Daily revenue with running total --")
    daily_rev = daily_revenue_with_running_total(enriched)
    daily_rev.show(20, truncate=False)

    print("\n  -- Top 3 products per category --")
    top_prods = top_products_per_category(enriched, top_n=3)
    top_prods.show(20, truncate=False)

    print("\n  -- Day-over-day revenue change --")
    dod = day_over_day_change(enriched)
    dod.show(20, truncate=False)

    print("\n  -- Customer lifetime value ranking --")
    clv = customer_lifetime_value_ranking(enriched)
    clv.show(20, truncate=False)

    # ── Phase 5: SparkSQL ───────────────────────────────────────
    print("\n[5/6] Running SparkSQL queries...")
    register_views(spark, enriched)

    print("\n  -- Monthly revenue by category (SQL) --")
    monthly_rev = monthly_revenue_by_category(spark)
    monthly_rev.show(20, truncate=False)

    print("\n  -- Above-average products (SQL subquery) --")
    above_avg = above_average_products(spark)
    above_avg.show(20, truncate=False)

    print("\n  -- Customer segments (SQL CASE WHEN) --")
    segments = customer_segments(spark)
    segments.show(20, truncate=False)

    # ── Phase 6: Load ──────────────────────────────────────────
    print("\n[6/6] Writing Parquet output...")
    output_path = write_partitioned_parquet(enriched)
    print(f"  Written to: {output_path}")

    # TODO(human): Unpersist the cached DataFrame to free memory.
    # In a real pipeline with many stages, this prevents memory pressure.
    #   cleaned.unpersist()
    #   print("  Cleaned DataFrame unpersisted.")

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    run_pipeline()
