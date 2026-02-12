"""Phase 3 — Enrichment: join sales with product catalog, compute derived columns.

Concepts:
- Broadcast join: when one side is small (< ~10MB), Spark sends it to every
  executor, avoiding a shuffle on the large side. Use F.broadcast(small_df).
- Inner join drops sales referencing unknown products (data quality gate).
- Derived columns (total_amount, profit) are computed via withColumn().

Docs: https://spark.apache.org/docs/3.5.0/api/python/reference/pyspark.sql/api/pyspark.sql.functions.broadcast.html
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches broadcast join optimization, a critical production pattern for Spark.
# Understanding when and how to use broadcast joins can reduce job runtime from hours to minutes
# by eliminating expensive shuffles when joining large fact tables with small dimension tables.

def enrich_sales(
    clean_sales: DataFrame,
    products: DataFrame,
) -> DataFrame:
    """Join sales with products (broadcast) and add computed columns.

    TODO(human): Three steps:

    1. Broadcast join — join clean_sales with products on "product_id".
       Use F.broadcast(products) to hint Spark that products is small:
         joined = clean_sales.join(F.broadcast(products), on="product_id", how="inner")

    2. Compute total_amount = quantity * unit_price:
         .withColumn("total_amount", F.col("quantity") * F.col("unit_price"))

    3. Compute profit = total_amount * margin:
         .withColumn("profit", F.col("total_amount") * F.col("margin"))

    Return the enriched DataFrame.

    Hint: Think about why "inner" join is the right choice here.
    If a sale references a product_id not in the catalog, that's bad data.
    """
    raise NotImplementedError("Implement enrich_sales — see TODO above")
