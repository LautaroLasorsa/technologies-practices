"""Phase 2 — Cleaning: handle nulls, duplicates, invalid values, type casts.

Concepts:
- dropDuplicates([cols]) removes exact duplicate rows by subset of columns
- dropna(subset=[cols]) removes rows where specified columns are null
- filter() / where() keeps only rows matching a condition
- withColumn() + to_date() casts string dates to proper DateType
- Cleaning BEFORE joins avoids propagating garbage into downstream analytics

Docs: https://spark.apache.org/docs/3.5.0/api/python/reference/pyspark.sql/dataframe.html
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches essential data cleaning operations in Spark DataFrames.
# Learning to chain deduplication, null handling, filtering, and type casting is fundamental
# for building production ETL pipelines where dirty data is the norm, not the exception.

def clean_sales(raw_sales: DataFrame) -> DataFrame:
    """Clean the raw sales DataFrame.

    TODO(human): Apply these cleaning steps IN ORDER:

    1. Deduplicate by transaction_id:
       raw_sales.dropDuplicates(["transaction_id"])

    2. Drop rows where critical columns are null:
       .dropna(subset=["customer_id", "product_id"])

    3. Filter out invalid quantities and prices:
       .filter((F.col("quantity") > 0) & (F.col("unit_price") > 0))

    4. Cast sale_date string to DateType:
       .withColumn("sale_date", F.to_date(F.col("sale_date"), "yyyy-MM-dd"))

    Chain all four steps and return the result.

    Hint: Each step returns a new DataFrame (immutable), so chain them:
      return (raw_sales
          .dropDuplicates(...)
          .dropna(...)
          .filter(...)
          .withColumn(...))
    """
    raise NotImplementedError("Implement clean_sales — see TODO above")
