"""Phase 6 — Load: write results to Parquet with partitioning.

Concepts:
- Parquet is a columnar format: great for analytics (reads only needed columns)
- Partitioning by date (year/month) creates a folder structure that enables
  "partition pruning" — Spark skips entire folders that don't match a WHERE clause
- Compression (snappy by default) reduces storage without heavy CPU cost
- mode("overwrite") replaces existing data; "append" adds to it

Docs: https://spark.apache.org/docs/3.5.0/sql-data-sources-parquet.html
"""

from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches writing optimized Parquet output with partitioning.
# Understanding partition pruning and columnar storage is critical for building data lakes
# where query performance depends on minimizing data scanned—partitioning can reduce reads by 100x.

def write_partitioned_parquet(enriched: DataFrame) -> Path:
    """Write enriched sales to Parquet, partitioned by year and month.

    TODO(human):

    Step 1 — Add partition columns derived from sale_date:
      partitioned = (enriched
          .withColumn("sale_year", F.year("sale_date"))
          .withColumn("sale_month", F.month("sale_date")))

    Step 2 — Write to Parquet with partitioning:
      output_path = str(OUTPUT_DIR / "enriched_sales")
      (partitioned
          .write
          .mode("overwrite")
          .partitionBy("sale_year", "sale_month")
          .parquet(output_path))

    Step 3 — Return output_path for logging.

    Hint: After writing, check the output directory. You'll see a folder
    structure like: enriched_sales/sale_year=2024/sale_month=1/part-00000.parquet
    That's partition pruning in action — a query with WHERE sale_month=1
    only reads files in that subfolder.
    """
    raise NotImplementedError("Implement write_partitioned_parquet")
