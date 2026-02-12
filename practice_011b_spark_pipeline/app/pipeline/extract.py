"""Phase 2 — Ingestion: read raw data from CSV and JSON sources.

Concepts:
- spark.read.csv() with header=True and explicit schema avoids inference overhead
- spark.read.json() reads newline-delimited JSON or JSON arrays
- Explicit schemas catch data quality issues at read time (fail-fast)

Docs: https://spark.apache.org/docs/3.5.0/sql-data-sources-csv.html
      https://spark.apache.org/docs/3.5.0/sql-data-sources-json.html
"""

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from pipeline.schemas import PRODUCTS_SCHEMA, SALES_SCHEMA

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches multi-source data ingestion with explicit schemas in Spark.
# Defining schemas upfront (instead of inferring) catches data quality issues early and
# avoids the performance overhead of schema inference, which is critical for large datasets.

def read_sales(spark: SparkSession) -> DataFrame:
    """Read sales transactions from CSV with explicit schema.

    TODO(human): Use spark.read.csv() with:
      - path: str(DATA_DIR / "sales.csv")
      - header=True  (first row is column names)
      - schema=SALES_SCHEMA  (enforce types, don't infer)

    Hint: spark.read.csv(path, header=True, schema=SALES_SCHEMA)

    Returns the raw sales DataFrame (not yet cleaned).
    """
    raise NotImplementedError("Implement read_sales — see TODO above")

# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches reading JSON data with explicit schemas in Spark.
# Understanding JSON ingestion (including multiLine mode) is essential for modern data pipelines
# where JSON is a common format for APIs, logs, and semi-structured data.

def read_products(spark: SparkSession) -> DataFrame:
    """Read product catalog from JSON with explicit schema.

    TODO(human): Use spark.read.json() with:
      - path: str(DATA_DIR / "products.json")
      - schema=PRODUCTS_SCHEMA
      - multiLine=True  (the file is a JSON array, not one-object-per-line)

    Hint: spark.read.json(path, schema=PRODUCTS_SCHEMA, multiLine=True)

    Returns the products DataFrame.
    """
    raise NotImplementedError("Implement read_products — see TODO above")
