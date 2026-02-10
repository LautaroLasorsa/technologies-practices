"""Explicit schemas for all data sources.

Why explicit schemas?
- Spark doesn't waste a pass over the data to infer types
- You catch mismatches early (fail-fast) instead of silent wrong types
- Schema is documentation: readers know exactly what the data looks like
"""

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

SALES_SCHEMA = StructType([
    StructField("transaction_id", StringType(), nullable=False),
    StructField("customer_id", StringType(), nullable=True),
    StructField("product_id", StringType(), nullable=True),
    StructField("quantity", IntegerType(), nullable=True),
    StructField("unit_price", DoubleType(), nullable=True),
    StructField("sale_date", StringType(), nullable=False),
    StructField("store_id", StringType(), nullable=False),
])

PRODUCTS_SCHEMA = StructType([
    StructField("product_id", StringType(), nullable=False),
    StructField("name", StringType(), nullable=False),
    StructField("category", StringType(), nullable=False),
    StructField("margin", DoubleType(), nullable=False),
])
