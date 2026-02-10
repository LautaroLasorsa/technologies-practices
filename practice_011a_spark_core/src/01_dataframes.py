"""Phase 2 -- DataFrame creation, schema inspection, and basic exploration.

Key concepts:
  - spark.read.csv / spark.read.json  -- data readers
  - inferSchema vs explicit StructType -- schema control
  - printSchema(), show(), count(), describe() -- inspection actions
  - Lazy evaluation: readers build a plan, actions trigger execution
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)

from spark_helpers import PRODUCTS_JSON, SALES_CSV, create_spark_session


# ── Boilerplate ───────────────────────────────────────────────────────

def load_sales(spark: SparkSession) -> "DataFrame":
    """Load sales.csv with header and schema inference.

    Hint: spark.read.csv() accepts `header=True` and `inferSchema=True`.
    """
    # TODO(human): Load SALES_CSV into a DataFrame.
    #   - Enable header (first row is column names)
    #   - Enable inferSchema so Spark guesses column types
    #   Return the DataFrame.
    raise NotImplementedError


def load_products(spark: SparkSession) -> "DataFrame":
    """Load products.json with an explicit schema.

    Hint: Define a StructType with StructField entries, then pass it via
    spark.read.schema(schema).json(path).

    The schema should match: product_id (str), name (str), brand (str),
    weight_kg (double).
    """
    # TODO(human): Define the StructType schema for products.json, then load it.
    #   Use the imported types: StringType, DoubleType, StructField, StructType.
    #   Return the DataFrame.
    raise NotImplementedError


def inspect(df: "DataFrame", label: str) -> None:
    """Print schema, first 5 rows, row count, and summary statistics."""
    # TODO(human): Call printSchema(), show(5), count(), and describe().show()
    #   Use `label` in print statements so the output is easy to identify.
    raise NotImplementedError


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    spark = create_spark_session("01-dataframes")

    sales_df = load_sales(spark)
    products_df = load_products(spark)

    inspect(sales_df, "Sales")
    inspect(products_df, "Products")

    spark.stop()


if __name__ == "__main__":
    main()
