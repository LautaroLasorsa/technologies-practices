"""Phase 3 -- Core transformations: select, withColumn, filter, orderBy.

Key concepts:
  - Transformations are lazy -- they return a new DataFrame without executing
  - Actions (show, collect, count) trigger the execution of the plan
  - explain() reveals the physical plan the Catalyst optimizer chose
  - Narrow transforms (map, filter) don't shuffle; wide ones (orderBy) do
"""

from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from spark_helpers import SALES_CSV, create_spark_session


# ── Helpers ───────────────────────────────────────────────────────────

def load_sales(spark) -> DataFrame:
    return spark.read.csv(str(SALES_CSV), header=True, inferSchema=True)


# ── Transformations ───────────────────────────────────────────────────

def add_total_price(df: DataFrame) -> DataFrame:
    """Add a 'total_price' column = quantity * unit_price.

    Hint: df.withColumn("total_price", F.col("quantity") * F.col("unit_price"))
    """
    # TODO(human): Use withColumn to create the total_price column.
    raise NotImplementedError


def expensive_sales(df: DataFrame, threshold: float = 100.0) -> DataFrame:
    """Keep only rows where total_price > threshold.

    Hint: df.filter(F.col("total_price") > threshold)
          Alternatively: df.where(...)
    """
    # TODO(human): Filter the DataFrame.
    raise NotImplementedError


def top_sales(df: DataFrame, n: int = 10) -> DataFrame:
    """Return the top-n rows ordered by total_price descending.

    Hint: df.orderBy(F.col("total_price").desc()).limit(n)
    """
    # TODO(human): Order descending by total_price and limit to n rows.
    raise NotImplementedError


def unique_categories(df: DataFrame) -> DataFrame:
    """Return distinct category values.

    Hint: df.select("category").distinct()
    """
    # TODO(human): Select the category column and deduplicate.
    raise NotImplementedError


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    spark = create_spark_session("02-transformations")
    sales = load_sales(spark)

    with_total = add_total_price(sales)
    print("=== With total_price column ===")
    with_total.show(5)

    expensive = expensive_sales(with_total)
    print("=== Sales over $100 ===")
    expensive.show()

    top = top_sales(with_total)
    print("=== Top 10 sales by total_price ===")
    top.show()

    categories = unique_categories(sales)
    print("=== Distinct categories ===")
    categories.show()

    # Inspect the physical plan for the expensive_sales query
    print("=== Physical plan for expensive + orderBy ===")
    top_sales(expensive_sales(with_total)).explain()

    spark.stop()


if __name__ == "__main__":
    main()
