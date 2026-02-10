"""Phase 4 -- Aggregations: groupBy, agg, and filtering grouped results.

Key concepts:
  - groupBy() returns a GroupedData object -- you must call an agg method on it
  - agg() accepts multiple column expressions: F.sum(), F.avg(), F.count(), etc.
  - alias() renames the aggregated column for readability
  - Filtering after groupBy is the DataFrame equivalent of SQL's HAVING clause
  - .count() on a DataFrame is an action; F.count() inside agg() is a column expr
"""

from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from spark_helpers import SALES_CSV, create_spark_session


# ── Helpers ───────────────────────────────────────────────────────────

def load_sales_with_total(spark) -> DataFrame:
    df = spark.read.csv(str(SALES_CSV), header=True, inferSchema=True)
    return df.withColumn("total_price", F.col("quantity") * F.col("unit_price"))


# ── Aggregations ──────────────────────────────────────────────────────

def revenue_by_category(df: DataFrame) -> DataFrame:
    """Group by category: total revenue (sum of total_price), average unit_price,
    and number of orders (count).

    Hint:
        df.groupBy("category").agg(
            F.sum("total_price").alias("total_revenue"),
            F.avg("unit_price").alias("avg_price"),
            F.count("sale_id").alias("order_count"),
        )
    """
    # TODO(human): groupBy category and compute the three aggregations above.
    raise NotImplementedError


def revenue_by_category_region(df: DataFrame) -> DataFrame:
    """Group by (category, region) with the same three aggregations.

    Hint: Same pattern, but groupBy("category", "region").
    """
    # TODO(human): Two-level groupBy with the same agg expressions.
    raise NotImplementedError


def high_revenue_categories(df: DataFrame, min_revenue: float = 500.0) -> DataFrame:
    """From revenue_by_category result, keep only rows where total_revenue > min_revenue.

    This is the HAVING equivalent in DataFrame API.

    Hint: Call revenue_by_category(df) first, then .filter(...).
    """
    # TODO(human): Chain aggregation + filter.
    raise NotImplementedError


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    spark = create_spark_session("03-aggregations")
    sales = load_sales_with_total(spark)

    print("=== Revenue by Category ===")
    revenue_by_category(sales).show()

    print("=== Revenue by Category + Region ===")
    revenue_by_category_region(sales).orderBy("category", "region").show(20)

    print("=== High-Revenue Categories (>$500) ===")
    high_revenue_categories(sales).show()

    spark.stop()


if __name__ == "__main__":
    main()
