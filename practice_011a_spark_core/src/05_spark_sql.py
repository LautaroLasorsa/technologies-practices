"""Phase 6 -- Spark SQL: register views and query with SQL strings.

Key concepts:
  - createOrReplaceTempView("name") registers a DataFrame as a SQL table
  - spark.sql("SELECT ...") returns a DataFrame -- same API, different syntax
  - The Catalyst optimizer produces equivalent plans for DataFrame and SQL code
  - You can freely mix: build part of the query with DataFrame API, register
    the intermediate result as a view, then continue with SQL (or vice versa)
"""

from pyspark.sql import DataFrame

from spark_helpers import PRODUCTS_JSON, SALES_CSV, create_spark_session


# ── Helpers ───────────────────────────────────────────────────────────

def load_and_register(spark) -> None:
    """Load CSVs and register them as temp views for SQL access."""
    sales = spark.read.csv(str(SALES_CSV), header=True, inferSchema=True)
    products = spark.read.json(str(PRODUCTS_JSON))

    # TODO(human): Register both DataFrames as temporary SQL views.
    #   Hint: sales.createOrReplaceTempView("sales")
    #         products.createOrReplaceTempView("products")
    raise NotImplementedError


def top_sales_by_category_sql(spark) -> DataFrame:
    """Write a SQL query that:
      1. Joins sales with products on product_id
      2. Computes total_price = quantity * unit_price
      3. Groups by category
      4. Aggregates: SUM(total_price) as total_revenue, COUNT(*) as order_count
      5. Filters groups where total_revenue > 200
      6. Orders by total_revenue DESC

    Hint: Use a single spark.sql(\"\"\"...\"\"\") call with standard SQL syntax.
    """
    # TODO(human): Write and execute the SQL query described above.
    #   Return the resulting DataFrame.
    raise NotImplementedError


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    spark = create_spark_session("05-spark-sql")

    load_and_register(spark)

    print("=== SQL: Top Sales by Category ===")
    result = top_sales_by_category_sql(spark)
    result.show()

    # Compare plans: SQL vs DataFrame
    print("=== SQL Physical Plan ===")
    result.explain()

    spark.stop()


if __name__ == "__main__":
    main()
