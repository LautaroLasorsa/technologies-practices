"""Phase 5 -- Joins: inner, left, and handling ambiguous columns.

Key concepts:
  - df1.join(df2, on="col")            -- equi-join on a shared column name
  - df1.join(df2, df1.col == df2.col)  -- explicit join condition (needed when
                                           column names differ or are ambiguous)
  - Join types: "inner", "left", "right", "cross", "semi", "anti"
  - After joining, duplicate column names cause AnalysisException on access;
    fix with alias() before joining or .drop() after
  - Spark auto-broadcasts DataFrames below spark.sql.autoBroadcastJoinThreshold
    (default 10 MB) -- you can see BroadcastHashJoin in explain()
"""

from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from spark_helpers import PRODUCTS_JSON, SALES_CSV, create_spark_session


# ── Helpers ───────────────────────────────────────────────────────────

def load_data(spark):
    sales = spark.read.csv(str(SALES_CSV), header=True, inferSchema=True)
    sales = sales.withColumn("total_price", F.col("quantity") * F.col("unit_price"))
    products = spark.read.json(str(PRODUCTS_JSON))
    return sales, products


# ── Joins ─────────────────────────────────────────────────────────────

def inner_join_sales_products(sales: DataFrame, products: DataFrame) -> DataFrame:
    """Inner join sales with products on product_id.

    Hint: sales.join(products, on="product_id", how="inner")
    When using `on="product_id"`, Spark automatically deduplicates the join column.
    """
    # TODO(human): Perform the inner join. Return the joined DataFrame.
    raise NotImplementedError


def left_join_find_missing(sales: DataFrame, products: DataFrame) -> DataFrame:
    """Left join to find sales whose product_id has no match in products.

    Hint:
      joined = sales.join(products, on="product_id", how="left")
      Then filter where a products-only column (e.g., "name") is null.
    """
    # TODO(human): Left join and filter for rows with no product info.
    raise NotImplementedError


def join_with_aliases(sales: DataFrame, products: DataFrame) -> DataFrame:
    """Join using aliased DataFrames and explicit condition to avoid ambiguity.

    Hint:
      s = sales.alias("s")
      p = products.alias("p")
      joined = s.join(p, F.col("s.product_id") == F.col("p.product_id"))
      Then select specific columns using "s.column" / "p.column" notation
      and drop one of the duplicate product_id columns.
    """
    # TODO(human): Alias both DataFrames, join with explicit condition,
    #   select useful columns, drop the duplicate product_id.
    raise NotImplementedError


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    spark = create_spark_session("04-joins")
    sales, products = load_data(spark)

    print("=== Inner Join: Sales + Products ===")
    joined = inner_join_sales_products(sales, products)
    joined.show(10)

    print("=== Left Join: Sales with Missing Product ===")
    missing = left_join_find_missing(sales, products)
    missing.show()
    print(f"Sales without product info: {missing.count()}")

    print("=== Join with Aliases ===")
    aliased = join_with_aliases(sales, products)
    aliased.show(10)

    # Check the join strategy in the physical plan
    print("=== Physical Plan (look for BroadcastHashJoin) ===")
    joined.explain()

    spark.stop()


if __name__ == "__main__":
    main()
