"""Phase 7 -- User-Defined Functions (UDFs).

Key concepts:
  - udf() wraps a Python function so Spark can apply it row-by-row
  - You MUST specify the return type (StringType, IntegerType, etc.)
  - UDFs serialize each row from JVM -> Python -> JVM, which is slow
  - Prefer built-in F.when().otherwise() over UDFs when possible
  - spark.udf.register("name", fn, returnType) makes a UDF available in SQL
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import pyspark.sql.functions as F

from spark_helpers import SALES_CSV, create_spark_session


# ── Helpers ───────────────────────────────────────────────────────────

def load_sales_with_total(spark: SparkSession) -> DataFrame:
    df = spark.read.csv(str(SALES_CSV), header=True, inferSchema=True)
    return df.withColumn("total_price", F.col("quantity") * F.col("unit_price"))


# ── UDF Definition ────────────────────────────────────────────────────

def price_tier(total_price: float) -> str:
    """Classify a total_price into a tier.

    Rules:
      - total_price <= 50   -> "low"
      - 50 < total_price <= 300 -> "medium"
      - total_price > 300   -> "high"
    """
    # TODO(human): Implement the classification logic.
    #   This is a plain Python function -- no Spark API needed here.
    raise NotImplementedError


# ── UDF Registration & Usage ──────────────────────────────────────────

def apply_udf_with_column(df: DataFrame) -> DataFrame:
    """Create a Spark UDF from price_tier and add a 'tier' column.

    Hint:
      price_tier_udf = udf(price_tier, StringType())
      df.withColumn("tier", price_tier_udf(col("total_price")))
    """
    # TODO(human): Wrap price_tier as a UDF, apply it with withColumn.
    raise NotImplementedError


def register_udf_for_sql(spark: SparkSession) -> None:
    """Register price_tier as a SQL-callable UDF named 'price_tier'.

    Hint: spark.udf.register("price_tier", price_tier, StringType())
    """
    # TODO(human): Register the UDF so it can be used in spark.sql() queries.
    raise NotImplementedError


def query_with_sql_udf(spark: SparkSession) -> DataFrame:
    """Use the registered UDF in a SQL query.

    Hint:
      spark.sql('''
          SELECT *, price_tier(quantity * unit_price) AS tier
          FROM sales
      ''')
    """
    # TODO(human): Write a SQL query that applies the price_tier UDF.
    raise NotImplementedError


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    spark = create_spark_session("06-udfs")
    sales = load_sales_with_total(spark)

    # DataFrame API usage
    print("=== UDF via withColumn ===")
    tiered = apply_udf_with_column(sales)
    tiered.select("sale_id", "category", "total_price", "tier").show(10)

    # SQL usage
    sales.createOrReplaceTempView("sales")
    register_udf_for_sql(spark)

    print("=== UDF via SQL ===")
    sql_result = query_with_sql_udf(spark)
    sql_result.select("sale_id", "category", "tier").show(10)

    spark.stop()


if __name__ == "__main__":
    main()
