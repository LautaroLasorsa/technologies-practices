"""Phase 4 — Window Functions: compute analytical metrics over partitions.

Concepts:
- Window.partitionBy() groups rows into partitions (like GROUP BY but keeps all rows)
- Window.orderBy() defines ordering within each partition
- row_number() assigns 1,2,3... per partition (no ties)
- rank() assigns same rank to ties, leaves gaps (1,1,3)
- dense_rank() assigns same rank to ties, no gaps (1,1,2)
- lag(col, offset) accesses a previous row's value within the window
- sum(col).over(window) computes running/cumulative aggregations

Docs: https://spark.apache.org/docs/3.5.0/api/python/reference/pyspark.sql/api/pyspark.sql.functions.row_number.html
      https://sparkbyexamples.com/pyspark/pyspark-window-functions/
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def daily_revenue_with_running_total(enriched: DataFrame) -> DataFrame:
    """Compute daily revenue per category with a cumulative running total.

    Expected output columns:
      category, sale_date, daily_revenue, running_total

    TODO(human):

    Step 1 — Aggregate daily revenue per category:
      daily = (enriched
          .groupBy("category", "sale_date")
          .agg(F.sum("total_amount").alias("daily_revenue")))

    Step 2 — Define a window partitioned by category, ordered by sale_date,
    with rows from the start of the partition to the current row:
      window = (Window
          .partitionBy("category")
          .orderBy("sale_date")
          .rowsBetween(Window.unboundedPreceding, Window.currentRow))

    Step 3 — Add running_total column:
      return daily.withColumn("running_total", F.sum("daily_revenue").over(window))

    Hint: rowsBetween(unboundedPreceding, currentRow) means "from the
    first row in the partition up to this row" — that's a running sum.
    """
    raise NotImplementedError("Implement daily_revenue_with_running_total")


def top_products_per_category(enriched: DataFrame, top_n: int = 3) -> DataFrame:
    """Rank products by total revenue within each category, keep top N.

    Expected output columns:
      category, product_id, name, product_revenue, rank

    TODO(human):

    Step 1 — Aggregate total revenue per product:
      product_rev = (enriched
          .groupBy("category", "product_id", "name")
          .agg(F.sum("total_amount").alias("product_revenue")))

    Step 2 — Define window partitioned by category, ordered by revenue DESC:
      window = Window.partitionBy("category").orderBy(F.desc("product_revenue"))

    Step 3 — Add rank column using row_number():
      ranked = product_rev.withColumn("rank", F.row_number().over(window))

    Step 4 — Filter to keep only top N:
      return ranked.filter(F.col("rank") <= top_n)

    Hint: row_number() guarantees unique ranks (1,2,3) even with ties.
    If you wanted ties to share a rank, you'd use rank() or dense_rank().
    """
    raise NotImplementedError("Implement top_products_per_category")


def day_over_day_change(enriched: DataFrame) -> DataFrame:
    """Compute day-over-day revenue change per category using lag().

    Expected output columns:
      category, sale_date, daily_revenue, prev_day_revenue, dod_change

    TODO(human):

    Step 1 — Aggregate daily revenue (same as in daily_revenue_with_running_total):
      daily = (enriched
          .groupBy("category", "sale_date")
          .agg(F.sum("total_amount").alias("daily_revenue")))

    Step 2 — Define window partitioned by category, ordered by sale_date:
      window = Window.partitionBy("category").orderBy("sale_date")

    Step 3 — Use lag() to get previous day's revenue (default=0 for first row):
      .withColumn("prev_day_revenue", F.lag("daily_revenue", 1, 0).over(window))

    Step 4 — Compute the change:
      .withColumn("dod_change", F.col("daily_revenue") - F.col("prev_day_revenue"))

    Hint: lag(col, offset=1, default=0) looks 1 row back in the window.
    The first row in each partition has no previous row, so default=0.
    """
    raise NotImplementedError("Implement day_over_day_change")


def customer_lifetime_value_ranking(enriched: DataFrame) -> DataFrame:
    """Rank customers by total spend across all purchases.

    Expected output columns:
      customer_id, total_spend, clv_rank

    TODO(human):

    Step 1 — Aggregate total spend per customer:
      clv = (enriched
          .groupBy("customer_id")
          .agg(F.sum("total_amount").alias("total_spend")))

    Step 2 — Define window ordered by total_spend DESC (no partitioning — global rank):
      window = Window.orderBy(F.desc("total_spend"))

    Step 3 — Add rank using rank() (ties share the same rank):
      return clv.withColumn("clv_rank", F.rank().over(window))

    Hint: rank() vs dense_rank() — rank() leaves gaps after ties (1,1,3),
    dense_rank() doesn't (1,1,2). Here rank() is more natural for leaderboards.
    """
    raise NotImplementedError("Implement customer_lifetime_value_ranking")
