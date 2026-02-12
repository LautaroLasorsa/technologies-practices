"""Phase 5 — SparkSQL: register temp views and query with SQL syntax.

Concepts:
- df.createOrReplaceTempView("name") registers a DataFrame as a SQL table
- spark.sql("SELECT ...") returns a DataFrame — full SQL power
- SQL is often more readable for complex aggregations, subqueries, CASE WHEN
- DataFrame API is better for programmatic / dynamic transformations

Docs: https://spark.apache.org/docs/3.5.0/sql-programming-guide.html
"""

from pyspark.sql import DataFrame, SparkSession


def register_views(spark: SparkSession, enriched: DataFrame) -> None:
    """Register the enriched DataFrame as a temp view for SQL queries.

    This is boilerplate — already implemented for you.
    """
    enriched.createOrReplaceTempView("sales")

# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches SQL aggregation with time-based grouping in Spark.
# Understanding how to extract date parts and perform multi-level grouping is essential for
# building time-series reports and dashboards that business stakeholders can actually read.

def monthly_revenue_by_category(spark: SparkSession) -> DataFrame:
    """SQL: Monthly revenue summary grouped by category.

    Expected output columns:
      sale_year, sale_month, category, monthly_revenue, transaction_count

    TODO(human): Write a SQL query that:
    1. Extracts year and month from sale_date
    2. Groups by year, month, category
    3. Computes SUM(total_amount) as monthly_revenue
    4. Computes COUNT(*) as transaction_count
    5. Orders by year, month, category

    Hint:
      spark.sql('''
          SELECT
              YEAR(sale_date) AS sale_year,
              MONTH(sale_date) AS sale_month,
              category,
              SUM(total_amount) AS monthly_revenue,
              COUNT(*) AS transaction_count
          FROM sales
          GROUP BY YEAR(sale_date), MONTH(sale_date), category
          ORDER BY sale_year, sale_month, category
      ''')
    """
    raise NotImplementedError("Implement monthly_revenue_by_category")

# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches subqueries in SQL for computing dynamic thresholds.
# Understanding how to use nested queries with aggregate comparisons is crucial for building
# filters like "above average", "top 10%", or "outliers" — common in analytics and reporting.

def above_average_products(spark: SparkSession) -> DataFrame:
    """SQL: Products whose total revenue exceeds the overall average product revenue.

    Expected output columns:
      product_id, name, category, product_revenue

    TODO(human): Write a SQL query with a subquery:
    1. Inner query: compute average revenue per product across ALL products
    2. Outer query: find products whose total revenue > that average

    Hint:
      spark.sql('''
          SELECT product_id, name, category,
                 SUM(total_amount) AS product_revenue
          FROM sales
          GROUP BY product_id, name, category
          HAVING SUM(total_amount) > (
              SELECT AVG(product_total)
              FROM (
                  SELECT SUM(total_amount) AS product_total
                  FROM sales
                  GROUP BY product_id
              )
          )
          ORDER BY product_revenue DESC
      ''')
    """
    raise NotImplementedError("Implement above_average_products")

# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches customer segmentation using CASE WHEN conditional logic in SQL.
# Understanding how to create categorical labels based on business rules is fundamental for
# building segmentation models, cohort analysis, and personalized marketing campaigns.

def customer_segments(spark: SparkSession) -> DataFrame:
    """SQL: Segment customers by purchase frequency using CASE WHEN.

    Expected output columns:
      customer_id, purchase_count, total_spend, segment

    Segments:
      - purchase_count >= 5  -> 'VIP'
      - purchase_count >= 3  -> 'Regular'
      - else                 -> 'Occasional'

    TODO(human): Write a SQL query that:
    1. Groups by customer_id
    2. Counts transactions as purchase_count
    3. Sums total_amount as total_spend
    4. Uses CASE WHEN to assign segment
    5. Orders by total_spend DESC

    Hint:
      spark.sql('''
          SELECT
              customer_id,
              COUNT(*) AS purchase_count,
              SUM(total_amount) AS total_spend,
              CASE
                  WHEN COUNT(*) >= 5 THEN 'VIP'
                  WHEN COUNT(*) >= 3 THEN 'Regular'
                  ELSE 'Occasional'
              END AS segment
          FROM sales
          GROUP BY customer_id
          ORDER BY total_spend DESC
      ''')
    """
    raise NotImplementedError("Implement customer_segments")
