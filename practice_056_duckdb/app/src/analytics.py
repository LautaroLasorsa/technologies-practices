"""
Exercise: Analytical SQL — Window Functions, CTEs, and QUALIFY.

This module teaches advanced analytical SQL features that are the core
reason DuckDB exists. Window functions let you compute values across
"windows" of related rows without collapsing the result set (unlike
GROUP BY). CTEs (Common Table Expressions) structure complex queries
into readable named steps. QUALIFY filters window results directly —
a feature unique to analytical databases like DuckDB, Snowflake, and
BigQuery.

These skills transfer directly to any analytical SQL engine (BigQuery,
Snowflake, Redshift, ClickHouse).

Run: uv run python src/analytics.py
"""

from pathlib import Path

import duckdb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_data(con: duckdb.DuckDBPyConnection) -> None:
    """Load CSV data into tables for analytics exercises."""
    con.sql(f"CREATE TABLE products AS SELECT * FROM read_csv('{DATA_DIR}/products.csv')")
    con.sql(f"CREATE TABLE stores AS SELECT * FROM read_csv('{DATA_DIR}/stores.csv')")
    con.sql(f"CREATE TABLE sales AS SELECT * FROM read_csv('{DATA_DIR}/sales.csv')")
    print(f"Loaded: {con.sql('SELECT COUNT(*) FROM sales').fetchone()[0]} sales")


# ---------------------------------------------------------------------------
# Exercise 1: Window Functions (ROW_NUMBER, RANK, LAG/LEAD)
# ---------------------------------------------------------------------------


def window_functions(con: duckdb.DuckDBPyConnection) -> None:
    """Implement window functions for ranking and row-level comparisons.

    TODO(human): Implement this function.

    Window functions operate on a "window" of rows related to the current
    row. Unlike GROUP BY (which collapses rows into groups), window functions
    PRESERVE all rows while adding computed columns. The syntax is:

        function_name() OVER (
            PARTITION BY <cols>   -- defines the window (like GROUP BY, but keeps rows)
            ORDER BY <cols>      -- defines ordering within each window
        )

    Query 1 — ROW_NUMBER, RANK, DENSE_RANK:
    For each product category, rank products by their total revenue:

        SELECT
            p.category,
            p.product_name,
            SUM(s.total_amount) AS total_revenue,
            ROW_NUMBER() OVER (PARTITION BY p.category ORDER BY SUM(s.total_amount) DESC) AS row_num,
            RANK()       OVER (PARTITION BY p.category ORDER BY SUM(s.total_amount) DESC) AS rank,
            DENSE_RANK() OVER (PARTITION BY p.category ORDER BY SUM(s.total_amount) DESC) AS dense_rank
        FROM sales s
        JOIN products p ON s.product_id = p.product_id
        GROUP BY p.category, p.product_name

    Show all rows. Print the result.

    The difference between the three:
    - ROW_NUMBER: Always unique (1, 2, 3, 4...) — ties broken arbitrarily
    - RANK: Ties get same rank, gaps after ties (1, 2, 2, 4...)
    - DENSE_RANK: Ties get same rank, no gaps (1, 2, 2, 3...)

    Query 2 — LAG and LEAD for month-over-month comparison:
    Calculate monthly revenue and compare each month to the previous month:

        WITH monthly_revenue AS (
            SELECT
                DATE_TRUNC('month', sale_date) AS month,
                SUM(total_amount) AS revenue
            FROM sales
            GROUP BY DATE_TRUNC('month', sale_date)
        )
        SELECT
            month,
            revenue,
            LAG(revenue) OVER (ORDER BY month)  AS prev_month_revenue,
            LEAD(revenue) OVER (ORDER BY month) AS next_month_revenue,
            ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
                  / LAG(revenue) OVER (ORDER BY month), 1) AS pct_change
        FROM monthly_revenue
        ORDER BY month

    LAG looks at the previous row; LEAD looks at the next row. These are
    essential for time-series analysis — month-over-month growth, year-over-year
    comparisons, detecting anomalies. The alternative (self-join on month - 1)
    is much more verbose and harder to read.

    Print both results using .show() or .fetchdf().

    Why this matters:
    - Window functions are the single most powerful analytical SQL feature
    - They appear in ~80% of analytical queries in production data pipelines
    - Understanding PARTITION BY vs GROUP BY is a common interview question
    - LAG/LEAD replaces complex self-joins for time-series analysis
    """
    raise NotImplementedError(
        "TODO(human): Implement ROW_NUMBER/RANK/DENSE_RANK ranking and LAG/LEAD month-over-month"
    )


# ---------------------------------------------------------------------------
# Exercise 2: Running Aggregates and Window Frames
# ---------------------------------------------------------------------------


def running_aggregates(con: duckdb.DuckDBPyConnection) -> None:
    """Implement running totals, moving averages, and cumulative percentages.

    TODO(human): Implement this function.

    Window frames specify WHICH rows within the partition to include in
    the calculation. The syntax is:

        function() OVER (
            PARTITION BY ...
            ORDER BY ...
            ROWS BETWEEN <start> AND <end>
        )

    Frame boundaries can be:
    - UNBOUNDED PRECEDING: from the first row of the partition
    - N PRECEDING: N rows before the current row
    - CURRENT ROW: the current row
    - N FOLLOWING: N rows after the current row
    - UNBOUNDED FOLLOWING: to the last row of the partition

    Query 1 — Running total of monthly revenue:
    Calculate a cumulative sum of revenue across months:

        WITH monthly_revenue AS (
            SELECT
                DATE_TRUNC('month', sale_date) AS month,
                SUM(total_amount) AS revenue
            FROM sales
            GROUP BY DATE_TRUNC('month', sale_date)
        )
        SELECT
            month,
            revenue,
            SUM(revenue) OVER (ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                AS running_total
        FROM monthly_revenue
        ORDER BY month

    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW means: "sum all rows
    from the start of the partition up to and including the current row."
    This is the standard running total pattern.

    Query 2 — 3-month moving average:
    Calculate a smoothed revenue trend using a 3-month sliding window:

        SELECT
            month,
            revenue,
            AVG(revenue) OVER (ORDER BY month ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING)
                AS moving_avg_3m
        FROM monthly_revenue
        ORDER BY month

    ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING creates a centered 3-month
    window. The first and last months only average 2 values (the frame is
    automatically clipped at partition boundaries).

    Query 3 — Cumulative percentage per category:
    For each category, show what percentage of the category's total revenue
    each product represents, ordered by revenue:

        SELECT
            p.category,
            p.product_name,
            SUM(s.total_amount) AS product_revenue,
            ROUND(100.0 * SUM(s.total_amount) /
                  SUM(SUM(s.total_amount)) OVER (PARTITION BY p.category), 2)
                AS pct_of_category,
            ROUND(100.0 * SUM(SUM(s.total_amount)) OVER (
                      PARTITION BY p.category
                      ORDER BY SUM(s.total_amount) DESC
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                  ) / SUM(SUM(s.total_amount)) OVER (PARTITION BY p.category), 2)
                AS cumulative_pct
        FROM sales s
        JOIN products p ON s.product_id = p.product_id
        GROUP BY p.category, p.product_name
        ORDER BY p.category, product_revenue DESC

    Note the nested aggregation: SUM(SUM(...)) OVER (...). The inner SUM
    is the GROUP BY aggregation; the outer SUM is the window function.
    This is valid SQL and very common in analytical queries.

    Print all three results.

    Why this matters:
    - Running totals and moving averages are fundamental in financial/business analytics
    - Window frames (ROWS BETWEEN) are the most powerful yet least understood SQL feature
    - Cumulative percentages enable Pareto analysis ("top 20% of products = 80% of revenue")
    - These patterns are used daily in data engineering and BI dashboards
    """
    raise NotImplementedError(
        "TODO(human): Implement running totals, moving averages, and cumulative percentages"
    )


# ---------------------------------------------------------------------------
# Exercise 3: CTEs and QUALIFY
# ---------------------------------------------------------------------------


def ctes_and_qualify(con: duckdb.DuckDBPyConnection) -> None:
    """Use CTEs for query structuring and QUALIFY for window filtering.

    TODO(human): Implement this function.

    CTEs (Common Table Expressions) are named temporary result sets defined
    with the WITH clause. They make complex queries readable by breaking
    them into logical steps — like naming intermediate variables in code.

    QUALIFY is a clause that filters the results of window functions,
    similar to how HAVING filters aggregate results. Without QUALIFY,
    you'd need to wrap the window query in a CTE or subquery just to
    add a WHERE filter on the window result. QUALIFY eliminates that
    boilerplate.

    SQL evaluation order:
        FROM → WHERE → GROUP BY → HAVING → WINDOW → QUALIFY → SELECT → ORDER BY

    Query 1 — Multi-step CTE:
    Find the top 3 products per category by total revenue, including
    the category's total for percentage calculation:

        WITH product_revenue AS (
            SELECT
                p.category,
                p.product_name,
                SUM(s.total_amount) AS revenue
            FROM sales s
            JOIN products p ON s.product_id = p.product_id
            GROUP BY p.category, p.product_name
        ),
        category_totals AS (
            SELECT
                category,
                SUM(revenue) AS category_revenue
            FROM product_revenue
            GROUP BY category
        )
        SELECT
            pr.category,
            pr.product_name,
            pr.revenue,
            ct.category_revenue,
            ROUND(100.0 * pr.revenue / ct.category_revenue, 1) AS pct_of_category
        FROM product_revenue pr
        JOIN category_totals ct ON pr.category = ct.category
        ORDER BY pr.category, pr.revenue DESC

    This CTE chain builds up from raw aggregation → category totals →
    final joined result. Each step is independently understandable.

    Query 2 — QUALIFY to replace CTE wrapping:
    Same goal (top 3 products per category), but using QUALIFY:

        SELECT
            p.category,
            p.product_name,
            SUM(s.total_amount) AS revenue,
            RANK() OVER (PARTITION BY p.category ORDER BY SUM(s.total_amount) DESC) AS rnk
        FROM sales s
        JOIN products p ON s.product_id = p.product_id
        GROUP BY p.category, p.product_name
        QUALIFY RANK() OVER (PARTITION BY p.category ORDER BY SUM(s.total_amount) DESC) <= 3
        ORDER BY p.category, revenue DESC

    Compare the QUALIFY version with the CTE approach:
    - CTE: 20+ lines, two named steps, a join
    - QUALIFY: ~10 lines, one query, direct filtering

    QUALIFY is evaluated AFTER window functions but BEFORE SELECT, so
    you can reference window expressions directly. This is like HAVING
    for windows.

    Query 3 — Deduplicate with QUALIFY:
    A very common real-world use case: keep only the latest sale per
    product per store (deduplication by recency):

        SELECT
            s.product_id,
            s.store_id,
            s.sale_date,
            s.total_amount
        FROM sales s
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY s.product_id, s.store_id
            ORDER BY s.sale_date DESC
        ) = 1
        ORDER BY s.product_id, s.store_id

    This pattern replaces the common "subquery with MAX(date)" anti-pattern.
    ROW_NUMBER with QUALIFY is the canonical way to deduplicate in analytical
    SQL.

    Print all three results.

    Why this matters:
    - CTEs are essential for readable, maintainable analytical SQL
    - QUALIFY is a DuckDB/Snowflake/BigQuery feature that saves significant boilerplate
    - The deduplication pattern (ROW_NUMBER + QUALIFY = 1) appears in nearly every data pipeline
    - Understanding SQL evaluation order prevents subtle bugs in complex queries
    """
    raise NotImplementedError(
        "TODO(human): Implement multi-step CTEs, QUALIFY filtering, and deduplication"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all analytics exercises."""
    con = duckdb.connect(":memory:")
    load_data(con)

    print()
    print("=" * 70)
    print("EXERCISE 1: Window Functions (ROW_NUMBER, RANK, LAG/LEAD)")
    print("=" * 70)
    window_functions(con)

    print()
    print("=" * 70)
    print("EXERCISE 2: Running Aggregates and Window Frames")
    print("=" * 70)
    running_aggregates(con)

    print()
    print("=" * 70)
    print("EXERCISE 3: CTEs and QUALIFY")
    print("=" * 70)
    ctes_and_qualify(con)

    con.close()
    print("\nAll analytics exercises complete.")


if __name__ == "__main__":
    main()
