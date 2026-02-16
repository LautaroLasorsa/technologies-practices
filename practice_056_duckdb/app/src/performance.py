"""
Exercise: Performance Comparison — DuckDB vs Pandas.

This module benchmarks DuckDB against Pandas for typical analytical
operations: aggregation, multi-table joins, and window functions.

The goal is NOT to declare a "winner" — both tools have their strengths.
The goal is to understand WHY DuckDB is faster for certain operations
(columnar storage, vectorized execution, query optimization) and when
Pandas is still the right choice (small data, complex reshaping, ML
feature engineering).

Benchmark methodology:
- Each operation is timed using time.perf_counter() for wall-clock accuracy
- Each benchmark runs 3 times and reports the median (avoids cold-start bias)
- Both tools operate on the same data (loaded from the same Parquet files)
- DuckDB uses in-memory mode for fair comparison (no disk I/O advantage)

Run: uv run python src/performance.py
"""

import statistics
import time
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
NUM_RUNS = 3


# ---------------------------------------------------------------------------
# Benchmark Infrastructure
# ---------------------------------------------------------------------------


def benchmark(name: str, fn: Callable[[], object], runs: int = NUM_RUNS) -> float:
    """Run a function multiple times and return the median duration in seconds."""
    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    median_time = statistics.median(times)
    print(f"  {name}: {median_time:.4f}s (median of {runs} runs)")
    return median_time


def print_comparison(duck_time: float, pandas_time: float) -> None:
    """Print speedup ratio between DuckDB and Pandas."""
    if duck_time > 0 and pandas_time > 0:
        if duck_time < pandas_time:
            ratio = pandas_time / duck_time
            print(f"  --> DuckDB is {ratio:.1f}x faster than Pandas")
        else:
            ratio = duck_time / pandas_time
            print(f"  --> Pandas is {ratio:.1f}x faster than DuckDB")
    print()


# ---------------------------------------------------------------------------
# Exercise 1: Aggregation Benchmark
# ---------------------------------------------------------------------------


def aggregation_benchmark(
    con: duckdb.DuckDBPyConnection,
    sales_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> None:
    """Compare DuckDB vs Pandas for GROUP BY aggregation.

    TODO(human): Implement this function.

    Operation: Calculate total revenue, average order value, and sale count
    per product category. This requires joining sales with products (to get
    category) and then aggregating.

    DuckDB approach:
    Write a SQL query that joins sales with products and groups by category:

        def duckdb_agg():
            return con.sql(\"\"\"
                SELECT
                    p.category,
                    SUM(s.total_amount) AS total_revenue,
                    AVG(s.total_amount) AS avg_order,
                    COUNT(*) AS num_sales
                FROM sales_df s
                JOIN products_df p ON s.product_id = p.product_id
                GROUP BY p.category
                ORDER BY total_revenue DESC
            \"\"\").fetchdf()

    Note: we query the Pandas DataFrames directly (zero-copy) and call
    .fetchdf() to materialize the result.

    Pandas approach:
    Implement the same operation using Pandas merge + groupby:

        def pandas_agg():
            merged = sales_df.merge(products_df[['product_id', 'category']], on='product_id')
            result = (
                merged
                .groupby('category')
                .agg(
                    total_revenue=('total_amount', 'sum'),
                    avg_order=('total_amount', 'mean'),
                    num_sales=('total_amount', 'count')
                )
                .sort_values('total_revenue', ascending=False)
                .reset_index()
            )
            return result

    Run both using the benchmark() function and print the comparison:
        duck_time = benchmark("DuckDB aggregation", duckdb_agg)
        pandas_time = benchmark("Pandas aggregation", pandas_agg)
        print_comparison(duck_time, pandas_time)

    Why DuckDB wins for aggregations:
    - Columnar storage: only reads the needed columns (product_id, total_amount, category)
    - Vectorized hash aggregation: processes 2048 values per batch
    - Query optimizer: chooses optimal join order and aggregation strategy
    - Pandas: creates intermediate DataFrames at each step (merge → groupby → sort),
      each requiring full materialization

    Why this matters:
    - GROUP BY aggregation is the most common analytical operation
    - Understanding why columnar + vectorized beats row-at-a-time helps you
      choose the right tool for the job
    """
    raise NotImplementedError(
        "TODO(human): Implement and benchmark DuckDB vs Pandas aggregation"
    )


# ---------------------------------------------------------------------------
# Exercise 2: Join Benchmark
# ---------------------------------------------------------------------------


def join_benchmark(
    con: duckdb.DuckDBPyConnection,
    sales_df: pd.DataFrame,
    products_df: pd.DataFrame,
    stores_df: pd.DataFrame,
) -> None:
    """Compare DuckDB vs Pandas for multi-table JOIN with aggregation.

    TODO(human): Implement this function.

    Operation: Join sales + products + stores, then compute revenue per
    region per category. This is a 3-table join followed by a 2-column
    GROUP BY — a typical star-schema analytical query.

    DuckDB approach:
    Write a SQL query that joins all three tables:

        def duckdb_join():
            return con.sql(\"\"\"
                SELECT
                    st.region,
                    p.category,
                    SUM(s.total_amount) AS revenue,
                    COUNT(*) AS num_sales,
                    AVG(s.quantity) AS avg_qty
                FROM sales_df s
                JOIN products_df p ON s.product_id = p.product_id
                JOIN stores_df st ON s.store_id = st.store_id
                GROUP BY st.region, p.category
                ORDER BY st.region, revenue DESC
            \"\"\").fetchdf()

    Pandas approach:
    Chain two merges and then groupby:

        def pandas_join():
            merged = (
                sales_df
                .merge(products_df[['product_id', 'category']], on='product_id')
                .merge(stores_df[['store_id', 'region']], on='store_id')
            )
            result = (
                merged
                .groupby(['region', 'category'])
                .agg(
                    revenue=('total_amount', 'sum'),
                    num_sales=('total_amount', 'count'),
                    avg_qty=('quantity', 'mean')
                )
                .sort_values(['region', 'revenue'], ascending=[True, False])
                .reset_index()
            )
            return result

    Run both with benchmark() and compare.

    Why DuckDB wins for joins:
    - Hash join with vectorized probing (processes batches, not single rows)
    - Query optimizer determines the optimal join order automatically
    - No intermediate DataFrame materialization between join steps
    - Pandas: each .merge() creates a full intermediate DataFrame in memory

    Why this matters:
    - Multi-table joins are fundamental in star/snowflake schema analytics
    - The performance gap grows with data size (DuckDB scales better)
    - Understanding join strategies (hash join vs sort-merge vs nested loop)
      is important for database optimization
    """
    raise NotImplementedError(
        "TODO(human): Implement and benchmark DuckDB vs Pandas 3-table join"
    )


# ---------------------------------------------------------------------------
# Exercise 3: Window Function Benchmark
# ---------------------------------------------------------------------------


def window_benchmark(
    con: duckdb.DuckDBPyConnection,
    sales_df: pd.DataFrame,
) -> None:
    """Compare DuckDB vs Pandas for window function operations.

    TODO(human): Implement this function.

    Operation: For each product, compute a running total of sales amount
    ordered by date, and assign a rank within each month by total_amount.

    DuckDB approach:
    Use SQL window functions:

        def duckdb_window():
            return con.sql(\"\"\"
                SELECT
                    sale_date,
                    product_id,
                    total_amount,
                    SUM(total_amount) OVER (
                        PARTITION BY product_id
                        ORDER BY sale_date
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS running_total,
                    RANK() OVER (
                        PARTITION BY DATE_TRUNC('month', sale_date)
                        ORDER BY total_amount DESC
                    ) AS monthly_rank
                FROM sales_df
            \"\"\").fetchdf()

    Pandas approach:
    Use groupby + cumsum and groupby + rank:

        def pandas_window():
            df = sales_df.copy()
            df = df.sort_values(['product_id', 'sale_date'])
            df['running_total'] = df.groupby('product_id')['total_amount'].cumsum()
            df['month'] = df['sale_date'].dt.to_period('M')
            df['monthly_rank'] = df.groupby('month')['total_amount'].rank(
                method='min', ascending=False
            )
            return df

    Run both with benchmark() and compare.

    Why DuckDB wins for window functions:
    - Native window function engine with optimized partitioning
    - Vectorized window operators process batches of rows
    - A single SQL query with multiple OVER clauses is optimized holistically
    - Pandas: cumsum and rank are separate operations, each iterating the data
    - Pandas has no native equivalent to SQL OVER(PARTITION BY ... ORDER BY ...)
      requiring manual sort + groupby workarounds

    Why this matters:
    - Window functions are the most performance-critical analytical operation
    - The gap between DuckDB and Pandas is typically largest for window operations
    - In production pipelines, replacing Pandas window operations with DuckDB SQL
      can reduce processing time from minutes to seconds
    """
    raise NotImplementedError(
        "TODO(human): Implement and benchmark DuckDB vs Pandas window functions"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all performance benchmarks."""
    print("Loading data...")
    sales_df = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["sale_date"])
    products_df = pd.read_csv(DATA_DIR / "products.csv")
    stores_df = pd.read_csv(DATA_DIR / "stores.csv")
    print(f"  Sales: {len(sales_df):,} rows")
    print(f"  Products: {len(products_df):,} rows")
    print(f"  Stores: {len(stores_df):,} rows")
    print()

    # DuckDB in-memory connection (fair comparison — no disk I/O advantage)
    con = duckdb.connect(":memory:")

    print("=" * 70)
    print("BENCHMARK 1: Aggregation (GROUP BY)")
    print("=" * 70)
    aggregation_benchmark(con, sales_df, products_df)

    print()
    print("=" * 70)
    print("BENCHMARK 2: Multi-Table Join")
    print("=" * 70)
    join_benchmark(con, sales_df, products_df, stores_df)

    print()
    print("=" * 70)
    print("BENCHMARK 3: Window Functions")
    print("=" * 70)
    window_benchmark(con, sales_df)

    con.close()
    print("All benchmarks complete.")


if __name__ == "__main__":
    main()
