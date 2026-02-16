"""
Exercise: DuckDB Basics — Create Tables, Load Data, Basic Queries.

This module teaches the fundamentals of DuckDB: creating an in-process
database, loading data from CSV files, and running basic SQL queries.

DuckDB runs entirely inside your Python process — there is no server to
start, no connection string to configure. You just call duckdb.connect()
and start querying. This is fundamentally different from PostgreSQL or
MySQL, where a separate server process manages the database.

Run: uv run python src/basics.py
"""

from pathlib import Path

import duckdb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Exercise 1: Create and Populate Tables
# ---------------------------------------------------------------------------


def create_and_populate_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create products, stores, and sales tables by loading CSV files.

    TODO(human): Implement this function.

    DuckDB can create tables directly from files using:
        CREATE TABLE <name> AS SELECT * FROM read_csv('<path>')

    The read_csv() function auto-detects:
    - Column names (from header row)
    - Data types (by sampling rows)
    - Delimiters and quoting

    This is a core DuckDB pattern: instead of defining a schema manually
    and then INSERT-ing data, you let DuckDB infer everything from the file.
    Compare this to PostgreSQL where you'd need CREATE TABLE with explicit
    column types, then COPY or INSERT statements.

    Steps:
    1. Create the 'products' table from products.csv
       - Columns: product_id (INT), product_name (VARCHAR), category (VARCHAR),
         base_price (DOUBLE)
    2. Create the 'stores' table from stores.csv
       - Columns: store_id (INT), store_name (VARCHAR), city (VARCHAR),
         region (VARCHAR)
    3. Create the 'sales' table from sales.csv
       - Columns: sale_id (INT), sale_date (DATE), product_id (INT),
         store_id (INT), quantity (INT), unit_price (DOUBLE),
         discount_pct (INT), total_amount (DOUBLE)
    4. Print the row count of each table to verify loading

    Hint: Use f-strings to embed the DATA_DIR path into the SQL.
    DuckDB accepts both forward slashes and backslashes in file paths.

    Why this matters:
    - Schema inference eliminates tedious DDL for exploratory analytics
    - read_csv() handles edge cases (quoted fields, type coercion) automatically
    - Creating tables from files is DuckDB's primary data ingestion pattern
    """
    raise NotImplementedError(
        "TODO(human): Create tables from CSV files using CREATE TABLE ... AS SELECT * FROM read_csv()"
    )


# ---------------------------------------------------------------------------
# Exercise 2: Basic Queries (SELECT, JOIN, GROUP BY)
# ---------------------------------------------------------------------------


def run_basic_queries(con: duckdb.DuckDBPyConnection) -> None:
    """Run basic analytical queries: JOINs, GROUP BY, HAVING, ORDER BY.

    TODO(human): Implement this function.

    Write and execute three queries, printing results for each:

    Query 1 — JOIN: Select the top 10 sales by total_amount, showing:
        sale_id, sale_date, product_name, category, store_name, city,
        quantity, total_amount
    Join sales with products (on product_id) and stores (on store_id).
    Order by total_amount DESC, limit to 10.

    This teaches multi-table JOINs — the bread-and-butter of relational
    analytics. DuckDB uses hash joins internally, which are extremely fast
    for analytical workloads (vs nested-loop joins used by some row stores).

    Query 2 — GROUP BY with aggregation: Calculate total revenue and
    average order value per product category:
        category, total_revenue (SUM of total_amount),
        avg_order_value (AVG of total_amount),
        num_sales (COUNT)
    Order by total_revenue DESC.

    This teaches columnar aggregation — DuckDB only reads the columns
    needed (category, total_amount), skipping all other columns. On a
    row store, the entire row would be read even though most columns
    are unused.

    Query 3 — HAVING: Find stores where the average discount exceeds 10%:
        store_name, city, avg_discount, total_sales
    Use GROUP BY store_id with HAVING AVG(discount_pct) > 10.
    Join with stores table for store_name and city.

    This teaches HAVING — filtering after aggregation. The WHERE clause
    filters rows before GROUP BY; HAVING filters groups after aggregation.

    Hint: Use con.sql("...").show() to print results directly, or
    con.sql("...").fetchdf() to get a Pandas DataFrame.

    Why this matters:
    - JOINs, GROUP BY, and HAVING are the foundation of all analytical SQL
    - DuckDB's columnar engine makes these operations significantly faster
      than equivalent Pandas merge/groupby chains for large datasets
    - Understanding the query execution flow (FROM → WHERE → GROUP BY →
      HAVING → SELECT → ORDER BY) is essential for writing correct SQL
    """
    raise NotImplementedError(
        "TODO(human): Write and execute JOIN, GROUP BY, and HAVING queries"
    )


# ---------------------------------------------------------------------------
# Exercise 3: DuckDB Introspection (DESCRIBE, SUMMARIZE, EXPLAIN)
# ---------------------------------------------------------------------------


def inspect_database(con: duckdb.DuckDBPyConnection) -> None:
    """Use DuckDB's introspection features to understand data and queries.

    TODO(human): Implement this function.

    DuckDB provides several introspection commands that are invaluable
    for data exploration:

    Step 1 — DESCRIBE: Show the schema of the sales table.
        DESCRIBE sales;
    This returns column names, types, nullability, defaults, and keys.
    Equivalent to \\d in PostgreSQL. Use this to verify that read_csv()
    inferred the correct types.

    Step 2 — SUMMARIZE: Generate summary statistics for the sales table.
        SUMMARIZE sales;
    This returns min, max, avg, count, null count, and approximate unique
    count for EVERY column — in a single command. This is a DuckDB-specific
    feature that replaces the need for multiple df.describe() calls in Pandas.
    It's extremely useful for initial data exploration.

    Step 3 — EXPLAIN: Show the query execution plan.
        EXPLAIN SELECT ... (use the GROUP BY query from Exercise 2)
    This shows the physical operators DuckDB will use: SEQ_SCAN (sequential
    column scan), HASH_GROUP_BY (hash-based aggregation), HASH_JOIN, etc.
    Understanding the plan helps you optimize queries.

    Step 4 — EXPLAIN ANALYZE: Show the plan WITH actual execution times.
        EXPLAIN ANALYZE SELECT ... (same query)
    This actually runs the query and reports time spent in each operator.
    Compare this to PostgreSQL's EXPLAIN ANALYZE — same concept, different
    operators (columnar scan vs heap scan, vectorized hash join, etc.).

    Print each result using .show() for readable console output.

    Why this matters:
    - DESCRIBE verifies schema inference (catching type mismatches early)
    - SUMMARIZE replaces 5-10 Pandas calls with one SQL command
    - EXPLAIN reveals how DuckDB's vectorized engine processes your query
    - EXPLAIN ANALYZE identifies bottlenecks (is it the scan? the join? the agg?)
    """
    raise NotImplementedError(
        "TODO(human): Use DESCRIBE, SUMMARIZE, EXPLAIN, and EXPLAIN ANALYZE"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all basics exercises."""
    # DuckDB in-memory database — no file, no server, no config
    con = duckdb.connect(":memory:")

    print("=" * 70)
    print("EXERCISE 1: Create and Populate Tables from CSV")
    print("=" * 70)
    create_and_populate_tables(con)

    print()
    print("=" * 70)
    print("EXERCISE 2: Basic Queries (JOIN, GROUP BY, HAVING)")
    print("=" * 70)
    run_basic_queries(con)

    print()
    print("=" * 70)
    print("EXERCISE 3: DuckDB Introspection (DESCRIBE, SUMMARIZE, EXPLAIN)")
    print("=" * 70)
    inspect_database(con)

    con.close()
    print("\nAll basics exercises complete.")


if __name__ == "__main__":
    main()
