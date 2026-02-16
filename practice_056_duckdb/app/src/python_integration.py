"""
Exercise: Python Integration — Pandas, Arrow, and the Relation API.

This module teaches DuckDB's seamless integration with the Python data
ecosystem. DuckDB can:
- Query Pandas DataFrames directly using SQL (zero-copy)
- Export results to Arrow tables (zero-copy)
- Build queries programmatically with the Relation API

The key insight: DuckDB is not an either/or replacement for Pandas —
it's a complement. Use Pandas for data wrangling (reshape, merge,
feature engineering) and DuckDB for SQL analytics on the same data,
in the same script, with zero serialization overhead.

Run: uv run python src/python_integration.py
"""

from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Exercise 1: Query Pandas DataFrames with SQL
# ---------------------------------------------------------------------------


def query_pandas_dataframes(con: duckdb.DuckDBPyConnection) -> None:
    """Query Pandas DataFrames directly using DuckDB SQL.

    TODO(human): Implement this function.

    DuckDB can query Pandas DataFrames by name — just reference the Python
    variable in your SQL query. DuckDB uses "replacement scans" to detect
    that a table name matches a local Python variable holding a DataFrame,
    and reads the DataFrame's underlying arrays directly (zero-copy when
    backed by Arrow or NumPy arrays).

    Step 1 — Create DataFrames:
    Load the CSV files into Pandas DataFrames:
        sales_df = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["sale_date"])
        products_df = pd.read_csv(DATA_DIR / "products.csv")
        stores_df = pd.read_csv(DATA_DIR / "stores.csv")

    Step 2 — Query a DataFrame with SQL:
    Run a SQL query that references the DataFrame variable name directly:

        result = con.sql(\"\"\"
            SELECT
                category,
                COUNT(*) AS num_sales,
                ROUND(SUM(total_amount), 2) AS total_revenue,
                ROUND(AVG(total_amount), 2) AS avg_order
            FROM sales_df s
            JOIN products_df p ON s.product_id = p.product_id
            GROUP BY category
            ORDER BY total_revenue DESC
        \"\"\")

    Note: the variable names `sales_df` and `products_df` are used directly
    in the SQL — no registration needed. DuckDB finds them in the Python
    process's local/global scope.

    Step 3 — Convert result back to Pandas:
    Convert the DuckDB result to a Pandas DataFrame:
        result_df = result.fetchdf()  # or result.df()
    Print the result_df and its dtypes to verify types are preserved.

    Step 4 — Parameterized query with DataFrame:
    Filter the sales DataFrame for a specific category using a parameter:
        category = "Electronics"
    Use a prepared statement or f-string to filter:
        con.sql(f"SELECT * FROM sales_df s JOIN products_df p ON ... WHERE p.category = '{category}'")

    Or use the DuckDB default connection (module-level duckdb.sql()):
        duckdb.sql("SELECT * FROM sales_df LIMIT 5")
    Note: the module-level duckdb.sql() uses an implicit default connection
    that automatically sees Python variables in the caller's scope.

    Print results from Steps 2, 3, and 4.

    Why this matters:
    - Zero-copy means no serialization overhead when switching between Pandas and SQL
    - SQL is often more readable than chained Pandas operations for analytics
    - You can use DuckDB for the "hard" queries (window functions, complex joins)
      and Pandas for everything else — best of both worlds
    - The variable-name-as-table pattern makes DuckDB a natural extension of
      Pandas-based notebooks and scripts
    """
    raise NotImplementedError(
        "TODO(human): Load DataFrames, query them with SQL, convert results back"
    )


# ---------------------------------------------------------------------------
# Exercise 2: Arrow Interchange
# ---------------------------------------------------------------------------


def arrow_interchange(con: duckdb.DuckDBPyConnection) -> None:
    """Export DuckDB results to Arrow and query Arrow tables.

    TODO(human): Implement this function.

    Apache Arrow is a columnar in-memory format designed for zero-copy
    data sharing between systems. DuckDB's internal format is Arrow-
    compatible, making Arrow the fastest pathway for data exchange.

    Step 1 — Query to Arrow:
    Run a query and export the result directly as an Arrow table:

        arrow_table = con.sql(\"\"\"
            SELECT
                sale_date,
                product_id,
                total_amount
            FROM read_parquet('{DATA_DIR}/sales.parquet')
            WHERE total_amount > 500
            ORDER BY total_amount DESC
        \"\"\").arrow()

    The .arrow() method returns a pyarrow.Table with zero-copy when
    possible. This is the fastest way to get data out of DuckDB.

    Print the Arrow table's schema (arrow_table.schema) and number of
    rows (len(arrow_table) or arrow_table.num_rows).

    Step 2 — Query an Arrow table:
    Query the Arrow table you just created using SQL:

        result = con.sql(\"\"\"
            SELECT
                product_id,
                COUNT(*) AS high_value_sales,
                ROUND(AVG(total_amount), 2) AS avg_amount
            FROM arrow_table
            GROUP BY product_id
            ORDER BY high_value_sales DESC
            LIMIT 10
        \"\"\")
        result.show()

    Just like DataFrames, Arrow tables are queryable by variable name.

    Step 3 — Arrow to Pandas round-trip:
    Convert Arrow to Pandas and back to demonstrate the interchange:
        pandas_df = arrow_table.to_pandas()         # Arrow → Pandas
        back_to_arrow = pa.Table.from_pandas(pandas_df)  # Pandas → Arrow

    Print the shape of each to verify data integrity.

    Print results from all steps.

    Why this matters:
    - Arrow is the universal columnar interchange format (used by Spark,
      Polars, DuckDB, pandas 2.0+, DataFusion, etc.)
    - Zero-copy Arrow export means you can pipe DuckDB results into
      ML libraries (PyTorch, scikit-learn) without serialization
    - Understanding Arrow is essential for modern data engineering — it's
      the lingua franca of columnar data systems
    """
    raise NotImplementedError(
        "TODO(human): Export to Arrow, query Arrow tables, demonstrate round-trips"
    )


# ---------------------------------------------------------------------------
# Exercise 3: Relation API
# ---------------------------------------------------------------------------


def relation_api(con: duckdb.DuckDBPyConnection) -> None:
    """Build queries programmatically using DuckDB's Relation API.

    TODO(human): Implement this function.

    The Relation API is an alternative to raw SQL strings. Instead of
    writing SQL, you chain method calls on relation objects:

        con.sql("SELECT * FROM 'file.parquet'")
           .filter("amount > 100")
           .aggregate("category, SUM(amount) AS total", "category")
           .order("total DESC")
           .limit(10)
           .show()

    Relations are LAZILY EVALUATED — nothing executes until you call
    a terminal method (.show(), .fetchall(), .fetchdf(), .arrow()).
    This allows DuckDB's optimizer to see the entire query before
    executing it.

    Step 1 — Create a relation from a file:
    Create a relation from the sales Parquet file:
        sales = con.sql(f"SELECT * FROM read_parquet('{DATA_DIR}/sales.parquet')")
    Or equivalently:
        sales = con.read_parquet(str(DATA_DIR / "sales.parquet"))

    Print the relation (this shows the schema, not the data):
        print(sales)

    Step 2 — Chain operations:
    Build a query using method chaining:
        result = (
            sales
            .filter("quantity >= 3")
            .aggregate("product_id, SUM(total_amount) AS revenue, COUNT(*) AS num_sales", "product_id")
            .order("revenue DESC")
            .limit(10)
        )
        result.show()

    This is equivalent to:
        SELECT product_id, SUM(total_amount) AS revenue, COUNT(*) AS num_sales
        FROM sales
        WHERE quantity >= 3
        GROUP BY product_id
        ORDER BY revenue DESC
        LIMIT 10

    Step 3 — Join relations:
    Create relations for products and join with sales:
        products = con.read_csv(str(DATA_DIR / "products.csv"))
        joined = sales.join(products, "sales.product_id = products.product_id")
    Then aggregate and show:
        joined.aggregate(
            "category, SUM(total_amount) AS revenue", "category"
        ).order("revenue DESC").show()

    Step 4 — Relation to SQL:
    Show the SQL that a relation chain generates:
        print(result.explain())
    This prints the query plan, showing what DuckDB will execute.

    Print results from all steps.

    Why this matters:
    - The Relation API enables dynamic query building (useful when filters/
      columns are determined at runtime)
    - Lazy evaluation means DuckDB optimizes the entire chain before executing
    - Method chaining is familiar to Pandas/Polars users
    - Understanding both SQL strings and the Relation API makes you versatile —
      use SQL for complex analytics, Relation API for programmatic query construction
    """
    raise NotImplementedError(
        "TODO(human): Build queries with the Relation API using filter/aggregate/join/order"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all Python integration exercises."""
    con = duckdb.connect(":memory:")

    print("=" * 70)
    print("EXERCISE 1: Query Pandas DataFrames with SQL")
    print("=" * 70)
    query_pandas_dataframes(con)

    print()
    print("=" * 70)
    print("EXERCISE 2: Arrow Interchange")
    print("=" * 70)
    arrow_interchange(con)

    print()
    print("=" * 70)
    print("EXERCISE 3: Relation API")
    print("=" * 70)
    relation_api(con)

    con.close()
    print("\nAll Python integration exercises complete.")


if __name__ == "__main__":
    main()
