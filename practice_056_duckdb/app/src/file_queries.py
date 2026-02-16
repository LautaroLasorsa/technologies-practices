"""
Exercise: File Queries — Direct Parquet/CSV Querying Without Import.

This module teaches DuckDB's "zero-ETL" philosophy: querying data files
directly without loading them into tables first. This is one of DuckDB's
most powerful features — you can run complex analytical SQL against
Parquet files, CSV files, and even remote files over HTTP/S3, all
without a single CREATE TABLE statement.

DuckDB achieves this through:
- Automatic file format detection and schema inference
- Predicate pushdown into Parquet files (only reading needed row groups)
- Column pruning (only reading needed columns from columnar files)
- Glob pattern matching for querying multiple files at once

Run: uv run python src/file_queries.py
"""

from pathlib import Path

import duckdb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Exercise 1: Direct Parquet and CSV Querying
# ---------------------------------------------------------------------------


def direct_file_queries(con: duckdb.DuckDBPyConnection) -> None:
    """Query Parquet and CSV files directly without creating tables.

    TODO(human): Implement this function.

    DuckDB can query files directly using either:
    1. String literal path:  SELECT * FROM 'path/to/file.parquet'
    2. Function call:        SELECT * FROM read_parquet('path/to/file.parquet')
                             SELECT * FROM read_csv('path/to/file.csv')

    The string literal approach (option 1) only works if the file extension
    is recognized (.parquet, .csv, .json). For non-standard extensions,
    use the explicit function (option 2).

    Query 1 — Query Parquet directly:
    Run a GROUP BY query against the Parquet file without loading it:

        SELECT
            product_id,
            COUNT(*) AS num_sales,
            SUM(total_amount) AS total_revenue,
            AVG(total_amount) AS avg_sale
        FROM '<DATA_DIR>/sales.parquet'
        GROUP BY product_id
        ORDER BY total_revenue DESC
        LIMIT 10

    This demonstrates DuckDB's ability to push the aggregation down into
    the Parquet reader. Only the columns needed (product_id, total_amount)
    are read from the file — other columns are skipped entirely. The Parquet
    format stores column metadata (min/max per row group), enabling DuckDB
    to skip entire row groups that don't match filter predicates.

    Query 2 — Query CSV directly:
    Run the same query against the CSV file:

        SELECT
            product_id,
            COUNT(*) AS num_sales,
            SUM(total_amount) AS total_revenue
        FROM '<DATA_DIR>/sales.csv'
        GROUP BY product_id
        ORDER BY total_revenue DESC
        LIMIT 10

    Compare this with Query 1: CSV requires reading ALL columns (it's
    row-oriented text), while Parquet allows column pruning. For large
    files, Parquet queries are significantly faster.

    Query 3 — Cross-file JOIN without tables:
    Join two files directly:

        SELECT
            p.product_name,
            p.category,
            COUNT(*) AS num_sales,
            SUM(s.total_amount) AS revenue
        FROM '<DATA_DIR>/sales.parquet' s
        JOIN '<DATA_DIR>/products.csv' p ON s.product_id = p.product_id
        GROUP BY p.product_name, p.category
        ORDER BY revenue DESC
        LIMIT 10

    This joins a Parquet file with a CSV file — no tables, no schema
    definitions, no import. DuckDB infers types from both files and
    executes the join with its vectorized hash join engine.

    Print all three results.

    Why this matters:
    - Zero-ETL: analyze data where it lives without import/export pipelines
    - Column pruning on Parquet = only read what you need (huge I/O savings)
    - Cross-format joins enable ad-hoc analysis across heterogeneous data
    - This pattern is ideal for data exploration, CI/CD data validation
    """
    raise NotImplementedError(
        "TODO(human): Query Parquet/CSV files directly and join across formats"
    )


# ---------------------------------------------------------------------------
# Exercise 2: Glob Patterns and Partitioned Data
# ---------------------------------------------------------------------------


def glob_and_partitioned_queries(con: duckdb.DuckDBPyConnection) -> None:
    """Query multiple files using glob patterns and track source files.

    TODO(human): Implement this function.

    DuckDB supports glob patterns in file paths, allowing you to query
    multiple files as a single virtual table. This is essential for
    partitioned datasets (common in data lakes: data split by date,
    region, etc.).

    Query 1 — Glob pattern over partitioned files:
    Query all year-partitioned Parquet files at once:

        SELECT
            COUNT(*) AS total_rows,
            MIN(sale_date) AS earliest,
            MAX(sale_date) AS latest,
            SUM(total_amount) AS total_revenue
        FROM '<DATA_DIR>/sales_by_year/*.parquet'

    The glob pattern `*.parquet` matches all Parquet files in the directory.
    DuckDB reads them all and treats them as a single table. This is how
    data lakes work — data is partitioned into many files for efficient
    parallel processing and incremental updates.

    Query 2 — filename column for source tracking:
    Use the `filename` virtual column to see which file each row came from:

        SELECT
            filename,
            COUNT(*) AS rows,
            SUM(total_amount) AS revenue
        FROM '<DATA_DIR>/sales_by_year/*.parquet'
        GROUP BY filename
        ORDER BY filename

    The `filename` column is automatically available when reading multiple
    files — it contains the full path of the source file for each row.
    This is invaluable for debugging data quality issues ("which file
    has the bad data?") and for partition-aware processing.

    Note: In DuckDB >= 1.3.0, `filename` is available by default. In
    older versions, you may need: read_parquet('...', filename=true).

    Query 3 — Selective glob with filter pushdown:
    Query only 2023 and 2024 data by listing specific files:

        SELECT
            DATE_TRUNC('month', sale_date) AS month,
            SUM(total_amount) AS revenue
        FROM read_parquet([
            '<DATA_DIR>/sales_by_year/sales_2023.parquet',
            '<DATA_DIR>/sales_by_year/sales_2024.parquet'
        ])
        GROUP BY DATE_TRUNC('month', sale_date)
        ORDER BY month

    You can pass a list of specific files to read_parquet() instead of
    a glob. This gives precise control over which partitions to include,
    without needing a WHERE filter (which would still read all files).

    Print all three results.

    Why this matters:
    - Glob patterns are the standard way to query partitioned data lakes
    - The filename column enables data lineage and debugging
    - Selective file lists avoid reading unnecessary partitions
    - This mirrors how tools like Spark and Trino handle partitioned data
    """
    raise NotImplementedError(
        "TODO(human): Query partitioned files with globs and track sources with filename"
    )


# ---------------------------------------------------------------------------
# Exercise 3: httpfs Extension — Remote File Querying
# ---------------------------------------------------------------------------


def remote_file_queries(con: duckdb.DuckDBPyConnection) -> None:
    """Query remote Parquet files over HTTPS using the httpfs extension.

    TODO(human): Implement this function.

    DuckDB can query files hosted on HTTP/HTTPS servers and S3-compatible
    storage using the `httpfs` extension. This enables querying public
    datasets and data lake files without downloading them first.

    Step 1 — Install and load the httpfs extension:
        INSTALL httpfs;
        LOAD httpfs;

    DuckDB extensions are installed once (cached locally) and loaded per
    session. The httpfs extension adds support for HTTP(S) and S3 URLs
    in file-reading functions.

    Step 2 — Query a remote Parquet file:
    Use a small public Parquet dataset (DuckDB's own test data):

        SELECT *
        FROM read_parquet('https://raw.githubusercontent.com/duckdb/duckdb-web/main/data/weather.parquet')
        LIMIT 20

    This downloads only the needed Parquet metadata and row groups over
    HTTP — not the entire file. Parquet's footer-based metadata allows
    DuckDB to determine which row groups to fetch based on predicates
    and which columns to read.

    Step 3 — Aggregate remote data with pushdown:
    Run an aggregation on the remote file:

        SELECT
            city,
            AVG(temp) AS avg_temp,
            COUNT(*) AS readings
        FROM read_parquet('https://raw.githubusercontent.com/duckdb/duckdb-web/main/data/weather.parquet')
        GROUP BY city
        ORDER BY avg_temp DESC

    Even for remote files, DuckDB applies column pruning and predicate
    pushdown — it only downloads the columns and row groups it needs.

    Note: If the network is unavailable, print a message and skip this
    exercise gracefully (don't crash the whole script). Use try/except.

    Print results from Steps 2 and 3.

    Why this matters:
    - Querying remote files enables zero-download exploration of data lakes
    - The httpfs extension works with S3, GCS, Azure Blob (with credentials)
    - Parquet's metadata-first design means DuckDB only downloads needed data
    - This pattern is used extensively with Hugging Face datasets, public data
      repositories, and cloud data lakes
    """
    raise NotImplementedError(
        "TODO(human): Install httpfs, query remote Parquet files over HTTPS"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all file query exercises."""
    con = duckdb.connect(":memory:")

    print("=" * 70)
    print("EXERCISE 1: Direct Parquet/CSV Querying")
    print("=" * 70)
    direct_file_queries(con)

    print()
    print("=" * 70)
    print("EXERCISE 2: Glob Patterns and Partitioned Data")
    print("=" * 70)
    glob_and_partitioned_queries(con)

    print()
    print("=" * 70)
    print("EXERCISE 3: httpfs — Remote File Querying")
    print("=" * 70)
    remote_file_queries(con)

    con.close()
    print("\nAll file query exercises complete.")


if __name__ == "__main__":
    main()
