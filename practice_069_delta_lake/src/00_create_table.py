"""Exercise 0: Create a Delta Table & Inspect Its Structure.

This exercise introduces the core Delta Lake concept: a Delta table is just
a directory containing Parquet data files plus a _delta_log/ transaction log.
You will create a table from a pandas DataFrame, read it back, and explore the
physical file layout to understand what Delta Lake adds on top of plain Parquet.

After completing this exercise you should understand:
- How write_deltalake() creates a Delta table from a DataFrame
- The physical layout: Parquet files + _delta_log/ directory
- How to read a Delta table back into pandas, PyArrow, and Polars
- What a transaction log commit (JSON file) looks like
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Sample data: e-commerce orders
# ---------------------------------------------------------------------------
# We use a realistic e-commerce orders dataset throughout all exercises.
# This makes DML operations (merge new orders, update status, delete cancelled)
# feel natural later.

TABLE_PATH = Path(__file__).resolve().parent.parent / "data" / "orders"


def generate_sample_orders() -> pd.DataFrame:
    """Generate a DataFrame of 20 sample e-commerce orders."""
    now = datetime.now()
    orders = []
    products = ["Laptop", "Mouse", "Keyboard", "Monitor", "Headphones",
                "Webcam", "USB Hub", "SSD Drive", "RAM Kit", "GPU"]
    statuses = ["completed", "completed", "completed", "pending", "shipped"]

    for i in range(1, 21):
        orders.append({
            "order_id": i,
            "customer_id": f"CUST-{(i % 8) + 1:03d}",
            "product": products[(i - 1) % len(products)],
            "quantity": (i % 5) + 1,
            "amount": round(49.99 + (i * 17.5) % 500, 2),
            "status": statuses[i % len(statuses)],
            "order_date": (now - timedelta(days=30 - i)).strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(orders)


def inspect_delta_log(table_path: Path) -> None:
    """Print the contents of the first transaction log commit.

    This is provided as boilerplate so you can see what Delta Lake records.
    Each JSON commit in _delta_log/ contains 'add' actions (files added),
    'remove' actions (files removed), 'metaData' (schema), and 'protocol'
    (reader/writer versions).
    """
    log_dir = table_path / "_delta_log"
    if not log_dir.exists():
        print("  [!] No _delta_log directory found. Table not created yet.")
        return

    # List all commit files
    commits = sorted(log_dir.glob("*.json"))
    print(f"\n  Transaction log commits: {len(commits)}")
    for c in commits:
        print(f"    {c.name}")

    # Show the first commit contents
    if commits:
        print(f"\n  Contents of {commits[0].name}:")
        print("  " + "-" * 56)
        with open(commits[0], "r") as f:
            for line in f:
                action = json.loads(line)
                # Pretty-print each action on one line, truncated for readability
                action_type = list(action.keys())[0]
                print(f"    [{action_type}] {json.dumps(action[action_type], indent=None)[:120]}")


def list_parquet_files(table_path: Path) -> None:
    """List the Parquet data files in the table directory."""
    parquet_files = list(table_path.glob("*.parquet"))
    print(f"\n  Parquet data files: {len(parquet_files)}")
    for pf in parquet_files:
        size_kb = pf.stat().st_size / 1024
        print(f"    {pf.name} ({size_kb:.1f} KB)")


# TODO(human): Implement create_and_read_delta_table
#
# This is the core exercise. You will:
#
# 1. CREATE A DELTA TABLE from the sample orders DataFrame.
#    Use the `write_deltalake` function from the `deltalake` package:
#
#      from deltalake import write_deltalake
#      write_deltalake(table_path, df)
#
#    - `table_path` can be a string or Path -- it's the directory where the
#      Delta table will be created.
#    - `df` is a pandas DataFrame (or PyArrow Table).
#    - By default, the mode is "error" -- it will fail if the table already exists.
#      For this first exercise, the table won't exist yet, so default mode is fine.
#    - Under the hood, write_deltalake converts the DataFrame to a PyArrow Table,
#      writes it as one or more Parquet files, and creates a _delta_log/ directory
#      with a JSON commit file recording the transaction.
#
# 2. READ THE TABLE BACK using DeltaTable and convert to different formats:
#
#    a) Using deltalake's DeltaTable class:
#         from deltalake import DeltaTable
#         dt = DeltaTable(table_path)
#
#       Once you have a DeltaTable object, you can:
#         - dt.version()          -> current version number (0 for first write)
#         - dt.schema()           -> the Delta schema object
#         - dt.to_pandas()        -> read into pandas DataFrame
#         - dt.to_pyarrow_table() -> read into PyArrow Table
#         - dt.files()            -> list of data file paths (relative)
#         - dt.file_uris()        -> list of data file paths (absolute)
#
#       Print the version, schema, number of files, and the first 5 rows
#       of the pandas DataFrame.
#
#    b) Using Polars (demonstrates cross-library interop):
#         import polars as pl
#         pl_df = pl.read_delta(str(table_path))
#
#       Polars can natively read Delta tables because it uses delta-rs under
#       the hood (same Rust library!). Print the first 5 rows from Polars
#       to confirm it matches.
#
#    c) Using DuckDB (demonstrates SQL queries on Delta tables):
#         import duckdb
#         result = duckdb.query(f"SELECT * FROM delta_scan('{table_path}')")
#
#       DuckDB has a built-in delta_scan() function (auto-loaded extension).
#       Run a query like: SELECT status, COUNT(*) as cnt, SUM(amount) as total
#                         FROM delta_scan('path') GROUP BY status
#       Print the result to show DuckDB can query Delta tables with SQL.
#
#       NOTE on Windows paths: DuckDB's delta_scan needs forward slashes.
#       Convert with: str(table_path).replace("\\", "/")
#
# 3. Print a summary showing:
#    - Table version
#    - Schema (column names and types)
#    - Number of rows
#    - Number of underlying Parquet files
#
# Function signature:
#   def create_and_read_delta_table(df: pd.DataFrame, table_path: Path) -> None
#
# Key learning points:
#   - A Delta table is NOT a single file -- it's a directory with Parquet files
#     and a _delta_log/ subdirectory.
#   - The _delta_log/ is what makes it "Delta" rather than just "Parquet files
#     in a folder." It provides ACID guarantees, schema enforcement, and versioning.
#   - Multiple engines (pandas, Polars, DuckDB, Spark) can all read the same
#     Delta table because the format is an open standard.
#   - write_deltalake does NOT need Spark or JVM -- delta-rs is a pure Rust
#     implementation of the Delta Lake protocol.

def create_and_read_delta_table(df: pd.DataFrame, table_path: Path) -> None:
    raise NotImplementedError(
        "TODO(human): Create a Delta table from the DataFrame, then read it back "
        "using DeltaTable, Polars, and DuckDB. See comments above for guidance."
    )


def main() -> None:
    print("=" * 60)
    print("Exercise 0: Create a Delta Table & Inspect Its Structure")
    print("=" * 60)

    # Generate sample data
    df = generate_sample_orders()
    print(f"\nGenerated {len(df)} sample orders:")
    print(df.to_string(index=False, max_rows=5))

    # Ensure clean state
    import shutil
    if TABLE_PATH.exists():
        shutil.rmtree(TABLE_PATH)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Core exercise
    print("\n" + "-" * 60)
    print("Creating and reading Delta table...")
    print("-" * 60)
    create_and_read_delta_table(df, TABLE_PATH)

    # Inspect physical layout (provided)
    print("\n" + "-" * 60)
    print("Physical layout of the Delta table:")
    print("-" * 60)
    list_parquet_files(TABLE_PATH)
    inspect_delta_log(TABLE_PATH)


if __name__ == "__main__":
    main()
