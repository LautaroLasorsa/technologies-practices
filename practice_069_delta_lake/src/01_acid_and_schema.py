"""Exercise 1: ACID Transactions & Schema Enforcement.

Delta Lake's core value proposition over plain Parquet is ACID transactions.
This exercise demonstrates:
- Appending data atomically (each append is a separate versioned commit)
- Schema enforcement (writes with incompatible schemas are rejected)
- Schema evolution (adding new columns with schema_mode="merge")

After completing this exercise you should understand:
- How each write_deltalake() call creates a new version in the transaction log
- Why schema enforcement prevents data corruption
- How schema evolution (merge mode) adds new columns while keeping old data
- The difference between mode="append" and mode="overwrite"
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa

TABLE_PATH = Path(__file__).resolve().parent.parent / "data" / "orders_acid"


def generate_initial_orders() -> pd.DataFrame:
    """Generate 10 initial orders for the base table."""
    now = datetime.now()
    orders = []
    products = ["Laptop", "Mouse", "Keyboard", "Monitor", "Headphones"]

    for i in range(1, 11):
        orders.append({
            "order_id": i,
            "customer_id": f"CUST-{(i % 5) + 1:03d}",
            "product": products[(i - 1) % len(products)],
            "quantity": (i % 3) + 1,
            "amount": round(49.99 + (i * 23.7) % 400, 2),
            "status": "completed",
            "order_date": (now - timedelta(days=20 - i)).strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(orders)


def generate_new_orders_batch() -> pd.DataFrame:
    """Generate 5 new orders to append (same schema as initial)."""
    now = datetime.now()
    orders = []
    products = ["SSD Drive", "RAM Kit", "GPU", "Webcam", "USB Hub"]

    for i in range(11, 16):
        orders.append({
            "order_id": i,
            "customer_id": f"CUST-{(i % 5) + 1:03d}",
            "product": products[(i - 11) % len(products)],
            "quantity": (i % 4) + 1,
            "amount": round(99.99 + (i * 31.2) % 600, 2),
            "status": "pending",
            "order_date": (now - timedelta(days=5 - (i - 11))).strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(orders)


def generate_wrong_schema_data() -> pd.DataFrame:
    """Generate data with a different schema (missing columns, wrong types).

    This is intentionally incompatible with the orders table to demonstrate
    Delta Lake's schema enforcement.
    """
    return pd.DataFrame({
        "id": [100, 101],           # wrong column name (order_id expected)
        "product_name": ["X", "Y"], # wrong column name (product expected)
        "price": [9.99, 19.99],     # wrong column name (amount expected)
    })


def generate_evolved_schema_data() -> pd.DataFrame:
    """Generate data with the original columns PLUS a new 'discount' column.

    This represents a natural schema evolution: a new business requirement
    adds a discount field. Existing rows will get null for this column.
    """
    now = datetime.now()
    return pd.DataFrame({
        "order_id": [16, 17, 18],
        "customer_id": ["CUST-002", "CUST-003", "CUST-001"],
        "product": ["Monitor", "Laptop", "GPU"],
        "quantity": [1, 1, 2],
        "amount": [349.99, 999.99, 1599.98],
        "status": ["pending", "pending", "shipped"],
        "order_date": [
            (now - timedelta(days=2)).strftime("%Y-%m-%d"),
            (now - timedelta(days=1)).strftime("%Y-%m-%d"),
            now.strftime("%Y-%m-%d"),
        ],
        "discount": [0.10, 0.0, 0.15],  # NEW column not in original schema
    })


def print_table_state(table_path: Path, label: str) -> None:
    """Helper to print current table version, row count, and schema."""
    from deltalake import DeltaTable

    dt = DeltaTable(table_path)
    df = dt.to_pandas()
    print(f"\n  [{label}]")
    print(f"    Version: {dt.version()}")
    print(f"    Rows:    {len(df)}")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Files:   {len(dt.files())}")


# TODO(human): Implement acid_and_schema_exercise
#
# This is the core exercise. You will perform 4 steps that demonstrate
# Delta Lake's transaction and schema guarantees:
#
# STEP 1 -- CREATE THE BASE TABLE
#   Use write_deltalake() to create a new Delta table from `initial_orders`.
#   After this, call print_table_state() to show version 0.
#
#     from deltalake import write_deltalake, DeltaTable
#     write_deltalake(table_path, initial_orders)
#
#   This creates version 0 with 10 rows. The table's schema is now locked:
#   {order_id: int64, customer_id: string, product: string, quantity: int64,
#    amount: double, status: string, order_date: string}
#
# STEP 2 -- APPEND NEW DATA (same schema)
#   Use write_deltalake() with mode="append" to add `new_orders` to the table.
#
#     write_deltalake(table_path, new_orders, mode="append")
#
#   This creates version 1 with 15 total rows (10 original + 5 new).
#   Key insight: the append is ATOMIC -- either all 5 rows are added or none.
#   If the process crashes mid-write, the transaction log won't have the
#   commit entry, so readers will never see partial data.
#
#   After appending, call print_table_state() to confirm version 1 with 15 rows.
#
# STEP 3 -- SCHEMA ENFORCEMENT (expected failure)
#   Try to append `wrong_schema` data to the table:
#
#     try:
#         write_deltalake(table_path, wrong_schema, mode="append")
#         print("    [!] Write succeeded unexpectedly")
#     except Exception as e:
#         print(f"    Schema enforcement caught error: {type(e).__name__}")
#         print(f"    {e}")
#
#   This SHOULD fail because the column names don't match. Delta Lake
#   enforces schema on write -- you cannot accidentally corrupt a table
#   by writing mismatched data. This is a critical safety feature that
#   plain Parquet files lack (you can dump any Parquet file into a folder
#   and break downstream readers).
#
#   Print a message confirming the error was caught and the table version
#   is still 1 (the failed write did NOT create a new version).
#
# STEP 4 -- SCHEMA EVOLUTION (adding a column)
#   Append `evolved_schema` data which has all original columns PLUS a
#   new "discount" column. Use schema_mode="merge" to allow the evolution:
#
#     write_deltalake(
#         table_path,
#         evolved_schema,
#         mode="append",
#         schema_mode="merge",
#     )
#
#   schema_mode="merge" tells Delta Lake: "if the incoming data has new
#   columns not in the table schema, add them to the schema. Existing rows
#   will have null values for the new columns."
#
#   After this, call print_table_state() to confirm:
#   - Version is now 2
#   - Total rows: 18 (10 + 5 + 3)
#   - Columns now include "discount"
#
#   Read the full table with DeltaTable(table_path).to_pandas() and print
#   a few rows to show that old rows have NaN/null for the discount column
#   while new rows have actual values.
#
# Function signature:
#   def acid_and_schema_exercise(
#       table_path: Path,
#       initial_orders: pd.DataFrame,
#       new_orders: pd.DataFrame,
#       wrong_schema: pd.DataFrame,
#       evolved_schema: pd.DataFrame,
#   ) -> None
#
# Key learning points:
#   - Every write_deltalake() call is a TRANSACTION that either fully
#     succeeds or fully fails. No partial writes.
#   - Schema enforcement prevents data corruption from mismatched schemas.
#   - Schema evolution with schema_mode="merge" is the controlled way to
#     add new columns -- it's explicit, not accidental.
#   - Each successful write increments the version number by 1.
#   - The _delta_log/ will now have 3 JSON files: 00...00.json, 00...01.json,
#     00...02.json -- one per committed transaction.

def acid_and_schema_exercise(
    table_path: Path,
    initial_orders: pd.DataFrame,
    new_orders: pd.DataFrame,
    wrong_schema: pd.DataFrame,
    evolved_schema: pd.DataFrame,
) -> None:
    raise NotImplementedError(
        "TODO(human): Implement the 4-step ACID and schema exercise. "
        "See the detailed comments above for each step."
    )


def main() -> None:
    print("=" * 60)
    print("Exercise 1: ACID Transactions & Schema Enforcement")
    print("=" * 60)

    # Generate all datasets
    initial_orders = generate_initial_orders()
    new_orders = generate_new_orders_batch()
    wrong_schema = generate_wrong_schema_data()
    evolved_schema = generate_evolved_schema_data()

    print(f"\nInitial orders:       {len(initial_orders)} rows")
    print(f"New orders batch:     {len(new_orders)} rows")
    print(f"Wrong schema data:    {len(wrong_schema)} rows, cols={list(wrong_schema.columns)}")
    print(f"Evolved schema data:  {len(evolved_schema)} rows, cols={list(evolved_schema.columns)}")

    # Ensure clean state
    import shutil
    if TABLE_PATH.exists():
        shutil.rmtree(TABLE_PATH)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Core exercise
    print("\n" + "-" * 60)
    print("Running ACID & schema exercise...")
    print("-" * 60)
    acid_and_schema_exercise(
        TABLE_PATH, initial_orders, new_orders, wrong_schema, evolved_schema
    )


if __name__ == "__main__":
    main()
