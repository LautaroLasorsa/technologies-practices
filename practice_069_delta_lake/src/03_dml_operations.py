"""Exercise 3: DML Operations (Merge, Update, Delete).

Plain Parquet files are immutable -- once written, you cannot update or delete
individual rows without rewriting the entire file. Delta Lake solves this by
implementing row-level DML (Data Manipulation Language) operations through
its transaction log: it writes new Parquet files with the changes and records
which old files were logically removed.

This exercise demonstrates:
- MERGE (upsert): insert new rows and update existing ones in a single operation
- UPDATE: modify specific rows matching a predicate
- DELETE: remove rows matching a predicate

These are the operations that make Delta Lake a "lakehouse" -- you get the
flexibility of a data lake with the DML capabilities of a database.

After completing this exercise you should understand:
- How merge() performs upsert operations using SQL-like predicates
- How update() modifies rows in place (actually writes new files)
- How delete() removes rows without rewriting the entire table
- How each DML operation creates a new version in the transaction log
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

TABLE_PATH = Path(__file__).resolve().parent.parent / "data" / "orders_dml"


def generate_base_orders() -> pd.DataFrame:
    """Generate 10 base orders for the DML exercise table."""
    base_date = datetime.now() - timedelta(days=7)
    return pd.DataFrame({
        "order_id": list(range(1, 11)),
        "customer_id": [
            "CUST-001", "CUST-002", "CUST-003", "CUST-001", "CUST-004",
            "CUST-005", "CUST-002", "CUST-006", "CUST-003", "CUST-007",
        ],
        "product": [
            "Laptop", "Mouse", "Keyboard", "Monitor", "Headphones",
            "GPU", "SSD Drive", "Webcam", "RAM Kit", "USB Hub",
        ],
        "amount": [
            999.99, 29.99, 79.99, 349.99, 149.99,
            799.99, 129.99, 69.99, 189.99, 39.99,
        ],
        "status": [
            "pending", "pending", "shipped", "pending", "completed",
            "pending", "shipped", "pending", "pending", "completed",
        ],
        "order_date": [(base_date + timedelta(days=i % 3)).strftime("%Y-%m-%d") for i in range(10)],
    })


def generate_upsert_data() -> pd.DataFrame:
    """Generate data for the merge/upsert operation.

    Contains:
    - 3 existing order_ids (2, 6, 8) with updated statuses -> should UPDATE
    - 3 new order_ids (11, 12, 13) -> should INSERT
    """
    base_date = datetime.now() - timedelta(days=2)
    return pd.DataFrame({
        "order_id": [2, 6, 8, 11, 12, 13],
        "customer_id": [
            "CUST-002", "CUST-005", "CUST-006",        # existing
            "CUST-008", "CUST-009", "CUST-001",         # new
        ],
        "product": [
            "Mouse", "GPU", "Webcam",                    # same products
            "Laptop", "Monitor", "Headphones",            # new orders
        ],
        "amount": [
            29.99, 799.99, 69.99,                        # same amounts
            1199.99, 449.99, 199.99,                      # new amounts
        ],
        "status": [
            "completed", "shipped", "completed",          # UPDATED statuses
            "pending", "pending", "pending",              # new orders
        ],
        "order_date": [(base_date + timedelta(days=i % 3)).strftime("%Y-%m-%d") for i in range(6)],
    })


def create_base_table(table_path: Path, df: pd.DataFrame) -> None:
    """Create the base Delta table. Provided as boilerplate."""
    from deltalake import write_deltalake
    write_deltalake(str(table_path), df)
    print(f"  Created base table: {len(df)} rows, version 0")


def print_table_snapshot(table_path: Path, label: str, show_rows: bool = True) -> None:
    """Print a snapshot of the table's current state."""
    from deltalake import DeltaTable
    dt = DeltaTable(table_path)
    df = dt.to_pandas()
    print(f"\n  [{label}] Version: {dt.version()}, Rows: {len(df)}")
    if show_rows:
        # Sort by order_id for consistent display
        print(df.sort_values("order_id").to_string(index=False))


# TODO(human): Implement dml_operations_exercise
#
# This is the core exercise. You will perform 3 DML operations on the
# orders table, each creating a new version in the transaction log.
#
# STEP 1 -- MERGE (Upsert)
#   The merge operation is Delta Lake's most powerful DML: it combines
#   INSERT and UPDATE in a single atomic transaction. You specify a
#   predicate to match source rows to target rows, then define what
#   happens for matched rows (update) and unmatched rows (insert).
#
#   Use the DeltaTable.merge() API:
#
#     from deltalake import DeltaTable
#     import pyarrow as pa
#
#     dt = DeltaTable(table_path)
#     source = pa.Table.from_pandas(upsert_data)
#
#     (
#         dt.merge(
#             source=source,
#             predicate="target.order_id = source.order_id",
#             source_alias="source",
#             target_alias="target",
#         )
#         .when_matched_update_all()       # if order_id exists: update ALL columns
#         .when_not_matched_insert_all()   # if order_id is new: insert ALL columns
#         .execute()
#     )
#
#   API details:
#     - merge() returns a TableMerger object (builder pattern)
#     - .when_matched_update_all() updates ALL columns from source for matched rows
#     - .when_not_matched_insert_all() inserts ALL columns from source for new rows
#     - .execute() commits the transaction
#
#   Alternative: instead of update_all/insert_all, you can be selective:
#     .when_matched_update(updates={
#         "status": "source.status",       # only update status
#     })
#     .when_not_matched_insert(updates={
#         "order_id": "source.order_id",
#         "customer_id": "source.customer_id",
#         "product": "source.product",
#         "amount": "source.amount",
#         "status": "source.status",
#         "order_date": "source.order_date",
#     })
#
#   After merge, call print_table_snapshot() to verify:
#   - 3 existing rows (2, 6, 8) should have updated statuses
#   - 3 new rows (11, 12, 13) should be inserted
#   - Total: 13 rows, version 1
#
#   Also print the merge metrics returned by execute():
#     metrics = dt.merge(...).execute()
#     The metrics dict includes:
#       num_target_rows_inserted, num_target_rows_updated,
#       num_target_rows_deleted, num_output_rows, etc.
#
# STEP 2 -- UPDATE
#   Update specific rows matching a predicate. Let's mark all "pending"
#   orders as "processing":
#
#     dt = DeltaTable(table_path)
#     dt.update(
#         predicate="status = 'pending'",
#         updates={"status": "'processing'"},
#     )
#
#   IMPORTANT: The `updates` dict maps column names to SQL EXPRESSIONS
#   (strings), not Python values. So to set a string value, you need
#   single quotes INSIDE the string: "'processing'" (not "processing").
#
#   For numeric updates, you can use expressions:
#     updates={"amount": "amount * 1.1"}   # 10% price increase
#
#   After update, verify that all previously "pending" rows now show
#   "processing". Version should be 2.
#
# STEP 3 -- DELETE
#   Delete rows matching a predicate. Let's remove all completed orders
#   (simulating an archival process):
#
#     dt = DeltaTable(table_path)
#     dt.delete(predicate="status = 'completed'")
#
#   If you omit the predicate, ALL rows are deleted (truncate).
#
#   After delete, verify:
#   - Completed orders are gone
#   - Only "processing" and "shipped" orders remain
#   - Version should be 3
#
#   Also show the full history to demonstrate the audit trail:
#     dt = DeltaTable(table_path)
#     for entry in dt.history():
#         version = entry.get("version", "?")
#         operation = entry.get("operation", "?")
#         params = entry.get("operationParameters", {})
#         print(f"    v{version}: {operation} | {params}")
#
# Function signature:
#   def dml_operations_exercise(
#       table_path: Path,
#       upsert_data: pd.DataFrame,
#   ) -> None
#
# Key learning points:
#   - MERGE is the workhorse for data pipelines: incoming data either
#     updates existing records or inserts new ones, in a single transaction.
#   - UPDATE and DELETE don't modify Parquet files in place -- they write
#     NEW files and mark old files as "removed" in the log. The old files
#     remain on disk until vacuum() is called.
#   - All DML operations use SQL-like expression syntax for predicates
#     and update expressions. This is consistent with Delta Lake on Spark.
#   - Each DML operation is atomic and creates exactly one new version.
#   - The history() audit trail shows the complete lineage of operations.

def dml_operations_exercise(
    table_path: Path,
    upsert_data: pd.DataFrame,
) -> None:
    raise NotImplementedError(
        "TODO(human): Implement the merge, update, and delete operations. "
        "See the detailed comments above for each step."
    )


def main() -> None:
    print("=" * 60)
    print("Exercise 3: DML Operations (Merge, Update, Delete)")
    print("=" * 60)

    # Generate data
    base_orders = generate_base_orders()
    upsert_data = generate_upsert_data()

    print(f"\nBase orders:  {len(base_orders)} rows")
    print(f"Upsert data:  {len(upsert_data)} rows (3 updates + 3 inserts)")

    # Ensure clean state
    import shutil
    if TABLE_PATH.exists():
        shutil.rmtree(TABLE_PATH)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Create base table (provided)
    print("\n" + "-" * 60)
    print("Creating base table...")
    print("-" * 60)
    create_base_table(TABLE_PATH, base_orders)
    print_table_snapshot(TABLE_PATH, "Base table (v0)")

    # Core exercise
    print("\n" + "-" * 60)
    print("Running DML operations...")
    print("-" * 60)
    dml_operations_exercise(TABLE_PATH, upsert_data)


if __name__ == "__main__":
    main()
