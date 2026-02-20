"""Exercise 2: Time Travel & History.

One of Delta Lake's most powerful features is time travel: the ability to
read any previous version of a table. Because the transaction log records
every change as an immutable versioned commit, you can "go back in time"
to see exactly what the table looked like at any point.

This exercise builds a table with multiple versions (through sequential
writes), then demonstrates reading historical versions and inspecting
the audit log.

After completing this exercise you should understand:
- How to read a specific historical version of a Delta table
- How to use load_as_version() for version-based and timestamp-based travel
- How history() provides a complete audit trail of all operations
- Why vacuum() can break time travel (and how retention periods protect it)
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

TABLE_PATH = Path(__file__).resolve().parent.parent / "data" / "orders_timetravel"


def generate_versioned_data() -> list[tuple[str, pd.DataFrame]]:
    """Generate 4 batches of data that will create 4 table versions.

    Returns a list of (description, DataFrame) tuples.
    Each batch represents a day of order processing.
    """
    base_date = datetime.now() - timedelta(days=10)

    # Version 0: Initial batch of 5 orders (Day 1)
    v0 = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "customer_id": ["CUST-001", "CUST-002", "CUST-003", "CUST-001", "CUST-004"],
        "product": ["Laptop", "Mouse", "Keyboard", "Monitor", "Headphones"],
        "amount": [999.99, 29.99, 79.99, 349.99, 149.99],
        "status": ["pending", "pending", "pending", "pending", "pending"],
        "order_date": [base_date.strftime("%Y-%m-%d")] * 5,
    })

    # Version 1: More orders arrive (Day 2)
    v1 = pd.DataFrame({
        "order_id": [6, 7, 8],
        "customer_id": ["CUST-005", "CUST-002", "CUST-006"],
        "product": ["GPU", "SSD Drive", "Webcam"],
        "amount": [799.99, 129.99, 69.99],
        "status": ["pending", "pending", "pending"],
        "order_date": [(base_date + timedelta(days=1)).strftime("%Y-%m-%d")] * 3,
    })

    # Version 2: Even more orders (Day 3)
    v2 = pd.DataFrame({
        "order_id": [9, 10, 11, 12],
        "customer_id": ["CUST-003", "CUST-007", "CUST-001", "CUST-008"],
        "product": ["RAM Kit", "USB Hub", "Laptop", "Monitor"],
        "amount": [189.99, 39.99, 1099.99, 279.99],
        "status": ["pending", "shipped", "pending", "pending"],
        "order_date": [(base_date + timedelta(days=2)).strftime("%Y-%m-%d")] * 4,
    })

    # Version 3: Overwrite entire table with updated statuses (represents
    # a nightly batch job that refreshes the full orders table)
    v3 = pd.DataFrame({
        "order_id": list(range(1, 13)),
        "customer_id": [
            "CUST-001", "CUST-002", "CUST-003", "CUST-001", "CUST-004",
            "CUST-005", "CUST-002", "CUST-006", "CUST-003", "CUST-007",
            "CUST-001", "CUST-008",
        ],
        "product": [
            "Laptop", "Mouse", "Keyboard", "Monitor", "Headphones",
            "GPU", "SSD Drive", "Webcam", "RAM Kit", "USB Hub",
            "Laptop", "Monitor",
        ],
        "amount": [
            999.99, 29.99, 79.99, 349.99, 149.99,
            799.99, 129.99, 69.99, 189.99, 39.99,
            1099.99, 279.99,
        ],
        "status": [
            "completed", "completed", "shipped", "completed", "completed",
            "shipped", "shipped", "completed", "pending", "shipped",
            "pending", "pending",
        ],
        "order_date": [
            (base_date).strftime("%Y-%m-%d")] * 5 + [
            (base_date + timedelta(days=1)).strftime("%Y-%m-%d")] * 3 + [
            (base_date + timedelta(days=2)).strftime("%Y-%m-%d")] * 4,
    })

    return [
        ("Day 1: Initial 5 orders (all pending)", v0),
        ("Day 2: 3 more orders arrive", v1),
        ("Day 3: 4 more orders arrive", v2),
        ("Day 4: Full refresh with updated statuses", v3),
    ]


def build_versioned_table(table_path: Path, batches: list[tuple[str, pd.DataFrame]]) -> None:
    """Create a Delta table and write 4 versions of data.

    This is provided as boilerplate. It creates:
    - Version 0: write (5 rows)
    - Version 1: append (8 total rows)
    - Version 2: append (12 total rows)
    - Version 3: overwrite (12 rows, statuses updated)
    """
    from deltalake import write_deltalake
    import time

    for i, (desc, df) in enumerate(batches):
        if i == 0:
            write_deltalake(str(table_path), df)
        elif i < 3:
            write_deltalake(str(table_path), df, mode="append")
        else:
            # Version 3 is an overwrite (simulating a full-table refresh)
            write_deltalake(str(table_path), df, mode="overwrite")

        print(f"  v{i}: {desc} ({len(df)} rows written)")
        # Small sleep so timestamps differ between versions
        time.sleep(1)


# TODO(human): Implement time_travel_exercise
#
# This is the core exercise. You will explore Delta Lake's time travel
# capabilities using the 4-version table created above.
#
# STEP 1 -- READ THE CURRENT VERSION
#   Load the table and print its current state:
#
#     from deltalake import DeltaTable
#     dt = DeltaTable(table_path)
#     print(f"Current version: {dt.version()}")
#     print(f"Current row count: {len(dt.to_pandas())}")
#
#   This should show version 3 with 12 rows (the overwritten refresh).
#
# STEP 2 -- TIME TRAVEL BY VERSION NUMBER
#   Load specific historical versions. There are two approaches:
#
#   Approach A -- Construct with version parameter:
#     dt_v0 = DeltaTable(table_path, version=0)
#     df_v0 = dt_v0.to_pandas()
#
#   Approach B -- Load then travel:
#     dt = DeltaTable(table_path)
#     dt.load_as_version(0)
#     df_v0 = dt.to_pandas()
#
#   NOTE: load_as_version() MUTATES the DeltaTable object in place. After
#   calling it, dt.version() returns the loaded version. If you need
#   multiple versions simultaneously, create separate DeltaTable instances.
#
#   For each version (0 through 3), print:
#     - Version number
#     - Row count
#     - Status distribution (use df["status"].value_counts())
#
#   Expected output pattern:
#     v0: 5 rows, all "pending"
#     v1: 8 rows, all "pending"
#     v2: 12 rows, mostly "pending" + 1 "shipped"
#     v3: 12 rows, mix of "completed", "shipped", "pending"
#
#   This demonstrates the key value of time travel: you can see exactly what
#   the table looked like at any point. In production, this is invaluable
#   for debugging data pipelines, auditing changes, and reproducing analyses.
#
# STEP 3 -- INSPECT THE HISTORY (AUDIT LOG)
#   DeltaTable.history() returns a list of dictionaries, one per version,
#   with metadata about each commit:
#
#     dt = DeltaTable(table_path)
#     history = dt.history()
#
#   Each entry contains keys like:
#     - "version": the version number
#     - "timestamp": when the commit was made (Unix timestamp in ms)
#     - "operation": "WRITE", "MERGE", "DELETE", etc.
#     - "operationParameters": details like mode, predicate
#
#   Print a formatted table of all history entries showing:
#     version | timestamp (human-readable) | operation | parameters
#
#   This is Delta Lake's built-in audit trail. In regulated industries
#   (finance, healthcare), this audit log is a compliance requirement.
#   With plain Parquet, you have no record of what changed when.
#
# STEP 4 -- DEMONSTRATE VERSION COMPARISON
#   Load version 0 and version 3 side-by-side. For orders that exist in
#   both versions (order_id 1-5), show how their status changed:
#
#     df_v0 = DeltaTable(table_path, version=0).to_pandas()
#     df_v3 = DeltaTable(table_path, version=3).to_pandas()
#
#     # Merge on order_id to compare statuses
#     comparison = df_v0[["order_id", "status"]].merge(
#         df_v3[["order_id", "status"]],
#         on="order_id",
#         suffixes=("_v0", "_v3"),
#     )
#     print(comparison)
#
#   This pattern (comparing versions) is commonly used in data engineering
#   to validate pipeline outputs: "did the nightly refresh change the data
#   in the expected way?"
#
# Function signature:
#   def time_travel_exercise(table_path: Path) -> None
#
# Key learning points:
#   - Every version is immutable -- loading version 0 will ALWAYS return the
#     same data, regardless of what happens to later versions.
#   - Time travel reads are cheap: Delta just reads the commit JSON for that
#     version to know which Parquet files to load.
#   - history() provides a complete audit trail with no extra configuration.
#   - load_as_version() mutates the DeltaTable instance in place -- create
#     separate instances if you need multiple versions simultaneously.
#   - vacuum() deletes old Parquet files -- after vacuuming, older versions
#     may become unreadable because their data files are gone.

def time_travel_exercise(table_path: Path) -> None:
    raise NotImplementedError(
        "TODO(human): Implement the time travel exercise with version loading, "
        "history inspection, and version comparison. See comments above."
    )


def main() -> None:
    print("=" * 60)
    print("Exercise 2: Time Travel & History")
    print("=" * 60)

    # Ensure clean state
    import shutil
    if TABLE_PATH.exists():
        shutil.rmtree(TABLE_PATH)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Build the multi-version table (provided)
    batches = generate_versioned_data()
    print("\nBuilding versioned table...")
    build_versioned_table(TABLE_PATH, batches)

    # Core exercise
    print("\n" + "-" * 60)
    print("Exploring time travel...")
    print("-" * 60)
    time_travel_exercise(TABLE_PATH)


if __name__ == "__main__":
    main()
