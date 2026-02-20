"""Exercise 4: Optimization -- Compaction, Z-Ordering, Vacuum & Partitioning.

Real-world Delta tables accumulate many small Parquet files over time,
especially when data arrives in small frequent batches (streaming, CDC, etc.).
Small files degrade query performance because each file has metadata overhead
and the engine must open/close many file handles.

Delta Lake provides three optimization mechanisms:
- COMPACT: merge small files into fewer larger files
- Z-ORDER: co-locate related data within files for better data skipping
- VACUUM: permanently remove old, unreferenced files to reclaim storage

This exercise also demonstrates PARTITIONING, which physically organizes
data into directories by column values (Hive-style partitioning).

After completing this exercise you should understand:
- How compaction reduces file count without changing data
- How Z-ordering improves query performance for specific columns
- How vacuum cleans up old files (and why it can break time travel)
- How partition_by creates a directory structure for efficient filtering
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

TABLE_PATH = Path(__file__).resolve().parent.parent / "data" / "orders_optimize"
PARTITIONED_TABLE_PATH = Path(__file__).resolve().parent.parent / "data" / "orders_partitioned"


def generate_many_small_batches() -> list[pd.DataFrame]:
    """Generate 15 small batches of orders to create many small files.

    Each batch has 3-5 rows, simulating a stream of incoming orders.
    After writing all batches, the table will have ~15 Parquet files
    (one per batch) which is inefficient for queries.
    """
    batches = []
    base_date = datetime.now() - timedelta(days=30)
    products = ["Laptop", "Mouse", "Keyboard", "Monitor", "Headphones",
                "GPU", "SSD Drive", "Webcam", "RAM Kit", "USB Hub"]
    categories = ["electronics", "accessories", "accessories", "electronics", "accessories",
                  "electronics", "storage", "accessories", "memory", "accessories"]
    order_id = 1

    for batch_idx in range(15):
        batch_size = 3 + (batch_idx % 3)  # 3, 4, or 5 rows
        rows = []
        for j in range(batch_size):
            prod_idx = (order_id - 1) % len(products)
            rows.append({
                "order_id": order_id,
                "customer_id": f"CUST-{(order_id % 20) + 1:03d}",
                "product": products[prod_idx],
                "category": categories[prod_idx],
                "amount": round(29.99 + (order_id * 13.7) % 1000, 2),
                "status": ["pending", "shipped", "completed"][order_id % 3],
                "order_date": (base_date + timedelta(days=batch_idx * 2)).strftime("%Y-%m-%d"),
            })
            order_id += 1
        batches.append(pd.DataFrame(rows))

    return batches


def write_many_batches(table_path: Path, batches: list[pd.DataFrame]) -> None:
    """Write each batch as a separate append, creating many small files."""
    from deltalake import write_deltalake

    for i, batch in enumerate(batches):
        if i == 0:
            write_deltalake(str(table_path), batch)
        else:
            write_deltalake(str(table_path), batch, mode="append")

    from deltalake import DeltaTable
    dt = DeltaTable(table_path)
    total_rows = len(dt.to_pandas())
    print(f"  Wrote {len(batches)} batches -> {len(dt.files())} files, {total_rows} rows, version {dt.version()}")


def count_data_files(table_path: Path) -> int:
    """Count the number of Parquet data files in the table directory.

    Note: this counts ALL parquet files in the directory tree, including
    inside partition directories.
    """
    return len(list(Path(table_path).rglob("*.parquet")))


def show_directory_tree(table_path: Path, max_depth: int = 3) -> None:
    """Print the directory structure of a Delta table."""
    base = Path(table_path)
    if not base.exists():
        print(f"  [!] {table_path} does not exist")
        return

    print(f"\n  Directory tree of {base.name}/:")
    for item in sorted(base.rglob("*")):
        rel = item.relative_to(base)
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        if item.name.startswith("."):
            continue
        # Skip _delta_log contents for brevity
        if "_delta_log" in str(rel) and depth > 1:
            continue
        indent = "    " + "  " * (depth - 1)
        if item.is_dir():
            print(f"{indent}{item.name}/")
        else:
            size_kb = item.stat().st_size / 1024
            print(f"{indent}{item.name} ({size_kb:.1f} KB)")


# TODO(human): Implement optimization_exercise
#
# This is the core exercise with 4 parts: compact, z-order, vacuum,
# and partitioned writes.
#
# PART 1 -- COMPACTION
#   The table was created with 15 small batches, resulting in ~15 Parquet
#   files. Compaction merges them into fewer, larger files.
#
#     from deltalake import DeltaTable
#
#     dt = DeltaTable(table_path)
#     print(f"Before compact: {len(dt.files())} files")
#
#     # Run compaction
#     compact_result = dt.optimize.compact()
#     print(f"Compact result: {compact_result}")
#
#   compact() returns a dict with metrics:
#     {"numFilesAdded": N, "numFilesRemoved": M, "filesAdded": {...}, ...}
#
#   After compaction, reload the table and check file count:
#     dt = DeltaTable(table_path)
#     print(f"After compact: {len(dt.files())} files")
#
#   Key insight: compaction does NOT change the data -- same rows, same
#   schema, same content. It only reorganizes the physical files for
#   better read performance. It creates a new version in the log.
#
#   IMPORTANT: After compaction, the old small files are NOT immediately
#   deleted. They're marked as "removed" in the transaction log but still
#   exist on disk (they're needed for time travel to pre-compaction versions).
#
# PART 2 -- Z-ORDERING
#   Z-ordering sorts data within files by one or more columns using a
#   space-filling Z-curve. This co-locates related values together,
#   enabling better "data skipping" -- when a query filters by a Z-ordered
#   column, the engine can skip entire files whose min/max statistics
#   don't overlap with the filter predicate.
#
#     dt = DeltaTable(table_path)
#     zorder_result = dt.optimize.z_order(columns=["customer_id"])
#     print(f"Z-order result: {zorder_result}")
#
#   Best practice for Z-order column selection:
#     - High-cardinality columns used in WHERE clauses (customer_id, order_id)
#     - Columns commonly used for joins
#     - NOT low-cardinality columns (status with 3 values = use partitioning instead)
#     - Maximum 2-3 columns (effectiveness decreases with more)
#
#   After Z-ordering, reload and show the table state. The version will
#   have incremented. Row count is unchanged.
#
# PART 3 -- VACUUM
#   Vacuum permanently deletes files that are no longer referenced by
#   any "recent" version of the table. The retention_hours parameter
#   controls what counts as "recent" (default: 168 hours = 7 days).
#
#     dt = DeltaTable(table_path)
#
#     # First: dry run to see what WOULD be deleted
#     files_to_delete = dt.vacuum(
#         retention_hours=0,
#         dry_run=True,
#         enforce_retention_duration=False,
#     )
#     print(f"Dry run: {len(files_to_delete)} files would be deleted")
#
#     # Then: actual vacuum (set dry_run=False)
#     deleted = dt.vacuum(
#         retention_hours=0,
#         dry_run=False,
#         enforce_retention_duration=False,
#     )
#     print(f"Vacuumed: {len(deleted)} files deleted")
#
#   IMPORTANT parameters:
#     - retention_hours=0: delete ALL unreferenced files (dangerous in
#       production -- usually you'd use 168 or more to preserve time travel)
#     - dry_run=True: only list files that would be deleted, don't delete
#     - enforce_retention_duration=False: required when retention_hours is
#       less than the configured minimum (default 168h). Without this flag,
#       vacuum will refuse to use retention_hours=0.
#
#   After vacuum, count files on disk (using count_data_files helper) and
#   compare to before. The unreferenced files from pre-compaction should
#   be gone.
#
#   WARNING: After vacuuming with retention_hours=0, time travel to old
#   versions will FAIL because the old Parquet files are gone. Try it:
#     try:
#         dt_old = DeltaTable(table_path, version=0)
#         dt_old.to_pandas()
#     except Exception as e:
#         print(f"Time travel broken after vacuum: {type(e).__name__}: {e}")
#
# PART 4 -- PARTITIONED TABLE
#   Create a NEW table at PARTITIONED_TABLE_PATH with partition_by.
#   Partitioning physically organizes data into directories by column value.
#
#   Combine all the small batches into one DataFrame, then write:
#
#     from deltalake import write_deltalake
#     all_data = pd.concat(batches, ignore_index=True)
#
#     write_deltalake(
#         str(partitioned_path),
#         all_data,
#         partition_by=["category"],
#     )
#
#   This creates a Hive-style directory structure:
#     orders_partitioned/
#       category=electronics/
#         part-00000-....parquet
#       category=accessories/
#         part-00000-....parquet
#       category=storage/
#         ...
#       _delta_log/
#         00000000000000000000.json
#
#   After writing, use show_directory_tree(partitioned_path) to see the
#   layout.
#
#   Then demonstrate partition pruning: read with a filter that targets
#   one partition:
#
#     dt = DeltaTable(partitioned_path)
#     # Using PyArrow dataset filters
#     electronics = dt.to_pandas(
#         partitions=[("category", "=", "electronics")]
#     )
#     print(f"Electronics orders: {len(electronics)} rows")
#
#   Also show DuckDB reading the partitioned table:
#     import duckdb
#     path_str = str(partitioned_path).replace("\\", "/")
#     result = duckdb.query(
#         f"SELECT category, COUNT(*) as cnt, SUM(amount) as total "
#         f"FROM delta_scan('{path_str}') GROUP BY category ORDER BY total DESC"
#     )
#     print(result.fetchdf())
#
# Function signature:
#   def optimization_exercise(
#       table_path: Path,
#       partitioned_path: Path,
#       batches: list[pd.DataFrame],
#   ) -> None
#
# Key learning points:
#   - COMPACT reduces file count, improving scan performance.
#   - Z-ORDER optimizes data layout for specific filter columns.
#   - VACUUM reclaims disk space but permanently breaks time travel
#     for versions older than the retention period.
#   - PARTITIONING is best for low-cardinality columns used in filters
#     (status, category, date). It creates physical directory boundaries
#     that engines can skip entirely.
#   - Compaction + Z-ordering create new versions (can be time-traveled).
#     Vacuum is destructive (deletes files, can break old version reads).

def optimization_exercise(
    table_path: Path,
    partitioned_path: Path,
    batches: list[pd.DataFrame],
) -> None:
    raise NotImplementedError(
        "TODO(human): Implement compaction, z-ordering, vacuum, and partitioned "
        "table creation. See the detailed comments above for each part."
    )


def main() -> None:
    print("=" * 60)
    print("Exercise 4: Optimization -- Compact, Z-Order, Vacuum, Partition")
    print("=" * 60)

    # Generate data
    batches = generate_many_small_batches()
    total_rows = sum(len(b) for b in batches)
    print(f"\nGenerated {len(batches)} batches with {total_rows} total rows")

    # Ensure clean state
    import shutil
    for p in [TABLE_PATH, PARTITIONED_TABLE_PATH]:
        if p.exists():
            shutil.rmtree(p)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write many small batches (provided)
    print("\n" + "-" * 60)
    print("Writing many small batches (simulating streaming ingestion)...")
    print("-" * 60)
    write_many_batches(TABLE_PATH, batches)

    # Core exercise
    print("\n" + "-" * 60)
    print("Running optimization exercise...")
    print("-" * 60)
    optimization_exercise(TABLE_PATH, PARTITIONED_TABLE_PATH, batches)


if __name__ == "__main__":
    main()
