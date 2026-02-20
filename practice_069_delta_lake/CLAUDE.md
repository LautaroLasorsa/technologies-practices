# Practice 069 -- Delta Lake: ACID Tables & Time Travel

## Technologies

- **delta-rs / deltalake** (Python) -- Rust-native Delta Lake implementation, no JVM or Spark required
- **PyArrow** -- columnar in-memory format, Delta's underlying data layer
- **Pandas** -- DataFrame creation and manipulation
- **Polars** -- high-performance DataFrame library with native Delta Lake support (via delta-rs)
- **DuckDB** -- embedded analytical SQL engine with built-in `delta_scan()` function

## Stack

Python 3.11+ (uv)

## Theoretical Context

### What Delta Lake Is & The Problem It Solves

Data lakes store massive amounts of data cheaply (typically as Parquet files in cloud storage or local filesystems). But raw Parquet has critical limitations:

1. **No ACID transactions** -- a crashed write can leave half-written files that corrupt downstream reads.
2. **No schema enforcement** -- any Parquet file can be dumped into a directory, even if its schema doesn't match. Consumers discover mismatches at read time.
3. **No DML** -- you cannot update or delete individual rows. Modifying one record requires rewriting the entire file.
4. **No versioning** -- once data is overwritten, the previous state is lost forever.

Delta Lake solves all four problems by adding a **transaction log** (`_delta_log/`) on top of Parquet files. A Delta table is physically just a directory containing:

```
my_table/
  _delta_log/
    00000000000000000000.json   # commit 0
    00000000000000000001.json   # commit 1
    00000000000000000002.json   # commit 2
    ...
  part-00000-xxxx.parquet       # data files
  part-00001-xxxx.parquet
  ...
```

The Parquet files hold the actual data. The `_delta_log/` JSON files record which Parquet files are currently "active" and what schema the table has. Together, they provide full ACID guarantees.

### The Transaction Log -- How ACID Works

Each commit is a JSON file in `_delta_log/` containing **actions**:

| Action | Purpose |
|--------|---------|
| **`add`** | Registers a new Parquet file as part of the table. Includes file path, size, partition values, and column-level statistics (min/max). |
| **`remove`** | Marks a Parquet file as logically deleted. The file stays on disk until vacuum. |
| **`metaData`** | Defines the table schema, partition columns, format, and configuration. Present in the first commit. |
| **`protocol`** | Specifies minimum reader and writer protocol versions. |
| **`txn`** | Application-specific transaction identifier for exactly-once semantics. |

**Atomicity**: A transaction either fully succeeds (its JSON file is written) or fully fails (no JSON file). There is no intermediate state.

**Consistency**: Every write validates against the table's schema before committing. Incompatible schemas are rejected.

**Isolation**: Delta Lake uses **optimistic concurrency control (OCC)**. Writers don't acquire locks. Instead, they write their data files first, then attempt to commit by writing the next sequentially-numbered JSON file. If another writer committed first (conflict), the transaction is aborted and the orphaned data files are cleaned up later by vacuum.

**Durability**: Once the JSON commit file is persisted, the transaction is permanent.

**Checkpoints**: Every 10 commits (by default), Delta Lake writes a Parquet-format checkpoint file (e.g., `00000000000000000010.checkpoint.parquet`) that captures the complete table state. This avoids having to replay all previous JSON files from the beginning when opening a table.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Delta table** | A directory containing Parquet data files and a `_delta_log/` transaction log. |
| **Transaction log** | The `_delta_log/` directory of sequentially-numbered JSON commit files. |
| **Version** | Each successful commit increments the version by 1 (starting from 0). |
| **Add action** | Records a new Parquet file as part of the table's current state. |
| **Remove action** | Marks a Parquet file as logically deleted (still on disk until vacuum). |
| **Checkpoint** | A Parquet-format snapshot of the entire table state at a version, for fast reads. |
| **Schema enforcement** | Writes with incompatible schemas are rejected before committing. |
| **Schema evolution** | Controlled addition of new columns via `schema_mode="merge"`. Existing rows get null for new columns. |
| **Time travel** | Reading any previous version of the table via `DeltaTable(path, version=N)` or `load_as_version()`. |
| **Compaction** | Merging many small Parquet files into fewer large ones (`optimize.compact()`). |
| **Z-ordering** | Sorting data within files by columns using a Z-curve for better data skipping (`optimize.z_order()`). |
| **Vacuum** | Permanently deleting unreferenced files older than a retention period (`vacuum()`). Breaks time travel for old versions. |
| **Partition** | Hive-style directory structure (`col=value/`) for coarse-grained data skipping. |

### How Delta Lake Compares

| Feature | Plain Parquet | Delta Lake | Apache Iceberg | Apache Hudi |
|---------|--------------|------------|----------------|-------------|
| ACID transactions | No | Yes | Yes | Yes |
| Schema enforcement | No | Yes | Yes | Yes |
| Time travel | No | Yes | Yes | Yes |
| Row-level DML (update/delete) | No | Yes | Yes | Yes |
| JVM required | No | No (delta-rs) / Yes (Spark) | Yes (mostly) | Yes |
| Ecosystem maturity | Ubiquitous | Most adopted lakehouse format | Growing fast | Smaller community |

Delta Lake, Iceberg, and Hudi solve similar problems with different architectures. Delta Lake uses a single transaction log directory; Iceberg uses metadata files that track "snapshots"; Hudi uses timeline-based metadata. The Delta Lake **UniForm** feature (v3.1+) can produce Iceberg-compatible metadata, enabling interoperability.

### delta-rs vs Delta Lake on Spark

The Delta Lake format is an open standard (protocol spec on GitHub). Two main implementations exist:

- **Delta Lake on Spark** (original): requires JVM, Apache Spark, large cluster infrastructure. Full feature set.
- **delta-rs** (Rust-native): no JVM required. The `deltalake` Python package wraps delta-rs. Suitable for single-machine workloads, ETL scripts, and applications where Spark is overkill. Supports most features including merge, update, delete, optimize, vacuum, and time travel.

Both read/write the same format -- a table created by delta-rs can be read by Spark and vice versa.

### The Lakehouse Paradigm

"Lakehouse" = data warehouse reliability (ACID, schema, SQL) + data lake flexibility (open formats, cheap storage, multi-engine access). Delta Lake is the storage layer that makes this possible. Multiple engines (Spark, DuckDB, Polars, pandas, Trino, Flink) can all read the same Delta table.

## Description

Build and query Delta Lake tables locally using delta-rs (no Spark/JVM). You will create tables from DataFrames, perform ACID transactions with schema enforcement, time-travel through table history, execute row-level DML operations (merge/update/delete), and optimize table storage through compaction, Z-ordering, and vacuum.

### What you'll learn

1. **Table creation and structure** -- how a Delta table is physically laid out (Parquet + `_delta_log/`)
2. **ACID transactions** -- atomic writes, schema enforcement, and controlled schema evolution
3. **Time travel** -- reading historical versions and inspecting the audit log
4. **DML operations** -- merge (upsert), update, and delete on existing tables
5. **Optimization** -- compacting small files, Z-ordering for query performance, vacuuming old files, and partitioning

## Instructions

### Phase 1: Table Creation & Structure (~20 min)

1. Install dependencies: `uv sync`
2. Run `uv run python src/00_create_table.py`
3. **Exercise (`src/00_create_table.py`):** Create a Delta table from a pandas DataFrame using `write_deltalake()`, then read it back using three different engines: `DeltaTable.to_pandas()`, `polars.read_delta()`, and `duckdb.query("SELECT ... FROM delta_scan(...)")`. This teaches the fundamental Delta Lake structure -- a directory of Parquet files plus a `_delta_log/` transaction log -- and demonstrates that the open format enables multi-engine access. The boilerplate inspects the physical file layout and transaction log contents so you can see exactly what Delta Lake creates.

### Phase 2: ACID Transactions & Schema (~20 min)

1. Run `uv run python src/01_acid_and_schema.py`
2. **Exercise (`src/01_acid_and_schema.py`):** Perform 4 operations: create a base table, append data atomically, attempt a write with an incompatible schema (expected failure), and evolve the schema by adding a new column with `schema_mode="merge"`. This teaches Delta Lake's core value proposition over plain Parquet: every write is an atomic transaction, schema mismatches are caught at write time (not read time), and schema changes are explicit and controlled.

### Phase 3: Time Travel & History (~20 min)

1. Run `uv run python src/02_time_travel.py`
2. **Exercise (`src/02_time_travel.py`):** After the boilerplate builds a 4-version table, you explore time travel by loading specific versions with `DeltaTable(path, version=N)` and `load_as_version()`, inspecting the complete audit trail with `history()`, and comparing data across versions. This teaches how Delta Lake's immutable versioning enables debugging data pipelines, auditing changes, and reproducing historical analyses -- capabilities that are impossible with plain Parquet.

### Phase 4: DML Operations (Merge/Update/Delete) (~20 min)

1. Run `uv run python src/03_dml_operations.py`
2. **Exercise (`src/03_dml_operations.py`):** Execute three DML operations on the orders table: a merge (upsert) that simultaneously updates existing orders and inserts new ones, an update that modifies rows matching a predicate, and a delete that removes rows. This is the key differentiator between Delta Lake and plain Parquet -- row-level mutations without rewriting entire datasets. The merge operation in particular is the workhorse of production data pipelines (CDC, slowly changing dimensions).

### Phase 5: Optimization & Cross-Engine Queries (~20 min)

1. Run `uv run python src/04_optimization.py`
2. **Exercise (`src/04_optimization.py`):** After the boilerplate creates a table with 15 small files (simulating streaming ingestion), you run compaction to merge them, Z-order by a high-cardinality column for better query performance, vacuum to reclaim disk space (and observe that time travel breaks for old versions), and create a partitioned table with `partition_by`. This teaches the operational side of Delta Lake -- how to maintain table health and query performance over time.

## Motivation

- **Production data engineering**: Delta Lake is the most widely adopted open lakehouse format. Understanding ACID tables, time travel, and DML operations is essential for building reliable data pipelines.
- **No JVM overhead**: delta-rs enables Delta Lake in lightweight Python/Rust environments without Spark infrastructure -- ideal for ETL scripts, microservices, and ML feature stores.
- **Multi-engine ecosystem**: The same Delta table can be queried by pandas, Polars, DuckDB, Spark, Trino, and Flink. Understanding the format means understanding the interoperability layer of the modern data stack.
- **Complements existing practices**: Builds on DataFrame knowledge (pandas/Polars) while introducing transactional guarantees, versioning, and storage optimization concepts that apply to any lakehouse technology (Iceberg, Hudi).

## Commands

All commands run from `practice_069_delta_lake/`.

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies from `pyproject.toml` |

### Exercises

| Command | Description |
|---------|-------------|
| `uv run python src/00_create_table.py` | Exercise 0: Create a Delta table and read it with pandas, Polars, DuckDB |
| `uv run python src/01_acid_and_schema.py` | Exercise 1: ACID transactions, schema enforcement, and schema evolution |
| `uv run python src/02_time_travel.py` | Exercise 2: Time travel by version, history audit log, version comparison |
| `uv run python src/03_dml_operations.py` | Exercise 3: Merge (upsert), update, and delete operations |
| `uv run python src/04_optimization.py` | Exercise 4: Compaction, Z-ordering, vacuum, and partitioned tables |

### Cleanup

| Command | Description |
|---------|-------------|
| `python clean.py` | Remove all generated data, caches, and virtual environments |

## References

- [delta-rs Documentation (delta-io.github.io)](https://delta-io.github.io/delta-rs/) -- Official docs for the Rust-native Delta Lake library
- [delta-rs Python API Reference](https://delta-io.github.io/delta-rs/api/delta_table/) -- DeltaTable class, write_deltalake, merge, optimize, vacuum
- [delta-rs Writing Delta Tables](https://delta-io.github.io/delta-rs/usage/writing/) -- write_deltalake modes, schema handling, partitioning
- [Delta Lake Protocol Specification](https://github.com/delta-io/delta/blob/master/PROTOCOL.md) -- Formal spec for the transaction log format
- [Delta Lake ACID Transactions (delta-rs)](https://delta-io.github.io/delta-rs/how-delta-lake-works/delta-lake-acid-transactions/) -- How optimistic concurrency control works
- [Understanding the Delta Lake Transaction Log (Databricks)](https://www.databricks.com/blog/2019/08/21/diving-into-delta-lake-unpacking-the-transaction-log.html) -- Deep dive into _delta_log internals
- [DuckDB Delta Extension](https://duckdb.org/docs/stable/core_extensions/delta) -- delta_scan() for SQL queries on Delta tables
- [Polars Delta Lake Integration](https://docs.pola.rs/api/python/stable/reference/api/polars.read_delta.html) -- pl.read_delta() and pl.scan_delta()
- [deltalake on PyPI](https://pypi.org/project/deltalake/) -- Package installation and version history

## State

`not-started`
