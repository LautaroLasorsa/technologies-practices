# Practice 056 — DuckDB: Embedded Analytics & Columnar Queries

## Technologies

- **DuckDB 1.x** — Embedded OLAP database engine ("SQLite for analytics")
- **duckdb** (Python package) — In-process Python client with DB-API and Relational API
- **Apache Arrow / PyArrow** — Zero-copy columnar interchange format
- **Pandas** — DataFrame library for comparison and integration exercises
- **Parquet** — Columnar file format for efficient analytical storage

## Stack

Python 3.12+ (uv)

## Theoretical Context

### What DuckDB Is & The Problem It Solves

DuckDB is an **embedded OLAP (Online Analytical Processing) database** — often called "the SQLite for analytics." Like SQLite, it runs **in-process** (no separate server, no network, no configuration), but unlike SQLite, it is optimized for **analytical queries** (aggregations, joins, window functions over large datasets) rather than transactional workloads (many small reads/writes).

The problem: data analysts and engineers commonly face a gap between "too big for Pandas" and "too small to justify Spark/Redshift." Setting up PostgreSQL, Spark, or a cloud warehouse for exploratory analytics on a 500MB Parquet file is overkill. Pandas loads everything into memory and processes row-at-a-time, which is slow for analytical queries on datasets that exceed a few hundred megabytes. DuckDB fills this gap: you `import duckdb` and immediately run SQL against local files, DataFrames, or an embedded database — with performance that often exceeds Pandas by 5-50x for analytical workloads.

### Row-Store vs Column-Store: Why Columnar Wins for Analytics

Traditional databases (PostgreSQL, MySQL, SQLite) use **row-oriented storage**: all columns of a single row are stored contiguously on disk/memory.

```
Row store:  [id=1, name="Alice", amount=100] [id=2, name="Bob", amount=200] ...
```

Columnar databases (DuckDB, ClickHouse, Redshift) store each **column** contiguously:

```
Col store:  id:     [1, 2, 3, 4, ...]
            name:   ["Alice", "Bob", "Carol", ...]
            amount: [100, 200, 150, 300, ...]
```

Why columnar is faster for analytics:

1. **Selective column reads**: `SELECT AVG(amount) FROM sales` only reads the `amount` column, skipping `id`, `name`, and every other column. Row stores must read entire rows even if only one column is needed.
2. **Cache efficiency**: Contiguous same-type values pack tightly into CPU cache lines. Processing a vector of integers is far faster than jumping between different-typed fields of row tuples.
3. **Compression**: Same-type contiguous data compresses extremely well (run-length encoding, dictionary encoding, bit-packing). A column of country codes might compress 10:1, while a row store with mixed types gets much less compression.
4. **SIMD vectorization**: Modern CPUs have Single Instruction Multiple Data instructions that can process 4-16 values simultaneously. Columnar layout feeds these instructions naturally.

**Trade-off**: Row stores are better for OLTP (transactional) workloads — inserting a single row is one write to one location, whereas a columnar store must write to N locations (one per column). DuckDB is not meant for high-concurrency transactional workloads.

### Vectorized Execution Engine

Traditional SQL engines process data **one row at a time** (Volcano/iterator model). Each operator (`scan`, `filter`, `aggregate`) processes a single tuple, passes it up, and the next operator processes that same single tuple. This has enormous per-row overhead: virtual function calls, branch mispredictions, poor cache utilization.

DuckDB uses **vectorized execution**: operators process **batches of 2,048 tuples** (called "vectors") at a time. Instead of passing one row through the operator pipeline, DuckDB passes a vector of column values. A `filter` operator applies the predicate to 2,048 values in a tight loop, producing a selection vector. The `aggregate` operator then processes 2,048 values using the selection vector.

Benefits of vectorized execution:
- **Amortized overhead**: Function call overhead is paid once per 2,048 rows, not once per row
- **Cache locality**: Processing a batch of same-type values keeps data in L1/L2 cache
- **SIMD friendly**: Tight loops over typed arrays enable auto-vectorization by the compiler
- **Branch prediction**: Processing uniform batches has predictable branch patterns

DuckDB combines vectorized execution with **morsel-driven parallelism**: the data is divided into "morsels" (chunks of row groups), and multiple threads each process their own morsels independently, then merge results. This achieves near-linear scaling with CPU cores.

The execution model is also **push-based**: operators push result vectors to the next operator (rather than the top operator pulling from below), which simplifies parallelism and enables adaptive scheduling.

### Zero-Copy Integration with the Python Ecosystem

One of DuckDB's killer features is **zero-copy data exchange** with Python data structures:

- **Pandas DataFrames**: DuckDB can query a Pandas DataFrame directly using SQL — no data copying. The DataFrame's underlying NumPy/Arrow arrays are read in-place via "replacement scans." Just reference the Python variable name in SQL: `duckdb.sql("SELECT * FROM my_dataframe")`.
- **Apache Arrow**: DuckDB's internal format is Arrow-compatible. Converting query results to Arrow (`result.arrow()`) is zero-copy, and querying Arrow tables is zero-copy in the other direction.
- **Parquet files**: DuckDB reads Parquet files directly (metadata-driven, column pruning, predicate pushdown). No need to load into memory first — `SELECT * FROM 'data.parquet'` just works.

### When to Use DuckDB vs Alternatives

| Scenario | Best Tool | Why |
|----------|-----------|-----|
| Analytical queries on local files (CSV, Parquet) | **DuckDB** | Zero setup, fast columnar processing, SQL interface |
| Small-medium DataFrames (< 100MB) | **Pandas** | Familiar API, rich ecosystem, adequate performance |
| Terabyte-scale distributed processing | **Spark** | Distributed execution across cluster |
| OLTP: many concurrent reads/writes | **PostgreSQL** | Row-store optimized for transactions, ACID, concurrency |
| Embedded transactional database | **SQLite** | Row-store, great for OLTP, poor for analytics |
| Real-time streaming analytics | **ClickHouse** | Server-based, optimized for insert-heavy analytical workloads |

DuckDB is ideal for: data exploration, feature engineering, ETL scripts, CI/CD data validation, Jupyter notebooks, CLI data analysis, and anywhere you need fast SQL on local data without infrastructure.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Columnar Storage** | Data stored column-by-column (not row-by-row), enabling selective reads, compression, and vectorized processing. DuckDB uses row groups of ~120K tuples internally. |
| **Vectorized Execution** | Processing batches of 2,048 values per operator instead of one row at a time. Reduces per-row overhead by ~1000x compared to Volcano model. |
| **In-Process** | Runs inside your application process (like SQLite). No server, no network, no configuration. `import duckdb` and go. |
| **Zero-Copy** | DuckDB reads Pandas/Arrow data without copying it. Query results convert to Arrow without copying. Eliminates serialization overhead. |
| **Parquet Integration** | Native ability to query Parquet files directly with predicate pushdown and column pruning — no import step needed. |
| **Replacement Scans** | Mechanism that allows DuckDB to resolve Python variable names (e.g., DataFrame names) as virtual tables in SQL queries. |
| **Morsel-Driven Parallelism** | Data is split into morsels; threads independently process morsels and merge results. Near-linear core scaling. |
| **QUALIFY** | SQL clause unique to analytical databases (DuckDB, Snowflake, BigQuery) that filters window function results — like HAVING for aggregates. Eliminates subquery wrapping. |
| **httpfs Extension** | DuckDB extension for reading files over HTTP/S3 — query remote Parquet files without downloading them first. |
| **Relation API** | Python API for building queries programmatically by chaining methods (filter, project, aggregate, join) on relation objects. Lazily evaluated. |

## Description

Build an analytics pipeline demonstrating DuckDB's core strengths: querying Parquet and CSV files directly without import, writing advanced analytical SQL (window functions, CTEs, QUALIFY), integrating with the Python data ecosystem (Pandas, Arrow), and benchmarking DuckDB against Pandas for typical analytics operations.

### What you'll learn

1. **DuckDB basics** — creating/querying tables, loading data, SQL dialect features
2. **Analytical SQL** — window functions (ROW_NUMBER, RANK, LAG/LEAD, running totals), CTEs, QUALIFY clause
3. **File-based querying** — query Parquet/CSV directly, glob patterns, httpfs extension
4. **Python integration** — query DataFrames with SQL, export to Arrow, Relation API
5. **Performance** — benchmark DuckDB vs Pandas for aggregations, joins, and window operations

## Instructions

### Phase 1: Setup & Data Generation (~10 min)

1. Initialize the Python project: `cd app && uv sync`
2. Generate sample data: `uv run python src/generate_data.py`
   - This creates ~100K rows of sales/products/stores data as both CSV and Parquet files in `data/`
3. Verify: check that `data/` contains `sales.parquet`, `sales.csv`, `products.csv`, `stores.csv`, and partitioned Parquet files under `data/sales_by_year/`
4. Key question: Why does DuckDB not need a server? What does "in-process" mean compared to PostgreSQL?

### Phase 2: DuckDB Basics (~20 min)

1. Open `src/basics.py` and review the scaffolding
2. **Exercise 1 — Create and populate tables:** Create `products`, `stores`, and `sales` tables by loading from CSV files. Use `CREATE TABLE ... AS SELECT * FROM read_csv('...')`. This teaches DuckDB's ability to infer schemas from files and the `read_csv` auto-detection.
3. **Exercise 2 — Basic queries:** Write SELECT queries with JOINs (sales + products + stores), GROUP BY aggregations (total revenue per category), and filtering with HAVING. This teaches DuckDB's standard SQL support and columnar query speed.
4. **Exercise 3 — DuckDB-specific features:** Use `DESCRIBE`, `SUMMARIZE`, `EXPLAIN` to inspect tables and query plans. This teaches how to understand what DuckDB is doing under the hood — critical for optimization.
5. Run: `uv run python src/basics.py`

### Phase 3: Analytical Queries (~25 min)

1. Open `src/analytics.py` and review the scaffolding
2. **Exercise 1 — Window functions:** Implement ROW_NUMBER, RANK, DENSE_RANK partitioned by category, and LAG/LEAD for month-over-month comparisons. Window functions are the bread-and-butter of analytical SQL — they compute values across related rows without collapsing the result set (unlike GROUP BY).
3. **Exercise 2 — Running aggregates:** Compute running totals, moving averages (3-month window), and cumulative percentages using window frames (ROWS BETWEEN). This teaches frame specifications — the most powerful (and confusing) part of window functions.
4. **Exercise 3 — CTEs and QUALIFY:** Write recursive and non-recursive CTEs, then use QUALIFY to filter window results directly. QUALIFY eliminates the common pattern of wrapping a window query in a CTE just to filter on rank — it is to window functions what HAVING is to GROUP BY.
5. Run: `uv run python src/analytics.py`

### Phase 4: File Queries (~20 min)

1. Open `src/file_queries.py` and review the scaffolding
2. **Exercise 1 — Direct Parquet/CSV querying:** Query Parquet and CSV files directly without creating tables. Use `read_parquet()` and `read_csv()` with schema detection. This teaches DuckDB's zero-ETL philosophy — analyze data where it lives.
3. **Exercise 2 — Glob patterns and partitioned data:** Query multiple files using glob patterns (`data/sales_by_year/*.parquet`) and use the `filename` column to track source files. This teaches how DuckDB handles partitioned datasets (Hive-style and filename-based).
4. **Exercise 3 — httpfs and remote files:** Install the httpfs extension and query a public Parquet file over HTTPS. This teaches DuckDB's ability to query remote data without downloading — useful for data lakes and public datasets.
5. Run: `uv run python src/file_queries.py`

### Phase 5: Python Integration (~20 min)

1. Open `src/python_integration.py` and review the scaffolding
2. **Exercise 1 — Query Pandas DataFrames:** Create a Pandas DataFrame and query it with DuckDB SQL (zero-copy). Convert results back to DataFrame. This teaches the seamless Pandas integration that makes DuckDB a "SQL engine for DataFrames."
3. **Exercise 2 — Arrow interchange:** Query data and export to Arrow tables. Convert Arrow to Pandas and back. This teaches the zero-copy Arrow pathway — the fastest way to move data between DuckDB and Python.
4. **Exercise 3 — Relation API:** Use `duckdb.sql()` and relation methods (`.filter()`, `.project()`, `.aggregate()`, `.order()`, `.limit()`) to build queries programmatically. This teaches the alternative to raw SQL strings — useful for dynamic query construction.
5. Run: `uv run python src/python_integration.py`

### Phase 6: Performance Comparison (~15 min)

1. Open `src/performance.py` and review the benchmarking scaffold
2. **Exercise 1 — Aggregation benchmark:** Compare DuckDB vs Pandas for GROUP BY aggregation on the full sales dataset. Time both approaches. This teaches why columnar + vectorized execution matters for analytical queries.
3. **Exercise 2 — Join benchmark:** Compare DuckDB vs Pandas for a 3-table join (sales + products + stores) with aggregation. This teaches how DuckDB's query optimizer and hash joins outperform Pandas merge chains.
4. **Exercise 3 — Window function benchmark:** Compare DuckDB vs Pandas for computing running totals and rankings. This teaches the performance gap for complex analytical operations where Pandas has no native SQL-style window.
5. Run: `uv run python src/performance.py`

### Phase 7: Reflection (~5 min)

1. Compare DuckDB with PostgreSQL (practice 023) — when would you use each?
2. Discuss: How does DuckDB's approach compare to Spark for local analytics? At what data size does Spark become necessary?
3. Consider: Where could DuckDB fit in your current data pipelines?

## Motivation

- **Modern analytics**: DuckDB adoption is rapidly growing in data engineering, ML feature pipelines, and embedded analytics (1M+ monthly downloads on PyPI)
- **Gap filler**: Fills the gap between "too big for Pandas" and "too small for Spark" — a common real-world scenario
- **SQL proficiency**: Advanced analytical SQL (window functions, CTEs, QUALIFY) is a transferable skill applicable to any analytical database
- **Complements other practices**: Pairs with PostgreSQL (023) for OLTP vs OLAP understanding, and Spark (011) for understanding when distributed processing is truly needed
- **Zero infrastructure**: No Docker, no server, no config — just `import duckdb`. Ideal for CI/CD pipelines, notebooks, and scripting
- **Industry trend**: Increasingly adopted by companies for local analytics, data validation, and ETL preprocessing (MotherDuck, dbt integration, Delta Lake support)

## Commands

All commands run from `practice_056_duckdb/app/`.

### Project Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies (duckdb, pandas, pyarrow) |
| `uv run python src/generate_data.py` | Generate sample sales/products/stores data as CSV and Parquet files |

### Phase 2: Basics

| Command | Description |
|---------|-------------|
| `uv run python src/basics.py` | Run table creation, basic queries, and DuckDB introspection exercises |

### Phase 3: Analytical Queries

| Command | Description |
|---------|-------------|
| `uv run python src/analytics.py` | Run window functions, running aggregates, CTEs, and QUALIFY exercises |

### Phase 4: File Queries

| Command | Description |
|---------|-------------|
| `uv run python src/file_queries.py` | Run direct Parquet/CSV querying, glob patterns, and httpfs exercises |

### Phase 5: Python Integration

| Command | Description |
|---------|-------------|
| `uv run python src/python_integration.py` | Run Pandas integration, Arrow interchange, and Relation API exercises |

### Phase 6: Performance

| Command | Description |
|---------|-------------|
| `uv run python src/performance.py` | Run DuckDB vs Pandas benchmarks for aggregation, joins, and windows |

## References

- [DuckDB Documentation](https://duckdb.org/docs/)
- [DuckDB Python API Overview](https://duckdb.org/docs/stable/clients/python/overview)
- [DuckDB Relational API](https://duckdb.org/docs/stable/clients/python/relational_api)
- [DuckDB SQL on Pandas](https://duckdb.org/docs/stable/guides/python/sql_on_pandas)
- [DuckDB Window Functions](https://duckdb.org/docs/stable/sql/functions/window_functions)
- [DuckDB QUALIFY Clause](https://duckdb.org/docs/stable/sql/query_syntax/qualify)
- [DuckDB Parquet Reading](https://duckdb.org/docs/stable/data/parquet/overview)
- [DuckDB Reading Multiple Files](https://duckdb.org/docs/stable/data/multiple_files/overview)
- [DuckDB Vectorized Execution Internals](https://duckdb.org/docs/stable/internals/vector)
- [CMU 15-721 DuckDB Lecture (2024)](https://15721.courses.cs.cmu.edu/spring2024/slides/20-duckdb.pdf)

## State

`not-started`
