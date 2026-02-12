# Practice 011b: Spark — Real Data Pipeline

## Technologies

- **Apache Spark 3.5** — Distributed data processing engine (local mode via Docker)
- **PySpark** — Python API for Spark (DataFrame API, SparkSQL, Window Functions)
- **Bitnami Spark Docker** — Pre-built Spark master/worker containers
- **Parquet** — Columnar storage format for analytical workloads

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Theoretical Context

Spark ETL pipelines orchestrate data extraction, transformation, and loading at scale using Spark's DataFrame API and SparkSQL. This practice builds on Spark fundamentals (011a) to implement production-grade patterns: multi-source ingestion, data cleaning, broadcast joins, window functions for time-series analytics, and partitioned Parquet output.

**ETL vs ELT:**

Traditional ETL (Extract-Transform-Load) transforms data before loading into the warehouse. Modern cloud systems often use ELT (Extract-Load-Transform), loading raw data first and transforming in the warehouse. Spark excels at both: it can transform complex data before loading (ETL) or read from data lakes and transform on-demand (ELT). This practice follows ETL: read CSV/JSON, clean and enrich in Spark, then write optimized Parquet.

**Broadcast Joins:**

Spark joins are expensive when both sides are large (requires shuffling both datasets). A **broadcast join** optimizes the common case of large-table-join-small-table by sending the small table to every executor, eliminating the shuffle on the large side. Spark auto-broadcasts tables under 10MB (configurable via `spark.sql.autoBroadcastJoinThreshold`), but you can hint explicitly with `F.broadcast(small_df)`. Broadcast joins are critical for dimension-table enrichment (joining fact tables with lookup tables).

**Window Functions:**

Window functions perform calculations across a "window" of rows related to the current row, without collapsing groups like `groupBy`. The window is defined by partitioning (which rows to group) and ordering (how to arrange them within each partition). Key functions: `row_number()` (unique sequential rank), `rank()` (ties share rank, gaps after), `dense_rank()` (ties share rank, no gaps), `lag()`/`lead()` (access previous/next row), and aggregations like `sum().over(window)` (running totals). Window functions are essential for time-series analysis, rankings, and period-over-period comparisons—common in business analytics.

**Parquet and Partitioning:**

Parquet is a columnar storage format optimized for analytics—it stores data by column instead of by row, enabling efficient reads when querying subsets of columns (select 3 out of 100 columns reads only 3% of data). Parquet also compresses well (Snappy, Gzip) and supports predicate pushdown (Spark skips reading chunks that don't match filters). **Partition pruning** takes this further: writing Parquet with `.partitionBy("year", "month")` creates a folder hierarchy (`year=2024/month=01/`) where Spark skips entire folders when queries filter on partition columns (`WHERE year=2024 AND month=1`).

**Key Concepts:**

| Concept | Description |
|---------|-------------|
| **ETL Pipeline** | Extract data from sources, transform (clean/enrich/aggregate), load into target storage |
| **Broadcast Join** | Optimization where a small table is sent to all executors, avoiding shuffle on the large side |
| **Window Function** | Computation over a "window" of rows (partitioned/ordered) without collapsing groups |
| **Partition (Window)** | Subset of rows for a window function (e.g., per category) |
| **Ordering (Window)** | Sort order within each partition (e.g., by date) |
| **Parquet** | Columnar storage format with compression, predicate pushdown, and schema evolution support |
| **Partition Pruning** | Skipping entire folders/files based on partition column filters (e.g., skip non-matching years) |
| **Caching** | Materializing a DataFrame in memory/disk to reuse across multiple actions |

**Ecosystem Context:**

Spark ETL pipelines are the backbone of data platforms at scale. They power data lakes (Databricks Delta Lake, AWS Lake Formation), warehouses (Snowflake ingestion, BigQuery ETL), and ML feature pipelines (Feast, Tecton). Window functions are a standard SQL feature (Postgres, Oracle, etc.), but Spark's distributed implementation lets you run them on petabyte-scale data. Parquet is the de facto standard for big data storage (used by AWS Athena, Google BigQuery, Snowflake, Delta Lake, Iceberg). Understanding Spark ETL patterns transfers directly to tools like dbt (SQL-based transformations), Databricks workflows, and Airflow-orchestrated pipelines.

Sources: [PySpark DataFrame API](https://spark.apache.org/docs/3.5.0/api/python/reference/pyspark.sql/dataframe.html), [Broadcast Joins](https://sparkbyexamples.com/pyspark/pyspark-broadcast-join-with-example/), [Window Functions](https://sparkbyexamples.com/pyspark/pyspark-window-functions/)

## Description

Build a complete **retail analytics ETL pipeline** that reads sales transactions (CSV) and a product catalog (JSON), cleans and joins the data, computes business metrics using window functions and SparkSQL, then writes optimized Parquet output partitioned by date.

### What you'll learn

1. **Multi-source ingestion** — reading CSV and JSON with explicit schemas
2. **Data cleaning** — handling nulls, deduplication, type casting, filtering invalid rows
3. **Broadcast joins** — joining a large fact table (sales) with a small dimension table (products) efficiently
4. **Window functions** — `row_number`, `rank`, `lag`, `sum().over()` for running totals, rankings, and period-over-period comparisons
5. **SparkSQL** — registering temp views and running SQL queries alongside DataFrame API
6. **Caching & persistence** — when and why to cache intermediate DataFrames, and how to unpersist
7. **Parquet output** — writing partitioned columnar data with compression
8. **Performance tuning** — reading explain plans, understanding shuffles, broadcast hints

## Instructions

### Phase 1: Setup & Spark Basics (~10 min)

1. Start the Spark cluster with `docker compose up -d`
2. Run `uv sync` inside `app/`
3. Run the pipeline: `uv run python -m pipeline.main`
4. Verify the Spark Master UI at `http://localhost:8080`
5. Key concept: Spark's lazy evaluation — transformations build a DAG, actions trigger execution

### Phase 2: Ingestion & Cleaning (~20 min)

1. **User implements:** `extract.py` — read `sales.csv` with explicit schema, read `products.json`
2. **User implements:** `clean.py` — drop nulls in critical columns, deduplicate by `transaction_id`, cast types, filter rows with `quantity <= 0` or `unit_price <= 0`
3. Verify: print schema and row counts before/after cleaning
4. Key question: Why define schemas explicitly instead of letting Spark infer them?

### Phase 3: Joins & Enrichment (~15 min)

1. **User implements:** `enrich.py` — broadcast join sales with products on `product_id`, compute `total_amount = quantity * unit_price`, add `profit = total_amount * margin`
2. Verify: check that the joined DataFrame has no null `category` values (inner join semantics)
3. Key question: When would a broadcast join hurt performance instead of help?

### Phase 4: Window Functions & Analytics (~25 min)

1. **User implements:** `analytics.py` — four analytical computations:
   - **Daily revenue by category** with running total using `sum().over(window)`
   - **Top-3 products per category by revenue** using `row_number().over(window)`
   - **Day-over-day revenue change** using `lag().over(window)`
   - **Customer lifetime value ranking** using `rank().over(window)`
2. Key question: What's the difference between `row_number`, `rank`, and `dense_rank`?

### Phase 5: SparkSQL (~15 min)

1. **User implements:** `sql_queries.py` — register temp views and write equivalent SQL for:
   - Monthly revenue summary grouped by category
   - Products with above-average revenue (subquery)
   - Customer purchase frequency segmentation (CASE WHEN)
2. Key question: When would you prefer SQL over DataFrame API (or vice versa)?

### Phase 6: Output & Performance (~15 min)

1. **User implements:** `load.py` — write enriched data to Parquet partitioned by `sale_year` and `sale_month`, with snappy compression
2. **User implements:** `main.py` — add caching for the cleaned DataFrame (used multiple times), call `explain(True)` on the enriched DataFrame to inspect the plan, unpersist at the end
3. Verify: check output directory for partitioned Parquet files
4. Key question: What does a "shuffle" in the explain plan mean, and how can you minimize them?

## Motivation

- **Industry-standard tool**: Spark is the dominant engine for batch data processing in data engineering roles (Databricks, EMR, Dataproc)
- **ETL pattern literacy**: Extract-Transform-Load is the most common data pipeline pattern; mastering it in Spark transfers to any data platform
- **SQL + DataFrame duality**: Real teams mix SQL and programmatic APIs; understanding both and when to use each is a practical skill
- **Window functions**: One of the most powerful and frequently tested SQL/data concepts in interviews and production analytics
- **Performance awareness**: Understanding broadcast joins, caching, and explain plans separates junior from senior data engineers

## References

- [PySpark DataFrame API — Spark 3.5 Docs](https://spark.apache.org/docs/3.5.0/api/python/reference/pyspark.sql/index.html)
- [SparkSQL Guide](https://spark.apache.org/docs/3.5.0/sql-programming-guide.html)
- [Window Functions — Spark Docs](https://spark.apache.org/docs/3.5.0/sql-ref-syntax-qry-select-window.html)
- [Broadcast Joins — sparkbyexamples.com](https://sparkbyexamples.com/pyspark/pyspark-broadcast-join-with-example/)
- [PySpark Window Functions Guide](https://sparkbyexamples.com/pyspark/pyspark-window-functions/)
- [Bitnami Spark Docker](https://hub.docker.com/r/bitnami/spark/)
- [Spark Performance Tuning Guide](https://spark.apache.org/docs/3.5.0/sql-performance-tuning.html)

## Commands

### Phase 1: Setup

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start the Spark cluster (1 master + 1 worker with volumes mounting `./app`) |
| `docker compose ps` | Verify both containers (spark-master, spark-worker) are running |
| `cd app && uv sync` | Install Python dependencies (pyspark) inside the `app/` directory |

### Phase 2-6: Run the Full Pipeline

| Command | Description |
|---------|-------------|
| `cd app && uv run python -m pipeline.main` | Execute the full ETL pipeline (extract, clean, enrich, analytics, SQL, parquet output) |

### Verification & Inspection

| Command | Description |
|---------|-------------|
| Open `http://localhost:8080` in browser | Spark Master Web UI -- verify worker registration and running applications |
| Open `http://localhost:8081` in browser | Spark Worker Web UI -- inspect executor details |
| `ls app/output/enriched_sales/` | Verify partitioned Parquet output directory structure (`sale_year=*/sale_month=*/`) |

### Teardown

| Command | Description |
|---------|-------------|
| `docker compose down` | Stop and remove the Spark cluster containers |
| `docker compose down -v` | Stop containers and remove volumes |

## State

`not-started`
