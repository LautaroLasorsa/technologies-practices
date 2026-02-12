# Practice 011a: Spark -- Core Transformations

## Technologies

- **Apache Spark 4.x** -- Distributed data processing engine with lazy evaluation and a Catalyst query optimizer
- **PySpark** -- Python API for Spark (SparkSession, DataFrames, SQL)
- **bitnami/spark** -- Lightweight Docker image for local Spark master + worker
- **Docker Compose** -- Orchestrates Spark cluster (1 master, 1 worker)

## Stack

- Python 3.11+ (uv)
- Docker / Docker Compose

## Theoretical Context

Apache Spark is a distributed data processing engine designed for batch and streaming analytics at scale. Spark's core innovation is **lazy evaluation combined with the Catalyst query optimizer**, which transforms user code into an optimized execution plan before running any computation.

**How Lazy Evaluation Works:**

In Spark, operations are divided into **transformations** (lazy operations that build a computation graph) and **actions** (eager operations that trigger execution). When you call `.filter()`, `.select()`, or `.join()`, Spark doesn't execute anything—it records the operation in a logical plan (a directed acyclic graph, or DAG). Only when you call an action like `.count()`, `.show()`, or `.write()` does Spark compile the entire chain of transformations into an optimized physical plan and execute it. This deferred execution enables powerful optimizations: Spark can reorder operations, combine stages, prune unnecessary columns, and push predicates down to data sources—optimizations impossible if each transformation executed immediately.

**The Catalyst Optimizer:**

Catalyst is Spark's extensible query optimizer, operating in four phases: (1) **Analysis** — resolve column references and table names against the catalog, (2) **Logical Optimization** — apply rule-based transformations like predicate pushdown (filter before join), constant folding (evaluate literals at compile-time), and projection pruning (read only needed columns), (3) **Physical Planning** — generate multiple physical plans (different join algorithms, partition strategies) and choose the best based on cost estimation, (4) **Code Generation** — compile the plan to JVM bytecode using Janino for runtime efficiency (whole-stage code generation).

**Narrow vs Wide Transformations:**

Transformations are classified by data movement. **Narrow transformations** (like `map`, `filter`, `withColumn`) operate on individual partitions independently—no data shuffles across the network. **Wide transformations** (like `groupBy`, `join`, `orderBy`) require redistributing data across executors (a **shuffle**), triggering stage boundaries. Shuffles are expensive (disk I/O, network transfer, serialization), so minimizing them is a key optimization strategy. The Catalyst optimizer tries to push narrow operations before wide ones (e.g., filter before join) to reduce shuffle volume.

**Key Concepts:**

| Concept | Description |
|---------|-------------|
| **DataFrame** | Distributed collection of rows with a schema (like a SQL table), the primary abstraction in Spark SQL |
| **Transformation** | Lazy operation that defines a new DataFrame from an existing one (e.g., `select`, `filter`, `join`) |
| **Action** | Eager operation that triggers computation and returns a result (e.g., `count`, `show`, `write`) |
| **DAG (Directed Acyclic Graph)** | Logical execution plan representing the sequence of transformations |
| **Catalyst Optimizer** | Rule-based + cost-based query optimizer that rewrites logical plans for efficiency |
| **Narrow Transformation** | No shuffle required (e.g., `map`, `filter`) — can pipeline within a single stage |
| **Wide Transformation** | Requires shuffle (e.g., `groupBy`, `join`) — triggers a new stage boundary |
| **Shuffle** | Redistributing data across partitions/executors over the network |
| **Partition** | A chunk of data processed by a single executor core (parallelism unit) |

**Ecosystem Context:**

Spark dominates large-scale batch processing in industry—used by virtually every data-intensive company (Uber, Netflix, Airbnb, etc.) for ETL, analytics, and ML pipelines. Databricks (founded by Spark's creators) offers managed Spark; AWS EMR, Google Dataproc, and Azure Synapse all run Spark. Alternatives exist: DuckDB and Polars for single-machine analytics, Trino/Presto for interactive SQL, Flink for low-latency streaming. But for "big data" batch workloads (100GB-100TB datasets), Spark remains the industry standard due to its mature ecosystem, unified API (SQL + DataFrame + RDD), and broad format support (Parquet, ORC, Delta Lake).

Sources: [Databricks: Catalyst Optimizer](https://www.databricks.com/glossary/catalyst-optimizer), [Deep Dive into Catalyst](https://www.databricks.com/blog/2015/04/13/deep-dive-into-spark-sqls-catalyst-optimizer.html), [Lazy Evaluation in Spark](https://medium.com/@john_tringham/spark-concepts-simplified-lazy-evaluation-d398891e0568)

## Description

Work through Spark's DataFrame API on a local single-worker cluster: loading CSV/JSON data, applying transformations (select, filter, withColumn, groupBy, agg, join), registering and querying SQL views, writing UDFs, and understanding lazy evaluation + the Catalyst optimizer. All exercises run against Docker-hosted Spark, so nothing external is required.

### What you'll learn

1. **SparkSession lifecycle** -- creating, configuring, and stopping a session; connecting from a local driver to a Docker-hosted master
2. **DataFrame creation** -- from CSV, JSON, and in-memory data; schema inference vs explicit schemas (StructType)
3. **Core transformations** -- select, filter/where, withColumn, drop, distinct, orderBy
4. **Aggregations** -- groupBy + agg (count, sum, avg, min, max), multiple aggregations in one call
5. **Joins** -- inner, left, right, cross; join conditions; handling ambiguous column names
6. **Spark SQL** -- createOrReplaceTempView, spark.sql(), mixing DataFrame and SQL APIs
7. **UDFs** -- registering Python UDFs with proper return types, performance tradeoffs vs built-in functions
8. **Lazy evaluation & Catalyst** -- explain() plans, narrow vs wide transformations, when shuffles happen

## Instructions

### Phase 1: Environment Setup (~10 min)

1. Start the Spark cluster: `docker compose up -d` from this folder
2. Verify the Spark UI at `http://localhost:8080` -- you should see 1 worker registered
3. Run `uv sync` to install pyspark
4. Run `uv run python src/00_connectivity_check.py` -- it should connect to `spark://localhost:7077` and print the Spark version

### Phase 2: DataFrames & Schema (~15 min)

1. Open `src/01_dataframes.py`
2. **User implements:** Load `data/sales.csv` with schema inference, then load `data/products.json` with an explicit StructType schema
3. **User implements:** Inspect the data -- printSchema(), show(), count(), describe()
4. Key question: When does Spark actually read the file -- at `spark.read.csv()` or at `.show()`?

### Phase 3: Core Transformations (~20 min)

1. Open `src/02_transformations.py`
2. **User implements:** select + withColumn to add a `total_price` column (quantity * unit_price)
3. **User implements:** filter rows where total_price > 100
4. **User implements:** orderBy total_price descending, then take the top 10
5. **User implements:** distinct product categories
6. Call `explain()` on the final DataFrame to see the physical plan
7. Key question: How many stages does `filter + orderBy` produce? Why does orderBy trigger a shuffle?

### Phase 4: Aggregations (~15 min)

1. Open `src/03_aggregations.py`
2. **User implements:** groupBy("category") with agg: total revenue (sum), average price, order count
3. **User implements:** groupBy("category", "region") with multiple aggregations in a single `.agg()` call
4. **User implements:** Filter grouped results (HAVING equivalent) -- only categories with total revenue > 500
5. Key question: What is the difference between `.count()` (action) and `F.count()` (aggregation function)?

### Phase 5: Joins (~15 min)

1. Open `src/04_joins.py`
2. **User implements:** Inner join sales with products on product_id
3. **User implements:** Left join to find sales with missing product info
4. **User implements:** Handle the ambiguous `product_id` column after join (alias or drop)
5. Key question: Why does Spark broadcast small DataFrames automatically? What is the broadcast threshold?

### Phase 6: Spark SQL (~10 min)

1. Open `src/05_spark_sql.py`
2. **User implements:** Register sales and products as temporary SQL views
3. **User implements:** Write a SQL query that joins, filters, groups, and orders -- the same logic from Phases 3-5 but in pure SQL
4. Key question: Does the Catalyst optimizer produce the same plan for equivalent DataFrame and SQL code?

### Phase 7: UDFs (~15 min)

1. Open `src/06_udfs.py`
2. **User implements:** A Python UDF that categorizes total_price into "low" / "medium" / "high" tiers
3. **User implements:** Register the UDF and apply it with withColumn
4. **User implements:** Register the same UDF for SQL usage and test it in a spark.sql() query
5. Key question: Why are Python UDFs slower than built-in functions? (Hint: serialization overhead between JVM and Python)

## Motivation

- **Industry standard**: Spark is the dominant engine for batch data processing at scale -- used at virtually every data-intensive company
- **AutoScheduler.AI relevance**: Large-scale supply chain optimization often involves ETL pipelines and analytical queries over millions of rows -- Spark is the go-to tool
- **Foundation for 011b**: This practice builds the core vocabulary needed for real data pipeline work in Practice 011b
- **Transferable mental model**: Lazy evaluation + query optimization concepts apply to Polars, DuckDB, Trino, and any query engine with a planner

## References

- [PySpark DataFrame API](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html)
- [Spark SQL Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [PySpark UDF Documentation](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.udf.html)
- [Catalyst Optimizer Overview](https://spark.apache.org/docs/latest/sql-programming-guide.html#catalyst-optimizer)
- [bitnami/spark Docker Image](https://hub.docker.com/r/bitnami/spark/)

## Commands

### Phase 1: Environment Setup

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start the Spark cluster (1 master + 1 worker) |
| `docker compose ps` | Verify both containers are running |
| `uv sync` | Install Python dependencies (pyspark) |
| `uv run python src/00_connectivity_check.py` | Verify driver connects to `spark://localhost:7077` and print Spark version |

### Phase 2: DataFrames & Schema

| Command | Description |
|---------|-------------|
| `uv run python src/01_dataframes.py` | Load sales.csv (inferred schema) and products.json (explicit StructType), inspect with printSchema/show/count/describe |

### Phase 3: Core Transformations

| Command | Description |
|---------|-------------|
| `uv run python src/02_transformations.py` | Run select, withColumn, filter, orderBy transformations and print explain() plan |

### Phase 4: Aggregations

| Command | Description |
|---------|-------------|
| `uv run python src/03_aggregations.py` | Run groupBy + agg (sum, avg, count) and HAVING-equivalent filter |

### Phase 5: Joins

| Command | Description |
|---------|-------------|
| `uv run python src/04_joins.py` | Run inner join, left join (find missing), aliased join, and print physical plan |

### Phase 6: Spark SQL

| Command | Description |
|---------|-------------|
| `uv run python src/05_spark_sql.py` | Register temp views, run SQL join+group+filter query, compare plans |

### Phase 7: UDFs

| Command | Description |
|---------|-------------|
| `uv run python src/06_udfs.py` | Define price_tier UDF, apply via withColumn and via spark.sql() |

### Teardown

| Command | Description |
|---------|-------------|
| `docker compose down` | Stop and remove the Spark cluster containers |
| `docker compose down -v` | Stop containers and remove volumes |

## State

`not-started`
