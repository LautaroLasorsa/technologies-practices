# Practice 023: Advanced SQL — Window Functions, CTEs & Query Optimization

## Technologies

- **PostgreSQL 17** — World's most advanced open-source relational database
- **psycopg 3** — Modern Python PostgreSQL adapter with Row Factories and native async
- **Docker** — Containerized PostgreSQL instance for reproducible local development

## Stack

- Python 3.12+ (uv)
- PostgreSQL 17 (Docker)

## Theoretical Context

### What are window functions and CTEs, and what problems do they solve?

**Window functions** and **CTEs (Common Table Expressions)** are advanced SQL features that solve the problem of **complex analytical queries without procedural code**. Before these features, calculating running totals, ranking within groups, or hierarchical recursion required either multiple self-joins (slow, hard to read) or moving logic into application code (loses database optimization). Window functions and CTEs express these operations declaratively in SQL, letting the query optimizer generate efficient execution plans.

**Window functions** operate on a "window" of rows related to the current row, enabling calculations like:
- Ranking (ROW_NUMBER, RANK, DENSE_RANK)
- Offsets (LAG, LEAD for previous/next row values)
- Running aggregates (cumulative SUM, rolling AVG)

**CTEs** organize complex queries into named, reusable subqueries. Recursive CTEs traverse hierarchical data (org charts, category trees) by referencing themselves.

### How they work internally

**Window function execution model:**

1. **Partition**: The query planner groups rows into partitions based on `PARTITION BY` columns (like GROUP BY, but doesn't collapse rows).
2. **Order**: Within each partition, rows are sorted by `ORDER BY` (defines the window frame's ordering).
3. **Frame**: For each row, a **window frame** (e.g., "all rows from partition start to current row" or "3 rows before to 3 rows after") is defined via `ROWS BETWEEN` or `RANGE BETWEEN`.
4. **Compute**: The window function (SUM, RANK, LAG, etc.) computes its result over the frame.
5. **Output**: Each input row produces one output row with the window function result appended — no collapsing like GROUP BY.

**Example:** `SUM(revenue) OVER (PARTITION BY employee_id ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)` computes a **running total** of revenue per employee, month by month.

**CTE execution model:**

- **Non-recursive CTE**: The database executes the CTE query first, materializes the result into a temporary table, then uses it in the main query. Acts like a named subquery or temp table, improving readability.
- **Recursive CTE**: Executes in two phases:
  1. **Base case** (non-recursive term): Computes the initial rows (e.g., the root of a tree).
  2. **Recursive step**: Repeatedly self-joins the working table with the CTE's recursive term until no new rows are produced (termination condition).

  Internally, the database maintains a working table that grows iteratively. Termination is guaranteed if each iteration filters out already-visited rows (e.g., via a JOIN condition that prevents infinite cycles).

**Query optimizer considerations:**

- **Window functions**: The optimizer may push down filters before partitioning, reorder window functions if they share PARTITION BY/ORDER BY, and use sorting/hashing optimizations.
- **CTEs**: In PostgreSQL 12+, CTEs are inline-able (optimizer can merge them into the main query) unless prefixed with `MATERIALIZED`. In older versions, CTEs were always materialized (optimization fence).

### Key concepts

| Concept | Description |
|---------|-------------|
| **Window function** | A function that operates on a set of rows (a "window") related to the current row without collapsing rows like GROUP BY |
| **PARTITION BY** | Divides rows into partitions; window function operates independently within each partition |
| **ORDER BY (window)** | Defines the ordering within each partition; affects frame boundaries and rank calculations |
| **Frame clause** | `ROWS/RANGE BETWEEN ... AND ...` defines which rows in the partition are included in the window for each row |
| **ROW_NUMBER vs RANK vs DENSE_RANK** | ROW_NUMBER: unique sequential numbering; RANK: same rank for ties, gaps after; DENSE_RANK: same rank for ties, no gaps |
| **LAG / LEAD** | Access the value from a previous/next row within the partition, offset by N rows |
| **CTE (Common Table Expression)** | A named temporary result set defined with `WITH`, used to simplify complex queries |
| **Recursive CTE** | A CTE that references itself, used for hierarchical or graph traversal (e.g., org charts, bill-of-materials) |
| **LATERAL JOIN** | A JOIN where the right side can reference columns from the left side; enables correlated subqueries in FROM clause |
| **EXPLAIN ANALYZE** | Shows the query execution plan with actual runtime statistics (cost, rows, time) for optimization analysis |

### Ecosystem context

**Alternatives and trade-offs:**

- **Self-joins for ranking**: Before window functions, ranking within groups required self-joins counting rows with higher/equal values. Slow (O(n²)), hard to maintain.
- **Application-side processing**: Moving aggregation logic to Python/Java. Loses database optimizations, requires full data transfer.
- **Procedural SQL (PL/pgSQL, stored procedures)**: Can implement recursion and complex logic, but procedural code is harder to optimize than declarative SQL.
- **OLAP cubes / BI tools**: Pre-aggregate common queries (data cubes, materialized views). Fast reads, but stale data and rigid schema.

**When to use window functions and CTEs:**

- **Window functions**: Analytics queries (running totals, top-N-per-group, time-series analysis, percentiles)
- **Recursive CTEs**: Hierarchical data (org charts, category trees, graph traversal, bill-of-materials)
- **Non-recursive CTEs**: Improving readability of complex multi-step queries (replace nested subqueries)

**Limitations:**

- **Window functions**: Can be expensive on large datasets without proper indexing (requires sorting within partitions).
- **Recursive CTEs**: Unbounded recursion risks infinite loops (always include a termination condition). Performance degrades on deep hierarchies without indexing.
- **Portability**: Window function syntax is standardized (SQL:2003), but frame clauses and specific functions vary across databases (PostgreSQL, MySQL, SQL Server).

## Description

Build a series of increasingly complex SQL queries against a Northwind-inspired trading company database. Practice window functions (ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, NTILE, running aggregates), CTEs (recursive and non-recursive), query optimization (EXPLAIN ANALYZE, indexing), advanced JOINs (LATERAL, self-joins), and PostgreSQL-specific features (GENERATE_SERIES, array operations, JSON aggregation).

All data stored in a Dockerized PostgreSQL 17 instance. Python (psycopg3) used as the query execution layer — focus stays on SQL, not Python.

### What you'll learn

1. **Window functions** — partition, rank, offset, and aggregate across rows without GROUP BY
2. **CTEs** — organize complex queries and traverse hierarchical data with recursive CTEs
3. **Query optimization** — read EXPLAIN ANALYZE plans, create indexes, measure speedups
4. **LATERAL joins** — correlated subqueries in FROM clause for top-N-per-group patterns
5. **PostgreSQL features** — GENERATE_SERIES for date dimensions, array_agg, jsonb_build_object

## Instructions

### Phase 1: Setup & Data Loading (~10 min)

1. Start PostgreSQL: `docker compose up -d`
2. Install dependencies: `uv sync`
3. Seed the database: `uv run python -m app.seed`
4. Verify tables: `docker compose exec postgres psql -U practice -d northwind -c '\dt'`
5. Explore the data: connect with `psql` and run a few `SELECT COUNT(*)` queries to understand the dataset

### Phase 2: Window Functions (~25 min)

Open `app/queries.py` and work through the window function section:

1. **User implements:** `window_rank_products_by_price()` — ROW_NUMBER, RANK, DENSE_RANK comparing behavior on ties
2. **User implements:** `window_top3_per_supplier()` — DENSE_RANK + CTE to filter top N per group
3. **User implements:** `window_lag_monthly_growth()` — LAG for month-over-month growth calculation
4. **User implements:** `window_running_total()` — SUM with frame clause for cumulative revenue per employee
5. **User implements:** `window_ntile_quartiles()` — NTILE(4) to bucket customers into spending quartiles
6. Key question: When would you use RANK vs DENSE_RANK vs ROW_NUMBER? What happens with ties in each?

### Phase 3: CTEs (~20 min)

Continue in `app/queries.py` with the CTE section:

1. **User implements:** `cte_high_value_customers()` — Multi-step non-recursive CTE for customer segmentation
2. **User implements:** `cte_recursive_employee_hierarchy()` — Recursive CTE traversing the reporting tree
3. **User implements:** `cte_recursive_category_tree()` — Recursive CTE traversing the category tree
4. Key question: How does WITH RECURSIVE terminate? What prevents infinite loops?

### Phase 4: Query Optimization (~20 min)

Continue in `app/queries.py` with the optimization section:

1. **User implements:** `optimization_slow_query()` — Run a date-range query, inspect EXPLAIN ANALYZE output
2. **User implements:** `optimization_create_index()` — Create a B-tree index, re-run, compare execution plans
3. **User implements:** `optimization_composite_index()` — Create a composite index, understand leftmost prefix rule
4. Key question: Why doesn't PostgreSQL always use an index even when one exists? (Hint: cost-based optimizer)

### Phase 5: Advanced Joins & PostgreSQL Features (~20 min)

Final section in `app/queries.py`:

1. **User implements:** `lateral_top3_per_customer()` — LATERAL JOIN for top-N-per-group
2. **User implements:** `selfjoin_products_ordered_together()` — Self-join for market basket analysis
3. **User implements:** `generate_series_date_gaps()` — GENERATE_SERIES to fill date gaps in time series
4. **User implements:** `json_aggregation()` — jsonb_build_object + jsonb_agg for JSON responses
5. Key question: When is LATERAL better than a window function for top-N-per-group?

## Motivation

- **Data engineering essential**: Window functions and CTEs are the #1 most tested skill in data engineering interviews
- **Separates junior from senior**: Most developers can SELECT/JOIN but few can write optimized window functions or recursive CTEs
- **Directly applicable**: AutoScheduler.AI uses PostgreSQL — these patterns improve production query quality
- **Universal skill**: SQL transcends frameworks and languages — these patterns work in any database

## References

- [PostgreSQL 17 Documentation](https://www.postgresql.org/docs/17/)
- [PostgreSQL Window Functions](https://www.postgresql.org/docs/17/tutorial-window.html)
- [PostgreSQL WITH Queries (CTEs)](https://www.postgresql.org/docs/17/queries-with.html)
- [PostgreSQL EXPLAIN](https://www.postgresql.org/docs/17/sql-explain.html)
- [psycopg 3 Documentation](https://www.psycopg.org/psycopg3/docs/)

## Commands

### Setup

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start PostgreSQL 17 container in background |
| `uv sync` | Install Python dependencies (psycopg3) |
| `uv run python -m app.seed` | Create tables and insert sample data (idempotent — safe to re-run) |

### Database

| Command | Description |
|---------|-------------|
| `docker compose exec postgres psql -U practice -d northwind` | Open interactive psql shell inside the PostgreSQL container |
| `docker compose exec postgres psql -U practice -d northwind -c '\dt'` | List all tables in the northwind database |
| `docker compose exec postgres psql -U practice -d northwind -c '\di'` | List all indexes in the northwind database |
| `docker compose exec postgres psql -U practice -d northwind -c 'SELECT COUNT(*) FROM orders'` | Quick row count for a specific table |

### Run Queries

| Command | Description |
|---------|-------------|
| `uv run python -m app.queries` | Run the main query module (uncomment functions as you implement them) |

### Cleanup

| Command | Description |
|---------|-------------|
| `docker compose down` | Stop and remove the PostgreSQL container (data preserved in volume) |
| `docker compose down -v` | Stop container and delete the volume (destroys all data) |

## State

`not-started`
