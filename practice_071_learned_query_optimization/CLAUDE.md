# Practice 071 — Learned Query Optimization

## Technologies

- **psycopg2-binary** — PostgreSQL adapter for Python (libpq-based)
- **PyTorch** — Neural network framework for the latency prediction model
- **pandas** — Data manipulation for plan features and training data
- **sqlglot** — SQL parsing and analysis (query complexity extraction)
- **numpy** — Numerical operations for feature vectors
- **matplotlib** — Plotting loss curves and comparison charts

## Stack

Python 3.12 (uv), Docker (PostgreSQL 16)

## Theoretical Context

### Traditional Query Optimization: How PostgreSQL Plans Queries

Every SQL query goes through four phases before producing results:

1. **Parsing** — The raw SQL string is tokenized and parsed into an abstract syntax tree (AST). Syntax errors are caught here. The output is a "parse tree" representing the query's structure.

2. **Rewriting** — The rewriter applies transformation rules: view expansion (replacing view references with their definitions), rule-based rewrites (e.g., `IN (subquery)` to semi-join), and security barrier application. The output is still a logical representation.

3. **Planning / Optimization** — This is where the critical decisions happen. The planner must decide:
   - **Access methods** for each table: sequential scan, index scan, bitmap index scan, index-only scan
   - **Join algorithms** for each pair of tables: nested loop join, hash join, merge join
   - **Join ordering**: which tables to join first (this is combinatorial — N tables have N! possible orderings)
   - **Aggregation strategy**: plain, sorted, or hashed
   - **Sort strategy**: explicit sort vs. leveraging index ordering

   PostgreSQL's planner is a **cost-based optimizer (CBO)**. It enumerates candidate plans (or a subset of them), estimates the cost of each, and picks the cheapest. Cost estimation relies on two models:
   - **Cardinality estimation**: predicting how many rows each operation produces. PostgreSQL uses histograms (collected by `ANALYZE`), most-common-values lists, distinct-value counts, and correlation statistics. For single-table predicates, this works reasonably well. For multi-column predicates and joins, PostgreSQL assumes **attribute independence** — a simplification that often breaks down on real data.
   - **Cost model**: converting cardinality estimates into I/O and CPU cost units. The cost model uses configurable parameters (`seq_page_cost`, `random_page_cost`, `cpu_tuple_cost`, etc.) that represent the relative expense of disk seeks, sequential reads, and CPU operations.

   For join enumeration, PostgreSQL uses **dynamic programming** (bottom-up) for queries with up to ~12 tables (the `geqo_threshold`). Beyond that, it switches to a **Genetic Query Optimizer (GEQO)** — a randomized search that trades optimality for speed.

4. **Execution** — The chosen plan is executed using a Volcano-style iterator model: each node in the plan tree implements `init()`, `next()`, and `close()`. Rows flow upward from scan nodes through join and aggregation nodes to the root.

**Reference**: [PostgreSQL Documentation: Using EXPLAIN](https://www.postgresql.org/docs/current/using-explain.html), [PostgreSQL Documentation: Query Planning Configuration](https://www.postgresql.org/docs/current/runtime-config-query.html)

### Why Traditional Optimizers Fail

Traditional cost-based optimizers have well-documented failure modes:

**Cardinality estimation errors propagate exponentially.** If a base table filter is estimated to produce 100 rows but actually produces 10,000, the join above it inherits that 100x error — and the join above *that* compounds it further. A landmark study by Leis et al. (2015) showed that PostgreSQL's cardinality estimates can be off by **orders of magnitude** on multi-join queries, and roughly 10% of test queries timed out because the optimizer chose catastrophically bad plans. The root causes:
- **Independence assumption**: PostgreSQL assumes columns are statistically independent. If `city = 'NYC'` and `state = 'NY'` both have selectivity 0.01, PostgreSQL estimates their conjunction at 0.0001 — but the true selectivity is ~0.01 (they're correlated).
- **Join crossing correlations**: Even if single-table estimates are accurate, the cardinality after a join depends on the *joint distribution* of join keys, which histograms cannot capture.
- **Missing statistics**: Complex expressions, UDFs, and CTEs often lack statistics entirely, forcing the optimizer to use default selectivity estimates (often 0.5% for equality, 33% for range).

**Static cost models don't adapt.** The cost parameters (`seq_page_cost = 1.0`, `random_page_cost = 4.0`, etc.) are static defaults. They don't reflect whether your data is cached in memory (making random I/O cheap), whether your SSD makes sequential vs. random costs nearly equal, or whether concurrent queries are saturating I/O bandwidth. A plan that's optimal on an idle server with cold caches can be terrible under production load.

**No learning from execution.** After executing a query, PostgreSQL discards the actual cardinalities and timing — it never feeds this information back to improve future estimates. Every query is planned from scratch using the same (potentially stale) statistics.

**Reference**: [Leis et al., "How Good Are Query Optimizers, Really?" (VLDB 2015)](https://15799.courses.cs.cmu.edu/spring2025/papers/13-cardinalities1/leis-vldbj2017.pdf), [A Survey on Advancing the DBMS Query Optimizer (2021)](https://link.springer.com/article/10.1007/s41019-020-00149-7)

### Learned Query Optimization: Using ML to Fix the Optimizer

The core insight of learned query optimization is: **if the optimizer's decisions depend on estimates that are often wrong, can we learn better estimates — or better decisions — from actual execution data?** Three landmark systems explore this:

**Neo (2019)** — The first end-to-end learned query optimizer. Neo uses a deep neural network to directly generate query execution plans, bypassing the traditional optimizer entirely. It bootstraps from an existing optimizer's plans, then improves by observing actual execution costs. Neo uses a tree-structured neural network (tree-LSTM) to encode query plan trees, preserving their hierarchical structure. The key contribution: demonstrating that a neural network can learn to produce competitive query plans.

**Bao (2021)** — "Bandit optimizer." Bao takes a fundamentally different approach: instead of *replacing* the optimizer, it **steers** it. Bao observes that PostgreSQL's optimizer already produces good plans *most of the time* — the problem is the ~10% of queries where it makes catastrophic choices. Bao's strategy:
1. Define a set of **hint configurations** — combinations of PostgreSQL's `enable_*` parameters (e.g., `enable_hashjoin=off`, `enable_nestloop=off`, `enable_seqscan=off`). Each configuration forces the optimizer to produce a *different* plan.
2. For each incoming query, get the optimizer's plan under each hint configuration using `EXPLAIN`.
3. Use a **tree convolutional neural network (tree-CNN)** to predict the latency of each plan.
4. Select the hint configuration with the lowest predicted latency.
5. Execute the query with that configuration and observe the actual latency.
6. Use Thompson sampling (a reinforcement learning technique) to balance exploitation (using what works) with exploration (trying uncertain configurations).

Bao's advantage over Neo: it requires **far less training data** (order of magnitude less) because it's choosing among a small set of plans rather than generating plans from scratch. It also gracefully degrades — in the worst case, it just uses the default optimizer's plan.

**Balsa (2022)** — Learns query optimization *without* expert demonstrations. Balsa first trains in a lightweight simulator to learn basic plan structure, then fine-tunes by executing queries on the real database with safety mechanisms (timeouts for catastrophic plans). Balsa matches commercial optimizer performance on the Join Order Benchmark within ~2 hours of real execution.

**Reference**: [Neo: A Learned Query Optimizer (VLDB 2019)](https://www.vldb.org/pvldb/vol12/p1705-marcus.pdf), [Bao: Making Learned Query Optimization Practical (SIGMOD 2021)](https://dl.acm.org/doi/10.1145/3448016.3452838), [Balsa: Learning a Query Optimizer Without Expert Demonstrations (SIGMOD 2022)](https://arxiv.org/abs/2201.01441)

### Plan Featurization: Turning Query Plans into Vectors

A query plan is a **tree** of operator nodes. To feed it into a neural network, we need to convert it to a numerical representation. Two main approaches:

**Flat featurization (what we do in this practice):**
Walk the plan tree and aggregate features into a fixed-size vector:
- **Node type counts**: how many Seq Scans, Index Scans, Hash Joins, Nested Loops, etc.
- **Cost aggregates**: total estimated cost, maximum single-node cost, startup cost
- **Row estimates**: total estimated rows, maximum rows at any node
- **Structural features**: tree depth, number of nodes, branching factor
- **Width**: estimated average row width (bytes)

This loses structural information but is simple and effective for plan comparison.

**Tree-structured featurization (used in Bao/Neo):**
Preserve the tree structure by encoding each node as a feature vector, then using a tree-CNN or tree-LSTM to propagate information bottom-up:
- Each leaf node (scan) gets features: table one-hot encoding, estimated rows, cost, scan type
- Each internal node (join/aggregate) gets features: operator type one-hot encoding, estimated rows, cost, join type
- The tree-CNN applies convolutions that respect parent-child relationships
- The root node's output vector represents the entire plan

### EXPLAIN Output: What PostgreSQL Tells Us

`EXPLAIN (FORMAT JSON)` returns a JSON tree with these key fields per node:

| Field | Description |
|-------|-------------|
| `Node Type` | Operator: "Seq Scan", "Index Scan", "Hash Join", "Nested Loop", "Merge Join", "Sort", "Aggregate", etc. |
| `Startup Cost` | Estimated cost before the first row is produced |
| `Total Cost` | Estimated cost to produce all rows |
| `Plan Rows` | Estimated number of output rows |
| `Plan Width` | Estimated average row width in bytes |
| `Relation Name` | Table being scanned (for scan nodes) |
| `Index Name` | Index being used (for index scans) |
| `Join Type` | "Inner", "Left", "Right", "Full", "Semi", "Anti" |
| `Hash Cond` / `Join Filter` | Join predicate |
| `Filter` | WHERE predicate applied at this node |
| `Plans` | Array of child plan nodes (recursive tree structure) |

With `EXPLAIN (ANALYZE, FORMAT JSON)`, additional fields appear:
- `Actual Startup Time` / `Actual Total Time` — real wall-clock milliseconds
- `Actual Rows` — actual rows produced (compare to `Plan Rows` to measure estimation error)
- `Actual Loops` — how many times the node was executed (important for nested loops)

### Hint-Based Steering in PostgreSQL

PostgreSQL doesn't have Oracle-style optimizer hints (`/*+ USE_HASH */`). Instead, it provides **session-level configuration parameters** that disable specific operators:

| Parameter | Effect when OFF |
|-----------|----------------|
| `enable_hashjoin` | Planner avoids hash joins (adds 10M penalty cost) |
| `enable_mergejoin` | Planner avoids merge joins |
| `enable_nestloop` | Planner avoids nested loop joins (cannot fully disable — it's the fallback) |
| `enable_seqscan` | Planner avoids sequential scans (prefers indexes) |
| `enable_indexscan` | Planner avoids index scans |
| `enable_bitmapscan` | Planner avoids bitmap index scans |
| `enable_sort` | Planner avoids explicit sorts (prefers index ordering) |
| `enable_hashagg` | Planner avoids hash aggregation (prefers sorted aggregation) |

Setting `enable_hashjoin = off` doesn't absolutely prevent hash joins — it adds a large penalty (10,000,000) to their cost estimate, making the optimizer prefer alternatives. In rare cases where no alternative exists, the hash join will still be chosen.

The Bao approach: enumerate a manageable set of hint configurations (not all 2^8 = 256 combinations, but ~8-16 meaningful ones), get the plan for each, predict which will be fastest, and execute with that configuration.

**Reference**: [PostgreSQL Documentation: Query Planning Configuration](https://www.postgresql.org/docs/current/runtime-config-query.html)

### Evaluation: Does It Actually Help?

The learned optimizer is evaluated by comparing total workload latency:
- **Baseline**: PostgreSQL's native optimizer with default settings
- **Learned**: PostgreSQL steered by the trained model's hint selections

Key metrics:
- **Mean/median/p99 latency** across the workload
- **Speedup ratio** per query (native_latency / learned_latency)
- **Regression rate** — fraction of queries where learned is *slower* (false positives of the model)
- **Training efficiency** — how many queries needed before the model outperforms the baseline

In the Bao paper, the system achieved 2-5x speedup on tail queries while rarely regressing on queries the optimizer already handled well. The practical value is in the tail — the ~10% of queries where the optimizer makes its worst mistakes.

## Description

Build a simplified Bao-style learned query optimizer for PostgreSQL. You'll create an e-commerce database, collect query execution plans via EXPLAIN, train a neural network to predict query latency from plan features, then use the model to select optimizer hint configurations that improve workload performance.

### What you'll learn

1. **EXPLAIN parsing** — extracting structured features from PostgreSQL's JSON execution plans
2. **Plan featurization** — converting hierarchical plan trees into fixed-size feature vectors suitable for ML
3. **Latency prediction** — training an MLP to predict query execution time from plan features
4. **Hint-based steering** — using PostgreSQL's `enable_*` parameters to generate alternative plans
5. **Workload evaluation** — comparing native optimizer performance against learned hint selection

## Instructions

### Setup: Database and Data (~15 min)

1. Start PostgreSQL: `docker compose up -d`
2. Install dependencies: `uv sync`
3. Create tables and load synthetic e-commerce data: `uv run python src/00_setup_database.py`
   This creates 5 tables (customers, products, orders, order_items, reviews) with realistic correlations — popular products attract more orders and reviews, active customers order more frequently. Appropriate indexes are created on foreign keys and common filter columns.

### Exercise 1: Parse EXPLAIN JSON into Feature Vectors (~25 min)

Run: `uv run python src/01_explain_parser.py` | TODO functions in: `src/explain_features.py`

This exercise teaches you to work with PostgreSQL's `EXPLAIN (ANALYZE, FORMAT JSON)` output. You'll implement two functions:
- `parse_plan_node()` — recursively parse the JSON plan tree, extracting per-node features (operator type, estimated rows, cost, width). This is the foundation of plan featurization — understanding what information the optimizer exposes about its decisions.
- `flatten_plan_tree()` — aggregate the recursive tree into a fixed-size feature vector. This is where you make the design decision of what information to preserve vs. discard. The feature vector is what the neural network will see.

**Why it matters:** Every learned optimizer needs to convert plans to numbers. The quality of your featurization determines how much signal the model can learn from. Bad features = bad predictions, no matter how good the model architecture.

### Exercise 2: Train Neural Net to Predict Query Latency (~25 min)

Run: `uv run python src/02_cost_model.py` | TODO functions in: `src/cost_model.py`

This exercise builds the core ML component: a neural network that takes plan feature vectors and predicts how long the query will take to execute. You'll implement:
- `LatencyPredictor` — an MLP architecture. Key design question: should you predict raw latency or log(latency)? Query latencies span orders of magnitude (0.1ms to 10s+), and MSE loss on raw values would be dominated by the slowest queries.
- `train_model()` — the training loop with proper normalization, validation tracking, and loss visualization.

**Why it matters:** This is the "learned" part of learned query optimization. The model replaces PostgreSQL's static cost model with a data-driven one trained on actual execution times. If the model predicts well, it can identify when the optimizer's plan choice is suboptimal.

### Exercise 3: Bao-Style Hint Selection (~25 min)

Run: `uv run python src/03_hint_selection.py` | TODO functions in: `src/hint_selector.py`

This exercise implements the Bao strategy: for each query, generate alternative plans by toggling optimizer parameters, predict latency for each, and select the best. You'll implement:
- `generate_hint_configs()` — design a set of meaningful hint configurations. Not all 256 combinations are useful — you need to identify which operator toggles actually produce different plans.
- `select_best_hints()` — the full pipeline: apply hints, get EXPLAIN plan, featurize, predict, select minimum.

**Why it matters:** This is where the learned optimizer makes decisions. The key insight is that we don't replace the optimizer — we just ask it to produce plans under different constraints, then use the model to pick the best one.

### Exercise 4: Evaluate Learned vs Native Optimizer (~15 min)

Run: `uv run python src/04_evaluation.py` | TODO functions in: `src/evaluator.py`

Compare the native optimizer against your learned hint selector on a test workload. You'll implement:
- `run_workload()` — execute queries with and without hint configurations, measuring actual wall-clock latency.
- `generate_comparison_report()` — compute summary statistics and generate comparison plots.

**Why it matters:** The ultimate test — does the learned optimizer actually improve query performance? You'll see which queries benefit from hint steering and which are already well-optimized by PostgreSQL's native planner.

## Motivation

- **Core database knowledge**: Understanding query optimization internals is essential for any backend/data engineer — it explains why queries are slow and how to fix them
- **ML for systems**: Learned query optimization is a flagship example of ML applied to systems problems — this intersection (ML for databases, ML for compilers, ML for networking) is a rapidly growing research and industry area
- **Bao is practical**: Unlike purely academic approaches, Bao's hint-steering approach works with unmodified PostgreSQL — it's the most deployable learned optimizer design
- **Complements practice 023 (Advanced SQL)**: Query optimization is the "why" behind the performance patterns taught in the SQL practice — understanding the optimizer explains *why* certain query rewrites help

## Commands

All commands run from `practice_071_learned_query_optimization/`.

### Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start PostgreSQL 16 container (port 5432, persistent volume) |
| `docker compose down` | Stop PostgreSQL container (preserves data volume) |
| `docker compose down -v` | Stop PostgreSQL container and delete data volume |

### Project Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies from `pyproject.toml` |
| `uv run python src/00_setup_database.py` | Create e-commerce schema and load synthetic data (10K customers, 1K products, 100K orders, 300K items, 50K reviews) |

### Exercises

| Command | Description |
|---------|-------------|
| `uv run python src/01_explain_parser.py` | Exercise 1: Parse EXPLAIN JSON output into plan feature vectors |
| `uv run python src/02_cost_model.py` | Exercise 2: Train MLP to predict query latency from plan features |
| `uv run python src/03_hint_selection.py` | Exercise 3: Bao-style hint selection — toggle optimizer params, pick best plan |
| `uv run python src/04_evaluation.py` | Exercise 4: Compare learned hint selection vs native optimizer on test workload |

### Cleanup

| Command | Description |
|---------|-------------|
| `docker compose down -v` | Stop PostgreSQL and delete data volume |
| `python clean.py` | Remove all generated files (data/, plots/, models/, caches) |

## References

- [Bao: Making Learned Query Optimization Practical (SIGMOD 2021)](https://dl.acm.org/doi/10.1145/3448016.3452838)
- [Neo: A Learned Query Optimizer (VLDB 2019)](https://www.vldb.org/pvldb/vol12/p1705-marcus.pdf)
- [Balsa: Learning a Query Optimizer Without Expert Demonstrations (SIGMOD 2022)](https://arxiv.org/abs/2201.01441)
- [PostgreSQL EXPLAIN Documentation](https://www.postgresql.org/docs/current/using-explain.html)
- [PostgreSQL Query Planning Configuration](https://www.postgresql.org/docs/current/runtime-config-query.html)
- [Leis et al., "How Good Are Query Optimizers, Really?" (VLDB 2015)](https://15799.courses.cs.cmu.edu/spring2025/papers/13-cardinalities1/leis-vldbj2017.pdf)
- [sqlglot — Python SQL Parser](https://github.com/tobymao/sqlglot)
- [BaoForPostgreSQL — Reference Implementation](https://github.com/learnedsystems/BaoForPostgreSQL)

## State

`not-started`
