# Practice 026b: ETL/ELT Orchestration -- Dagster Assets & dbt

## Technologies

- **Dagster 1.6+** -- Asset-based orchestration: Software-Defined Assets, partitions, sensors, auto-materialization
- **dbt-core 1.7+** -- SQL transformation framework: models, refs, tests, incremental materialization
- **dbt-postgres** -- dbt adapter for PostgreSQL
- **PostgreSQL 16** -- Shared database for Dagster metadata, dbt warehouse, and application data
- **Docker Compose** -- Multi-container orchestration (Dagster webserver, daemon, PostgreSQL)

## Stack

- Python 3.12+ (uv for local development)
- Docker (Dagster + PostgreSQL containers)

## Description

Learn Dagster's **asset-based orchestration model** -- a paradigm shift from Airflow's task-centric approach. Instead of defining "run this task at this time," you define "this data asset exists and depends on these other assets." Dagster handles materialization, incremental updates, and lineage automatically.

Integrate **dbt as the transformation layer**, treating dbt models as Dagster assets with full lineage tracking. Compare the same weather ETL pipeline from 026a, now modeled as assets with time-partitions and incremental materialization.

### What you'll learn

1. **Software-Defined Assets** (`@asset`) -- Declare data assets as Python functions; Dagster infers the DAG from parameter dependencies
2. **Asset dependencies** -- Wire assets together via function parameters, not explicit `>>` operators
3. **Resources & I/O Managers** -- Inject database connections, API clients, and serialization strategies
4. **Partitions** -- `DailyPartitionsDefinition` for time-windowed processing, targeted backfills
5. **Sensors** -- React to external events or upstream asset changes
6. **dbt integration** (`@dbt_assets`) -- Load dbt models as first-class Dagster assets with unified lineage
7. **Incremental materialization** -- Process only new/changed data, not the entire dataset
8. **Dagster vs Airflow comparison** -- Understand when and why to choose one over the other

### Key paradigm shift from Airflow (026a)

| Concept | Airflow (026a) | Dagster (026b) |
|---------|----------------|----------------|
| Core unit | Task (operator) | Asset (data) |
| DAG definition | Explicit `>>` / `set_downstream` | Implicit from function parameters |
| Scheduling | Cron-based ("run at 2am") | Freshness-based ("data should be <6h old") |
| Backfills | Manual, error-prone | First-class partitioned backfills |
| Testing | Difficult (needs running Airflow) | Native: `materialize([asset])` in pytest |
| Data lineage | External tools (Atlas, DataHub) | Built-in, automatic from asset graph |
| dbt integration | BashOperator or cosmos plugin | Native `@dbt_assets` with lineage |

## Instructions

### Phase 1: Dagster Basics & First Asset (~15 min)

1. Start the stack: `docker compose up -d` -- brings up PostgreSQL, Dagster webserver (Dagit), and daemon
2. Open Dagit UI at `http://localhost:3000` -- explore the asset catalog, runs, and launchpad
3. Read `phase1_basic_assets.py` -- understand the `@asset` decorator and how `raw_weather_data` is defined
4. **Implement** `cleaned_weather_data` -- an asset that depends on `raw_weather_data` by accepting it as a function parameter
5. Materialize both assets from Dagit -- observe the lineage graph
6. **Key insight:** You defined WHAT data exists and its dependencies. Dagster figured out the execution order. In Airflow, you'd explicitly wire `task1 >> task2`.

### Phase 2: Asset Graph & Dependencies (~20 min)

1. Read `phase2_asset_graph.py` -- a multi-step weather pipeline with 4 assets
2. **Implement** the middle assets: `validated_readings` and `city_aggregates`
3. Materialize the full chain from Dagit -- observe the 4-node asset graph
4. Experiment: re-materialize only `city_aggregates` -- Dagster knows it can skip upstream if already materialized
5. **Key insight:** Asset graphs are declarative. Adding a new downstream asset doesn't require modifying upstream code. In Airflow, you'd edit the DAG file to add new tasks.

### Phase 3: Partitioned Assets (~20 min)

1. Read `phase3_partitions.py` -- `DailyPartitionsDefinition` and partition-aware assets
2. **Implement** partitioned assets that process weather data for a specific date via `context.partition_key`
3. Materialize a single day's partition from Dagit -- observe the partition status matrix
4. Launch a backfill for a date range -- Dagster queues one run per partition
5. **Key insight:** Partitions are first-class. In Airflow, you'd use `execution_date` and hope your operators handle it correctly. Dagster tracks which partitions are materialized and which need refresh.

### Phase 4: dbt Integration (~20 min)

1. Read the dbt project structure: `dbt_project/models/staging/` and `dbt_project/models/marts/`
2. Read `phase4_dbt_assets.py` -- how `@dbt_assets` loads dbt models as Dagster assets
3. **Implement** the `@dbt_assets` function and wire upstream/downstream Python assets
4. **Implement** the `daily_weather_summary.sql` dbt mart model (aggregation SQL)
5. Materialize the full chain: Python asset -> dbt staging -> dbt mart -> Python asset
6. **Key insight:** dbt models appear in the same asset graph as Python assets. In Airflow, dbt runs are opaque BashOperator calls with no lineage visibility.

### Phase 5: Sensors & Automation (~15 min)

1. Read `phase5_automation.py` -- asset sensors, freshness policies, auto-materialize
2. **Implement** a sensor that detects new data files and triggers materialization
3. **Implement** a `FreshnessPolicy` on a downstream asset
4. Enable auto-materialization in Dagit and observe the daemon picking up work
5. **Key insight:** In Airflow, you schedule DAGs on cron and hope data is ready. In Dagster, you declare freshness requirements and Dagster figures out what to run and when.

## Motivation

- **Fastest-growing orchestrator**: Dagster adoption is accelerating as teams hit Airflow's limitations (poor testing, no native assets, manual lineage)
- **Asset thinking**: Modern data engineering is moving from "run tasks on schedule" to "ensure data assets are fresh" -- Dagster is built for this
- **dbt-native**: Dagster's dbt integration is the most seamless in the market, treating dbt models as first-class assets
- **Better testing**: Assets can be unit-tested with `materialize()` in pytest -- no need for a running orchestrator
- **Complements 026a**: Understanding both Airflow and Dagster lets you make informed architecture decisions and migrate between them
- **Industry demand**: Companies like Elementl (Dagster creators), plus adopters in fintech, ML platforms, and data teams are actively hiring for Dagster expertise

## References

- [Dagster Concepts: Software-Defined Assets](https://docs.dagster.io/concepts/assets/software-defined-assets)
- [Dagster Tutorial](https://docs.dagster.io/tutorial)
- [Dagster Partitions](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions)
- [Dagster dbt Integration](https://docs.dagster.io/integrations/dbt)
- [Dagster Resources](https://docs.dagster.io/concepts/resources)
- [Dagster Auto-Materialization](https://docs.dagster.io/concepts/assets/asset-auto-execution)
- [dbt Core Documentation](https://docs.getdbt.com/docs/introduction)
- [dbt ref() function](https://docs.getdbt.com/reference/dbt-jinja-functions/ref)
- [Dagster vs Airflow](https://dagster.io/blog/dagster-airflow)

## Commands

All commands are run from the `practice_026b_etl_dagster/` folder root.

### Docker Compose -- Stack Management

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start all services: PostgreSQL, Dagster webserver (port 3000), Dagster daemon |
| `docker compose up -d --build` | Rebuild images and start (use after changing Dockerfile or dependencies) |
| `docker compose logs -f dagster-webserver` | Follow Dagster webserver logs (Dagit UI) |
| `docker compose logs -f dagster-daemon` | Follow Dagster daemon logs (sensors, auto-materialization) |
| `docker compose logs -f postgres` | Follow PostgreSQL logs |
| `docker compose ps` | Show running containers and their status |
| `docker compose down` | Stop all containers (preserves volumes) |
| `docker compose down -v` | Stop all containers AND delete volumes (clean reset) |

### Dagit UI

| Command | Description |
|---------|-------------|
| Open `http://localhost:3000` | Dagit web UI -- asset catalog, runs, launchpad, sensors |

### Data Generation

| Command | Description |
|---------|-------------|
| `docker compose exec dagster-webserver python /app/data/generate_weather.py` | Generate mock weather data into PostgreSQL |
| `docker compose exec dagster-webserver python /app/data/generate_weather.py --days 30` | Generate 30 days of historical weather data |

### dbt Commands (inside container)

| Command | Description |
|---------|-------------|
| `docker compose exec dagster-webserver dbt debug --project-dir /app/dbt_project --profiles-dir /app/dbt_project` | Verify dbt can connect to PostgreSQL |
| `docker compose exec dagster-webserver dbt run --project-dir /app/dbt_project --profiles-dir /app/dbt_project` | Run all dbt models manually (outside Dagster) |
| `docker compose exec dagster-webserver dbt test --project-dir /app/dbt_project --profiles-dir /app/dbt_project` | Run dbt tests manually |

### Local Development (optional, with uv)

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies locally for IDE support |
| `uv run dagster dev -f dagster_project/definitions.py` | Run Dagster dev server locally (requires local PostgreSQL) |

## State

`not-started`
