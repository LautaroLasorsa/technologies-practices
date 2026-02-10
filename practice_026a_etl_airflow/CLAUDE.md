# Practice 026a: ETL/ELT Orchestration — Airflow Fundamentals

**State:** `not-started`

## Technologies

- **Apache Airflow 2.8+** — The industry-standard workflow orchestration platform
- **PostgreSQL 15** — Airflow metadata database + ETL target database
- **Docker Compose** — Full Airflow stack (webserver, scheduler, metadata DB)

## Stack

- **Python 3.12+** (uv for local helper scripts)
- **Docker** (Airflow runs entirely in containers)

## Description

Learn Apache Airflow's core concepts by building DAGs that orchestrate a simple ETL pipeline. Start with basic task dependencies, progress to the modern TaskFlow API with decorators, handle data passing with XComs, implement sensors and scheduling, and finally see where Airflow's task-centric model hits its limits.

The domain is a **weather data ETL**: extract from mock JSON files, transform/aggregate readings, and load summaries into PostgreSQL.

### What You'll Learn

- **DAGs** — Directed Acyclic Graphs: the core abstraction for defining workflows
- **Operators** — PythonOperator, BashOperator, and how they wrap task logic
- **TaskFlow API** — The modern `@task` decorator approach (Airflow 2.0+)
- **XComs** — Cross-communication: how tasks pass data to each other
- **Sensors** — Tasks that wait for external conditions (files, APIs, time)
- **Scheduling** — Cron expressions, catchup, backfill, and execution dates
- **Dynamic DAGs** — DAG factory patterns and parameterized workflows
- **Airflow Limitations** — Task-centric pain points that motivate modern alternatives (Dagster in 026b)

## Instructions

### Phase 1: Airflow Setup & First DAG (~15 min)

1. Start the Airflow stack with `docker compose up -d`
2. Wait for initialization to complete (~30-60 seconds)
3. Open the Airflow UI at `http://localhost:8080` (login: admin/admin)
4. Explore the UI: DAGs list, Graph view, Tree view, task logs
5. The `phase1_hello` DAG is pre-deployed — trigger it manually and watch it execute
6. Read through `dags/phase1_hello.py` to understand the structure

### Phase 2: Classic Operators & Dependencies (~20 min)

1. Open `dags/phase2_classic_operators.py`
2. Read the existing BashOperator tasks and DAG skeleton
3. Implement the TODO(human) sections: PythonOperator callables and dependency chain
4. Save the file — Airflow auto-detects changes (scheduler re-parses every ~30s)
5. Trigger the DAG in the UI, inspect task logs and execution order
6. Experiment with trigger_rule options to see how they affect execution flow

### Phase 3: TaskFlow API & XComs (~20 min)

1. Open `dags/phase3_taskflow.py`
2. Study the reference `@task` function provided
3. Implement the TODO(human) `@task` functions: extract, transform, load
4. Notice how return values automatically become XComs — no explicit push/pull
5. Trigger the DAG and inspect XCom values in the UI (Admin > XComs)
6. Compare the code verbosity with Phase 2's explicit operator approach

### Phase 4: ETL Pipeline with Sensors & Scheduling (~20 min)

1. Generate mock weather data: `uv run python data/mock_weather.py`
2. Open `dags/phase4_etl_pipeline.py`
3. Implement the TODO(human) extract, transform, and load functions
4. Trigger the DAG manually (or wait for schedule if you set catchup=True)
5. Verify results: `uv run python scripts/check_results.py`
6. Understand the FileSensor: what happens if the data file doesn't exist?

### Phase 5: Dynamic DAGs & Limitations (~15 min)

1. Open `dags/phase5_dynamic_dag.py`
2. Implement the TODO(human) DAG factory function
3. Observe how multiple DAGs appear in the UI from a single .py file
4. Read the extensive limitation comments — these set up the motivation for Dagster (026b)
5. Reflect: what would change if you needed 50 cities? 500? Real-time data?

## Motivation

Apache Airflow is the **de facto standard** for workflow orchestration in data engineering:

- **Industry adoption**: Used by Airbnb (creators), Uber, Lyft, Spotify, Twitter, and thousands of companies
- **Job market**: "Airflow" appears in ~60% of data engineering job postings
- **Foundation knowledge**: Understanding Airflow is a prerequisite for appreciating WHY modern alternatives (Dagster, Prefect, Mage) exist and what problems they solve
- **Professional relevance**: At AutoScheduler.AI, ETL/ELT pipelines are core infrastructure — knowing the standard tool and its limitations informs better architecture decisions
- **Gateway to 026b**: This practice intentionally exposes Airflow's pain points so that Dagster's asset-centric model (026b) feels like a revelation, not just "another tool"

## References

- [Apache Airflow Documentation](https://airflow.apache.org/docs/apache-airflow/stable/index.html)
- [TaskFlow API Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/taskflow.html)
- [Astronomer Guides](https://docs.astronomer.io/learn)
- [Airflow Docker Compose](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)

## Commands

All commands are run from `practice_026a_etl_airflow/`.

### Setup

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start the full Airflow stack (webserver, scheduler, postgres, init) |
| `docker compose logs -f airflow-init` | Watch initialization progress (wait for "Admin user created") |
| `docker compose ps` | Verify all services are healthy |
| `docker compose logs -f airflow-scheduler` | Watch scheduler logs for DAG parsing errors |

### Development Workflow

| Command | Description |
|---------|-------------|
| `docker compose logs -f airflow-webserver` | Stream webserver logs (API errors, UI issues) |
| `docker compose logs -f airflow-scheduler` | Stream scheduler logs (DAG parse errors, task execution) |
| `docker compose restart airflow-scheduler` | Force re-parse of DAG files after changes |
| `docker compose exec airflow-webserver airflow dags list` | List all recognized DAGs from CLI |
| `docker compose exec airflow-webserver airflow tasks list <dag_id>` | List tasks in a specific DAG |

### Mock Data & Verification

| Command | Description |
|---------|-------------|
| `uv run python data/mock_weather.py` | Generate mock weather JSON files in `data/` |
| `uv run python data/mock_weather.py --days 7` | Generate 7 days of mock weather data |
| `uv run python scripts/check_results.py` | Query PostgreSQL to verify ETL results |

### Cleanup

| Command | Description |
|---------|-------------|
| `docker compose down` | Stop all containers (preserves volumes) |
| `docker compose down -v` | Stop all containers AND delete volumes (full reset) |
