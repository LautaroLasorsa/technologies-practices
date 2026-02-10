"""
Phase 5: Dynamic DAGs & Airflow Limitations
=============================================

This module demonstrates DAG factories — generating multiple DAGs from configuration.
It also serves as a detailed discussion of Airflow's limitations, setting up the
motivation for modern alternatives like Dagster (Practice 026b).

KEY CONCEPTS:

    DAG FACTORIES:
        A "DAG factory" is a function that creates DAG objects from parameters.
        Since Airflow discovers DAGs by scanning Python files at IMPORT TIME,
        you can generate multiple DAGs from a single .py file by calling the
        factory in a loop.

        This is powerful for:
        - Multi-tenant pipelines (same logic, different configs per client)
        - Multi-region ETL (same pipeline, different data sources per region)
        - Environment-specific DAGs (dev/staging/prod with different schedules)

    HOW AIRFLOW DISCOVERS DAGS:
        1. The scheduler scans every .py file in the dags/ folder
        2. It IMPORTS each file as a Python module
        3. It looks for DAG objects in the module's global scope
        4. Any DAG object found is registered

        This means:
        - ALL top-level code runs during parsing (be careful with side effects!)
        - Database queries, API calls, or slow imports in DAG files SLOW DOWN
          the entire scheduler (it re-parses every dag_dir_list_interval seconds)
        - A syntax error in ONE file can break ALL DAGs in that file
        - Dynamic DAG generation happens at PARSE TIME, not at runtime

    LIMITATIONS OF DYNAMIC DAGS:
        - The number of DAGs must be determinable at parse time
          (you can't create DAGs based on runtime data easily)
        - All DAGs from a file share the same parse cycle
        - Too many dynamic DAGs (>100s) can slow down the scheduler
        - Each DAG appears separately in the UI — no "group" concept

WHAT TO DO:
    1. Implement the TODO(human) DAG factory function
    2. Observe multiple DAGs appearing in the Airflow UI from this single file
    3. Read the limitations discussion at the bottom — this is the motivation for 026b

===========================================================================
AIRFLOW LIMITATIONS — Why Modern Alternatives Exist
===========================================================================

After building DAGs in Phases 1-4, you've experienced Airflow's core model.
Here are the fundamental limitations that drive teams to alternatives:

1. TASK-CENTRIC, NOT DATA-CENTRIC:
    Airflow tracks TASKS (did this function run successfully at this time?)
    but NOT DATA (is this dataset fresh? what produced it? who consumes it?).

    Pain: You schedule "run ETL at 6am" but can't ask "is the weather_summary
    table up to date?" Airflow knows task_X ran at 6am, but doesn't know
    whether the data it was supposed to produce actually exists or is valid.

    In Dagster (026b), the core primitive is a "software-defined asset" —
    you define WHAT data exists and HOW to compute it. The orchestrator
    tracks data freshness, lineage, and dependencies automatically.

2. SCHEDULE-DRIVEN, NOT EVENT-DRIVEN:
    Airflow runs DAGs on a SCHEDULE (cron). If data arrives late or early,
    Airflow doesn't care — it runs at the scheduled time regardless.

    Sensors help (wait for a file), but they're awkward:
    - They consume a worker slot while waiting (or reschedule mode adds latency)
    - They can timeout and fail the entire pipeline
    - You're still fundamentally on a schedule with sensors as guardrails

    Dagster supports "eager materialization" — compute an asset whenever
    its upstream dependencies change, not on a fixed schedule.

3. XCOM IS NOT FOR DATA:
    XCom is stored in the metadata database (PostgreSQL). It's designed for
    small values: file paths, row counts, config dicts. The default limit
    is ~64KB per XCom value.

    Pain: If your ETL produces a 1GB DataFrame, you can't pass it via XCom.
    You must write it to shared storage and pass the PATH. This means:
    - Manual serialization/deserialization in every task
    - No automatic type checking
    - No data lineage tracking

    Dagster's IO Managers handle serialization/deserialization transparently.
    You return a DataFrame from one asset, and the downstream asset receives
    a DataFrame — the framework handles the storage.

4. DAG PARSING AT IMPORT TIME:
    Every DAG file is re-imported every ~30 seconds by the scheduler.
    This means:
    - Slow imports (heavy libraries) slow the ENTIRE scheduler
    - Side effects in DAG files (API calls, DB queries) run repeatedly
    - You can't easily do "dynamic DAGs based on a database query"
      without querying the DB every 30 seconds during parsing

    Dagster separates "definition" from "execution" more cleanly.
    Asset definitions are loaded once, not re-parsed on a tight loop.

5. TESTING IS AWKWARD:
    To test a DAG, you need a running Airflow instance (or mock one).
    Testing individual tasks requires setting up the Airflow context,
    XCom backend, and connections.

    # Typical Airflow test — lots of ceremony:
    from airflow.models import DagBag
    dagbag = DagBag()
    dag = dagbag.get_dag('my_dag')
    dag.test()  # Runs all tasks — no isolation

    # Testing a single task is harder:
    from airflow.models import TaskInstance
    ti = TaskInstance(task=dag.get_task('my_task'), run_id='test')
    ti.run()  # Needs DB connection, XCom backend, etc.

    Dagster assets are plain Python functions — test them like any function:
    result = my_asset(input_data)
    assert result.shape == (100, 5)

6. NO INCREMENTAL MATERIALIZATION:
    Airflow runs tasks for time intervals. If you need to reprocess just
    one partition of data, you backfill the entire interval.

    Dagster has first-class "partitions" — you can materialize a single
    partition without affecting others, and the UI shows which partitions
    are fresh vs stale.

7. LINEAGE AND OBSERVABILITY:
    Airflow shows task dependencies (the DAG graph), but doesn't natively
    show DATA dependencies (which tables feed into which downstream tables).

    You can't ask: "if I change the weather_readings source, which downstream
    tables are affected?" You'd have to trace through DAG code manually.

    Dagster's asset graph IS the data lineage graph. You see, at a glance,
    every data asset and its upstream/downstream dependencies.

IMPORTANT CONTEXT:
    Airflow is NOT a bad tool. It's the INDUSTRY STANDARD for good reasons:
    - Battle-tested at massive scale (Airbnb, Uber, Spotify)
    - Huge ecosystem of providers (every cloud service, database, etc.)
    - Large community, extensive documentation, lots of hiring demand
    - Works well for traditional schedule-driven ETL

    The limitations above matter most for MODERN data platforms where:
    - Data freshness > schedule adherence
    - Data lineage and observability are first-class requirements
    - Testing and development velocity matter
    - The "unit of work" is a data asset, not a task

    Understanding these limitations is WHY we're doing this practice:
    you need to feel the pain before appreciating the cure (026b: Dagster).
"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task


# =============================================================================
# Configuration for dynamic DAG generation
# =============================================================================
# In a real system, this might come from a config file, database, or API.
# For learning, we hardcode it here. The KEY insight: this list is read at
# PARSE TIME (when the scheduler imports this file), NOT at task runtime.

CITY_CONFIGS = [
    {
        "city_id": "nyc",
        "city_name": "New York",
        "timezone": "America/New_York",
        "schedule": "0 7 * * *",  # 7 AM local time
    },
    {
        "city_id": "london",
        "city_name": "London",
        "timezone": "Europe/London",
        "schedule": "0 8 * * *",  # 8 AM local time
    },
    {
        "city_id": "tokyo",
        "city_name": "Tokyo",
        "timezone": "Asia/Tokyo",
        "schedule": "0 9 * * *",  # 9 AM local time
    },
]


# =============================================================================
# TODO(human): Implement the DAG factory function
# =============================================================================
def create_city_weather_dag(city_config: dict) -> None:
    """
    DAG factory: creates one DAG per city configuration.

    TODO(human): Implement this function. Here's what it should do:

    1. Create a @dag-decorated function with a UNIQUE dag_id per city:
       - dag_id = f"phase5_weather_{city_config['city_id']}"
       - This is CRITICAL — dag_ids must be globally unique across ALL DAGs
       - If two DAGs have the same dag_id, Airflow silently picks one (undefined behavior!)

    2. The @dag decorator should have:
       - default_args with owner="learner", retries=1
       - description: f"Dynamic DAG for {city_config['city_name']} weather"
       - start_date: datetime(2024, 1, 1)
       - schedule: city_config["schedule"]  (each city has its own schedule!)
       - catchup: False
       - tags: ["phase5", "dynamic", city_config["city_id"]]

    3. Inside the @dag function, define these @task functions:

       @task
       def extract() -> dict:
           # Simulates extracting weather for this specific city.
           # Use city_config from the enclosing scope (closure).
           # Return a dict with city_name, readings (mock list), and timezone.
           print(f"Extracting weather for {city_config['city_name']}")
           return {
               "city": city_config["city_name"],
               "timezone": city_config["timezone"],
               "readings": [
                   {"temp_f": 45.2, "humidity": 60},
                   {"temp_f": 48.1, "humidity": 55},
               ],
           }

       @task
       def transform(raw: dict) -> dict:
           # Convert temps to Celsius, add city metadata
           # This is the same logic for all cities — the factory parameterizes the DATA
           city = raw["city"]
           readings = raw["readings"]
           avg_temp_c = round(
               sum((r["temp_f"] - 32) * 5/9 for r in readings) / len(readings), 1
           )
           return {
               "city": city,
               "avg_temp_c": avg_temp_c,
               "reading_count": len(readings),
           }

       @task
       def load(summary: dict) -> None:
           # Print the result (in a real system, write to DB)
           print(f"[{summary['city']}] Avg temp: {summary['avg_temp_c']}C "
                 f"({summary['reading_count']} readings)")

    4. Wire up the dependency chain inside the @dag function:
           raw = extract()
           summary = transform(raw)
           load(summary)

    5. CALL the @dag function to register it:
       This is the step most people forget!
       The @dag decorator turns the function into a factory.
       You must call it to actually create and register the DAG.

    Key design principle:
       The factory function captures `city_config` via CLOSURE. Each DAG gets
       its own copy of the config. The tasks themselves are identical in structure
       but operate on different data because the config differs.

    Why is this a factory and not a parameterized DAG?
       Airflow doesn't have first-class "parameterized DAGs" (it has dag_run.conf
       for runtime params, but the DAG structure itself must be static).
       DAG factories are the standard pattern for "same pipeline, different config".

       In Dagster, you'd use "partitions" or "config schemas" instead —
       cleaner, with UI support for selecting which partition to run.

    Gotcha: variable scoping in loops
       If you create DAGs in a for-loop, be careful with closures:

       ❌ BAD — all DAGs share the same variable reference:
           for config in configs:
               @dag(dag_id=config["id"])
               def my_dag():
                   # config here is ALWAYS the last element of configs!
                   # Python closures capture the VARIABLE, not the VALUE
                   pass

       ✅ GOOD — use a factory function (like this one) to capture the value:
           for config in configs:
               create_dag(config)  # config is passed as a parameter, not a closure

       Or use a default argument:
           for config in configs:
               @dag(dag_id=config["id"])
               def my_dag(cfg=config):  # default arg captures current value
                   pass
    """
    # --- YOUR CODE HERE ---
    # Create a @dag-decorated function and call it to register the DAG.
    #
    # Skeleton to get you started:
    #
    # @dag(
    #     dag_id=f"phase5_weather_{city_config['city_id']}",
    #     ...
    # )
    # def city_weather_pipeline():
    #     @task
    #     def extract() -> dict:
    #         ...
    #
    #     @task
    #     def transform(raw: dict) -> dict:
    #         ...
    #
    #     @task
    #     def load(summary: dict) -> None:
    #         ...
    #
    #     raw = extract()
    #     summary = transform(raw)
    #     load(summary)
    #
    # city_weather_pipeline()  # DON'T FORGET THIS LINE
    pass


# =============================================================================
# Generate DAGs from configuration
# =============================================================================
# This loop runs at IMPORT TIME (when the scheduler parses this file).
# Each call to create_city_weather_dag registers a new DAG.
# After parsing, Airflow sees 3 DAGs: phase5_weather_nyc, _london, _tokyo.
#
# WARNING: Keep this fast! This code runs every ~30 seconds.
# Don't do API calls, DB queries, or heavy computation here.
# Read from a file or hardcoded config only.

for config in CITY_CONFIGS:
    create_city_weather_dag(config)
