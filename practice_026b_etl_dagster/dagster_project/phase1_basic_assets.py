"""Phase 1: Dagster Basics & First Asset (~15 min)

THE BIG PARADIGM SHIFT:
    In Airflow (026a), you defined a DAG like this:
        with DAG("weather_pipeline", schedule="@daily"):
            extract = PythonOperator(task_id="extract", python_callable=extract_fn)
            transform = PythonOperator(task_id="transform", python_callable=transform_fn)
            extract >> transform  # explicit dependency wiring

    In Dagster, you define ASSETS (data), not TASKS (operations):
        @asset
        def raw_weather_data():       # This asset EXISTS
            return fetch_weather()

        @asset
        def clean_weather(raw_weather_data):  # This asset DEPENDS ON raw_weather_data
            return clean(raw_weather_data)     # Dagster infers the dependency from the parameter name!

    The key insight: Airflow asks "what should I RUN?" Dagster asks "what DATA should exist?"

WHY ASSETS MATTER:
    1. Dependencies are IMPLICIT from function signatures (no >> operator)
    2. Dagster tracks the STATE of each asset (when was it last materialized? with what metadata?)
    3. You can materialize one asset without re-running the whole pipeline
    4. Testing is trivial: just call the function with test data
    5. The asset graph IS your data catalog -- no external lineage tool needed

WHAT IS "MATERIALIZATION"?
    In Airflow, you "trigger a DAG run" and tasks execute.
    In Dagster, you "materialize an asset" -- the function runs, produces data,
    and Dagster records the result (metadata, timestamp, partition info).

    Think of it like this:
    - Airflow: "Run the extract task" (imperative)
    - Dagster: "Ensure raw_weather_data is up to date" (declarative)
"""

import json
import os
from datetime import datetime
from typing import Any

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from dagster_project.resources import PostgresResource


# ---------------------------------------------------------------------------
# REFERENCE IMPLEMENTATION: raw_weather_data
# ---------------------------------------------------------------------------
# This asset is FULLY IMPLEMENTED as a learning reference.
# Study how it:
#   1. Uses the @asset decorator to declare itself
#   2. Accepts a Dagster resource (PostgresResource) via parameter injection
#   3. Returns MaterializeResult with metadata (shown in Dagit UI)
#   4. Uses context.log for structured logging (visible in Dagit run viewer)
# ---------------------------------------------------------------------------


@asset(
    description="Raw weather readings loaded from the source PostgreSQL table. "
    "This is the entry point of the weather data pipeline.",
    group_name="phase1_weather",
    # 'compute_kind' shows up as a badge in the Dagit asset graph
    compute_kind="python",
)
def raw_weather_data(
    context: AssetExecutionContext,
    postgres: PostgresResource,
) -> MaterializeResult:
    """Load raw weather readings from the source table.

    COMPARISON WITH AIRFLOW:
        In Airflow, this would be a PythonOperator task inside a DAG. The data
        would be passed to the next task via XCom (serialized to the metadata DB,
        limited to ~48KB by default, fragile).

        In Dagster, this asset PRODUCES data. Downstream assets declare they need
        this data by listing `raw_weather_data` as a function parameter. Dagster
        handles the plumbing.

    NOTE ON I/O MANAGERS:
        In a production setup, you'd use an I/O Manager to automatically store/load
        the asset's output (e.g., to S3, a database table, or a Parquet file).
        For this practice, we store data directly in PostgreSQL and pass metadata
        via MaterializeResult. Phase 2 explores I/O managers in more depth.
    """
    # Ensure the raw data table exists (the generate_weather.py script populates it)
    postgres.ensure_table(
        "raw_weather_readings",
        """
        id SERIAL PRIMARY KEY,
        city TEXT NOT NULL,
        temperature_c REAL NOT NULL,
        humidity_pct REAL NOT NULL,
        wind_speed_kmh REAL NOT NULL,
        reading_date DATE NOT NULL,
        recorded_at TIMESTAMP DEFAULT NOW()
        """,
    )

    # Query the raw data
    rows = postgres.execute_query(
        "SELECT COUNT(*), MIN(reading_date), MAX(reading_date) FROM raw_weather_readings"
    )
    count, min_date, max_date = rows[0]

    context.log.info(
        f"Raw weather data: {count} readings from {min_date} to {max_date}"
    )

    # MaterializeResult lets you attach metadata that's visible in Dagit.
    # This is MUCH richer than Airflow's XCom -- you can attach row counts,
    # data previews, links, Markdown tables, etc.
    return MaterializeResult(
        metadata={
            "row_count": MetadataValue.int(count or 0),
            "date_range": MetadataValue.text(f"{min_date} to {max_date}"),
            "source_table": MetadataValue.text("raw_weather_readings"),
        }
    )


# ---------------------------------------------------------------------------
# TODO(human): Implement cleaned_weather_data
# ---------------------------------------------------------------------------
# ── Exercise Context ──────────────────────────────────────────────────
# This teaches Dagster's core innovation: implicit dependency wiring via function parameters.
# By naming the parameter `raw_weather_data`, Dagster infers the asset graph automatically.
# This demonstrates the paradigm shift from task-centric (Airflow) to data-centric (Dagster).
#
# YOUR TASK: Implement the `cleaned_weather_data` asset below.
#
# This asset depends on `raw_weather_data` -- and here's the magic of Dagster:
# you express that dependency simply by naming the parameter `raw_weather_data`.
# Dagster sees the parameter name matches another asset and wires them together.
#
# WHAT THIS ASSET SHOULD DO:
#   1. Read raw weather readings from the `raw_weather_readings` PostgreSQL table
#   2. Create a `cleaned_weather_readings` table with the same schema plus:
#      - Filter out rows where temperature_c is outside [-60, 60] (sensor errors)
#      - Filter out rows where humidity_pct is outside [0, 100]
#      - Filter out rows where wind_speed_kmh is negative
#   3. INSERT the valid rows into `cleaned_weather_readings` (TRUNCATE first to
#      make this idempotent -- re-running produces the same result)
#   4. Return a MaterializeResult with metadata:
#      - "row_count": number of valid rows inserted
#      - "rows_filtered": number of invalid rows removed
#      - "filter_rules": text description of the filters applied
#
# IMPORTANT DAGSTER CONCEPTS AT PLAY:
#
#   DEPENDENCY VIA PARAMETER NAME:
#     The parameter `raw_weather_data` matches the function name of the upstream
#     asset. Dagster uses this to build the dependency graph automatically.
#     In Airflow, you'd write: clean_task.set_upstream(raw_task) or raw >> clean.
#     In Dagster, the dependency is implicit from the function signature.
#
#   IDEMPOTENCY:
#     Assets should be idempotent -- materializing them twice produces the same
#     result. That's why we TRUNCATE before INSERT. In Airflow, idempotency is
#     your responsibility and often forgotten, leading to duplicate data.
#
#   MaterializeResult:
#     Instead of returning the data directly, we return metadata ABOUT the data.
#     The actual data lives in PostgreSQL. This is a common pattern when assets
#     represent database tables rather than in-memory DataFrames.
#
# HINT:
#   The function signature should look like this:
#
#   @asset(
#       description="...",
#       group_name="phase1_weather",
#       compute_kind="python",
#       deps=[raw_weather_data],  # <-- Alternative: explicit deps instead of parameter name
#   )
#   def cleaned_weather_data(
#       context: AssetExecutionContext,
#       postgres: PostgresResource,
#   ) -> MaterializeResult:
#
#   Note: When using `deps=[raw_weather_data]`, the upstream asset's output is
#   NOT passed as a parameter. This is useful when assets communicate via a
#   shared database (like we do here) rather than passing Python objects.
#
#   Alternatively, you CAN use the parameter name approach:
#
#   def cleaned_weather_data(
#       context: AssetExecutionContext,
#       postgres: PostgresResource,
#       raw_weather_data: MaterializeResult,  # <-- Dagster passes the upstream output
#   ) -> MaterializeResult:
#
#   But since both assets read/write to PostgreSQL tables directly, the `deps=`
#   approach is cleaner here. The parameter name approach shines when assets pass
#   DataFrames or Python objects between them (Phase 2 shows this).
#
# WORKFLOW AFTER IMPLEMENTING:
#   1. Save this file
#   2. Dagit auto-reloads (watch the "Code location" status in the top bar)
#   3. Navigate to Assets > phase1_weather group
#   4. You should see two assets: raw_weather_data -> cleaned_weather_data
#   5. Click "Materialize all" to run both in dependency order
#   6. Click on cleaned_weather_data to see your metadata in the "Latest materialization" panel
# ---------------------------------------------------------------------------


@asset(
    description="Weather readings with invalid sensor data filtered out. "
    "Removes readings with impossible temperature, humidity, or wind values.",
    group_name="phase1_weather",
    compute_kind="python",
    deps=[raw_weather_data],
)
def cleaned_weather_data(
    context: AssetExecutionContext,
    postgres: PostgresResource,
) -> MaterializeResult:
    """Filter raw weather readings, removing invalid sensor data.

    TODO(human): Implement this asset. See the detailed instructions above.

    Steps:
      1. Ensure the cleaned_weather_readings table exists (same schema as raw)
      2. TRUNCATE the table (idempotency)
      3. INSERT INTO cleaned_weather_readings SELECT ... FROM raw_weather_readings
         WHERE temperature_c BETWEEN -60 AND 60
           AND humidity_pct BETWEEN 0 AND 100
           AND wind_speed_kmh >= 0
      4. Query the counts and return MaterializeResult with metadata
    """
    # ------------------------------------------------------------------
    # STUB: Remove this and implement the real logic.
    # This stub ensures Dagster can load the asset even before you implement it.
    # ------------------------------------------------------------------
    context.log.warning("cleaned_weather_data is not yet implemented -- returning stub result")
    return MaterializeResult(
        metadata={
            "row_count": MetadataValue.int(0),
            "rows_filtered": MetadataValue.int(0),
            "status": MetadataValue.text("NOT IMPLEMENTED -- see TODO(human) above"),
        }
    )
