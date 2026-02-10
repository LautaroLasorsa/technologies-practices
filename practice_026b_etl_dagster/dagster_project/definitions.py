"""Dagster Definitions -- the entry point for all assets, resources, and sensors.

WHAT IS Definitions?
    In Airflow, the scheduler scans a `dags/` folder and discovers DAG objects
    automatically. This can lead to import errors, stale DAGs, and surprises.

    In Dagster, you have a SINGLE entry point: the Definitions object.
    It explicitly registers everything Dagster should know about:
    - Assets (data assets with dependencies)
    - Resources (database connections, API clients)
    - Jobs (explicit groupings of assets for manual triggering)
    - Sensors (event-driven triggers)
    - Schedules (cron-based triggers)

    This is explicit and predictable: nothing runs unless it's registered here.

HOW DAGSTER LOADS CODE:
    1. workspace.yaml points to this file (dagster_project/definitions.py)
    2. Dagster imports this module and looks for a `defs` variable of type Definitions
    3. All assets, resources, sensors, etc. registered in `defs` become available in Dagit
    4. Hot-reloading: when you save this file, Dagster reloads the code location

    Compare with Airflow:
    - Airflow: drops .py files in dags/ -> scheduler picks them up (minutes delay)
    - Dagster: changes to definitions.py -> instant reload in Dagit
"""

from pathlib import Path

from dagster import Definitions, EnvVar

# ---------------------------------------------------------------------------
# Import assets from each phase module
# ---------------------------------------------------------------------------
# Each phase is a separate Python module with its own assets.
# This is a clean pattern: phases are independent, can be developed/tested
# in isolation, and imported here for registration.
# ---------------------------------------------------------------------------

from dagster_project.phase1_basic_assets import (
    cleaned_weather_data,
    raw_weather_data,
)
from dagster_project.phase2_asset_graph import (
    city_aggregates,
    daily_summary,
    raw_readings,
    validated_readings,
)
from dagster_project.phase3_partitions import (
    partitioned_city_daily_stats,
    partitioned_raw_readings,
    partitioned_validated_readings,
)
from dagster_project.phase4_dbt_assets import (
    weather_alerts,
    weather_source,
)
from dagster_project.phase5_automation import (
    automated_weather_report,
    formatted_weather_email,
    new_data_sensor,
)

# ---------------------------------------------------------------------------
# Import resources
# ---------------------------------------------------------------------------

from dagster_project.resources import PostgresResource

# ---------------------------------------------------------------------------
# Conditionally import dbt assets and resource
# ---------------------------------------------------------------------------

dbt_assets_list = []
dbt_resources = {}

try:
    from dagster_project.phase4_dbt_assets import _dbt_assets_loaded, DBT_PROJECT_DIR, DBT_PROFILES_DIR

    if _dbt_assets_loaded:
        from dagster_project.phase4_dbt_assets import dbt_weather_models
        from dagster_dbt import DbtCliResource

        dbt_assets_list = [dbt_weather_models]
        dbt_resources = {
            "dbt": DbtCliResource(
                project_dir=str(DBT_PROJECT_DIR),
                profiles_dir=str(DBT_PROFILES_DIR),
            ),
        }
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Assemble the Definitions object
# ---------------------------------------------------------------------------
# This is the ONLY thing Dagster loads. Everything must be registered here.
#
# COMPARISON WITH AIRFLOW:
#   Airflow: Each .py file in dags/ is independently scanned for DAG objects.
#            No single registry. Hard to see "everything that's registered."
#   Dagster: One Definitions object = one truth. Easy to audit, version, and test.
# ---------------------------------------------------------------------------

all_assets = [
    # Phase 1: Basic assets
    raw_weather_data,
    cleaned_weather_data,
    # Phase 2: Asset graph
    raw_readings,
    validated_readings,
    city_aggregates,
    daily_summary,
    # Phase 3: Partitioned assets
    partitioned_raw_readings,
    partitioned_validated_readings,
    partitioned_city_daily_stats,
    # Phase 4: dbt pipeline (Python assets; dbt assets added separately)
    weather_source,
    weather_alerts,
    # Phase 5: Automation
    automated_weather_report,
    formatted_weather_email,
] + dbt_assets_list

all_resources = {
    # PostgresResource is injected into any asset/sensor that declares it as a parameter.
    # The resource is instantiated ONCE and shared across all assets in a run.
    #
    # COMPARISON WITH AIRFLOW:
    #   Airflow: PostgresHook(postgres_conn_id="my_conn") -- connection configured in UI
    #   Dagster: postgres: PostgresResource -- configured here, type-safe, testable
    "postgres": PostgresResource(
        host="postgres",
        port=5432,
        user="dagster",
        password="dagster",
        dbname="dagster",
    ),
    **dbt_resources,
}

all_sensors = [
    new_data_sensor,
]

defs = Definitions(
    assets=all_assets,
    resources=all_resources,
    sensors=all_sensors,
)
