"""Phase 2: Asset Graph & Dependencies (~20 min)

BUILDING A MULTI-ASSET PIPELINE:
    In Phase 1, we had two assets: raw -> cleaned. Now we build a deeper graph:

        raw_readings --> validated_readings --> city_aggregates --> daily_summary

    Each asset is a Python function. Dependencies flow through function parameters.
    Dagster builds the graph automatically -- no manual wiring needed.

ASSET GRAPH vs AIRFLOW DAG:
    In Airflow, adding a new downstream task means editing the DAG file:
        extract >> transform >> load
        # Later, someone adds:
        extract >> transform >> load >> notify  # must edit existing DAG

    In Dagster, you just create a new asset that depends on an existing one:
        @asset
        def notify(daily_summary):  # depends on daily_summary
            send_alert(daily_summary)

    The upstream assets don't change. This is the Open/Closed principle applied
    to data pipelines: open for extension, closed for modification.

I/O MANAGERS -- HOW DAGSTER PASSES DATA BETWEEN ASSETS:
    When one asset returns a value and another asset takes it as a parameter,
    Dagster uses an "I/O Manager" to serialize/deserialize the data between them.

    The default I/O Manager pickles objects to the local filesystem. In production,
    you'd configure an I/O manager that writes to S3, GCS, or a database.

    For this phase, we use DIRECT PYTHON OBJECTS (dicts, lists) passed between
    assets via the default I/O manager. This shows the simplest Dagster pattern.

    In Phase 3, we switch back to PostgreSQL-based assets with explicit I/O.

WHAT YOU'LL LEARN:
    1. Multi-step asset chains with automatic dependency resolution
    2. Passing Python objects between assets (Dagster handles serialization)
    3. Asset metadata for observability
    4. Selective re-materialization (only rebuild what changed)
"""

from typing import Any

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from dagster_project.resources import PostgresResource


# ---------------------------------------------------------------------------
# REFERENCE IMPLEMENTATION: raw_readings
# ---------------------------------------------------------------------------
# Entry point of the Phase 2 pipeline. Reads from PostgreSQL and returns
# a Python list of dicts -- demonstrating how assets can pass data objects
# (not just database table references) to downstream assets.
# ---------------------------------------------------------------------------


@asset(
    description="Raw weather readings fetched from PostgreSQL as Python dicts. "
    "Entry point of the Phase 2 asset graph.",
    group_name="phase2_weather_graph",
    compute_kind="python",
)
def raw_readings(
    context: AssetExecutionContext,
    postgres: PostgresResource,
) -> list[dict[str, Any]]:
    """Fetch raw weather readings from PostgreSQL.

    IMPORTANT: This asset RETURNS a Python object (list of dicts).
    Dagster's I/O manager will serialize it and pass it to downstream assets
    that declare `raw_readings` as a parameter.

    COMPARISON WITH AIRFLOW:
        In Airflow, you'd push this to XCom: `ti.xcom_push(key="data", value=data)`.
        XCom has size limits (~48KB default), is stored in the metadata DB, and
        requires explicit push/pull calls.

        In Dagster, just `return` the value. Dagster handles the rest.
    """
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

    rows = postgres.execute_query(
        """
        SELECT city, temperature_c, humidity_pct, wind_speed_kmh, reading_date::text
        FROM raw_weather_readings
        ORDER BY reading_date, city
        """
    )

    readings = [
        {
            "city": row[0],
            "temperature_c": row[1],
            "humidity_pct": row[2],
            "wind_speed_kmh": row[3],
            "reading_date": row[4],
        }
        for row in rows
    ]

    context.log.info(f"Fetched {len(readings)} raw readings")
    return readings


# ---------------------------------------------------------------------------
# TODO(human): Implement validated_readings
# ---------------------------------------------------------------------------
# YOUR TASK: Implement the `validated_readings` asset.
#
# This asset depends on `raw_readings` -- notice how you declare the dependency
# by simply naming the parameter `raw_readings` (matching the upstream asset name).
# Dagster passes the LIST OF DICTS that `raw_readings` returned.
#
# WHAT THIS ASSET SHOULD DO:
#   1. Accept `raw_readings` as a parameter (Dagster injects the upstream output)
#   2. Filter out invalid readings:
#      - temperature_c must be between -60 and 60
#      - humidity_pct must be between 0 and 100
#      - wind_speed_kmh must be >= 0
#   3. Add a "validated" flag to each reading: reading["is_valid"] = True
#   4. Return the list of valid readings (as list[dict])
#
# DAGSTER CONCEPT -- DEPENDENCY VIA PARAMETER NAME:
#   When you write:
#       def validated_readings(raw_readings: list[dict]) -> list[dict]:
#   Dagster sees that `raw_readings` matches the name of another asset and:
#     a) Adds an edge in the asset graph: raw_readings -> validated_readings
#     b) When materializing, runs raw_readings first, then passes its output here
#     c) If raw_readings was already materialized, loads it from the I/O manager
#
#   This is profoundly different from Airflow where you'd write:
#       data = ti.xcom_pull(task_ids="raw_readings", key="return_value")
#   Dagster eliminates the XCom boilerplate and makes dependencies type-safe.
#
# HINT:
#   The implementation is a simple list comprehension:
#   valid = [r for r in raw_readings if -60 <= r["temperature_c"] <= 60 and ...]
#   Return that list. Dagster handles the serialization.
#
# METADATA:
#   Log how many readings passed/failed validation using context.log.info().
#   You can also return the data AND attach metadata using the `Output` type,
#   but for simplicity, just return the list and log the counts.
# ---------------------------------------------------------------------------


@asset(
    description="Weather readings that passed validation filters. "
    "Invalid sensor data (extreme temperatures, impossible humidity) is removed.",
    group_name="phase2_weather_graph",
    compute_kind="python",
)
def validated_readings(
    context: AssetExecutionContext,
    raw_readings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter raw readings, keeping only valid sensor data.

    TODO(human): Implement the validation logic.

    The parameter `raw_readings` is automatically populated by Dagster with
    the output of the `raw_readings` asset. You don't need to fetch it yourself.

    Steps:
      1. Filter the list: keep readings where temperature, humidity, wind are in valid ranges
      2. Add {"is_valid": True} to each valid reading
      3. Log the counts: total, valid, filtered
      4. Return the list of valid readings
    """
    # ------------------------------------------------------------------
    # STUB: Remove this and implement the real logic.
    # ------------------------------------------------------------------
    context.log.warning("validated_readings is not yet implemented -- returning all readings as-is")
    return raw_readings


# ---------------------------------------------------------------------------
# TODO(human): Implement city_aggregates
# ---------------------------------------------------------------------------
# YOUR TASK: Implement the `city_aggregates` asset.
#
# This asset depends on `validated_readings` -- again, just name the parameter.
#
# WHAT THIS ASSET SHOULD DO:
#   1. Accept `validated_readings` as a parameter (list of valid reading dicts)
#   2. Group readings by city
#   3. For each city, compute:
#      - avg_temperature: average of temperature_c
#      - avg_humidity: average of humidity_pct
#      - max_wind: maximum wind_speed_kmh
#      - reading_count: number of readings
#   4. Return a dict[str, dict] mapping city name -> aggregated stats
#
# EXAMPLE OUTPUT:
#   {
#       "New York": {"avg_temperature": 15.2, "avg_humidity": 65.0, "max_wind": 45.0, "reading_count": 120},
#       "London":   {"avg_temperature": 10.5, "avg_humidity": 78.0, "max_wind": 30.0, "reading_count": 115},
#   }
#
# DAGSTER CONCEPT -- ASSET GRAPH GROWS NATURALLY:
#   At this point, Dagster's asset graph looks like:
#       raw_readings -> validated_readings -> city_aggregates
#   Each arrow was inferred from parameter names. No explicit wiring.
#
#   If you later want to add a "city_rankings" asset that also depends on
#   city_aggregates, you just create a new function with `city_aggregates`
#   as a parameter. The upstream assets don't change at all.
#
# HINT:
#   Use collections.defaultdict or a simple loop to group by city.
#   This is basic Python -- the interesting part is how Dagster chains it.
# ---------------------------------------------------------------------------


@asset(
    description="Aggregated weather statistics per city: average temperature, "
    "humidity, max wind speed, and reading count.",
    group_name="phase2_weather_graph",
    compute_kind="python",
)
def city_aggregates(
    context: AssetExecutionContext,
    validated_readings: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate validated readings by city.

    TODO(human): Implement the aggregation logic.

    Steps:
      1. Group readings by city
      2. Compute avg_temperature, avg_humidity, max_wind, reading_count per city
      3. Log the number of cities found
      4. Return the aggregated dict
    """
    # ------------------------------------------------------------------
    # STUB: Remove this and implement the real logic.
    # ------------------------------------------------------------------
    context.log.warning("city_aggregates is not yet implemented -- returning empty dict")
    return {}


# ---------------------------------------------------------------------------
# REFERENCE IMPLEMENTATION: daily_summary
# ---------------------------------------------------------------------------
# This is the final asset in the Phase 2 graph. It consumes city_aggregates
# and writes a summary to PostgreSQL.
#
# Study how it:
#   1. Depends on city_aggregates via parameter name
#   2. Writes results to a PostgreSQL table (production pattern)
#   3. Returns MaterializeResult with rich metadata for Dagit
# ---------------------------------------------------------------------------


@asset(
    description="Daily weather summary written to PostgreSQL. "
    "Combines city aggregates into a single summary table.",
    group_name="phase2_weather_graph",
    compute_kind="python",
)
def daily_summary(
    context: AssetExecutionContext,
    postgres: PostgresResource,
    city_aggregates: dict[str, dict[str, Any]],
) -> MaterializeResult:
    """Write city aggregates to a PostgreSQL summary table.

    COMPARISON WITH AIRFLOW:
        In Airflow, loading to a database is typically a separate "load" task
        at the end of the DAG. The data arrives via XCom (fragile, size-limited)
        or via a shared filesystem/S3 path passed between tasks.

        In Dagster, `city_aggregates` is passed directly as a Python dict.
        No XCom, no shared filesystem, no serialization headaches.

    NOTE: This asset takes BOTH a resource (postgres) AND an upstream asset
    (city_aggregates) as parameters. Dagster distinguishes them by type:
    resources are ConfigurableResource subclasses, assets are data types.
    """
    # Create the summary table
    postgres.ensure_table(
        "weather_daily_summary",
        """
        city TEXT NOT NULL,
        avg_temperature REAL,
        avg_humidity REAL,
        max_wind REAL,
        reading_count INTEGER,
        computed_at TIMESTAMP DEFAULT NOW(),
        PRIMARY KEY (city)
        """,
    )

    # Truncate for idempotency
    postgres.execute_command("TRUNCATE TABLE weather_daily_summary")

    # Insert aggregates
    for city, stats in city_aggregates.items():
        postgres.execute_command(
            """
            INSERT INTO weather_daily_summary (city, avg_temperature, avg_humidity, max_wind, reading_count)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                city,
                stats.get("avg_temperature", 0),
                stats.get("avg_humidity", 0),
                stats.get("max_wind", 0),
                stats.get("reading_count", 0),
            ),
        )

    context.log.info(f"Wrote summary for {len(city_aggregates)} cities")

    # Rich metadata visible in Dagit
    # In Airflow, you'd need a separate monitoring tool (DataDog, Grafana) to see this.
    # Dagster builds observability into the orchestrator itself.
    return MaterializeResult(
        metadata={
            "cities_count": MetadataValue.int(len(city_aggregates)),
            "cities": MetadataValue.text(", ".join(sorted(city_aggregates.keys()))),
            "target_table": MetadataValue.text("weather_daily_summary"),
        }
    )
