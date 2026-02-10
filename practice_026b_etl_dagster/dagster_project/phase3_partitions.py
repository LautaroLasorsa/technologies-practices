"""Phase 3: Partitioned Assets (~20 min)

PARTITIONS -- DAGSTER'S KILLER FEATURE:
    In Airflow (026a), you process data by date using `execution_date` (or `data_interval_start`).
    But Airflow doesn't truly understand partitions:
        - You can't easily see which dates have been processed and which haven't
        - Backfilling is a separate CLI command with different semantics
        - Re-running a specific date requires navigating the "clear task" UI
        - There's no guarantee your operators actually use execution_date correctly

    In Dagster, partitions are FIRST-CLASS:
        - Each asset can declare a partition scheme (daily, hourly, static, custom)
        - Dagster tracks which partitions are materialized (green) and which are missing (gray)
        - Backfills are a single click in the UI: select a date range, hit "Backfill"
        - The partition key is injected into your asset function via context.partition_key
        - The asset graph shows a PARTITION STATUS MATRIX: asset x partition -> status

    This is the single biggest practical improvement over Airflow for batch ETL.

PARTITION TYPES:
    Dagster supports several partition schemes:
    - DailyPartitionsDefinition  -- one partition per day (most common for ETL)
    - HourlyPartitionsDefinition -- one partition per hour
    - WeeklyPartitionsDefinition -- one partition per week
    - MonthlyPartitionsDefinition -- one partition per month
    - StaticPartitionsDefinition -- fixed set of partitions (e.g., regions, categories)
    - DynamicPartitionsDefinition -- partitions added at runtime

    For this practice, we use DailyPartitionsDefinition since our weather data
    is organized by date.

HOW PARTITIONED ASSETS WORK:
    1. You declare the partition scheme on the asset
    2. When materializing, you choose WHICH partition(s) to process
    3. Inside the asset function, context.partition_key gives you the specific partition
    4. Dagster records the materialization PER PARTITION
    5. You can re-materialize a specific partition without affecting others

    Example flow:
    - Monday: Materialize partition "2024-01-15" --> processes Monday's data
    - Tuesday: Materialize partition "2024-01-16" --> processes Tuesday's data
    - Wednesday: Discover Monday's data was bad --> re-materialize "2024-01-15" only
    - In Airflow, that last step means: clear the task, re-run the DAG for that date,
      hope nothing else breaks. In Dagster: click the partition, click materialize.
"""

from datetime import datetime, timedelta
from typing import Any

from dagster import (
    AssetExecutionContext,
    DailyPartitionsDefinition,
    MaterializeResult,
    MetadataValue,
    asset,
)

from dagster_project.resources import PostgresResource


# ---------------------------------------------------------------------------
# Define the partition scheme shared across all Phase 3 assets.
# All assets in this pipeline use the SAME partition definition, which means
# Dagster can track the partition status matrix across the entire graph.
# ---------------------------------------------------------------------------

daily_partitions = DailyPartitionsDefinition(
    start_date="2024-01-01",
    # No end_date means partitions extend to today (and tomorrow, for scheduling)
)


# ---------------------------------------------------------------------------
# REFERENCE IMPLEMENTATION: partitioned_raw_readings
# ---------------------------------------------------------------------------
# This asset uses `daily_partitions` and reads only the data for the
# partition's date. Study how `context.partition_key` is used.
# ---------------------------------------------------------------------------


@asset(
    description="Raw weather readings for a specific date partition. "
    "Each materialization processes exactly one day of data.",
    group_name="phase3_partitioned",
    compute_kind="python",
    partitions_def=daily_partitions,
)
def partitioned_raw_readings(
    context: AssetExecutionContext,
    postgres: PostgresResource,
) -> list[dict[str, Any]]:
    """Fetch raw weather readings for a single date partition.

    COMPARISON WITH AIRFLOW:
        In Airflow, you'd access the execution date like this:
            def my_task(**kwargs):
                date = kwargs["data_interval_start"].strftime("%Y-%m-%d")
                # ... query WHERE date = ...

        In Dagster, the partition key is explicitly provided:
            date = context.partition_key  # "2024-01-15"

        The difference seems small, but Dagster's approach means:
        - The partition key is typed and validated
        - Dagster tracks WHICH partitions have been materialized
        - You can see partition status in a matrix view in Dagit
        - Backfills automatically iterate over partition keys
    """
    # context.partition_key is a string like "2024-01-15"
    partition_date = context.partition_key
    context.log.info(f"Processing partition: {partition_date}")

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
        "SELECT city, temperature_c, humidity_pct, wind_speed_kmh, reading_date::text "
        "FROM raw_weather_readings WHERE reading_date = %s",
        (partition_date,),
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

    context.log.info(f"Partition {partition_date}: {len(readings)} readings")
    return readings


# ---------------------------------------------------------------------------
# TODO(human): Implement partitioned_validated_readings
# ---------------------------------------------------------------------------
# YOUR TASK: Implement a PARTITIONED version of the validation asset from Phase 2.
#
# WHAT THIS ASSET SHOULD DO:
#   1. Accept `partitioned_raw_readings` as a parameter (Dagster passes the list
#      of dicts for the SAME partition -- partition keys propagate through the graph!)
#   2. Filter out invalid readings (same rules as Phase 1/2):
#      - temperature_c between -60 and 60
#      - humidity_pct between 0 and 100
#      - wind_speed_kmh >= 0
#   3. Return the filtered list of dicts
#
# CRITICAL DAGSTER CONCEPT -- PARTITION KEY PROPAGATION:
#   When you materialize partition "2024-01-15" of `partitioned_validated_readings`,
#   Dagster automatically materializes partition "2024-01-15" of its upstream dependency
#   `partitioned_raw_readings` (if not already materialized).
#
#   The partition key flows through the entire graph. You don't need to manually
#   pass dates around or coordinate which partition to process -- Dagster handles it.
#
#   In Airflow, you'd need to ensure every task in the DAG uses the same
#   execution_date, and if any task ignores it, you get data for the wrong day.
#   Dagster makes this impossible: the partition key is injected by the framework.
#
# IMPORTANT -- SAME partitions_def:
#   Both assets MUST use the same `daily_partitions` definition. This tells Dagster
#   that partition "2024-01-15" of this asset maps to partition "2024-01-15" of the
#   upstream asset. If they used different partition definitions, Dagster would need
#   a PartitionMapping to translate between them.
#
# HINT:
#   This is almost identical to Phase 2's validated_readings, but with:
#   - `partitions_def=daily_partitions` in the @asset decorator
#   - context.partition_key available for logging
#   - The data only contains ONE day's readings (thanks to partitioning)
# ---------------------------------------------------------------------------


@asset(
    description="Validated weather readings for a specific date partition. "
    "Invalid sensor data is filtered out.",
    group_name="phase3_partitioned",
    compute_kind="python",
    partitions_def=daily_partitions,
)
def partitioned_validated_readings(
    context: AssetExecutionContext,
    partitioned_raw_readings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter raw readings for a single partition, removing invalid sensor data.

    TODO(human): Implement the validation logic.

    Steps:
      1. Get the partition date from context.partition_key (for logging)
      2. Filter the list using the same validation rules as Phase 2
      3. Log: "Partition {date}: {valid}/{total} readings passed validation"
      4. Return the valid readings
    """
    # ------------------------------------------------------------------
    # STUB: Remove this and implement the real logic.
    # ------------------------------------------------------------------
    partition_date = context.partition_key
    context.log.warning(
        f"partitioned_validated_readings not yet implemented for {partition_date} -- "
        "returning all readings as-is"
    )
    return partitioned_raw_readings


# ---------------------------------------------------------------------------
# TODO(human): Implement partitioned_city_daily_stats
# ---------------------------------------------------------------------------
# YOUR TASK: Implement a PARTITIONED aggregation asset.
#
# WHAT THIS ASSET SHOULD DO:
#   1. Accept `partitioned_validated_readings` as a parameter
#   2. Group readings by city (remember: all readings are for the same date)
#   3. For each city, compute: avg_temperature, avg_humidity, max_wind, reading_count
#   4. Write the results to a `city_daily_stats` PostgreSQL table with columns:
#      - city TEXT
#      - reading_date DATE
#      - avg_temperature REAL
#      - avg_humidity REAL
#      - max_wind REAL
#      - reading_count INTEGER
#   5. Use DELETE + INSERT for the specific partition date (not TRUNCATE!)
#      because other partitions' data should be preserved.
#   6. Return MaterializeResult with metadata (city count, partition date)
#
# CRITICAL DAGSTER CONCEPT -- PARTITION-AWARE WRITES:
#   When writing partitioned data to a database, you MUST only modify the
#   rows belonging to the current partition. DO NOT truncate the entire table!
#
#   Pattern:
#     DELETE FROM city_daily_stats WHERE reading_date = '{partition_key}'
#     INSERT INTO city_daily_stats (...) VALUES (...)
#
#   This ensures:
#   - Idempotency: re-materializing a partition replaces only that partition's data
#   - Isolation: other partitions remain untouched
#   - Backfills: you can re-process a date range without affecting today's data
#
#   In Airflow, getting this right requires discipline. In Dagster, the partition_key
#   is always available and the framework encourages this pattern.
#
# HINT:
#   You need BOTH `partitioned_validated_readings` (upstream data) AND
#   `postgres` (resource for writing) as parameters. Dagster handles both:
#
#   def partitioned_city_daily_stats(
#       context: AssetExecutionContext,
#       postgres: PostgresResource,
#       partitioned_validated_readings: list[dict],
#   ) -> MaterializeResult:
# ---------------------------------------------------------------------------


@asset(
    description="Per-city daily weather statistics for a specific date partition. "
    "Aggregates validated readings into avg temperature, humidity, and max wind.",
    group_name="phase3_partitioned",
    compute_kind="python",
    partitions_def=daily_partitions,
)
def partitioned_city_daily_stats(
    context: AssetExecutionContext,
    postgres: PostgresResource,
    partitioned_validated_readings: list[dict[str, Any]],
) -> MaterializeResult:
    """Aggregate validated readings by city for a single date partition.

    TODO(human): Implement the aggregation and PostgreSQL write.

    Steps:
      1. Get the partition date from context.partition_key
      2. Ensure the city_daily_stats table exists
      3. DELETE existing rows for this partition date
      4. Group readings by city and compute aggregates
      5. INSERT the aggregated rows
      6. Return MaterializeResult with metadata
    """
    # ------------------------------------------------------------------
    # STUB: Remove this and implement the real logic.
    # ------------------------------------------------------------------
    partition_date = context.partition_key
    context.log.warning(
        f"partitioned_city_daily_stats not yet implemented for {partition_date} -- "
        "returning stub result"
    )
    return MaterializeResult(
        metadata={
            "partition_date": MetadataValue.text(partition_date),
            "status": MetadataValue.text("NOT IMPLEMENTED -- see TODO(human) above"),
        }
    )
