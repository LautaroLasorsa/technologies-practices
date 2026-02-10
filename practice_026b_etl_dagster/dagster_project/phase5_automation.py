"""Phase 5: Sensors & Automation (~15 min)

THE AUTOMATION SPECTRUM: CRON vs EVENTS vs FRESHNESS

    In Airflow (026a), automation means CRON:
        schedule="0 2 * * *"  # Run at 2am every day
        # Problem: What if the upstream data arrives at 3am? You get empty results.
        # Solution: Add a sensor (Airflow FileSensor, ExternalTaskSensor) that polls.
        # But sensors in Airflow CONSUME a worker slot while waiting!

    In Dagster, you have THREE automation mechanisms:

    1. SCHEDULES (like Airflow cron):
       @schedule(cron_schedule="0 2 * * *", job=my_job)
       Same as Airflow. Simple, but has the same "data might not be ready" problem.

    2. SENSORS (event-driven, efficient):
       @sensor(asset_selection=...)
       def my_sensor(context):
           if new_data_available():
               yield RunRequest(...)
       Dagster sensors are LIGHTWEIGHT -- they run on the daemon process, not on workers.
       In Airflow, a sensor task occupies a worker slot for its entire wait time.

    3. AUTO-MATERIALIZATION (declarative, freshness-based):
       @asset(auto_materialize_policy=AutoMaterializePolicy.eager())
       def my_asset(...):
           ...
       Dagster automatically materializes assets when:
       - Their upstream dependencies have been updated
       - They're missing or stale according to a FreshnessPolicy
       - A partition is missing

       THIS IS THE KILLER FEATURE. You declare "this data should be fresh within 6 hours"
       and Dagster figures out the execution plan. No cron, no sensors, no manual wiring.

    Comparison table:
    | Feature | Airflow | Dagster |
    |---------|---------|---------|
    | Cron scheduling | Native (@daily, cron) | Schedules (similar) |
    | Event-driven | FileSensor, ExternalTaskSensor (blocks worker) | @sensor (lightweight, daemon) |
    | Freshness-based | Not native (need external tools) | FreshnessPolicy + AutoMaterialize |
    | Cost of waiting | Sensor task occupies worker slot | Sensor runs on daemon, no slot cost |
    | Backfill automation | CLI-only, manual | Auto-materialize handles missing partitions |

WHAT YOU'LL LEARN:
    1. Asset sensors -- react to upstream asset materializations
    2. FreshnessPolicy -- declare how fresh data should be
    3. AutoMaterializePolicy -- let Dagster decide when to materialize
    4. Combining sensors with partitioned assets
"""

from datetime import timedelta
from typing import Any

from dagster import (
    AssetExecutionContext,
    AssetKey,
    AssetSelection,
    AutoMaterializePolicy,
    AutoMaterializeRule,
    DefaultSensorStatus,
    FreshnessPolicy,
    MaterializeResult,
    MetadataValue,
    RunRequest,
    SensorEvaluationContext,
    SensorResult,
    asset,
    asset_sensor,
    sensor,
)

from dagster_project.resources import PostgresResource


# ---------------------------------------------------------------------------
# REFERENCE IMPLEMENTATION: Asset with FreshnessPolicy
# ---------------------------------------------------------------------------
# A FreshnessPolicy declares: "This asset should have been materialized within
# the last N hours/minutes." The Dagster daemon monitors freshness and, with
# auto-materialization enabled, will automatically trigger materialization
# when the asset goes stale.
#
# COMPARISON WITH AIRFLOW:
#   In Airflow, there's no concept of "data freshness." You schedule a DAG
#   to run at 2am and HOPE the data is fresh. If the DAG fails, you don't
#   know the data is stale until someone checks a dashboard.
#
#   In Dagster, freshness is DECLARED and MONITORED:
#   - Green: asset was materialized within the freshness window
#   - Yellow: asset is approaching staleness
#   - Red: asset is stale and needs re-materialization
#   This shows up in Dagit's asset health dashboard.
# ---------------------------------------------------------------------------


@asset(
    description="Weather report that should be refreshed every 6 hours. "
    "Dagster monitors freshness and auto-materializes when stale.",
    group_name="phase5_automation",
    compute_kind="python",
    # FreshnessPolicy: this asset should be materialized at least every 6 hours.
    # If it hasn't been materialized in 6 hours, Dagster marks it as stale.
    freshness_policy=FreshnessPolicy(maximum_lag_minutes=360),  # 6 hours
    # AutoMaterializePolicy: Dagster will automatically materialize this asset
    # when its upstream dependencies are updated OR when it becomes stale.
    auto_materialize_policy=AutoMaterializePolicy.eager(),
)
def automated_weather_report(
    context: AssetExecutionContext,
    postgres: PostgresResource,
) -> MaterializeResult:
    """Generate a weather report that auto-refreshes based on freshness policy.

    COMPARISON WITH AIRFLOW:
        In Airflow, you'd schedule this as:
            @dag(schedule="0 */6 * * *")  # every 6 hours
            def weather_report_dag():
                ...

        Problems with the Airflow approach:
        1. If upstream data isn't ready at the scheduled time, you get stale results
        2. If upstream data arrives early, you wait until the next schedule
        3. If the DAG fails, nothing retries it until the next schedule
        4. There's no monitoring of data freshness between runs

        In Dagster with FreshnessPolicy + AutoMaterializePolicy:
        1. If upstream data updates, this asset auto-materializes
        2. If the asset hasn't been refreshed in 6 hours, Dagster triggers it
        3. Failed materializations are retried by the daemon
        4. Dagit shows real-time freshness status
    """
    rows = postgres.execute_query(
        """
        SELECT city, AVG(temperature_c), AVG(humidity_pct), MAX(wind_speed_kmh), COUNT(*)
        FROM raw_weather_readings
        WHERE reading_date >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY city
        ORDER BY city
        """
    )

    report = {
        row[0]: {
            "avg_temp": round(float(row[1]), 1),
            "avg_humidity": round(float(row[2]), 1),
            "max_wind": round(float(row[3]), 1),
            "readings": row[4],
        }
        for row in rows
    }

    context.log.info(f"Generated weather report for {len(report)} cities")

    return MaterializeResult(
        metadata={
            "city_count": MetadataValue.int(len(report)),
            "report_preview": MetadataValue.json(
                {k: v for k, v in list(report.items())[:3]}
            ),
        }
    )


# ---------------------------------------------------------------------------
# TODO(human): Implement a sensor that detects new weather data
# ---------------------------------------------------------------------------
# YOUR TASK: Implement the `new_data_sensor` that checks for new weather
# data in PostgreSQL and triggers materialization of the weather report.
#
# WHAT THIS SENSOR SHOULD DO:
#   1. Query PostgreSQL for the maximum `recorded_at` timestamp in raw_weather_readings
#   2. Compare it with the last known timestamp (stored in the sensor cursor)
#   3. If new data is found (timestamp > cursor), yield a RunRequest to
#      materialize `automated_weather_report`
#   4. Update the cursor with the new timestamp
#
# HOW DAGSTER SENSORS WORK:
#   A sensor is a function that Dagster's daemon calls periodically (default: every 30s).
#   The function checks some external condition and optionally yields RunRequests
#   to trigger asset materializations or job runs.
#
#   Sensors have a CURSOR -- a persistent string value that survives between evaluations.
#   Use it to track state: "what was the last data I saw?"
#
#   @sensor(asset_selection=AssetSelection.assets(my_asset))
#   def my_sensor(context: SensorEvaluationContext):
#       last_seen = context.cursor or "1970-01-01"
#       new_data = check_for_new_data(since=last_seen)
#       if new_data:
#           context.update_cursor(new_data.timestamp)
#           yield RunRequest(run_key=f"new_data_{new_data.timestamp}")
#
# COMPARISON WITH AIRFLOW SENSORS:
#   In Airflow, sensors are TASKS that block a worker slot:
#       FileSensor(
#           task_id="wait_for_data",
#           filepath="/data/weather_*.csv",
#           poke_interval=60,  # check every 60 seconds
#           timeout=3600,  # give up after 1 hour
#       )
#   This sensor OCCUPIES a worker slot for up to 1 hour! If you have 10 sensors
#   waiting, that's 10 worker slots wasted on polling.
#
#   Dagster sensors run on the DAEMON process, not on workers. They're lightweight
#   Python functions that execute quickly. No worker slots consumed.
#
# STEP BY STEP:
#   1. Use the @sensor decorator with:
#      - asset_selection=AssetSelection.assets(automated_weather_report)
#      - default_status=DefaultSensorStatus.RUNNING (auto-start)
#      - minimum_interval_seconds=30
#
#   2. Inside the function:
#      a. Get the PostgresResource (sensors don't get resources injected --
#         you'll need to create a connection manually or use build_sensor_context)
#      b. Actually, for simplicity, use psycopg2 directly:
#         import psycopg2
#         conn = psycopg2.connect(host="postgres", user="dagster", ...)
#      c. Query: SELECT MAX(recorded_at) FROM raw_weather_readings
#      d. Compare with context.cursor
#      e. If new data: yield RunRequest, update cursor
#
# HINT:
#   The run_key prevents duplicate runs. If the same run_key is yielded twice,
#   Dagster skips the second one. Use the timestamp as the run_key.
# ---------------------------------------------------------------------------


@sensor(
    description="Detects new weather data in PostgreSQL and triggers report generation. "
    "Runs every 30 seconds on the Dagster daemon (no worker slots consumed).",
    asset_selection=AssetSelection.assets(automated_weather_report),
    default_status=DefaultSensorStatus.STOPPED,  # Start manually in Dagit for safety
    minimum_interval_seconds=30,
)
def new_data_sensor(context: SensorEvaluationContext) -> SensorResult:
    """Check for new weather data and trigger report materialization.

    TODO(human): Implement the sensor logic.

    Steps:
      1. Connect to PostgreSQL (use psycopg2 directly since sensors don't get resources)
      2. Query MAX(recorded_at) from raw_weather_readings
      3. Compare with context.cursor (the last seen timestamp)
      4. If new data found:
         - Yield a RunRequest with run_key=str(max_timestamp)
         - Update the cursor: return SensorResult(run_requests=[...], cursor=str(max_timestamp))
      5. If no new data: return SensorResult(run_requests=[])

    Compare with Airflow:
      In Airflow, you'd use a SqlSensor:
          SqlSensor(
              task_id="check_new_data",
              conn_id="postgres_default",
              sql="SELECT COUNT(*) FROM raw_weather_readings WHERE recorded_at > '{{ prev_ds }}'",
              poke_interval=60,
              timeout=3600,  # blocks a worker for up to 1 hour!
          )
      In Dagster, this sensor runs on the daemon and consumes zero worker resources.
    """
    # ------------------------------------------------------------------
    # STUB: Remove this and implement the real logic.
    # ------------------------------------------------------------------
    context.log.warning("new_data_sensor is not yet implemented -- skipping")
    return SensorResult(run_requests=[])


# ---------------------------------------------------------------------------
# TODO(human): Implement an asset with AutoMaterializePolicy
# ---------------------------------------------------------------------------
# YOUR TASK: Create an asset with an AutoMaterializePolicy that ONLY
# materializes when its upstream dependency has been updated.
#
# WHAT THIS ASSET SHOULD DO:
#   1. Depend on `automated_weather_report` (by parameter name or deps=)
#   2. Read the report and format it as a text summary (e.g., for an email)
#   3. Use AutoMaterializePolicy.eager() so it auto-runs when the report updates
#   4. Use a FreshnessPolicy to also ensure it runs at least every 12 hours
#
# AUTO-MATERIALIZE POLICIES EXPLAINED:
#   Dagster has two built-in policies:
#
#   AutoMaterializePolicy.eager():
#     Materialize as soon as ANY of these are true:
#     - The asset has never been materialized
#     - Any upstream dependency has been materialized since the last time this ran
#     - A FreshnessPolicy declares the asset is stale
#
#   AutoMaterializePolicy.lazy():
#     Only materialize when a DOWNSTREAM asset requests it (e.g., a downstream
#     asset with eager() needs this one to be fresh).
#
#   You can also build custom policies with AutoMaterializeRule:
#     AutoMaterializePolicy(
#         rules={
#             AutoMaterializeRule.materialize_on_parent_updated(),
#             AutoMaterializeRule.materialize_on_missing(),
#         }
#     )
#
# COMPARISON WITH AIRFLOW:
#   Airflow has NO equivalent. The closest is:
#   - TriggerDagRunOperator (one DAG triggers another -- but it's imperative, not declarative)
#   - ExternalTaskSensor (wait for another DAG's task -- blocks a worker slot)
#   - Dataset-triggered DAGs (Airflow 2.4+ -- similar concept but less mature than Dagster)
#
#   In Dagster, auto-materialization is built into the core model. You declare
#   "this asset should be fresh" and the daemon handles the rest.
#
# HINT:
#   @asset(
#       description="...",
#       group_name="phase5_automation",
#       auto_materialize_policy=AutoMaterializePolicy.eager(),
#       freshness_policy=FreshnessPolicy(maximum_lag_minutes=720),  # 12 hours
#   )
#   def formatted_weather_email(
#       context: AssetExecutionContext,
#       automated_weather_report: MaterializeResult,  # depends on upstream
#   ) -> MaterializeResult:
#       ...
#
# NOTE: Since automated_weather_report returns MaterializeResult (metadata only),
# you'll need to read the data from PostgreSQL. Alternatively, you can use
# deps=[automated_weather_report] and accept postgres as a resource instead.
# ---------------------------------------------------------------------------


@asset(
    description="Formatted weather email body, auto-materialized when the weather "
    "report is updated. Demonstrates AutoMaterializePolicy.",
    group_name="phase5_automation",
    compute_kind="python",
    freshness_policy=FreshnessPolicy(maximum_lag_minutes=720),
    auto_materialize_policy=AutoMaterializePolicy.eager(),
    deps=[automated_weather_report],
)
def formatted_weather_email(
    context: AssetExecutionContext,
    postgres: PostgresResource,
) -> MaterializeResult:
    """Format the weather report as an email body.

    TODO(human): Implement this asset.

    Steps:
      1. Read the latest weather data from PostgreSQL (same query as the report)
      2. Format it as a readable text summary:
         "Weather Report - {date}
          City: New York | Avg Temp: 15.2C | Humidity: 65% | Max Wind: 45 km/h
          City: London   | Avg Temp: 10.5C | Humidity: 78% | Max Wind: 30 km/h
          ..."
      3. Return MaterializeResult with the formatted text as metadata

    Compare with Airflow:
      In Airflow, you'd chain: extract >> dbt_run >> report >> email
      with a cron schedule. If the upstream data arrives late, the email
      goes out with stale data.

      In Dagster, this asset auto-materializes when automated_weather_report
      is updated. It's event-driven, not time-driven.
    """
    # ------------------------------------------------------------------
    # STUB: Remove this and implement the real logic.
    # ------------------------------------------------------------------
    context.log.warning("formatted_weather_email not yet implemented -- returning stub")
    return MaterializeResult(
        metadata={
            "status": MetadataValue.text("NOT IMPLEMENTED -- see TODO(human) above"),
        }
    )
