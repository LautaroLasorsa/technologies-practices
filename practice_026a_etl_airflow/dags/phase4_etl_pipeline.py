"""
Phase 4: Weather ETL Pipeline with Sensors & Scheduling
=========================================================

This is the most realistic DAG in the practice. It combines:
    - FileSensor: waits for a data file to appear before proceeding
    - @task functions: extract, transform, load steps
    - PostgreSQL interaction: actually writes results to a database table
    - Cron scheduling: runs daily at 6am (or manually for testing)
    - Catchup: whether to "backfill" missed runs

KEY CONCEPTS:

    SENSORS:
        A sensor is a special type of operator that WAITS for a condition to be met.
        It periodically checks (called "poking") until the condition is true,
        then succeeds and lets downstream tasks run.

        Common sensors:
        - FileSensor      — waits for a file to exist at a path
        - ExternalTaskSensor — waits for a task in ANOTHER DAG to complete
        - HttpSensor      — waits for an HTTP endpoint to return success
        - SqlSensor       — waits for a SQL query to return rows
        - S3KeySensor     — waits for a key to exist in an S3 bucket
        - TimeSensor      — waits until a specific time of day

        Sensor parameters:
        - poke_interval: how often to check (seconds). Default: 60
        - timeout: how long to wait before FAILING. Default: 7 days (!)
        - mode: "poke" (blocks a worker slot) or "reschedule" (frees the slot between checks)
          Use "reschedule" for long waits — "poke" holds a worker slot the entire time

    CRON SCHEDULING:
        Airflow uses cron expressions for scheduling:

        ┌───── minute (0-59)
        │ ┌───── hour (0-23)
        │ │ ┌───── day of month (1-31)
        │ │ │ ┌───── month (1-12)
        │ │ │ │ ┌───── day of week (0-6, Sun=0)
        │ │ │ │ │
        * * * * *

        Examples:
            "0 6 * * *"   — daily at 6:00 AM
            "0 */2 * * *" — every 2 hours
            "30 9 * * 1"  — Monday at 9:30 AM
            "0 0 1 * *"   — first day of every month at midnight

        Shortcuts: @daily, @hourly, @weekly, @monthly, @yearly, @once

    CATCHUP & BACKFILL:
        catchup=True (default):
            Airflow creates a run for EVERY missed schedule interval between
            start_date and now. If start_date is 30 days ago and schedule is
            @daily, Airflow queues 30 runs immediately!

        catchup=False:
            Only schedule from "now" forward. No backfilling of old intervals.

        Manual backfill:
            `airflow dags backfill -s 2024-01-01 -e 2024-01-07 phase4_weather_etl`
            This creates runs for a specific date range, useful for reprocessing.

    EXECUTION DATE vs LOGICAL DATE:
        This is Airflow's most confusing concept:

        A daily DAG scheduled at midnight processes the PREVIOUS day's data.
        The "logical_date" (formerly "execution_date") is the START of the interval:
            - Run triggered at 2024-01-02 00:00 → logical_date = 2024-01-01
            - The run REPRESENTS Jan 1, even though it EXECUTES on Jan 2

        Why? Because the data for Jan 1 isn't complete until midnight Jan 2.
        The run waits for the interval to END before executing.

        For manually triggered runs, logical_date ≈ "now" (less confusing).

    POSTGRESQL INTERACTION:
        In this DAG, we write ETL results to the same PostgreSQL that Airflow uses
        for its metadata. This is fine for learning but in production you'd use a
        separate database (data warehouse) for ETL outputs.

        We use psycopg2 (bundled with the Airflow image) to connect directly.
        In production, you'd use PostgresOperator or SqlAlchemyOperator.

WHAT TO DO:
    1. Generate mock data: `uv run python data/mock_weather.py` (from practice root)
    2. Implement the three TODO(human) @task functions
    3. Trigger the DAG manually
    4. Verify results: `uv run python scripts/check_results.py`
"""

import json
import os
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.sensors.filesystem import FileSensor


@dag(
    dag_id="phase4_weather_etl",
    default_args={
        "owner": "learner",
        "retries": 2,
        "retry_delay": timedelta(minutes=1),
    },
    description="Phase 4: Weather ETL — sensors, scheduling, PostgreSQL",
    start_date=datetime(2024, 1, 1),
    # Cron schedule: daily at 6:00 AM
    # For testing, you'll trigger manually — the schedule is here to demonstrate syntax.
    # Change to None if the scheduler keeps creating runs you don't want.
    schedule="0 6 * * *",
    catchup=False,  # Don't backfill — we only want manual triggers for now
    tags=["phase4", "learning", "etl"],
)
def phase4_weather_etl():
    """
    Weather ETL Pipeline:
        1. Wait for data file (FileSensor)
        2. Create target table (if not exists)
        3. Extract from JSON file
        4. Transform: aggregate by city
        5. Load into PostgreSQL
    """

    # =========================================================================
    # FileSensor — waits for the weather data file to exist
    # =========================================================================
    # The sensor checks every `poke_interval` seconds whether the file exists.
    # If not found within `timeout` seconds, the task FAILS.
    #
    # `filepath` is relative to the `fs_conn_id` connection's base path.
    # The default fs_conn_id="fs_default" uses "/" as base path.
    # We use an absolute path inside the container: /opt/airflow/data/
    #
    # For testing: generate the file BEFORE triggering the DAG,
    # or the sensor will wait (and eventually timeout).
    #
    # mode="reschedule": frees the worker slot between pokes.
    # Use this for sensors that might wait a long time.
    # mode="poke" (default): holds the worker slot — faster but wastes resources.

    today_str = "{{ ds }}"  # Jinja template: renders to YYYY-MM-DD at runtime

    wait_for_data = FileSensor(
        task_id="wait_for_weather_data",
        filepath="/opt/airflow/data/weather_{{ ds }}.json",
        poke_interval=10,   # Check every 10 seconds (short for learning)
        timeout=120,         # Fail after 2 minutes if file not found
        mode="poke",         # Fine for short waits
    )

    # =========================================================================
    # FULLY IMPLEMENTED: Create the target table in PostgreSQL
    # =========================================================================
    @task
    def create_table() -> str:
        """
        Create the weather_summary table if it doesn't exist.
        Returns the table name for downstream reference.

        Uses psycopg2 (bundled with Airflow) to connect to PostgreSQL directly.
        The connection string comes from the ETL_DB_CONN environment variable,
        which points to the same Postgres used by Airflow metadata.
        """
        import psycopg2

        # Parse connection from environment (set in docker-compose.yml)
        # Format: postgresql+psycopg2://user:pass@host:port/db
        conn_str = os.environ.get(
            "ETL_DB_CONN",
            "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow",
        )
        # psycopg2 uses a different format than SQLAlchemy
        # Convert: postgresql+psycopg2://user:pass@host:port/db
        #       → host=host port=port dbname=db user=user password=pass
        parts = conn_str.replace("postgresql+psycopg2://", "").split("@")
        user_pass = parts[0].split(":")
        host_db = parts[1].split("/")
        host_port = host_db[0].split(":")

        conn = psycopg2.connect(
            host=host_port[0],
            port=int(host_port[1]),
            dbname=host_db[1],
            user=user_pass[0],
            password=user_pass[1],
        )
        conn.autocommit = True

        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS weather_summary (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    city VARCHAR(100) NOT NULL,
                    avg_temp_c FLOAT,
                    min_temp_c FLOAT,
                    max_temp_c FLOAT,
                    avg_humidity FLOAT,
                    reading_count INTEGER,
                    loaded_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(date, city)
                )
            """)
            print("Table 'weather_summary' ready (created or already exists)")

        conn.close()
        return "weather_summary"

    # =========================================================================
    # TODO(human): Implement extract_weather
    # =========================================================================
    @task
    def extract_weather(table_name: str) -> dict:
        """
        Extract weather data from the JSON file.

        TODO(human): Implement this function. Here's what it should do:

        1. Build the file path to the weather data:
           - The data files are mounted at /opt/airflow/data/ inside the container
           - File naming: weather_YYYY-MM-DD.json
           - Get the current logical date from Airflow context:

             from airflow.operators.python import get_current_context
             context = get_current_context()
             ds = context["ds"]  # "ds" = date stamp in YYYY-MM-DD format

           - Construct: f"/opt/airflow/data/weather_{ds}.json"

        2. Read and parse the JSON file:
           - Use json.load() to parse the file
           - The file structure (created by mock_weather.py) is:
             {
                 "date": "2024-01-15",
                 "readings": [
                     {
                         "city": "New York",
                         "timestamp": "2024-01-15T06:00:00",
                         "temp_f": 32.5,
                         "humidity": 65.0,
                         "wind_mph": 12.3
                     },
                     ...
                 ],
                 "metadata": {
                     "source": "mock_weather_api",
                     "generated_at": "2024-01-15T10:00:00"
                 }
             }

        3. Print extraction summary:
           - Date being processed
           - Number of readings extracted
           - List of unique cities

        4. Return the parsed dict (Airflow stores it as XCom)

        Note: the `table_name` parameter creates a dependency on create_table()
        (we want the table to exist before we start the ETL).
        You don't actually need to USE table_name in this function —
        it's just for dependency ordering. But it's good practice to
        have it as a parameter to make the dependency explicit.

        Context access in TaskFlow:
            Unlike classic operators where you get **kwargs, in @task functions
            you use `get_current_context()` to access the Airflow context:
                from airflow.operators.python import get_current_context
                context = get_current_context()
                ds = context["ds"]           # "2024-01-15"
                logical_date = context["logical_date"]  # datetime object
                run_id = context["run_id"]   # "manual__2024-01-15T..."
        """
        # --- YOUR CODE HERE ---
        return {"date": "", "readings": []}  # Replace with your implementation

    # =========================================================================
    # TODO(human): Implement transform_weather
    # =========================================================================
    @task
    def transform_weather(raw_data: dict) -> list[dict]:
        """
        Transform weather data: aggregate readings by city.

        TODO(human): Implement this function. Here's what it should do:

        1. Group readings by city:
           - `raw_data["readings"]` is a list of individual weather readings
           - Multiple readings per city (different times of day)
           - Group them: {"New York": [reading1, reading2, ...], ...}

           Hint: use a defaultdict(list):
               from collections import defaultdict
               by_city = defaultdict(list)
               for reading in raw_data["readings"]:
                   by_city[reading["city"]].append(reading)

        2. For each city, calculate aggregates:
           - avg_temp_c: average temperature in CELSIUS
             Convert each temp_f to Celsius: (temp_f - 32) * 5/9
             Then average all Celsius values for the city
             Round to 1 decimal place
           - min_temp_c: minimum temperature (Celsius), rounded to 1 decimal
           - max_temp_c: maximum temperature (Celsius), rounded to 1 decimal
           - avg_humidity: average humidity, rounded to 1 decimal
           - reading_count: number of readings for this city

        3. Build a list of summary dicts, one per city:
           [
               {
                   "date": raw_data["date"],
                   "city": "New York",
                   "avg_temp_c": 0.3,
                   "min_temp_c": -2.1,
                   "max_temp_c": 3.5,
                   "avg_humidity": 67.2,
                   "reading_count": 4,
               },
               ...
           ]

        4. Print a summary table (city, avg temp, readings count)

        5. Return the list of summaries

        Why aggregate in a separate step?
            Separation of concerns: extract gets raw data, transform reshapes it.
            If the aggregation logic changes (e.g., add wind speed), only this
            function changes. The extract and load functions stay the same.

            In Airflow's task-centric model, each step is independently retryable.
            If the transform fails, you can re-run JUST the transform without
            re-extracting the data (XCom still has the raw data from the extract).
        """
        # --- YOUR CODE HERE ---
        return []  # Replace with your implementation

    # =========================================================================
    # TODO(human): Implement load_weather
    # =========================================================================
    @task
    def load_weather(summaries: list[dict]) -> dict:
        """
        Load weather summaries into PostgreSQL.

        TODO(human): Implement this function. Here's what it should do:

        1. Connect to PostgreSQL (same pattern as create_table):
               import psycopg2

               conn_str = os.environ.get(
                   "ETL_DB_CONN",
                   "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow",
               )
               # Parse and connect (copy the parsing logic from create_table)

        2. For each summary dict in `summaries`, INSERT into weather_summary:
               INSERT INTO weather_summary (date, city, avg_temp_c, min_temp_c,
                   max_temp_c, avg_humidity, reading_count)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (date, city) DO UPDATE SET
                   avg_temp_c = EXCLUDED.avg_temp_c,
                   min_temp_c = EXCLUDED.min_temp_c,
                   max_temp_c = EXCLUDED.max_temp_c,
                   avg_humidity = EXCLUDED.avg_humidity,
                   reading_count = EXCLUDED.reading_count,
                   loaded_at = NOW()

           The ON CONFLICT clause makes the load IDEMPOTENT — re-running
           the pipeline for the same date overwrites instead of duplicating.
           This is critical for ETL reliability: if a pipeline fails partway
           and you re-run it, you get the same result as running it once.

        3. Commit the transaction and close the connection

        4. Print how many rows were inserted/updated

        5. Return a summary:
           {
               "rows_loaded": len(summaries),
               "cities": [s["city"] for s in summaries],
               "date": summaries[0]["date"] if summaries else None,
               "loaded_at": str(datetime.now()),
           }

        Why use ON CONFLICT (upsert)?
            Idempotency is the #1 rule of ETL pipelines. If a task fails and
            Airflow retries it, the retry should produce the SAME result.
            Without ON CONFLICT, a retry would insert DUPLICATE rows.

            Airflow will retry failed tasks automatically (based on `retries`
            in default_args). Your load function MUST handle being called
            multiple times for the same data.

        Connection management:
            In production, you'd use Airflow's "Connection" system (Admin > Connections)
            and the PostgresHook/PostgresOperator. For this learning exercise,
            we connect directly to keep it simple and visible.
        """
        # --- YOUR CODE HERE ---
        return {}  # Replace with your implementation

    # =========================================================================
    # DAG dependency graph
    # =========================================================================
    # FileSensor (classic operator) and @task functions can coexist in the same DAG.
    # Use >> to connect a classic operator to a TaskFlow task.

    table_name = create_table()
    raw_data = extract_weather(table_name)
    summaries = transform_weather(raw_data)
    load_result = load_weather(summaries)

    # The FileSensor must complete before extract runs.
    # Since extract depends on table_name (from create_table), we need
    # the sensor to gate BOTH create_table and extract.
    # We put the sensor before create_table in the chain:
    wait_for_data >> table_name


# Register the DAG
phase4_weather_etl()
