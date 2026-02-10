"""
Phase 2: Classic Operators & Dependencies
==========================================

This DAG demonstrates the "classic" (pre-TaskFlow) way of building Airflow workflows:
    - BashOperator for shell commands
    - PythonOperator for Python callables
    - Dependency chains with >> and <<
    - Trigger rules for conditional execution

KEY CONCEPTS:

    OPERATORS:
        Operators are the building blocks of Airflow tasks. Each operator type
        wraps a specific kind of work:

        - PythonOperator  → runs a Python function
        - BashOperator    → runs a shell command
        - EmailOperator   → sends an email
        - SQLOperator     → executes a SQL query
        - S3ToGCSOperator → transfers data between cloud providers
        ... and hundreds more (Airflow has a rich ecosystem of "provider packages")

        The key insight: operators DEFINE what a task does, but Airflow MANAGES
        when and where it runs. You declare the work; Airflow handles scheduling,
        retries, logging, and dependency management.

    TRIGGER RULES:
        By default, a task runs only if ALL upstream tasks SUCCEEDED.
        But you can change this with `trigger_rule`:

        - "all_success"   (default) — run if all parents succeeded
        - "all_failed"    — run if all parents failed
        - "all_done"      — run regardless of parent state (success, failed, skipped)
        - "one_success"   — run if at least one parent succeeded
        - "one_failed"    — run if at least one parent failed
        - "none_failed"   — run if no parent failed (success or skipped is OK)
        - "none_skipped"  — run if no parent was skipped

        Use case: "cleanup" tasks that should run even if the pipeline fails.
        Use case: "alert" tasks that only run when something fails.

    XCOM (CLASSIC STYLE):
        XCom = "Cross-Communication". It's how tasks pass small data to each other.

        Classic style (explicit push/pull):
            # In task A — push a value:
            kwargs['ti'].xcom_push(key='my_key', value='my_value')

            # In task B — pull the value:
            value = kwargs['ti'].xcom_pull(task_ids='task_a', key='my_key')

        The TaskFlow API (Phase 3) makes this automatic via return values.

        IMPORTANT: XComs are stored in the Airflow metadata database (PostgreSQL).
        They're meant for SMALL data (config values, file paths, row counts).
        Do NOT pass large DataFrames or file contents through XComs.
        For large data, write to shared storage (S3, GCS, local volume) and pass
        the PATH via XCom.

WHAT TO DO:
    1. Read the fully implemented BashOperator tasks and DAG structure
    2. Implement the TODO(human) sections:
       a. `validate_data()` — a PythonOperator callable that validates extracted data
       b. `transform_data()` — a PythonOperator callable that transforms data
       c. Wire up the dependency chain (the >> operators at the bottom)
    3. Save the file and trigger the DAG in the Airflow UI
    4. Inspect task logs to see the execution order and XCom values
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


default_args = {
    "owner": "learner",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


# =============================================================================
# FULLY IMPLEMENTED: Example PythonOperator callable that uses XCom push
# =============================================================================
def extract_data(**kwargs):
    """
    Simulates data extraction. Pushes the 'extracted data' to XCom
    so downstream tasks can access it.

    In a real pipeline, this might:
        - Query an API and return the response
        - Read from a database and return row count
        - Download a file and return the file path
    """
    # Simulate extracted data
    data = {
        "records": [
            {"city": "NYC", "temp_f": 32, "humidity": 65},
            {"city": "LA", "temp_f": 72, "humidity": 30},
            {"city": "Chicago", "temp_f": 18, "humidity": 70},
        ],
        "source": "mock_api",
        "extracted_at": str(datetime.now()),
    }

    # Classic XCom push — explicitly store a value for downstream tasks
    # key='extracted_data' is the label; downstream tasks reference this key
    ti = kwargs["ti"]
    ti.xcom_push(key="extracted_data", value=data)
    print(f"Extracted {len(data['records'])} records from {data['source']}")


# =============================================================================
# TODO(human): Implement validate_data
# =============================================================================
def validate_data(**kwargs):
    """
    Validate the extracted data by pulling it from XCom.

    TODO(human): Implement this function. Here's what it should do:

    1. Pull the extracted data from XCom:
       - Use `kwargs['ti'].xcom_pull(task_ids='extract', key='extracted_data')`
       - This retrieves whatever `extract_data()` pushed to XCom

    2. Validate the data:
       - Check that `data` is not None (extraction didn't fail silently)
       - Check that `data['records']` is a non-empty list
       - Check that each record has the required keys: 'city', 'temp_f', 'humidity'
       - Print the validation results

    3. Push the validation result to XCom:
       - `kwargs['ti'].xcom_push(key='is_valid', value=True)` if valid
       - `kwargs['ti'].xcom_push(key='is_valid', value=False)` if invalid

    Why explicit xcom_push/xcom_pull?
       In the classic (pre-TaskFlow) style, you manually push and pull values.
       This is verbose but explicit — you control exactly what gets stored and
       with what key. In Phase 3, you'll see how @task makes this automatic.

    Why validate in a separate task?
       Single Responsibility: extract does one thing, validate does another.
       If validation fails, you can retry or alert without re-extracting.
       In the Airflow UI, you can see exactly WHICH step failed.
    """
    # --- YOUR CODE HERE ---
    pass


# =============================================================================
# TODO(human): Implement transform_data
# =============================================================================
def transform_data(**kwargs):
    """
    Transform the extracted data: convert Fahrenheit to Celsius, categorize humidity.

    TODO(human): Implement this function. Here's what it should do:

    1. Pull the extracted data from XCom:
       - `data = kwargs['ti'].xcom_pull(task_ids='extract', key='extracted_data')`

    2. (Optional) Pull the validation result to check if data is valid:
       - `is_valid = kwargs['ti'].xcom_pull(task_ids='validate', key='is_valid')`
       - If not valid, you could raise an exception or skip processing

    3. Transform each record:
       - Convert temp_f to temp_c: (temp_f - 32) * 5/9, rounded to 1 decimal
       - Categorize humidity: "low" (<40), "medium" (40-60), "high" (>60)
       - Add both new fields to each record

    4. Push the transformed data to XCom:
       - `kwargs['ti'].xcom_push(key='transformed_data', value=transformed_records)`

    Example output record:
        {"city": "NYC", "temp_f": 32, "temp_c": 0.0, "humidity": 65, "humidity_cat": "high"}

    Hint: Python's list comprehension makes this clean:
        transformed = [
            {**record, "temp_c": ..., "humidity_cat": ...}
            for record in data["records"]
        ]
    """
    # --- YOUR CODE HERE ---
    pass


# =============================================================================
# FULLY IMPLEMENTED: Load function (prints results, simulates DB insert)
# =============================================================================
def load_data(**kwargs):
    """
    Load transformed data — in this phase, just prints it.
    Phase 4 will actually write to PostgreSQL.
    """
    ti = kwargs["ti"]
    transformed = ti.xcom_pull(task_ids="transform", key="transformed_data")

    if transformed is None:
        print("WARNING: No transformed data found in XCom!")
        print("Did you implement transform_data() and push to XCom?")
        return

    print(f"\nLoading {len(transformed)} records:")
    print("-" * 60)
    for record in transformed:
        print(f"  {record}")
    print("-" * 60)
    print("Load complete! (In Phase 4, this writes to PostgreSQL)")


# =============================================================================
# FULLY IMPLEMENTED: Cleanup function — demonstrates trigger_rule
# =============================================================================
def cleanup(**kwargs):
    """
    Cleanup task that runs regardless of upstream success/failure.
    This uses trigger_rule="all_done" — see the task definition below.

    Use cases for cleanup tasks:
        - Delete temporary files
        - Release database locks
        - Send a "pipeline finished" notification (success or failure)
        - Update a monitoring dashboard
    """
    print("Cleanup: releasing resources...")
    print(f"DAG run ended at: {datetime.now()}")

    # You can check upstream task states if needed:
    ti = kwargs["ti"]
    extract_state = ti.xcom_pull(task_ids="extract", key="extracted_data")
    if extract_state:
        print("Cleanup: extraction had produced data")
    else:
        print("Cleanup: extraction produced no data (or failed)")


# =============================================================================
# DAG Definition
# =============================================================================
with DAG(
    dag_id="phase2_classic_operators",
    default_args=default_args,
    description="Phase 2: Classic Operators — BashOperator, PythonOperator, XComs",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase2", "learning"],
) as dag:

    # -------------------------------------------------------------------------
    # BashOperator tasks — fully implemented as reference
    # -------------------------------------------------------------------------
    # BashOperator runs a shell command. Useful for:
    #   - Running scripts (ETL scripts, data processing tools)
    #   - File operations (move, copy, check existence)
    #   - Calling external tools (spark-submit, dbt, aws cli)
    #
    # The `bash_command` is a Jinja template — you can use {{ ds }}, {{ params }}, etc.
    # {{ ds }} = the logical date in YYYY-MM-DD format

    print_date = BashOperator(
        task_id="print_date",
        bash_command='echo "Pipeline started at $(date). Logical date: {{ ds }}"',
    )

    check_disk = BashOperator(
        task_id="check_disk_space",
        bash_command='echo "Disk space:" && df -h / | tail -1',
    )

    # -------------------------------------------------------------------------
    # PythonOperator tasks
    # -------------------------------------------------------------------------
    extract = PythonOperator(
        task_id="extract",
        python_callable=extract_data,
    )

    validate = PythonOperator(
        task_id="validate",
        python_callable=validate_data,
    )

    transform = PythonOperator(
        task_id="transform",
        python_callable=transform_data,
    )

    load = PythonOperator(
        task_id="load",
        python_callable=load_data,
    )

    # trigger_rule="all_done" means this task runs even if upstream tasks fail
    cleanup_task = PythonOperator(
        task_id="cleanup",
        python_callable=cleanup,
        trigger_rule="all_done",
    )

    # -------------------------------------------------------------------------
    # TODO(human): Wire up the dependency chain
    # -------------------------------------------------------------------------
    #
    # Define the execution order using >> (bitshift) operators.
    #
    # The desired graph shape:
    #
    #   print_date ──┐
    #                ├──→ extract → validate → transform → load → cleanup
    #   check_disk ──┘
    #
    # This means:
    #   1. print_date and check_disk run IN PARALLEL (no dependency between them)
    #   2. extract runs AFTER BOTH print_date and check_disk complete
    #   3. validate, transform, load run sequentially after extract
    #   4. cleanup runs after load (with trigger_rule="all_done")
    #
    # Hint: Use list syntax for fan-in:
    #   [task_a, task_b] >> task_c   means task_c waits for both A and B
    #
    # Hint: Chain syntax:
    #   a >> b >> c >> d   creates a linear chain
    #
    # Write your dependency chain below:
    # --- YOUR CODE HERE ---
    pass  # Remove this line when you add the dependencies
