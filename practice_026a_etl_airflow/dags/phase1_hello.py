"""
Phase 1: Hello World DAG — Your First Airflow Workflow
=======================================================

This DAG is FULLY IMPLEMENTED as a reference. Deploy it and explore the Airflow UI.

KEY CONCEPTS:
    - A DAG (Directed Acyclic Graph) is Airflow's core abstraction for a workflow.
      It defines WHAT tasks exist and HOW they depend on each other, but NOT when
      individual tasks run (that's the scheduler's job).

    - Each node in the graph is a "task instance" — one execution of a task for a
      specific date (the "logical date" or "execution_date").

    - Airflow uses "operators" to define tasks. PythonOperator wraps a Python
      callable, BashOperator runs a shell command, etc. Think of operators as
      templates: "this task will run Python code" or "this task will run bash."

    - default_args are inherited by ALL tasks in the DAG. This avoids repeating
      owner, retries, retry_delay on every single task.

    - schedule=None means this DAG won't run on a timer — you trigger it manually
      from the Airflow UI (or CLI). This is useful for development/testing.

    - tags=["phase1"] help you filter DAGs in the UI when you have many.

WHAT TO DO:
    1. `docker compose up -d` to start Airflow
    2. Go to http://localhost:8080 (admin/admin)
    3. Find "phase1_hello_world" in the DAGs list
    4. Unpause it (toggle switch on the left)
    5. Click the play button (▶) → "Trigger DAG"
    6. Click on the DAG name → "Graph" tab to see the task dependency graph
    7. Click on each task → "Logs" to see the print() output
    8. Explore: Tree view, Gantt chart, task duration

AIRFLOW UI TOUR:
    - DAGs page: overview of all DAGs, their schedule, last run status
    - Graph view: visual representation of task dependencies (the DAG shape)
    - Grid view: historical runs as a grid (rows = tasks, columns = dates)
    - Gantt: timeline of task execution — useful for spotting bottlenecks
    - Task Instance Details: click any task → see logs, XCom, rendered template
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


# =============================================================================
# default_args — inherited by every task in this DAG
# =============================================================================
# Why default_args?
#   Instead of setting owner, retries, retry_delay on EACH task, you set them
#   once here and every task inherits them. Individual tasks can override.
#
# Key fields:
#   owner        — shows in the UI; useful for team attribution
#   retries      — how many times to retry a failed task before marking it FAILED
#   retry_delay  — wait time between retries (exponential backoff is also available)
#   start_date   — the earliest date Airflow will schedule this DAG for
#                  (NOT when the DAG was created — a common confusion!)
default_args = {
    "owner": "learner",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


# =============================================================================
# Task callables — plain Python functions that PythonOperator will call
# =============================================================================
# These are normal Python functions. Airflow calls them inside a subprocess.
# The `**kwargs` receives Airflow context (execution_date, task_instance, etc.)
# but you can ignore it for simple tasks.


def start_task(**kwargs):
    """First task: just logs a greeting."""
    print("=" * 60)
    print("  Hello from Airflow!")
    print(f"  Logical date: {kwargs['logical_date']}")
    print("=" * 60)
    # `logical_date` (Airflow 2.2+) is the date this run represents.
    # For manually triggered DAGs, it's roughly "now".
    # For scheduled DAGs, it's the START of the interval being processed.
    # Example: a daily DAG scheduled at midnight processes the PREVIOUS day,
    # so the logical_date for the Jan 2 run is Jan 1.


def process_task(**kwargs):
    """Middle task: simulates some processing."""
    import time

    print("Processing data...")
    time.sleep(2)  # Simulate work
    print("Processing complete!")

    # Task Instance (ti) is available in kwargs — we'll use it for XComs later.
    ti = kwargs["ti"]
    print(f"Task ID: {ti.task_id}")
    print(f"DAG ID: {ti.dag_id}")
    print(f"Run ID: {kwargs['run_id']}")


def end_task(**kwargs):
    """Final task: confirms the workflow completed."""
    print("=" * 60)
    print("  Workflow completed successfully!")
    print("  Check the Graph view in the UI to see the task flow.")
    print("=" * 60)


# =============================================================================
# DAG definition
# =============================================================================
# The `with DAG(...)` context manager creates the DAG and associates tasks to it.
#
# Why `with DAG(...)` instead of `@dag` decorator?
#   Both work! The `with` statement is the "classic" style (Airflow 1.x+).
#   The `@dag` decorator is the "TaskFlow" style (Airflow 2.0+), covered in Phase 3.
#   Use whichever you prefer — the result is identical.
#
# schedule=None:
#   This DAG does NOT run on a schedule — manual trigger only.
#   Other options: "@daily", "@hourly", "0 6 * * *" (cron), timedelta(hours=2)
#
# catchup=False:
#   If True, Airflow would create runs for every missed schedule interval
#   between start_date and now. For learning, we don't want that.
#
# Important: the variable name (hello_dag) doesn't matter to Airflow.
#   Airflow identifies DAGs by the `dag_id` parameter.
#   But the file MUST be importable — if it has a syntax error, no DAGs load.

with DAG(
    dag_id="phase1_hello_world",
    default_args=default_args,
    description="Phase 1: Hello World — your first Airflow DAG",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # Manual trigger only
    catchup=False,
    tags=["phase1", "learning"],
) as dag:

    # -------------------------------------------------------------------------
    # Task definitions using PythonOperator
    # -------------------------------------------------------------------------
    # PythonOperator takes a `python_callable` — the function to run.
    # `task_id` is the unique identifier for this task WITHIN the DAG.
    # It must be unique per DAG (not globally).

    start = PythonOperator(
        task_id="start",
        python_callable=start_task,
        # provide_context=True is the DEFAULT in Airflow 2.x — kwargs are always passed
    )

    process = PythonOperator(
        task_id="process",
        python_callable=process_task,
    )

    end = PythonOperator(
        task_id="end",
        python_callable=end_task,
    )

    # -------------------------------------------------------------------------
    # Dependencies — the "graph" part of DAG
    # -------------------------------------------------------------------------
    # `>>` is Airflow's bitshift operator overload. It means "runs before".
    #   start >> process  means: process runs AFTER start completes successfully
    #
    # This creates: start → process → end (a linear chain)
    #
    # Other dependency patterns:
    #   [a, b] >> c       — c runs after BOTH a and b complete (fan-in)
    #   a >> [b, c]       — b and c run in parallel after a (fan-out)
    #   a >> b >> [c, d]  — chain then fan-out
    #   a << b            — same as b >> a (less common, harder to read)

    start >> process >> end
