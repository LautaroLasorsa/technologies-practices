"""
Phase 3: TaskFlow API & XComs — The Modern Airflow
====================================================

The TaskFlow API (introduced in Airflow 2.0) is a PARADIGM SHIFT from the classic
operator style. Instead of creating operators and manually pushing/pulling XComs,
you write plain Python functions with @task decorators.

KEY CONCEPTS:

    @dag DECORATOR:
        Replaces `with DAG(...) as dag:`. You decorate a function that returns
        the task dependency graph. The function name becomes the dag_id (or you
        can override it).

    @task DECORATOR:
        Replaces PythonOperator + manual XCom push/pull.

        Classic style (Phase 2):
            def my_func(**kwargs):
                result = do_work()
                kwargs['ti'].xcom_push(key='result', value=result)

            task = PythonOperator(task_id='my_task', python_callable=my_func)

        TaskFlow style (this phase):
            @task
            def my_func():
                return do_work()   # Return value is AUTOMATICALLY stored as XCom

        When you call a @task function inside a @dag function, Airflow:
            1. Creates a PythonOperator under the hood
            2. Stores the return value as an XCom automatically
            3. When another @task uses this return value as a parameter,
               Airflow automatically pulls the XCom — no manual pull needed!

    AUTOMATIC XCOM PASSING:
        This is the magic of TaskFlow:

            @task
            def extract():
                return {"data": [1, 2, 3]}  # Stored as XCom automatically

            @task
            def transform(raw_data):         # raw_data = XCom from extract
                return [x * 2 for x in raw_data["data"]]

            # Inside @dag function:
            raw = extract()                  # This is NOT a dict — it's an XCom reference
            result = transform(raw)          # Airflow resolves the XCom at runtime

        The variable `raw` looks like a dict but it's actually a lazy XCom reference.
        Airflow resolves it when `transform` actually executes.

    RETURN VALUE SERIALIZATION:
        XCom values must be JSON-serializable (dicts, lists, strings, numbers, booleans).
        If you return a non-serializable object, Airflow will fail.
        For large data, return a FILE PATH (string) and let the next task read the file.

COMPARISON — Phase 2 (Classic) vs Phase 3 (TaskFlow):

    Classic (Phase 2):
        - Explicit: you see every xcom_push/xcom_pull call
        - Verbose: ~3x more boilerplate
        - Flexible: works with any operator type
        - Familiar: closer to traditional programming

    TaskFlow (Phase 3):
        - Implicit: XCom passing happens via function arguments/returns
        - Concise: looks like normal Python function calls
        - Pythonic: functions return values, callers receive them
        - Limited: only works for PythonOperator-style tasks
          (you can't use @task for BashOperator or SQL tasks...
           though @task.bash and @task.sql exist in newer versions!)

    Use TaskFlow for Python tasks, classic operators for everything else.
    You can MIX both styles in the same DAG.

WHAT TO DO:
    1. Study the reference @task function (`get_pipeline_config`)
    2. Implement the three TODO(human) @task functions:
       a. extract_orders() — returns mock order data
       b. transform_orders(raw_orders) — processes and enriches orders
       c. load_orders(transformed) — "loads" data (prints summary)
    3. Notice: NO xcom_push/xcom_pull anywhere — it's all automatic!
    4. Save, trigger the DAG, inspect XComs in Admin > XComs
"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task


# =============================================================================
# @dag decorator — replaces `with DAG(...) as dag:`
# =============================================================================
# The function decorated with @dag defines the DAG.
# Inside it, you call @task functions to build the dependency graph.
# The function must be called at module level to register the DAG.
#
# Parameters are the same as the DAG() constructor.
@dag(
    dag_id="phase3_taskflow_api",
    default_args={
        "owner": "learner",
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
    },
    description="Phase 3: TaskFlow API — modern @task decorators with automatic XCom",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase3", "learning"],
)
def phase3_taskflow():
    """
    An order-processing mini-pipeline using the TaskFlow API.

    Pipeline: get_config → extract → transform → load
                              ↑ config is passed to extract too
    """

    # =========================================================================
    # FULLY IMPLEMENTED: Reference @task function
    # =========================================================================
    @task
    def get_pipeline_config() -> dict:
        """
        Returns pipeline configuration. Demonstrates that @task functions
        are just normal Python functions that happen to return XCom values.

        The return type hint (-> dict) is optional but good practice.
        Airflow doesn't enforce it, but it documents what the XCom contains.
        """
        config = {
            "source": "mock_order_system",
            "min_order_value": 10.0,
            "currency": "USD",
            "regions": ["north", "south", "east", "west"],
        }
        print(f"Pipeline config: {config}")
        return config

    # =========================================================================
    # TODO(human): Implement extract_orders
    # =========================================================================
    @task
    def extract_orders(config: dict) -> list[dict]:
        """
        Extract mock order data from the "source system".

        TODO(human): Implement this function. Here's what it should do:

        1. Use the `config` parameter (automatically injected from XCom!):
           - Print which source system you're extracting from: config["source"]
           - You DON'T need xcom_pull — the config dict is passed as a function argument
             Airflow resolves the XCom reference automatically at runtime

        2. Create and return a list of mock order dicts:
           Return at least 5 orders with these fields:
           - "order_id": str (e.g., "ORD-001")
           - "customer": str (e.g., "Alice")
           - "items": list of dicts, each with "name" (str) and "price" (float)
           - "region": str (pick from config["regions"])
           - "timestamp": str (ISO format, e.g., "2024-01-15T10:30:00")

           Example order:
           {
               "order_id": "ORD-001",
               "customer": "Alice",
               "items": [
                   {"name": "Widget", "price": 29.99},
                   {"name": "Gadget", "price": 49.99},
               ],
               "region": "north",
               "timestamp": "2024-01-15T10:30:00",
           }

        3. Print how many orders were extracted
        4. Return the list — Airflow stores it as XCom automatically

        Remember: the return value must be JSON-serializable.
        Lists of dicts are fine. NumPy arrays or DataFrames are NOT.

        Why is `config` a function parameter instead of xcom_pull?
           When you write `extract_orders(config)` in the DAG function below,
           `config` is actually an XCom reference (from get_pipeline_config's return).
           At runtime, Airflow resolves it to the actual dict. This is the
           TaskFlow magic — it LOOKS like a normal function call but the data
           passes through XCom behind the scenes.
        """
        # --- YOUR CODE HERE ---
        return []  # Replace with your implementation

    # =========================================================================
    # TODO(human): Implement transform_orders
    # =========================================================================
    @task
    def transform_orders(raw_orders: list[dict], config: dict) -> list[dict]:
        """
        Transform raw orders: calculate totals, filter by min value, enrich.

        TODO(human): Implement this function. Here's what it should do:

        1. For each order in `raw_orders`:
           a. Calculate the total: sum of all item prices
           b. Add a "total" field with the calculated sum
           c. Add a "currency" field from config["currency"]
           d. Add an "item_count" field: number of items in the order

        2. Filter out orders where total < config["min_order_value"]
           - Print how many orders were filtered out and why

        3. Sort the remaining orders by total (descending)

        4. Print a summary:
           - How many orders passed the filter
           - Total revenue across all orders
           - Average order value

        5. Return the list of transformed orders

        Hint: list comprehension with conditional:
            valid = [
                {**order, "total": sum(...), "currency": config["currency"]}
                for order in raw_orders
                if sum(item["price"] for item in order["items"]) >= config["min_order_value"]
            ]

        Or do it in two steps for clarity:
            enriched = [{**o, "total": calc_total(o)} for o in raw_orders]
            filtered = [o for o in enriched if o["total"] >= config["min_order_value"]]

        Notice: both `raw_orders` and `config` are XCom references resolved at runtime.
        You can have MULTIPLE XCom inputs — each @task parameter maps to an upstream output.
        """
        # --- YOUR CODE HERE ---
        return []  # Replace with your implementation

    # =========================================================================
    # TODO(human): Implement load_orders
    # =========================================================================
    @task
    def load_orders(transformed: list[dict]) -> dict:
        """
        Load transformed orders — simulate writing to a database.

        TODO(human): Implement this function. Here's what it should do:

        1. Print a header: "Loading {len(transformed)} orders to database..."

        2. For each order, print a formatted line:
           f"  [{order['order_id']}] {order['customer']}: "
           f"{order['item_count']} items, {order['currency']} {order['total']:.2f}"

        3. Calculate and return a summary dict:
           {
               "total_orders": len(transformed),
               "total_revenue": sum of all totals,
               "avg_order_value": average total,
               "loaded_at": str(datetime.now()),
           }

        4. Print the summary

        The returned dict becomes an XCom. Even though nothing downstream reads it
        in this DAG, it's visible in Admin > XComs — useful for monitoring/debugging.

        Why return a summary instead of just printing?
           In a real pipeline, downstream tasks might need this info:
           - A notification task could include "loaded 42 orders, $12,345 revenue"
           - A monitoring task could alert if total_orders is unexpectedly low
           - An audit log could record what was loaded and when
        """
        # --- YOUR CODE HERE ---
        return {}  # Replace with your implementation

    # =========================================================================
    # DAG dependency graph — TaskFlow style
    # =========================================================================
    # In TaskFlow, you build the graph by "calling" @task functions.
    # The return values are XCom references, not actual data.
    # Passing them as arguments to other @task functions creates dependencies.

    # Step 1: Get config (no dependencies — runs first)
    config = get_pipeline_config()

    # Step 2: Extract (depends on config)
    raw_orders = extract_orders(config)

    # Step 3: Transform (depends on raw_orders AND config)
    # Notice: transform_orders takes TWO XCom inputs from different upstream tasks!
    transformed = transform_orders(raw_orders, config)

    # Step 4: Load (depends on transformed)
    load_orders(transformed)

    # The dependency graph is:
    #   get_pipeline_config → extract_orders → transform_orders → load_orders
    #                      ↘                 ↗
    #                       (config also flows to transform)
    #
    # Airflow infers this from the function call graph — no >> needed!


# =============================================================================
# IMPORTANT: You must CALL the @dag function to register the DAG
# =============================================================================
# This is a common gotcha. If you forget this line, Airflow won't see the DAG.
# The @dag decorator turns the function into a DAG factory.
# Calling it creates and registers the DAG with Airflow's DagBag.
phase3_taskflow()
