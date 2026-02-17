"""Kafka Connect REST API management.

The Kafka Connect REST API (port 8083) is the control plane for connectors.
All connector lifecycle operations -- create, update, delete, monitor --
go through this HTTP API. Understanding this API is essential because
Kafka Connect has NO CLI tool; everything is REST-based.

Run standalone:
    uv run python connect_manager.py
"""

import time

import requests

import config


# ── Connector CRUD operations ────────────────────────────────────────


def list_connectors(connect_url: str) -> list[str]:
    """List all registered connector names.

    # ── TODO(human) ──────────────────────────────────────────────────
    # Kafka Connect REST API endpoint: GET /connectors
    #
    # This is the simplest Connect REST call. It returns a JSON array
    # of connector name strings, e.g.: ["my-source", "my-sink"]
    #
    # Steps:
    #   1. Send a GET request to f"{connect_url}/connectors"
    #   2. Raise an exception if the HTTP status is not 2xx
    #      (use response.raise_for_status())
    #   3. Parse the JSON response body (response.json())
    #   4. Return the resulting list[str]
    #
    # If Connect is not running or unhealthy, requests will raise
    # a ConnectionError. The caller should handle that.
    #
    # Docs: https://docs.confluent.io/platform/current/connect/references/restapi.html#get--connectors
    # ─────────────────────────────────────────────────────────────────
    """
    raise NotImplementedError("TODO(human)")


def create_connector(connect_url: str, name: str, connector_config: dict) -> dict:
    """Create or update a connector using PUT (idempotent).

    # ── TODO(human) ──────────────────────────────────────────────────
    # Kafka Connect REST API endpoint: PUT /connectors/{name}/config
    #
    # Why PUT instead of POST?
    #   - POST /connectors creates a NEW connector and fails if it exists.
    #   - PUT /connectors/{name}/config is IDEMPOTENT: it creates the
    #     connector if it doesn't exist, or updates it if it does.
    #   This makes your scripts re-runnable without "connector already
    #   exists" errors -- essential for development workflows.
    #
    # Steps:
    #   1. Build the URL: f"{connect_url}/connectors/{name}/config"
    #   2. Send a PUT request with:
    #      - headers: {"Content-Type": "application/json"}
    #      - json body: the connector_config dict (NOT wrapped in
    #        {"name": ..., "config": ...} -- PUT /config takes raw config)
    #   3. Raise on non-2xx status (response.raise_for_status())
    #   4. Return the parsed JSON response
    #
    # The response will be the full connector info including name,
    # config, tasks, and type.
    #
    # Docs: https://docs.confluent.io/platform/current/connect/references/restapi.html#put--connectors-(string-name)-config
    # ─────────────────────────────────────────────────────────────────
    """
    raise NotImplementedError("TODO(human)")


def get_connector_status(connect_url: str, name: str) -> dict:
    """Get the current status of a connector and its tasks.

    # ── TODO(human) ──────────────────────────────────────────────────
    # Kafka Connect REST API endpoint: GET /connectors/{name}/status
    #
    # This endpoint returns a JSON object with the structure:
    # {
    #   "name": "my-connector",
    #   "connector": {
    #     "state": "RUNNING",       <-- "RUNNING", "PAUSED", "FAILED", "UNASSIGNED"
    #     "worker_id": "connect:8083"
    #   },
    #   "tasks": [
    #     {
    #       "id": 0,
    #       "state": "RUNNING",     <-- same possible states
    #       "worker_id": "connect:8083"
    #     }
    #   ]
    # }
    #
    # Steps:
    #   1. Send GET to f"{connect_url}/connectors/{name}/status"
    #   2. Raise on non-2xx status
    #   3. Parse and return the JSON response
    #
    # Key insight: a connector can be RUNNING but its tasks FAILED.
    # Always check BOTH connector.state AND tasks[*].state.
    # A failed task usually means a configuration error (wrong DB
    # credentials, unreachable host, etc.).
    #
    # Docs: https://docs.confluent.io/platform/current/connect/references/restapi.html#get--connectors-(string-name)-status
    # ─────────────────────────────────────────────────────────────────
    """
    raise NotImplementedError("TODO(human)")


def delete_connector(connect_url: str, name: str) -> bool:
    """Delete a connector and stop its tasks.

    # ── TODO(human) ──────────────────────────────────────────────────
    # Kafka Connect REST API endpoint: DELETE /connectors/{name}
    #
    # Deleting a connector:
    #   - Stops all running tasks immediately
    #   - Removes the connector config from the _connect-configs topic
    #   - Does NOT delete any Kafka topics the connector created
    #   - Does NOT delete the replication slot in PostgreSQL
    #     (Debezium cleans up the slot on graceful shutdown, but
    #     delete via REST may leave orphaned slots)
    #
    # Steps:
    #   1. Send DELETE to f"{connect_url}/connectors/{name}"
    #   2. If status is 204 (No Content), return True (success)
    #   3. If status is 404 (Not Found), return False (already gone)
    #   4. For any other non-2xx status, raise an exception
    #
    # Common pitfall: calling delete on a connector that's mid-restart
    # can return a 409 (Conflict). You might want to retry after a
    # short delay in production code.
    #
    # Docs: https://docs.confluent.io/platform/current/connect/references/restapi.html#delete--connectors-(string-name)-
    # ─────────────────────────────────────────────────────────────────
    """
    raise NotImplementedError("TODO(human)")


def wait_for_connector_running(
    connect_url: str, name: str, timeout: int = 60
) -> None:
    """Poll connector status until RUNNING or timeout.

    # ── TODO(human) ──────────────────────────────────────────────────
    # After registering a connector, it takes a few seconds for Kafka
    # Connect to start the tasks. This function polls the status
    # endpoint until both the connector AND all its tasks are RUNNING.
    #
    # Steps:
    #   1. Record the start time (time.time())
    #   2. In a loop:
    #      a. Call get_connector_status() to get current status
    #      b. Extract connector_state from status["connector"]["state"]
    #      c. Extract task states from status["tasks"] (list of dicts)
    #      d. If connector_state == "RUNNING" and ALL task states are
    #         "RUNNING", print success and return
    #      e. If connector_state == "FAILED" or any task is "FAILED",
    #         raise RuntimeError with the status details (include the
    #         "trace" field from failed tasks if present -- it contains
    #         the Java stack trace with the actual error)
    #      f. If elapsed time > timeout, raise TimeoutError
    #      g. Otherwise, sleep 2 seconds and retry
    #   3. Print status updates during polling so user sees progress
    #
    # Typical timing: connector reaches RUNNING within 5-15 seconds.
    # If a task stays in UNASSIGNED for >30s, something is wrong.
    # ─────────────────────────────────────────────────────────────────
    """
    raise NotImplementedError("TODO(human)")


# ── Informational endpoints ──────────────────────────────────────────


def list_connector_plugins(connect_url: str) -> list[dict]:
    """List all available connector plugins installed in the Connect worker.

    Returns a list of dicts, each with 'class' and 'type' keys.
    This is useful to verify Debezium plugins are loaded.
    """
    resp = requests.get(f"{connect_url}/connector-plugins")
    resp.raise_for_status()
    return resp.json()


def print_connector_status(connect_url: str, name: str) -> None:
    """Pretty-print the status of a connector and its tasks."""
    try:
        status = get_connector_status(connect_url, name)
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            print(f"  Connector '{name}' not found.")
            return
        raise

    connector_state = status.get("connector", {}).get("state", "UNKNOWN")
    print(f"  Connector '{name}': {connector_state}")

    for task in status.get("tasks", []):
        task_id = task.get("id", "?")
        task_state = task.get("state", "UNKNOWN")
        trace = task.get("trace", "")
        print(f"    Task {task_id}: {task_state}")
        if trace:
            # Print first 3 lines of Java stack trace for debugging
            trace_lines = trace.strip().split("\n")[:3]
            for line in trace_lines:
                print(f"      {line.strip()}")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    """List available plugins and registered connectors."""
    print("=== Kafka Connect Manager ===\n")

    print("Available connector plugins:")
    try:
        plugins = list_connector_plugins(config.CONNECT_REST_URL)
        for plugin in plugins:
            plugin_class = plugin.get("class", "unknown")
            plugin_type = plugin.get("type", "unknown")
            # Show short class name for readability
            short_name = plugin_class.split(".")[-1]
            print(f"  [{plugin_type}] {short_name} ({plugin_class})")
    except requests.exceptions.ConnectionError:
        print("  ERROR: Cannot connect to Kafka Connect at "
              f"{config.CONNECT_REST_URL}")
        print("  Make sure 'docker compose up -d' is running and healthy.")
        return

    print("\nRegistered connectors:")
    connectors = list_connectors(config.CONNECT_REST_URL)
    if not connectors:
        print("  (none)")
    else:
        for name in connectors:
            print_connector_status(config.CONNECT_REST_URL, name)


if __name__ == "__main__":
    main()
