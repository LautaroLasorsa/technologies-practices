"""Faust application entry point for the Sensor Metrics Pipeline.

This module creates the Faust app and imports all agents so they are
registered when the worker starts.

Usage (from the app/ directory):
    uv run faust -A app.main worker -l info

The Faust worker will:
  1. Connect to Kafka at the configured broker address.
  2. Discover all agents registered via @app.agent decorators.
  3. Create/subscribe to the required topics.
  4. Start processing events.

Docs: https://faust.readthedocs.io/en/latest/userguide/application.html
"""

import faust

from app import config

# ── Create the Faust application ─────────────────────────────────────

app = faust.App(
    id="sensor-metrics-pipeline",
    broker=config.KAFKA_BROKER,
    # Store table state in a local directory (default: RocksDB if installed,
    # otherwise in-memory). For this practice, in-memory is fine.
    store="memory://",
    # Topic configuration
    topic_replication_factor=1,
    topic_partitions=1,
    # Processing guarantee: start with "at_least_once" (default).
    # Phase 5 will switch to "exactly_once".
    processing_guarantee="at_least_once",
)

# ── Define topics ────────────────────────────────────────────────────
# Topics are declared here so all agents share the same instances.
# Faust auto-creates them on the broker if they don't exist.

from app.models import EnrichedReading, SensorReading, WindowAggregate

raw_readings_topic = app.topic(
    config.SENSOR_READINGS_TOPIC,
    value_type=SensorReading,
)

enriched_readings_topic = app.topic(
    config.ENRICHED_READINGS_TOPIC,
    value_type=EnrichedReading,
)

window_aggregates_topic = app.topic(
    config.WINDOW_AGGREGATES_TOPIC,
    value_type=WindowAggregate,
)

dead_letter_topic = app.topic(
    config.DEAD_LETTER_TOPIC,
    value_type=bytes,
    value_serializer="raw",
)


# ── Import agents (registers them with the app) ─────────────────────
# These imports MUST come after app and topics are defined, because the
# agent modules reference them.

from app.agents import counting, enrichment, windowing  # noqa: F401, E402


# ── Phase 5: Web endpoint ────────────────────────────────────────────


@app.page("/status/")
async def status_page(self, request):
    """
    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches Faust's web endpoints: exposing internal table state
    # over HTTP for monitoring, debugging, or integration with external systems.
    # This demonstrates how stream processors can be queryable services, not just pipelines.

    TODO(human): Return current table state as JSON.

    This is a Faust web endpoint accessible at http://localhost:6066/status/
    while the worker is running.

    Steps:
      1. Import the `sensor_counts` table from app.agents.counting.
      2. Build a dict from the table: {sensor_id: count for sensor_id, count in table.items()}
      3. Import the `json` module and `aiohttp.web.Response`.
      4. Return a web.Response with content_type="application/json" and
         body=json.dumps(data).

    Hint:
        from aiohttp.web import Response
        from app.agents.counting import sensor_counts
        data = {k: v for k, v in sensor_counts.items()}
        return Response(text=json.dumps(data), content_type="application/json")

    Docs: https://faust.readthedocs.io/en/latest/userguide/livecheck.html
    """
    raise NotImplementedError("TODO(human): implement status_page")
