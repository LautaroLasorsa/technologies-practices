"""Counting agent: tracks total reading count per sensor using a Faust Table.

Pipeline:
    enriched-readings  -->  [count_by_sensor]  -->  (updates sensor_counts table)

This agent demonstrates:
  - Faust Tables: distributed key/value stores backed by Kafka changelog topics
  - Stateful processing: maintaining an incrementing counter per key
  - Faust timers: periodic tasks that inspect table state

A Faust Table is like a Python dict that:
  - Survives worker restarts (backed by a Kafka changelog topic)
  - Is automatically partitioned across workers (each worker sees its partition's keys)
  - Can be windowed (tumbling/hopping) for time-based aggregation (see windowing.py)

Docs:
  - Tables: https://faust.readthedocs.io/en/latest/userguide/tables.html
  - Timers: https://faust.readthedocs.io/en/latest/userguide/agents.html#timers
"""

import logging

from app import config
from app.main import app, enriched_readings_topic

logger = logging.getLogger(__name__)


# ── Table definition (boilerplate) ───────────────────────────────────
# Creates a table named "sensor_counts" with a default value of 0 (int).
# Each key is a sensor_id string, each value is the total count of readings.
# The table is backed by a Kafka changelog topic: sensor-metrics-pipeline-sensor_counts-changelog

sensor_counts = app.Table(
    "sensor_counts",
    default=int,
    help="Total reading count per sensor_id (lifetime, not windowed).",
)


# ── TODO(human): Implement these ─────────────────────────────────────


@app.agent(enriched_readings_topic)
async def count_by_sensor(stream):
    """Increment the per-sensor reading count for each enriched event.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches stateful stream processing with Faust Tables.
    # Unlike stateless agents, this one maintains distributed key-value state
    # backed by a Kafka changelog topic, enabling fault-tolerant counting and
    # demonstrating the core mechanism behind joins, aggregations, and windowing.

    TODO(human): Implement the body of this agent.

    Steps:
      1. Use `async for reading in stream:` to iterate over EnrichedReading events.

      2. For each reading, increment the count in the sensor_counts table:
             sensor_counts[reading.sensor_id] += 1

         This works exactly like a Python dict because Faust Tables implement
         __getitem__ and __setitem__. The default=int factory ensures that
         accessing a missing key returns 0 (just like collections.defaultdict(int)).

      3. Log the updated count:
             current = sensor_counts[reading.sensor_id]
             logger.info(f"Sensor {reading.sensor_id}: count={current}")

    Why no yield?
      This agent only updates internal state (the table). It doesn't produce
      to a downstream topic, so there's no sink and no yield. The table's
      changelog topic is updated automatically by Faust.

    Where is the data stored?
      - In memory (we configured store="memory://" in main.py)
      - Faust also writes to a Kafka changelog topic so that if the worker
        restarts, it replays the changelog to rebuild the table state.
      - In production, you'd use store="rocksdb://" for persistence.

    Docs: https://faust.readthedocs.io/en/latest/userguide/tables.html#basics
    """
    raise NotImplementedError("TODO(human): implement count_by_sensor agent")


@app.timer(interval=30.0)
async def check_high_count_sensors():
    """Periodic task: log any sensors that have exceeded the count threshold.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches Faust timers: background periodic tasks that run
    # independently of event streams. Timers are essential for scheduled monitoring,
    # cleanup, and emitting windowed aggregates — a pattern you'll extend in windowing.py.

    TODO(human): Implement this timer function.

    Steps:
      1. Iterate over the sensor_counts table using:
             for sensor_id, count in sensor_counts.items():

      2. For each entry, check if count >= config.SENSOR_COUNT_ALERT_THRESHOLD.

      3. If the threshold is exceeded, log a warning:
             logger.warning(
                 f"ALERT: Sensor {sensor_id} has {count} readings "
                 f"(threshold={config.SENSOR_COUNT_ALERT_THRESHOLD})"
             )

      4. If no sensors exceed the threshold, log at info level:
             logger.info("Periodic check: no sensors above threshold.")

    How Faust timers work:
      The @app.timer(interval=30.0) decorator runs this coroutine every
      30 seconds while the worker is active. It's not triggered by events --
      it's a background periodic task (like a cron job within the worker).

    Docs: https://faust.readthedocs.io/en/latest/userguide/agents.html#timers
    """
    raise NotImplementedError("TODO(human): implement check_high_count_sensors timer")
