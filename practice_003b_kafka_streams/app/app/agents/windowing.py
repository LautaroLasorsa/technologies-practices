"""Windowing agent: tumbling-window aggregation of sensor metrics.

Pipeline:
    enriched-readings  -->  [aggregate_windows]  -->  window-aggregates

This agent demonstrates:
  - Tumbling window tables: fixed-size, non-overlapping time windows
  - Aggregating multiple fields (min, max, sum, count) within a window
  - Producing a summary when the window closes

Tumbling windows in Faust:
  A tumbling window of 60 seconds creates non-overlapping buckets:
    [0s-60s] [60s-120s] [120s-180s] ...
  Each event falls into exactly ONE window based on its timestamp.
  When the window "closes" (the stream advances past it), you can
  read the final aggregate.

Accessing windowed table values:
  - table[key].value()    -> value in the most recent window
  - table[key].now()      -> value in the window containing wall-clock now
  - table[key].current()  -> value in the window containing the stream's
                             current event timestamp
  - table[key].delta(30)  -> value from 30 seconds ago

Docs:
  - Windowing: https://faust.readthedocs.io/en/latest/userguide/tables.html#windowing
  - Tumbling:  https://faust.readthedocs.io/en/latest/userguide/tables.html#tumbling-windows
"""

import logging
import time
from datetime import timedelta

from app import config
from app.main import app, enriched_readings_topic, window_aggregates_topic
from app.models import WindowAggregate

logger = logging.getLogger(__name__)


# ── Windowed table definition (boilerplate) ──────────────────────────
# This table stores per-sensor aggregation state within tumbling windows.
#
# The default value is a dict with accumulators:
#   {"count": 0, "sum_temp": 0.0, "min_temp": inf, "max_temp": -inf,
#    "sum_hum": 0.0, "window_start": 0.0}
#
# Using a dict as the table value lets us track multiple aggregation
# fields in a single table (Faust tables store one value per key).


def _default_agg() -> dict:
    """Factory for the default aggregation state."""
    return {
        "count": 0,
        "sum_temp": 0.0,
        "min_temp": float("inf"),
        "max_temp": float("-inf"),
        "sum_hum": 0.0,
        "window_start": 0.0,
    }


window_agg_table = app.Table(
    "window_aggregates",
    default=_default_agg,
    help="Per-sensor tumbling window aggregation state.",
).tumbling(
    size=timedelta(seconds=config.WINDOW_SIZE_SECONDS),
    expires=timedelta(seconds=config.WINDOW_EXPIRES_SECONDS),
    key_index=True,
).relative_to_now()


# ── TODO(human): Implement this agent ────────────────────────────────


@app.agent(enriched_readings_topic)
async def aggregate_windows(stream):
    """Update tumbling-window aggregates for each enriched reading.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches tumbling-window aggregation: partitioning time into
    # fixed buckets and computing running statistics per window. This is the foundation
    # of time-series analytics in stream processing (metrics, dashboards, anomaly detection).

    TODO(human): Implement the body of this agent.

    Steps:
      1. Use `async for reading in stream:` to iterate over EnrichedReading events.

      2. Retrieve the current window's aggregation state for this sensor:
             agg = window_agg_table[reading.sensor_id].value()

         The .value() accessor returns the dict for the most recent window
         that this key belongs to. If the key is new to this window, the
         _default_agg() factory creates the initial state.

      3. Update the aggregation accumulators:
             agg["count"] += 1
             agg["sum_temp"] += reading.temperature
             agg["min_temp"] = min(agg["min_temp"], reading.temperature)
             agg["max_temp"] = max(agg["max_temp"], reading.temperature)
             agg["sum_hum"] += reading.humidity
             if agg["window_start"] == 0.0:
                 agg["window_start"] = reading.timestamp

      4. Write the updated state back to the windowed table:
             window_agg_table[reading.sensor_id] = agg

         IMPORTANT: You must reassign the full dict back to the table.
         Faust tables detect changes via __setitem__, not via mutation
         of the existing dict. Without this line, the update is lost.

      5. Log the running aggregate:
             logger.info(
                 f"Window agg {reading.sensor_id}: "
                 f"count={agg['count']}, avg_temp={agg['sum_temp']/agg['count']:.1f}"
             )

      6. (Optional, but recommended) Emit a WindowAggregate to the output
         topic when the count reaches a meaningful threshold, or rely on
         the timer below to emit periodic snapshots.

    Why reassign the dict?
      Faust's table proxy intercepts __setitem__ to mark the key as dirty
      and schedule a changelog write. If you only mutate the internal dict
      (agg["count"] += 1) without reassigning, the changelog never updates.
      Think of it like SQLAlchemy's "mutable tracking" problem.

    Docs: https://faust.readthedocs.io/en/latest/userguide/tables.html#basics
    """
    raise NotImplementedError("TODO(human): implement aggregate_windows agent")


@app.timer(interval=float(config.WINDOW_SIZE_SECONDS))
async def emit_window_snapshots():
    """Periodically emit window aggregate summaries to the output topic.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches how to emit windowed aggregates: combining timers
    # with windowed table accessors to produce summary events when windows close.
    # This pattern bridges stateful processing and downstream consumers (dashboards, alerting).

    TODO(human): Implement this timer function.

    Steps:
      1. Get the current time: now = time.time()

      2. Iterate over the windowed table. Because it's a tumbling window,
         you iterate over the underlying (non-windowed) items:
             for sensor_id in window_agg_table.keys():

      3. For each sensor_id, get the current window value:
             try:
                 agg = window_agg_table[sensor_id].value()
             except KeyError:
                 continue

      4. Skip entries with count == 0 (empty window).

      5. Build a WindowAggregate record:
             aggregate = WindowAggregate(
                 sensor_id=sensor_id,
                 window_start=agg["window_start"],
                 window_end=now,
                 count=agg["count"],
                 avg_temperature=agg["sum_temp"] / agg["count"],
                 min_temperature=agg["min_temp"],
                 max_temperature=agg["max_temp"],
                 avg_humidity=agg["sum_hum"] / agg["count"],
             )

      6. Send it to the output topic:
             await window_aggregates_topic.send(value=aggregate)

      7. Log the emission:
             logger.info(f"Emitted window aggregate for {sensor_id}: {aggregate}")

    Note:
      This timer emits snapshots at regular intervals. In a production
      system, you might instead detect window boundaries using Faust's
      relative_to_stream() or relative_to_now() on the table. The timer
      approach is simpler and sufficient for learning.

    Docs: https://faust.readthedocs.io/en/latest/userguide/tables.html#windowing
    """
    raise NotImplementedError("TODO(human): implement emit_window_snapshots timer")
