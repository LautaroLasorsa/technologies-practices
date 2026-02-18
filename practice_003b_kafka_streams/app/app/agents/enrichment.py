"""Enrichment agent: validates and classifies raw sensor readings.

Pipeline:
    sensor-readings  -->  [enrich_readings]  -->  enriched-readings
                                |
                                +---> sensor-dead-letter  (malformed events)

This agent demonstrates:
  - Consuming from a typed topic (SensorReading records)
  - Stateless transformation (compute status from thresholds)
  - Producing to a sink topic (EnrichedReading records)
  - Dead-letter routing for events that fail validation

Docs:
  - Agents: https://faust.readthedocs.io/en/latest/userguide/agents.html
  - Sinks:  https://faust.readthedocs.io/en/latest/userguide/agents.html#sinks
"""

import json
import logging

from app import config
from app.main import app, dead_letter_topic, enriched_readings_topic, raw_readings_topic
from app.models import EnrichedReading, SensorReading

logger = logging.getLogger(__name__)


# ── Validation helper (boilerplate) ──────────────────────────────────


def validate_reading(reading: SensorReading) -> str | None:
    """Return an error message if the reading is invalid, or None if valid.

    Validation rules:
      - sensor_id must be a non-empty string
      - temperature must be between -50 and 60 Celsius (physical limits)
      - humidity must be between 0 and 100 (percentage)
    """
    if not reading.sensor_id or not reading.sensor_id.strip():
        return "missing or empty sensor_id"
    if reading.temperature < -50.0 or reading.temperature > 60.0:
        return f"temperature {reading.temperature} out of physical range [-50, 60]"
    if reading.humidity < 0.0 or reading.humidity > 100.0:
        return f"humidity {reading.humidity} out of range [0, 100]"
    return None


# ── TODO(human): Implement these two functions ───────────────────────


def classify_status(reading: SensorReading) -> tuple[str, str]:
    """Classify a sensor reading as "normal", "warning", or "critical".

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches threshold-based classification and priority-based
    # decision logic. It's a common pattern in streaming pipelines: incoming
    # data gets enriched with computed fields based on business rules.

    TODO(human): Implement this function.

    Steps:
      1. Check CRITICAL conditions first (highest priority):
         - Temperature below config.TEMP_CRITICAL_LOW or above config.TEMP_CRITICAL_HIGH
         - Humidity above config.HUMIDITY_CRITICAL_HIGH
         If any critical condition is met, return ("critical", "<reason>").

      2. Check WARNING conditions next:
         - Temperature below config.TEMP_WARNING_LOW or above config.TEMP_WARNING_HIGH
         - Humidity above config.HUMIDITY_WARNING_HIGH
         If any warning condition is met, return ("warning", "<reason>").

      3. If no thresholds are exceeded, return ("normal", "all readings within range").

    The reason string should describe which threshold was breached, e.g.:
        "temperature 52.3 exceeds critical high 50.0"
        "humidity 82.5 exceeds warning high 80.0"

    Returns:
        A tuple of (status, reason) where status is one of
        "normal", "warning", "critical".

    Hint:
        if reading.temperature > config.TEMP_CRITICAL_HIGH:
            return ("critical", f"temperature {reading.temperature} exceeds critical high {config.TEMP_CRITICAL_HIGH}")
    """
    if reading.temperature < config.TEMP_CRITICAL_LOW:
        return ("critical",f"Temperature {reading.temperature} C is too low")
    if reading.temperature > config.TEMP_CRITICAL_HIGH:
        return ("critical",f"Temperature {reading.temperature} C is too high")
    if reading.humidity > config.HUMIDITY_CRITICAL_HIGH:
        return ("critical", f"Humidity {reading.humidity} % is too high")

    if reading.temperature < config.TEMP_WARNING_LOW:
        return ("warning",f"Temperature {reading.temperature} C is too low")
    if reading.temperature > config.TEMP_WARNING_HIGH:
        return ("warning",f"Temperature {reading.temperature} C is too high")
    if reading.humidity > config.HUMIDITY_WARNING_HIGH:
        return ("warning", f"Humidity {reading.humidity} % is too high")

    return ("normal","all readings within range")


@app.agent(raw_readings_topic, sink=[enriched_readings_topic])
async def enrich_readings(stream):
    """Consume raw readings, validate, classify, and produce enriched readings.

    # ── Exercise Context ──────────────────────────────────────────────────
    # This exercise teaches the consume-transform-produce pattern, the foundation
    # of stateless stream processing. You'll learn how Faust agents consume typed
    # events, route bad data to a dead-letter queue, and yield enriched events to
    # a sink topic — all using async/await for backpressure-aware processing.

    TODO(human): Implement the body of this agent.

    This agent iterates over the raw_readings_topic stream. For each
    SensorReading event:

    Steps:
      1. Use `async for reading in stream:` to iterate over events.
         Each `reading` is a SensorReading instance (Faust deserializes it).

      2. Validate the reading by calling validate_reading(reading).
         If validation returns an error string (not None):
           a. Log a warning: logger.warning(f"Invalid reading from {reading.sensor_id}: {error}")
           b. Send the malformed event to the dead-letter topic:
                  await dead_letter_topic.send(
                      value=json.dumps({"error": error, "raw": reading.asdict()}).encode()
                  )
           c. `continue` to skip to the next event (do NOT yield).

      3. If valid, classify the reading by calling classify_status(reading).
         This returns a (status, reason) tuple.

      4. Build an EnrichedReading from the original reading plus the status:
             enriched = EnrichedReading(
                 sensor_id=reading.sensor_id,
                 temperature=reading.temperature,
                 humidity=reading.humidity,
                 timestamp=reading.timestamp,
                 location=reading.location,
                 status=status,
                 status_reason=reason,
             )

      5. Log the enriched reading:
             logger.info(f"Enriched {reading.sensor_id}: status={status}")

      6. `yield enriched` -- this sends the EnrichedReading to the sink topic
         (enriched_readings_topic).

    Why yield?
      When an agent has a `sink=[topic]`, Faust takes whatever the agent
      yields and sends it to that sink topic. This is the idiomatic
      consume-transform-produce pattern in Faust.

    Docs: https://faust.readthedocs.io/en/latest/userguide/agents.html#the-stream
    """

    async for reading in stream:
        reading: SensorReading

        error = validate_reading(reading)
        if error is not None:
            logger.warning(f"Invalid reading from {reading.sensor_id}: {error}")
            await dead_letter_topic.send(
                value = json.dumps({"error":error, "raw":reading.asdict()}).encode("utf-8")
            )
            continue

        (status, reason) = classify_status(reading)

        enriched = EnrichedReading(
            sensor_id = reading.sensor_id,
            temperature = reading.temperature,
            humidity = reading.humidity,
            timestamp = reading.timestamp,
            location = reading.location,
            status = status,
            status_reason = reason
        )

        logger.info(f"Enriched {reading.sensor_id}: status={status}")
        yield enriched
