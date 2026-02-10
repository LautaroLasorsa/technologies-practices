"""Event schemas for the Sensor Metrics Pipeline.

Faust Records are typed, serializable event schemas (like Kafka Streams'
Serdes). They serialize to JSON by default and provide field validation.

These models define the shape of events flowing through the pipeline:

    SensorReading  -->  [enrichment agent]  -->  EnrichedReading
                                                        |
                                                        v
                                            [windowing agent]  -->  WindowAggregate

Malformed readings are routed to the dead-letter topic as raw dicts.

Docs: https://faust.readthedocs.io/en/latest/userguide/models.html
"""

import faust


class SensorReading(faust.Record, serializer="json"):
    """Raw sensor event produced by IoT devices.

    Fields:
        sensor_id:   Unique identifier for the sensor (e.g., "sensor-01").
        temperature: Temperature reading in Celsius.
        humidity:    Relative humidity as a percentage (0-100).
        timestamp:   Unix epoch timestamp (seconds) when the reading was taken.
        location:    Optional human-readable location string.
    """

    sensor_id: str
    temperature: float
    humidity: float
    timestamp: float
    location: str = "unknown"


class EnrichedReading(faust.Record, serializer="json"):
    """Sensor reading enriched with computed status and metadata.

    Produced by the enrichment agent after validating and classifying
    the raw SensorReading.

    Fields:
        sensor_id:   Copied from SensorReading.
        temperature: Copied from SensorReading.
        humidity:    Copied from SensorReading.
        timestamp:   Copied from SensorReading.
        location:    Copied from SensorReading.
        status:      Computed classification: "normal", "warning", or "critical".
        status_reason: Human-readable explanation for the status.
    """

    sensor_id: str
    temperature: float
    humidity: float
    timestamp: float
    location: str
    status: str
    status_reason: str


class WindowAggregate(faust.Record, serializer="json"):
    """Summary statistics for a sensor over a tumbling time window.

    Produced by the windowing agent when a window closes.

    Fields:
        sensor_id:       The sensor this aggregate belongs to.
        window_start:    Unix epoch timestamp of the window start.
        window_end:      Unix epoch timestamp of the window end.
        count:           Number of readings in the window.
        avg_temperature: Mean temperature across the window.
        min_temperature: Minimum temperature in the window.
        max_temperature: Maximum temperature in the window.
        avg_humidity:    Mean humidity across the window.
    """

    sensor_id: str
    window_start: float
    window_end: float
    count: int
    avg_temperature: float
    min_temperature: float
    max_temperature: float
    avg_humidity: float
