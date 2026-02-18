"""Shared configuration for the Sensor Metrics Pipeline.

Defines Kafka broker address, topic names, and processing thresholds
used across all agents and the producer script.
"""

# ── Kafka connection ─────────────────────────────────────────────────

KAFKA_BROKER = "kafka://localhost:9092"

# ── Topic names ──────────────────────────────────────────────────────

SENSOR_READINGS_TOPIC = "sensor-readings"
ENRICHED_READINGS_TOPIC = "enriched-readings"
WINDOW_AGGREGATES_TOPIC = "window-aggregates"
DEAD_LETTER_TOPIC = "sensor-dead-letter"

# ── Processing thresholds ────────────────────────────────────────────

# Temperature thresholds for status classification (Celsius)
TEMP_WARNING_LOW = 5.0
TEMP_WARNING_HIGH = 35.0
TEMP_CRITICAL_LOW = -10.0
TEMP_CRITICAL_HIGH = 50.0

# Humidity thresholds (percentage)
HUMIDITY_WARNING_HIGH = 80.0
HUMIDITY_CRITICAL_HIGH = 95.0

# Sensor count alert threshold (for Phase 3 timer)
SENSOR_COUNT_ALERT_THRESHOLD = 20

# ── Windowing ────────────────────────────────────────────────────────

# Tumbling window size in seconds (Phase 4)
WINDOW_SIZE_SECONDS = 60
WINDOW_EXPIRES_SECONDS = 300
