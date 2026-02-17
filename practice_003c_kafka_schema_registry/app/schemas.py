"""Avro schema definitions for practice 003c.

All schemas are defined as Python dicts that match the Avro JSON schema
specification. These are used by both the Schema Registry REST API
(registry_explorer.py) and the confluent-kafka AvroSerializer/Deserializer.

Schema evolution path for User:
  v1: {id, name}
  v2: {id, name, email (optional)} -- backward compatible (new optional field)
  v3: {id, name, email (optional), age (optional)} -- backward compatible

Sensor schemas demonstrate multi-schema-per-topic patterns:
  SensorReading: telemetry data from sensors
  SensorAlert: alert events triggered by sensor thresholds
"""

# ── User schemas (evolution chain) ───────────────────────────────────

USER_V1 = {
    "type": "record",
    "name": "User",
    "namespace": "com.practice.users",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "name", "type": "string"},
    ],
}

USER_V2 = {
    "type": "record",
    "name": "User",
    "namespace": "com.practice.users",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "name", "type": "string"},
        {"name": "email", "type": ["null", "string"], "default": None},
    ],
}

USER_V3 = {
    "type": "record",
    "name": "User",
    "namespace": "com.practice.users",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "name", "type": "string"},
        {"name": "email", "type": ["null", "string"], "default": None},
        {"name": "age", "type": ["null", "int"], "default": None},
    ],
}

# ── Incompatible schema (for demonstrating compatibility failures) ───

USER_BREAKING = {
    "type": "record",
    "name": "User",
    "namespace": "com.practice.users",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "name", "type": "string"},
        {"name": "email", "type": ["null", "string"], "default": None},
        # This field is REQUIRED (no default) -- breaks backward compatibility
        # because old consumers using v2 schema won't know about this field
        # and there's no default to fall back on.
        {"name": "phone", "type": "string"},
    ],
}

# ── Sensor schemas (multi-schema-per-topic) ──────────────────────────

SENSOR_READING = {
    "type": "record",
    "name": "SensorReading",
    "namespace": "com.practice.sensors",
    "fields": [
        {"name": "sensor_id", "type": "string"},
        {"name": "temperature", "type": "double"},
        {"name": "timestamp", "type": "long"},
    ],
}

SENSOR_ALERT = {
    "type": "record",
    "name": "SensorAlert",
    "namespace": "com.practice.sensors",
    "fields": [
        {"name": "sensor_id", "type": "string"},
        {"name": "alert_type", "type": "string"},
        {"name": "temperature", "type": "double"},
        {"name": "timestamp", "type": "long"},
    ],
}
