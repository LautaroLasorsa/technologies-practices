"""Shared configuration for Kafka Schema Registry practice 003c.

Defines broker connection settings, Schema Registry URL, topic names,
and consumer group IDs used across all scripts in this practice.
"""

# ── Broker connection ────────────────────────────────────────────────

BOOTSTRAP_SERVERS = "localhost:9092"

# ── Schema Registry ──────────────────────────────────────────────────

SCHEMA_REGISTRY_URL = "http://localhost:8081"

# ── Topic names ──────────────────────────────────────────────────────

USERS_TOPIC = "users"
USERS_PARTITIONS = 3

SENSOR_READINGS_TOPIC = "sensor-readings"
SENSOR_READINGS_PARTITIONS = 3

SENSOR_ALERTS_TOPIC = "sensor-alerts"
SENSOR_ALERTS_PARTITIONS = 3

# ── Consumer group IDs ───────────────────────────────────────────────

USERS_GROUP = "users-avro-consumer-group"
SENSOR_GROUP = "sensor-avro-consumer-group"
