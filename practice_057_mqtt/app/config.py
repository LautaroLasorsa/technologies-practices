"""Shared configuration for MQTT practice exercises.

All exercises connect to the same local Mosquitto broker.
"""

# ── Broker connection ────────────────────────────────────────────────

BROKER_HOST = "localhost"
BROKER_PORT = 1883
KEEPALIVE = 60

# ── Topic hierarchy for the IoT telemetry system ─────────────────────

TOPIC_PREFIX = "sensors"
ROOMS = ["room1", "room2", "room3"]
SENSOR_TYPES = ["temperature", "humidity", "pressure"]

# ── Device presence topics ───────────────────────────────────────────

DEVICE_STATUS_PREFIX = "devices"
DEVICE_IDS = ["sensor-001", "sensor-002", "sensor-003"]

# ── MQTT 5.0 exercise topics ────────────────────────────────────────

SHARED_SUB_TOPIC = "tasks/processing"
REQUEST_TOPIC = "services/calculator/request"
RESPONSE_TOPIC_PREFIX = "services/calculator/response"
