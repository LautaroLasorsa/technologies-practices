"""Shared configuration for Kafka practice 003a.

Defines broker connection settings, topic names, and consumer group IDs
used across all scripts in this practice.
"""

# ── Broker connection ────────────────────────────────────────────────

BOOTSTRAP_SERVERS = "localhost:9092"

# ── Topic names ──────────────────────────────────────────────────────

EVENTS_TOPIC = "events"
EVENTS_PARTITIONS = 3

ORDERS_TOPIC = "orders"
ORDERS_PARTITIONS = 4

# ── Consumer group IDs ───────────────────────────────────────────────

EVENTS_GROUP = "events-consumer-group"
ORDERS_GROUP = "orders-consumer-group"
