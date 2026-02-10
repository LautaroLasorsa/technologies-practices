"""Shared configuration: topic names, bootstrap servers, and consumer groups."""

import os

# --- Redpanda / Kafka connection ---
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:19092")

# --- Topic names ---
TOPIC_SAGA_COMMANDS = "saga.commands"  # Orchestrator -> Services (do this)
TOPIC_SAGA_EVENTS = "saga.events"  # Services -> Orchestrator (this happened)
TOPIC_ORDER_STATUS = "order.status"  # Orchestrator -> Order Service (final status)

# --- Consumer group IDs ---
GROUP_ORCHESTRATOR = "orchestrator-group"
GROUP_ORDER_SERVICE = "order-service-group"
GROUP_PAYMENT_SERVICE = "payment-service-group"
GROUP_INVENTORY_SERVICE = "inventory-service-group"
