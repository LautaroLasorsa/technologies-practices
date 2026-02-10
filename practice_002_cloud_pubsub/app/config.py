"""Shared configuration for Pub/Sub practice.

Sets the emulator environment variable and defines all resource names
(topics, subscriptions) used across publisher and subscriber scripts.
"""

import os

# --- Emulator connection ---
EMULATOR_HOST = "localhost:8085"
os.environ["PUBSUB_EMULATOR_HOST"] = EMULATOR_HOST

# --- GCP project (fake, used only by emulator) ---
PROJECT_ID = "test-project"

# --- Topic names ---
ORDERS_TOPIC = "orders"
DEAD_LETTER_TOPIC = "dead-letter-topic"

# --- Subscription names ---
INVENTORY_SUB = "inventory-sub"
NOTIFICATION_SUB = "notification-sub"
ORDERED_SUB = "ordered-sub"
DEAD_LETTER_SUB = "dead-letter-sub"
