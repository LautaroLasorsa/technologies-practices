"""Shared configuration for RabbitMQ practice 055.

Defines broker connection settings, exchange names, queue names,
and routing keys used across all scripts in this practice.
"""

import pika

# -- Broker connection --------------------------------------------------------

RABBITMQ_HOST = "localhost"
RABBITMQ_PORT = 5672
RABBITMQ_USER = "guest"
RABBITMQ_PASSWORD = "guest"
RABBITMQ_VHOST = "/"


def get_connection() -> pika.BlockingConnection:
    """Create a blocking connection to RabbitMQ.

    BlockingConnection is the simplest pika adapter -- it provides a
    synchronous API where each call blocks until complete. This is ideal
    for scripts and learning. For production async workloads, consider
    SelectConnection or AsyncioConnection.
    """
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        virtual_host=RABBITMQ_VHOST,
        credentials=credentials,
        # Heartbeat every 30s to keep the connection alive
        heartbeat=30,
    )
    return pika.BlockingConnection(parameters)


# -- Exchange names -----------------------------------------------------------

DIRECT_EXCHANGE = "practice.direct"
FANOUT_EXCHANGE = "practice.fanout"
TOPIC_EXCHANGE = "practice.topic"
HEADERS_EXCHANGE = "practice.headers"
DEAD_LETTER_EXCHANGE = "practice.dlx"

# -- Queue names --------------------------------------------------------------

# Direct exchange queues
QUEUE_PAYMENTS = "payments"
QUEUE_NOTIFICATIONS = "notifications"
QUEUE_AUDIT = "audit"

# Fanout exchange queues
QUEUE_FANOUT_LOGGER = "fanout.logger"
QUEUE_FANOUT_ANALYTICS = "fanout.analytics"
QUEUE_FANOUT_BACKUP = "fanout.backup"

# Topic exchange queues
QUEUE_TOPIC_ALL_ORDERS = "topic.all_orders"
QUEUE_TOPIC_US_ORDERS = "topic.us_orders"
QUEUE_TOPIC_CREATED = "topic.created_events"

# Ack/QoS exercise queues
QUEUE_ACK_TASKS = "ack.tasks"
QUEUE_DURABLE_TASKS = "durable.tasks"

# Dead letter queues
QUEUE_WORK = "work.queue"
QUEUE_DEAD_LETTER = "dead_letter.queue"
QUEUE_TTL_WORK = "ttl.work.queue"

# -- Routing keys (direct exchange) -------------------------------------------

ROUTING_KEY_PAYMENT = "payment.processed"
ROUTING_KEY_NOTIFICATION = "notification.send"
ROUTING_KEY_AUDIT = "audit.log"

# -- Topic exchange routing key patterns --------------------------------------

# Publishing keys (concrete, no wildcards)
# e.g., "order.created.us", "order.shipped.eu", "order.cancelled.us"

# Binding patterns (with wildcards)
TOPIC_BIND_ALL_ORDERS = "order.#"           # matches ALL order events
TOPIC_BIND_US_ORDERS = "order.*.us"         # matches any order event in US
TOPIC_BIND_CREATED = "*.created.*"          # matches any "created" event in any region
