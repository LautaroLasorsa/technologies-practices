"""Shared configuration for Kafka Connect & CDC practice 003d.

Defines broker connection settings, Kafka Connect REST URL,
source/sink database connection strings, and topic naming conventions
used across all scripts in this practice.
"""

# ── Kafka broker ─────────────────────────────────────────────

BOOTSTRAP_SERVERS = "localhost:9092"

# ── Kafka Connect REST API ───────────────────────────────────
# The Debezium Connect container exposes a REST API on port 8083.
# All connector management (create, delete, status) goes through this.

CONNECT_REST_URL = "http://localhost:8083"

# ── Source database (PostgreSQL) ─────────────────────────────
# The source database where changes originate. Debezium reads
# the WAL from this database via a logical replication slot.

SOURCE_DB_HOST = "localhost"
SOURCE_DB_PORT = 5432
SOURCE_DB_NAME = "source_db"
SOURCE_DB_USER = "cdc_user"
SOURCE_DB_PASSWORD = "cdc_pass"

SOURCE_DB_CONN_STRING = (
    f"host={SOURCE_DB_HOST} port={SOURCE_DB_PORT} "
    f"dbname={SOURCE_DB_NAME} user={SOURCE_DB_USER} password={SOURCE_DB_PASSWORD}"
)

# ── Sink database (PostgreSQL) ───────────────────────────────
# The destination database where CDC events are replayed.
# Kafka Connect's JDBC sink writes to this database.

SINK_DB_HOST = "localhost"
SINK_DB_PORT = 5433
SINK_DB_NAME = "sink_db"
SINK_DB_USER = "sink_user"
SINK_DB_PASSWORD = "sink_pass"

SINK_DB_CONN_STRING = (
    f"host={SINK_DB_HOST} port={SINK_DB_PORT} "
    f"dbname={SINK_DB_NAME} user={SINK_DB_USER} password={SINK_DB_PASSWORD}"
)

# JDBC URL for Kafka Connect's JDBC sink connector.
# Note: inside Docker, the sink database is reachable at "postgres-sink:5432"
# (the Docker service name), NOT localhost.
SINK_JDBC_URL = "jdbc:postgresql://postgres-sink:5432/sink_db"

# ── Debezium topic naming ────────────────────────────────────
# Debezium creates topics named: {topic.prefix}.{schema}.{table}
# With topic.prefix="cdc" and PostgreSQL's default "public" schema:
#   cdc.public.products
#   cdc.public.orders

TOPIC_PREFIX = "cdc"
PRODUCTS_CDC_TOPIC = f"{TOPIC_PREFIX}.public.products"
ORDERS_CDC_TOPIC = f"{TOPIC_PREFIX}.public.orders"

# ── Connector names ──────────────────────────────────────────

SOURCE_CONNECTOR_NAME = "postgres-source-connector"
SINK_CONNECTOR_NAME = "jdbc-sink-connector"

# ── Compaction demo ──────────────────────────────────────────

COMPACTION_TOPIC = "compaction-demo"
