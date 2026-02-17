"""Register JDBC sink connector for database replication.

The JDBC sink connector reads CDC events from Kafka topics and writes
them to a destination database (postgres-sink). This creates a
real-time replica of the source tables -- a common CDC use case for
analytics databases, read replicas, or cross-system synchronization.

Run standalone:
    uv run python sink_connector.py
"""

import config
import connect_manager


# ── Connector configuration ──────────────────────────────────────────


def build_jdbc_sink_config(source_topics: list[str]) -> dict:
    """Build the Debezium JDBC sink connector configuration.

    # ── TODO(human) ──────────────────────────────────────────────────
    # Build and return a dict containing the JDBC sink connector
    # configuration. The Debezium image (debezium/connect:2.7) ships
    # with the JDBC sink connector pre-installed.
    #
    # Required properties:
    #
    #   "connector.class": "io.debezium.connector.jdbc.JdbcSinkConnector"
    #       The Debezium JDBC sink connector class. Note: this is
    #       different from Confluent's kafka-connect-jdbc. The Debezium
    #       version can consume native Debezium envelope events directly
    #       BUT we'll use the ExtractNewRecordState SMT to flatten them,
    #       which is the more common production pattern.
    #
    #   "connection.url": config.SINK_JDBC_URL
    #       JDBC connection string for the sink database.
    #       Uses Docker hostname: "jdbc:postgresql://postgres-sink:5432/sink_db"
    #
    #   "connection.username": config.SINK_DB_USER
    #   "connection.password": config.SINK_DB_PASSWORD
    #
    #   "topics": ",".join(source_topics)
    #       Comma-separated list of Kafka topics to consume.
    #       Example: "cdc.public.products,cdc.public.orders"
    #
    #   "insert.mode": "upsert"
    #       How to handle incoming records:
    #       - "insert": Always INSERT (fails on duplicates)
    #       - "upsert": INSERT or UPDATE based on primary key
    #       Upsert is essential for CDC because snapshot events (op='r')
    #       and insert events (op='c') for the same key must not conflict.
    #
    #   "primary.key.mode": "record_key"
    #       How to determine the primary key for upsert operations:
    #       - "record_key": Use the Kafka message key (Debezium sets
    #         this to the source table's PK automatically)
    #       - "record_value": Use a field from the message value
    #       record_key is the standard choice for Debezium CDC.
    #
    #   "schema.evolution": "basic"
    #       Auto-create and alter destination tables:
    #       - "none": Tables must pre-exist with correct schema
    #       - "basic": Auto-CREATE tables and ADD columns as needed
    #       "basic" is convenient for development but risky in prod
    #       (no column type changes, no drops).
    #
    #   "table.name.format": "${topic}"
    #       Pattern for destination table names. "${topic}" uses the
    #       full Kafka topic name (e.g., "cdc.public.products" becomes
    #       the table name). Dots in topic names are replaced by
    #       underscores in the table name automatically.
    #
    # ── Single Message Transform (SMT): ExtractNewRecordState ──
    #
    # Debezium events have a nested envelope (before/after/op/source).
    # Most sink connectors expect flat records. The ExtractNewRecordState
    # SMT "unwraps" the envelope, extracting just the "after" payload
    # for inserts/updates, and enabling proper delete handling.
    #
    #   "transforms": "unwrap"
    #       Name(s) of transforms to apply (comma-separated).
    #
    #   "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState"
    #       The SMT class. This is Debezium-specific (not a standard
    #       Kafka Connect SMT).
    #
    #   "transforms.unwrap.drop.tombstones": "false"
    #       Keep tombstone records (key + null value) so the sink
    #       connector can handle deletes properly.
    #
    #   "transforms.unwrap.delete.handling.mode": "rewrite"
    #       How to handle DELETE events:
    #       - "drop": Ignore deletes entirely
    #       - "rewrite": Add a "__deleted" field set to "true" and
    #         keep the record. The sink can then decide what to do.
    #       - "none": Pass the delete event as-is (may cause errors
    #         in some sink connectors)
    #       "rewrite" is safest for JDBC sink with upsert mode.
    #
    # Docs:
    #   - https://debezium.io/documentation/reference/stable/connectors/jdbc.html
    #   - https://debezium.io/documentation/reference/stable/transformations/event-flattening.html
    # ─────────────────────────────────────────────────────────────────
    """
    raise NotImplementedError("TODO(human)")


def register_sink_connector(connect_url: str) -> None:
    """Register the JDBC sink connector.

    # ── TODO(human) ──────────────────────────────────────────────────
    # Register the JDBC sink connector to replicate CDC events from
    # Kafka to the sink PostgreSQL database.
    #
    # Steps:
    #   1. Define the list of source topics to replicate:
    #      [config.PRODUCTS_CDC_TOPIC, config.ORDERS_CDC_TOPIC]
    #   2. Call build_jdbc_sink_config(topics) to get the config dict
    #   3. Call connect_manager.create_connector() with:
    #      - connect_url: the Kafka Connect REST URL
    #      - name: config.SINK_CONNECTOR_NAME
    #      - connector_config: the dict from step 2
    #   4. Print a confirmation message
    #   5. Call connect_manager.wait_for_connector_running() to block
    #      until the sink connector is RUNNING
    #   6. Call connect_manager.print_connector_status() to display
    #      the final status
    #
    # After this completes, any CDC events already in the Kafka topics
    # will be replayed to the sink database. New events will flow
    # in near-real-time (typically <1 second latency).
    #
    # To verify replication, query the sink database:
    #   psql -h localhost -p 5433 -U sink_user -d sink_db
    #   SELECT * FROM "cdc.public.products";
    # ─────────────────────────────────────────────────────────────────
    """
    raise NotImplementedError("TODO(human)")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    """Register the JDBC sink connector and verify it's running."""
    print("=== Registering JDBC Sink Connector ===\n")
    register_sink_connector(config.CONNECT_REST_URL)
    print("\nSink connector is active. CDC events are being replicated.")
    print(f"Verify: psql -h localhost -p {config.SINK_DB_PORT} "
          f"-U {config.SINK_DB_USER} -d {config.SINK_DB_NAME}")


if __name__ == "__main__":
    main()
