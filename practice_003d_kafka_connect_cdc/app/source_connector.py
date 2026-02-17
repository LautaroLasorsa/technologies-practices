"""Register Debezium PostgreSQL source connector.

Debezium reads PostgreSQL's Write-Ahead Log (WAL) via a logical
replication slot and converts each INSERT/UPDATE/DELETE into a
structured Kafka message. This script builds the connector
configuration and registers it with Kafka Connect.

Run standalone:
    uv run python source_connector.py
"""

import config
import connect_manager


# ── Connector configuration ──────────────────────────────────────────


def build_debezium_config() -> dict:
    """Build the Debezium PostgreSQL source connector configuration.

    # ── TODO(human) ──────────────────────────────────────────────────
    # Build and return a dict containing the Debezium source connector
    # configuration. Each key is a Debezium/Kafka Connect config property.
    #
    # Required properties (return a dict with ALL of these):
    #
    #   "connector.class": "io.debezium.connector.postgresql.PostgresConnector"
    #       The Java class that implements the connector. This tells
    #       Kafka Connect which plugin to instantiate.
    #
    #   "database.hostname": "postgres-source"
    #       The hostname of the source database. IMPORTANT: use the
    #       Docker service name, NOT "localhost", because the Connect
    #       worker runs inside Docker and resolves hostnames via
    #       Docker's internal DNS.
    #
    #   "database.port": "5432"
    #       PostgreSQL port inside the Docker network.
    #
    #   "database.user": config.SOURCE_DB_USER
    #   "database.password": config.SOURCE_DB_PASSWORD
    #   "database.dbname": config.SOURCE_DB_NAME
    #       Database credentials. The user needs REPLICATION privilege
    #       (PostgreSQL's default superuser has this).
    #
    #   "topic.prefix": config.TOPIC_PREFIX
    #       Debezium creates Kafka topics named:
    #         {topic.prefix}.{schema}.{table}
    #       Example: "cdc.public.products"
    #       This prefix also serves as the logical server name in
    #       Debezium's internal bookkeeping.
    #
    #   "table.include.list": "public.products,public.orders"
    #       Comma-separated list of {schema}.{table} to capture.
    #       Only these tables will generate CDC events. Without this,
    #       Debezium captures ALL tables in the database.
    #
    #   "plugin.name": "pgoutput"
    #       The PostgreSQL logical decoding plugin. pgoutput is the
    #       built-in plugin since PostgreSQL 10+ (no extra installation).
    #       Alternative: decoderbufs (requires extension install).
    #
    #   "slot.name": "debezium_cdc_slot"
    #       Name of the PostgreSQL replication slot. A replication slot
    #       is a server-side cursor that tracks which WAL position has
    #       been consumed. PostgreSQL retains WAL segments until the
    #       slot confirms consumption -- this prevents data loss but
    #       can cause disk bloat if the consumer is down for too long.
    #
    #   "publication.name": "dbz_publication"
    #       PostgreSQL publication (pgoutput concept). A publication
    #       defines which tables' changes are included in the logical
    #       replication stream. Debezium auto-creates this if it
    #       doesn't exist (with publication.autocreate.mode=all_tables).
    #
    #   "snapshot.mode": "initial"
    #       Controls what happens on first startup:
    #       - "initial": Take a snapshot of existing data, then stream
    #         WAL changes. Snapshot events have op='r' (read).
    #       - "never": Skip snapshot, only stream new WAL changes.
    #       - "initial_only": Snapshot only, don't stream WAL after.
    #
    #   "tombstones.on.delete": "true"
    #       When a row is deleted, Debezium emits TWO messages:
    #       1. A delete event (op='d') with the before image
    #       2. A tombstone (key=row PK, value=null)
    #       The tombstone enables Kafka log compaction to remove the
    #       key entirely during cleanup. Set "false" to skip tombstones.
    #
    #   "decimal.handling.mode": "string"
    #       How to represent DECIMAL/NUMERIC columns:
    #       - "string": "999.99" (safe, no precision loss)
    #       - "double": 999.99 (may lose precision for large decimals)
    #       - "precise": uses org.apache.kafka.connect.data.Decimal
    #         (requires schema-aware converters like Avro)
    #       Use "string" with JSON converters to avoid precision issues.
    #
    # Docs: https://debezium.io/documentation/reference/stable/connectors/postgresql.html#postgresql-connector-properties
    # ─────────────────────────────────────────────────────────────────
    """
    raise NotImplementedError("TODO(human)")


def register_source_connector(connect_url: str) -> None:
    """Register the Debezium PostgreSQL source connector.

    # ── TODO(human) ──────────────────────────────────────────────────
    # This function ties together config building and connector
    # registration using the connect_manager module.
    #
    # Steps:
    #   1. Call build_debezium_config() to get the connector config dict
    #   2. Call connect_manager.create_connector() with:
    #      - connect_url: the Kafka Connect REST URL
    #      - name: config.SOURCE_CONNECTOR_NAME
    #      - connector_config: the dict from step 1
    #   3. Print a confirmation message with the connector name
    #   4. Call connect_manager.wait_for_connector_running() to block
    #      until the connector and all tasks are RUNNING
    #   5. Call connect_manager.print_connector_status() to display
    #      the final status
    #
    # If wait_for_connector_running raises (timeout or failure),
    # let the exception propagate -- the user needs to see the error.
    #
    # After this function completes, Debezium will:
    #   1. Create the replication slot in PostgreSQL
    #   2. Take an initial snapshot (all existing rows as op='r')
    #   3. Start streaming WAL changes in real time
    # ─────────────────────────────────────────────────────────────────
    """
    raise NotImplementedError("TODO(human)")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    """Register the Debezium source connector and verify it's running."""
    print("=== Registering Debezium PostgreSQL Source Connector ===\n")
    register_source_connector(config.CONNECT_REST_URL)
    print("\nSource connector is active. CDC events are flowing to Kafka.")
    print(f"Topics: {config.PRODUCTS_CDC_TOPIC}, {config.ORDERS_CDC_TOPIC}")


if __name__ == "__main__":
    main()
