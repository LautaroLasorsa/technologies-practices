"""Python consumer for Debezium CDC events.

Consumes CDC events from Kafka topics produced by Debezium and
displays them in a human-readable format. This is useful for
debugging and understanding the Debezium event envelope structure.

Run standalone:
    uv run python cdc_consumer.py
"""

import json
import time

from confluent_kafka import Consumer, KafkaError

import config


# ── CDC event parsing ────────────────────────────────────────────────


def parse_cdc_event(raw_value: dict) -> dict:
    """Parse a Debezium CDC envelope into a structured summary.

    # ── TODO(human) ──────────────────────────────────────────────────
    # Debezium wraps every change event in an "envelope" structure:
    #
    # {
    #   "before": { ... } | null,    <-- row state BEFORE the change
    #   "after":  { ... } | null,    <-- row state AFTER the change
    #   "source": {                  <-- metadata about the change
    #     "version": "2.7.0.Final",
    #     "connector": "postgresql",
    #     "name": "cdc",             <-- topic.prefix value
    #     "ts_ms": 1700000000000,    <-- timestamp of the DB change
    #     "db": "source_db",
    #     "schema": "public",
    #     "table": "products",
    #     "txId": 501,               <-- PostgreSQL transaction ID
    #     "lsn": 23456789            <-- WAL Log Sequence Number
    #   },
    #   "op": "c",                   <-- operation type
    #   "ts_ms": 1700000001000       <-- when Debezium processed it
    # }
    #
    # Operation types:
    #   "r" = read (initial snapshot)
    #   "c" = create (INSERT)
    #   "u" = update (UPDATE)
    #   "d" = delete (DELETE)
    #
    # Steps:
    #   1. Extract the "op" field from raw_value
    #   2. Map op code to a human-readable name:
    #      {"r": "SNAPSHOT", "c": "INSERT", "u": "UPDATE", "d": "DELETE"}
    #   3. Extract "before" and "after" dicts (may be None)
    #   4. Extract source metadata: table name, transaction ID, LSN
    #   5. Return a dict with keys:
    #      {
    #        "operation": "INSERT",          # human-readable op
    #        "table": "products",            # from source.table
    #        "before": {...} or None,        # before image
    #        "after": {...} or None,         # after image
    #        "tx_id": 501,                   # PostgreSQL txn ID
    #        "lsn": 23456789,               # WAL position
    #        "source_ts_ms": 1700000000000   # when DB change happened
    #      }
    #
    # For INSERT (op="c"): before is null, after has the new row.
    # For UPDATE (op="u"): before has old values, after has new values.
    # For DELETE (op="d"): before has the deleted row, after is null.
    # For SNAPSHOT (op="r"): before is null, after has the existing row.
    # ─────────────────────────────────────────────────────────────────
    """
    raise NotImplementedError("TODO(human)")


def consume_cdc_events(topic: str) -> None:
    """Consume and display CDC events from a Debezium topic.

    # ── TODO(human) ──────────────────────────────────────────────────
    # This function creates a Kafka consumer, subscribes to the given
    # CDC topic, and prints each change event in a readable format.
    #
    # Steps:
    #   1. Create a Consumer with this config dict (positional arg!):
    #      {
    #        "bootstrap.servers": config.BOOTSTRAP_SERVERS,
    #        "group.id": "cdc-monitor-group",
    #        "auto.offset.reset": "earliest",
    #        "enable.auto.commit": True,
    #      }
    #      Note: "earliest" ensures we see snapshot events too.
    #
    #   2. Subscribe to the topic: consumer.subscribe([topic])
    #
    #   3. Poll loop:
    #      a. Call consumer.poll(timeout=1.0)
    #      b. If msg is None, continue (no message available)
    #      c. If msg.error():
    #         - If error code is _PARTITION_EOF, continue (normal)
    #         - Otherwise, print the error and break
    #      d. Deserialize the message value:
    #         raw_value = json.loads(msg.value().decode("utf-8"))
    #      e. Call parse_cdc_event(raw_value) to get structured data
    #      f. Print a formatted summary, for example:
    #         "[INSERT] products: {name: 'Keyboard', price: 49.99}"
    #         "[UPDATE] products: price 999.99 -> 899.99"
    #         "[DELETE] products: removed id=3"
    #         Be creative with the format -- the goal is readability.
    #      g. Use time.sleep(0.2) between polls for signal handling
    #         on Windows (see practice 003a notes)
    #
    #   4. On KeyboardInterrupt, close the consumer gracefully:
    #      consumer.close()
    #
    # Tip: The "after" dict contains column values. For UPDATE events,
    # compare "before" and "after" to show what actually changed.
    # ─────────────────────────────────────────────────────────────────
    """
    raise NotImplementedError("TODO(human)")


# ── Display helpers ──────────────────────────────────────────────────


def format_row_summary(row: dict | None, max_fields: int = 5) -> str:
    """Format a row dict as a compact string, showing up to max_fields."""
    if row is None:
        return "(null)"
    items = list(row.items())[:max_fields]
    fields = ", ".join(f"{k}: {v!r}" for k, v in items)
    if len(row) > max_fields:
        fields += f", ... (+{len(row) - max_fields} more)"
    return f"{{{fields}}}"


def format_changes(before: dict | None, after: dict | None) -> str:
    """Show which fields changed between before and after images."""
    if before is None or after is None:
        return ""
    changes = []
    for key in after:
        old_val = before.get(key)
        new_val = after.get(key)
        if old_val != new_val:
            changes.append(f"{key}: {old_val!r} -> {new_val!r}")
    return ", ".join(changes) if changes else "(no field changes)"


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    """Consume CDC events from the products topic."""
    print(f"=== CDC Consumer: {config.PRODUCTS_CDC_TOPIC} ===")
    print("Listening for change events (Ctrl+C to stop)...\n")
    consume_cdc_events(config.PRODUCTS_CDC_TOPIC)


if __name__ == "__main__":
    main()
