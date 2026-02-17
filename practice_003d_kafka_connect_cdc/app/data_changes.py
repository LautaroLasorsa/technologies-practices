"""Generate database changes for CDC capture.

This script performs a series of INSERT, UPDATE, and DELETE operations
on the source PostgreSQL database. Debezium captures these via WAL
and produces CDC events to Kafka. Run this while cdc_consumer.py
is listening to observe real-time change data capture.

Run standalone:
    uv run python data_changes.py

FULLY IMPLEMENTED -- no TODO(human) in this file.
"""

import time

import psycopg2

import config


# ── Database helpers ─────────────────────────────────────────────────


def get_source_connection():
    """Create a connection to the source PostgreSQL database."""
    return psycopg2.connect(config.SOURCE_DB_CONN_STRING)


def execute_and_report(conn, description: str, sql: str, params=None) -> None:
    """Execute a SQL statement, commit, and report."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()
    print(f"  [OK] {description}")


# ── Change operations ────────────────────────────────────────────────


def insert_new_products(conn) -> None:
    """Insert new products to generate CDC INSERT (op='c') events."""
    print("\n--- Phase 1: INSERT new products ---")

    execute_and_report(
        conn,
        "Inserted 'Keyboard' ($49.99, Electronics)",
        "INSERT INTO products (name, price, category, stock) "
        "VALUES (%s, %s, %s, %s)",
        ("Keyboard", 49.99, "Electronics", 150),
    )
    time.sleep(2)

    execute_and_report(
        conn,
        "Inserted 'Webcam' ($79.99, Electronics)",
        "INSERT INTO products (name, price, category, stock) "
        "VALUES (%s, %s, %s, %s)",
        ("Webcam", 79.99, "Electronics", 60),
    )
    time.sleep(2)

    execute_and_report(
        conn,
        "Inserted 'Standing Desk' ($599.99, Furniture)",
        "INSERT INTO products (name, price, category, stock) "
        "VALUES (%s, %s, %s, %s)",
        ("Standing Desk", 599.99, "Furniture", 15),
    )
    time.sleep(2)


def update_product_prices(conn) -> None:
    """Update product prices to generate CDC UPDATE (op='u') events."""
    print("\n--- Phase 2: UPDATE product prices ---")

    execute_and_report(
        conn,
        "Laptop price: 999.99 -> 899.99 (sale!)",
        "UPDATE products SET price = %s, updated_at = NOW() "
        "WHERE name = %s",
        (899.99, "Laptop"),
    )
    time.sleep(2)

    execute_and_report(
        conn,
        "Mouse price: 29.99 -> 24.99",
        "UPDATE products SET price = %s, updated_at = NOW() "
        "WHERE name = %s",
        (24.99, "Mouse"),
    )
    time.sleep(2)


def update_product_stock(conn) -> None:
    """Update stock quantities to generate more UPDATE events."""
    print("\n--- Phase 3: UPDATE stock levels ---")

    execute_and_report(
        conn,
        "Monitor stock: 80 -> 75 (5 sold)",
        "UPDATE products SET stock = stock - 5, updated_at = NOW() "
        "WHERE name = %s",
        ("Monitor",),
    )
    time.sleep(2)

    execute_and_report(
        conn,
        "Chair stock: 45 -> 0 (sold out!)",
        "UPDATE products SET stock = 0, updated_at = NOW() "
        "WHERE name = %s",
        ("Chair",),
    )
    time.sleep(2)


def delete_product(conn) -> None:
    """Delete a product to generate CDC DELETE (op='d') + tombstone events."""
    print("\n--- Phase 4: DELETE a product ---")

    # First, delete any orders referencing this product (FK constraint)
    execute_and_report(
        conn,
        "Removed orders for 'Desk' (FK cleanup)",
        "DELETE FROM orders WHERE product_id = "
        "(SELECT id FROM products WHERE name = %s)",
        ("Desk",),
    )
    time.sleep(1)

    execute_and_report(
        conn,
        "Deleted product 'Desk'",
        "DELETE FROM products WHERE name = %s",
        ("Desk",),
    )
    time.sleep(2)


def create_orders(conn) -> None:
    """Create orders to generate CDC events on the orders table."""
    print("\n--- Phase 5: INSERT orders ---")

    execute_and_report(
        conn,
        "Order: 2x Laptop for alice@example.com",
        "INSERT INTO orders (product_id, quantity, total_price, status, customer_email) "
        "VALUES ((SELECT id FROM products WHERE name = %s), %s, %s, %s, %s)",
        ("Laptop", 2, 1799.98, "pending", "alice@example.com"),
    )
    time.sleep(2)

    execute_and_report(
        conn,
        "Order: 5x Mouse for bob@example.com",
        "INSERT INTO orders (product_id, quantity, total_price, status, customer_email) "
        "VALUES ((SELECT id FROM products WHERE name = %s), %s, %s, %s, %s)",
        ("Mouse", 5, 124.95, "pending", "bob@example.com"),
    )
    time.sleep(2)


def update_order_status(conn) -> None:
    """Update order statuses to generate UPDATE events on orders."""
    print("\n--- Phase 6: UPDATE order statuses ---")

    execute_and_report(
        conn,
        "Order #1: pending -> shipped",
        "UPDATE orders SET status = %s WHERE id = "
        "(SELECT MIN(id) FROM orders WHERE status = 'pending')",
        ("shipped",),
    )
    time.sleep(2)

    execute_and_report(
        conn,
        "Order #2: pending -> confirmed",
        "UPDATE orders SET status = %s WHERE id = "
        "(SELECT MAX(id) FROM orders WHERE status = 'pending')",
        ("confirmed",),
    )
    time.sleep(2)


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    """Run all change operations sequentially with delays."""
    print("=== Generating Database Changes for CDC ===")
    print("Run cdc_consumer.py in another terminal to observe events.\n")

    conn = get_source_connection()
    try:
        insert_new_products(conn)
        update_product_prices(conn)
        update_product_stock(conn)
        delete_product(conn)
        create_orders(conn)
        update_order_status(conn)

        print("\n=== All changes complete ===")
        print("Check the CDC consumer terminal for captured events.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
