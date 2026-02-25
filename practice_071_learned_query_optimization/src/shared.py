"""Shared constants, connection config, and utilities across all exercises."""

import json
import time
from pathlib import Path

import numpy as np
import psycopg2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

DB_CONFIG = {
    "dbname": "queryopt",
    "user": "practice",
    "password": "practice",
    "host": "localhost",
    "port": 5432,
}


def connect():
    """Connect to PostgreSQL, retrying up to 10 times for container startup."""
    for attempt in range(10):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.autocommit = True
            return conn
        except psycopg2.OperationalError:
            if attempt < 9:
                print(f"  Connection attempt {attempt + 1}/10 failed, retrying...")
                time.sleep(2)
            else:
                raise


# ---------------------------------------------------------------------------
# Known node types for one-hot encoding
# ---------------------------------------------------------------------------

NODE_TYPES = [
    "Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan",
    "Bitmap Index Scan", "Nested Loop", "Hash Join", "Merge Join",
    "Sort", "Hash", "Aggregate", "Group Aggregate", "HashAggregate",
    "Limit", "Materialize", "Gather", "Gather Merge",
    "Append", "Subquery Scan", "CTE Scan", "Result",
]

NODE_TYPE_TO_IDX = {nt: i for i, nt in enumerate(NODE_TYPES)}


# ---------------------------------------------------------------------------
# Template queries for data collection
# ---------------------------------------------------------------------------

TEMPLATE_QUERIES = [
    # Simple scans
    "SELECT * FROM customers WHERE city = 'New York'",
    "SELECT * FROM products WHERE category = 'Electronics' AND price > 500",
    "SELECT * FROM orders WHERE status = 'delivered' AND order_date > '2023-01-01'",
    "SELECT * FROM reviews WHERE rating >= 4",

    # Simple joins
    "SELECT o.order_id, c.name FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE c.city = 'Chicago' LIMIT 100",
    "SELECT oi.item_id, p.name FROM order_items oi JOIN products p ON oi.product_id = p.product_id WHERE p.category = 'Books'",
    "SELECT r.review_id, p.name FROM reviews r JOIN products p ON r.product_id = p.product_id WHERE r.rating = 5",

    # Multi-table joins
    "SELECT c.name, o.order_id, p.name AS product FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id WHERE c.state = 'CA' AND p.category = 'Electronics' LIMIT 50",
    "SELECT c.name, COUNT(*) AS num_orders FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.status = 'delivered' GROUP BY c.customer_id, c.name HAVING COUNT(*) > 5",
    "SELECT p.category, AVG(r.rating) AS avg_rating, COUNT(*) AS num_reviews FROM products p JOIN reviews r ON p.product_id = r.product_id GROUP BY p.category",

    # Aggregations
    "SELECT customer_id, SUM(total_amount) AS total_spent FROM orders GROUP BY customer_id ORDER BY total_spent DESC LIMIT 20",
    "SELECT product_id, COUNT(*) AS times_ordered, SUM(quantity) AS total_qty FROM order_items GROUP BY product_id ORDER BY times_ordered DESC LIMIT 20",
    "SELECT DATE_TRUNC('month', order_date) AS month, COUNT(*) AS num_orders, SUM(total_amount) AS revenue FROM orders GROUP BY month ORDER BY month",

    # Subqueries
    "SELECT * FROM customers WHERE customer_id IN (SELECT customer_id FROM orders WHERE total_amount > 1000)",
    "SELECT * FROM products WHERE product_id IN (SELECT product_id FROM reviews WHERE rating = 5 GROUP BY product_id HAVING COUNT(*) > 3)",

    # Complex joins with filters
    "SELECT c.city, c.state, COUNT(DISTINCT o.order_id) AS orders, SUM(oi.quantity * oi.unit_price) AS revenue FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id WHERE o.order_date BETWEEN '2022-01-01' AND '2022-12-31' GROUP BY c.city, c.state ORDER BY revenue DESC LIMIT 10",
    "SELECT p.category, p.subcategory, COUNT(DISTINCT r.customer_id) AS reviewers, AVG(r.rating) AS avg_rating FROM products p JOIN reviews r ON p.product_id = r.product_id GROUP BY p.category, p.subcategory HAVING COUNT(DISTINCT r.customer_id) > 10 ORDER BY avg_rating DESC",

    # Self-join style / correlated
    "SELECT c.name, c.city FROM customers c WHERE c.is_premium = true AND EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id AND o.total_amount > 500)",
    "SELECT p.name, p.price FROM products p WHERE p.price > (SELECT AVG(price) FROM products WHERE category = p.category)",

    # Wide joins
    "SELECT c.name, c.email, o.order_date, o.status, p.name AS product, p.category, oi.quantity, oi.unit_price, r.rating FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id LEFT JOIN reviews r ON p.product_id = r.product_id AND c.customer_id = r.customer_id WHERE c.state = 'TX' AND o.order_date > '2023-06-01' LIMIT 100",
]


# ---------------------------------------------------------------------------
# EXPLAIN helper
# ---------------------------------------------------------------------------

def get_explain_json(conn, query: str, analyze: bool = True) -> dict:
    """Run EXPLAIN on a query and return the JSON plan."""
    explain_cmd = "EXPLAIN (FORMAT JSON"
    if analyze:
        explain_cmd += ", ANALYZE, BUFFERS"
    explain_cmd += f") {query}"
    with conn.cursor() as cur:
        cur.execute(explain_cmd)
        result = cur.fetchone()[0]
    # EXPLAIN JSON returns a list with one element
    return result[0]["Plan"]
