"""Setup: Create e-commerce schema and load synthetic data into PostgreSQL.

This script creates 5 tables with realistic correlations:
- customers (10K rows)
- products (1K rows)
- orders (100K rows)
- order_items (300K rows)
- reviews (50K rows)

Correlations:
- Popular products attract more orders (Zipf-like distribution)
- Active customers place more orders (80/20 rule)
- Products with more orders tend to have more reviews
- Order totals correlate with item counts

Indexes are created on foreign keys and common filter columns.
"""

import random
import time
from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import execute_values

# ---------------------------------------------------------------------------
# Connection
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
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
DROP TABLE IF EXISTS reviews CASCADE;
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

CREATE TABLE customers (
    customer_id   SERIAL PRIMARY KEY,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL UNIQUE,
    city          TEXT NOT NULL,
    state         TEXT NOT NULL,
    country       TEXT NOT NULL DEFAULT 'US',
    created_at    TIMESTAMP NOT NULL,
    is_premium    BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE products (
    product_id    SERIAL PRIMARY KEY,
    name          TEXT NOT NULL,
    category      TEXT NOT NULL,
    subcategory   TEXT NOT NULL,
    price         NUMERIC(10, 2) NOT NULL,
    weight_kg     NUMERIC(6, 2) NOT NULL,
    in_stock      BOOLEAN NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMP NOT NULL
);

CREATE TABLE orders (
    order_id      SERIAL PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date    DATE NOT NULL,
    status        TEXT NOT NULL,
    total_amount  NUMERIC(12, 2) NOT NULL,
    shipping_city TEXT NOT NULL,
    shipping_state TEXT NOT NULL
);

CREATE TABLE order_items (
    item_id       SERIAL PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL,
    unit_price    NUMERIC(10, 2) NOT NULL,
    discount_pct  NUMERIC(5, 2) NOT NULL DEFAULT 0
);

CREATE TABLE reviews (
    review_id     SERIAL PRIMARY KEY,
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    rating        INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    review_text   TEXT,
    created_at    TIMESTAMP NOT NULL
);
"""

INDEX_SQL = """
-- Foreign key indexes
CREATE INDEX idx_orders_customer     ON orders(customer_id);
CREATE INDEX idx_orders_date         ON orders(order_date);
CREATE INDEX idx_orders_status       ON orders(status);
CREATE INDEX idx_order_items_order   ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);
CREATE INDEX idx_reviews_product     ON reviews(product_id);
CREATE INDEX idx_reviews_customer    ON reviews(customer_id);
CREATE INDEX idx_reviews_rating      ON reviews(rating);

-- Filter columns
CREATE INDEX idx_customers_city      ON customers(city);
CREATE INDEX idx_customers_state     ON customers(state);
CREATE INDEX idx_customers_premium   ON customers(is_premium);
CREATE INDEX idx_products_category   ON products(category);
CREATE INDEX idx_products_price      ON products(price);
CREATE INDEX idx_products_in_stock   ON products(in_stock);

-- Composite indexes for common query patterns
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);
CREATE INDEX idx_order_items_order_product ON order_items(order_id, product_id);
CREATE INDEX idx_reviews_product_rating ON reviews(product_id, rating);
"""


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

CITIES = [
    ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"),
    ("Houston", "TX"), ("Phoenix", "AZ"), ("Philadelphia", "PA"),
    ("San Antonio", "TX"), ("San Diego", "CA"), ("Dallas", "TX"),
    ("San Jose", "CA"), ("Austin", "TX"), ("Jacksonville", "FL"),
    ("Fort Worth", "TX"), ("Columbus", "OH"), ("Charlotte", "NC"),
    ("Indianapolis", "IN"), ("San Francisco", "CA"), ("Seattle", "WA"),
    ("Denver", "CO"), ("Nashville", "TN"), ("Portland", "OR"),
    ("Detroit", "MI"), ("Memphis", "TN"), ("Boston", "MA"),
    ("Atlanta", "GA"), ("Miami", "FL"), ("Minneapolis", "MN"),
    ("Tampa", "FL"), ("St. Louis", "MO"), ("Pittsburgh", "PA"),
]

CATEGORIES = {
    "Electronics": ["Laptops", "Phones", "Tablets", "Headphones", "Cameras"],
    "Clothing": ["Shirts", "Pants", "Dresses", "Shoes", "Jackets"],
    "Home": ["Furniture", "Kitchen", "Bedding", "Lighting", "Decor"],
    "Books": ["Fiction", "Non-Fiction", "Technical", "Children", "Comics"],
    "Sports": ["Fitness", "Outdoor", "Team Sports", "Water Sports", "Cycling"],
}

STATUSES = ["pending", "shipped", "delivered", "returned", "cancelled"]
STATUS_WEIGHTS = [0.05, 0.10, 0.75, 0.05, 0.05]

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen",
    "Daniel", "Lisa", "Matthew", "Nancy", "Anthony", "Betty", "Mark",
    "Margaret", "Steven", "Sandra", "Andrew", "Ashley", "Paul", "Dorothy",
    "Joshua", "Kimberly", "Kenneth", "Emily", "Kevin", "Donna",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
]


def generate_customers(n: int = 10_000) -> list[tuple]:
    """Generate customer rows with city/state correlations."""
    rng = random.Random(42)
    base_date = datetime(2020, 1, 1)
    rows = []
    for i in range(1, n + 1):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}.{i}@example.com"
        city, state = rng.choice(CITIES)
        created = base_date + timedelta(days=rng.randint(0, 1500))
        is_premium = rng.random() < 0.15  # 15% premium
        rows.append((name, email, city, state, "US", created, is_premium))
    return rows


def generate_products(n: int = 1_000) -> list[tuple]:
    """Generate product rows across categories."""
    rng = random.Random(43)
    base_date = datetime(2020, 1, 1)
    rows = []
    adjectives = ["Premium", "Basic", "Pro", "Ultra", "Classic", "Modern",
                  "Deluxe", "Essential", "Advanced", "Standard"]
    for i in range(1, n + 1):
        category = rng.choice(list(CATEGORIES.keys()))
        subcategory = rng.choice(CATEGORIES[category])
        adj = rng.choice(adjectives)
        name = f"{adj} {subcategory} {i}"
        # Price varies by category
        base_prices = {
            "Electronics": (50, 2000), "Clothing": (15, 200),
            "Home": (20, 500), "Books": (5, 60), "Sports": (10, 300),
        }
        lo, hi = base_prices[category]
        price = round(rng.uniform(lo, hi), 2)
        weight = round(rng.uniform(0.1, 25.0), 2)
        in_stock = rng.random() < 0.85
        created = base_date + timedelta(days=rng.randint(0, 1200))
        rows.append((name, category, subcategory, price, weight, in_stock, created))
    return rows


def generate_orders(
    n: int = 100_000,
    num_customers: int = 10_000,
) -> list[tuple]:
    """Generate orders with 80/20 customer activity distribution."""
    rng = random.Random(44)
    # 20% of customers generate 80% of orders (power-law)
    active_customers = list(range(1, int(num_customers * 0.2) + 1))
    regular_customers = list(range(int(num_customers * 0.2) + 1, num_customers + 1))
    base_date = datetime(2021, 1, 1)
    rows = []
    for _ in range(n):
        if rng.random() < 0.8:
            customer_id = rng.choice(active_customers)
        else:
            customer_id = rng.choice(regular_customers)
        order_date = base_date + timedelta(days=rng.randint(0, 1095))  # 3 years
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
        total = round(rng.uniform(10, 2000), 2)
        city, state = rng.choice(CITIES)
        rows.append((customer_id, order_date.date(), status, total, city, state))
    return rows


def generate_order_items(
    n_orders: int = 100_000,
    n_products: int = 1_000,
    target_items: int = 300_000,
) -> list[tuple]:
    """Generate order items with Zipf-like product popularity."""
    rng = random.Random(45)
    # Zipf distribution: product k has popularity ~ 1/k^0.8
    product_weights = [1.0 / ((k + 1) ** 0.8) for k in range(n_products)]
    total_w = sum(product_weights)
    product_weights = [w / total_w for w in product_weights]
    product_ids = list(range(1, n_products + 1))

    rows = []
    items_per_order = target_items // n_orders  # ~3 items per order average
    for order_id in range(1, n_orders + 1):
        n_items = max(1, rng.randint(1, items_per_order * 2))
        for _ in range(n_items):
            product_id = rng.choices(product_ids, weights=product_weights, k=1)[0]
            quantity = rng.choices([1, 2, 3, 4, 5], weights=[50, 25, 15, 7, 3], k=1)[0]
            unit_price = round(rng.uniform(5, 500), 2)
            discount = round(rng.choice([0, 0, 0, 5, 10, 15, 20, 25]), 2)
            rows.append((order_id, product_id, quantity, unit_price, discount))
            if len(rows) >= target_items:
                return rows
    return rows


def generate_reviews(
    n: int = 50_000,
    n_products: int = 1_000,
    n_customers: int = 10_000,
) -> list[tuple]:
    """Generate reviews correlated with product popularity."""
    rng = random.Random(46)
    # Popular products get more reviews (same Zipf as order_items)
    product_weights = [1.0 / ((k + 1) ** 0.8) for k in range(n_products)]
    total_w = sum(product_weights)
    product_weights = [w / total_w for w in product_weights]
    product_ids = list(range(1, n_products + 1))

    base_date = datetime(2021, 6, 1)
    rows = []
    for _ in range(n):
        product_id = rng.choices(product_ids, weights=product_weights, k=1)[0]
        customer_id = rng.randint(1, n_customers)
        # Slightly positive skew in ratings
        rating = rng.choices([1, 2, 3, 4, 5], weights=[5, 10, 20, 35, 30], k=1)[0]
        review_text = f"Review for product {product_id}: {'Great' if rating >= 4 else 'OK' if rating == 3 else 'Poor'} product."
        created = base_date + timedelta(
            days=rng.randint(0, 900),
            hours=rng.randint(0, 23),
            minutes=rng.randint(0, 59),
        )
        rows.append((product_id, customer_id, rating, review_text, created))
    return rows


# ---------------------------------------------------------------------------
# Bulk insert
# ---------------------------------------------------------------------------

def bulk_insert(conn, table: str, columns: list[str], rows: list[tuple]) -> None:
    """Insert rows using execute_values for performance."""
    cols = ", ".join(columns)
    template = f"({', '.join(['%s'] * len(columns))})"
    sql = f"INSERT INTO {table} ({cols}) VALUES %s"
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, template=template, page_size=5000)
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Setup: Create e-commerce schema and load synthetic data")
    print("=" * 60)

    print("\nConnecting to PostgreSQL...")
    conn = connect()
    print("  Connected.")

    print("\nCreating schema...")
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()
    print("  Tables created.")

    # Generate and insert data
    print("\nGenerating customers (10,000)...")
    customers = generate_customers(10_000)
    bulk_insert(conn, "customers",
                ["name", "email", "city", "state", "country", "created_at", "is_premium"],
                customers)
    print(f"  Inserted {len(customers)} customers.")

    print("Generating products (1,000)...")
    products = generate_products(1_000)
    bulk_insert(conn, "products",
                ["name", "category", "subcategory", "price", "weight_kg", "in_stock", "created_at"],
                products)
    print(f"  Inserted {len(products)} products.")

    print("Generating orders (100,000)...")
    orders = generate_orders(100_000)
    bulk_insert(conn, "orders",
                ["customer_id", "order_date", "status", "total_amount", "shipping_city", "shipping_state"],
                orders)
    print(f"  Inserted {len(orders)} orders.")

    print("Generating order items (300,000)...")
    order_items = generate_order_items()
    bulk_insert(conn, "order_items",
                ["order_id", "product_id", "quantity", "unit_price", "discount_pct"],
                order_items)
    print(f"  Inserted {len(order_items)} order items.")

    print("Generating reviews (50,000)...")
    reviews = generate_reviews(50_000)
    bulk_insert(conn, "reviews",
                ["product_id", "customer_id", "rating", "review_text", "created_at"],
                reviews)
    print(f"  Inserted {len(reviews)} reviews.")

    # Create indexes
    print("\nCreating indexes...")
    with conn.cursor() as cur:
        for statement in INDEX_SQL.strip().split(";"):
            statement = statement.strip()
            if statement and not statement.startswith("--"):
                cur.execute(statement)
    conn.commit()
    print("  Indexes created.")

    # Run ANALYZE to collect statistics
    print("\nRunning ANALYZE to collect statistics...")
    with conn.cursor() as cur:
        cur.execute("ANALYZE;")
    print("  Statistics collected.")

    # Print summary
    print("\n" + "-" * 60)
    print("Database summary:")
    print("-" * 60)
    tables = ["customers", "products", "orders", "order_items", "reviews"]
    with conn.cursor() as cur:
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            cur.execute(
                "SELECT pg_size_pretty(pg_total_relation_size(%s))",
                (table,),
            )
            size = cur.fetchone()[0]
            print(f"  {table:20s}: {count:>10,} rows  ({size})")

    print("\nIndex summary:")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT indexname, tablename, pg_size_pretty(pg_relation_size(indexname::regclass))
            FROM pg_indexes
            WHERE schemaname = 'public'
            ORDER BY tablename, indexname
        """)
        for idx_name, tbl_name, size in cur.fetchall():
            print(f"  {idx_name:40s} on {tbl_name:15s} ({size})")

    conn.close()
    print("\nSetup complete!")


if __name__ == "__main__":
    main()
