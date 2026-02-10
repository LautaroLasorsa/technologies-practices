"""Seed the Northwind-inspired database with sample data.

Creates all tables (dropping existing ones first for idempotency)
and inserts realistic sample data for the Advanced SQL practice.

Tables: categories, employees, suppliers, products, customers, orders, order_details

Run: uv run python -m app.seed
"""

import random
from datetime import date, timedelta

from app.connection import get_connection

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

DROP_TABLES = """
DROP TABLE IF EXISTS order_details CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS suppliers CASCADE;
DROP TABLE IF EXISTS employees CASCADE;
DROP TABLE IF EXISTS categories CASCADE;
"""

CREATE_TABLES = """
CREATE TABLE categories (
    category_id   SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    description   TEXT,
    parent_category_id INT REFERENCES categories(category_id)
);

CREATE TABLE employees (
    employee_id   SERIAL PRIMARY KEY,
    first_name    VARCHAR(50) NOT NULL,
    last_name     VARCHAR(50) NOT NULL,
    title         VARCHAR(100),
    reports_to    INT REFERENCES employees(employee_id),
    hire_date     DATE
);

CREATE TABLE suppliers (
    supplier_id   SERIAL PRIMARY KEY,
    company_name  VARCHAR(100) NOT NULL,
    contact_name  VARCHAR(100),
    country       VARCHAR(50)
);

CREATE TABLE products (
    product_id    SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    supplier_id   INT REFERENCES suppliers(supplier_id),
    category_id   INT REFERENCES categories(category_id),
    unit_price    NUMERIC(10,2) NOT NULL,
    units_in_stock INT NOT NULL DEFAULT 0,
    discontinued  BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE customers (
    customer_id   SERIAL PRIMARY KEY,
    company_name  VARCHAR(100) NOT NULL,
    contact_name  VARCHAR(100),
    country       VARCHAR(50),
    city          VARCHAR(50)
);

CREATE TABLE orders (
    order_id      SERIAL PRIMARY KEY,
    customer_id   INT REFERENCES customers(customer_id),
    employee_id   INT REFERENCES employees(employee_id),
    order_date    DATE NOT NULL,
    shipped_date  DATE,
    freight       NUMERIC(10,2)
);

CREATE TABLE order_details (
    order_detail_id SERIAL PRIMARY KEY,
    order_id      INT REFERENCES orders(order_id),
    product_id    INT REFERENCES products(product_id),
    unit_price    NUMERIC(10,2) NOT NULL,
    quantity      INT NOT NULL,
    discount      NUMERIC(3,2) NOT NULL DEFAULT 0.00
);
"""

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

CATEGORIES = [
    # (name, description, parent_category_id or None)
    ("Beverages", "Soft drinks, coffees, teas, beers, and ales", None),                   # 1
    ("Condiments", "Sweet and savory sauces, relishes, spreads, and seasonings", None),    # 2
    ("Confections", "Desserts, candies, and sweet breads", None),                          # 3
    ("Dairy Products", "Cheeses, milk, and yogurt", None),                                 # 4
    ("Grains & Cereals", "Breads, crackers, pasta, and cereal", None),                     # 5
    ("Meat & Poultry", "Prepared meats and poultry products", None),                       # 6
    ("Produce", "Dried fruit, vegetables, and bean curd", None),                           # 7
    # Subcategories (have parent_category_id for recursive CTE practice)
    ("Hot Beverages", "Coffees and teas", 1),                                              # 8
    ("Cold Beverages", "Sodas, juices, and iced teas", 1),                                 # 9
    ("Organic Produce", "Certified organic fruits and vegetables", 7),                     # 10
]

EMPLOYEES = [
    # (first_name, last_name, title, reports_to, hire_date)
    # 1: CEO (no manager)
    ("Andrew", "Fuller", "VP of Sales", None, "2020-01-15"),
    # 2-3: Managers (report to CEO)
    ("Janet", "Leverling", "Sales Manager - East", 1, "2020-06-01"),
    ("Steven", "Buchanan", "Sales Manager - West", 1, "2020-08-20"),
    # 4-8: Staff (report to managers)
    ("Nancy", "Davolio", "Sales Representative", 2, "2021-03-10"),
    ("Margaret", "Peacock", "Sales Representative", 2, "2021-05-15"),
    ("Michael", "Suyama", "Sales Representative", 3, "2021-07-01"),
    ("Robert", "King", "Sales Representative", 3, "2022-01-10"),
    ("Laura", "Callahan", "Inside Sales Coordinator", 2, "2022-04-20"),
]

SUPPLIERS = [
    ("Exotic Liquids", "Charlotte Cooper", "United Kingdom"),
    ("New Orleans Cajun Delights", "Shelley Burke", "United States"),
    ("Grandma Kelly's Homestead", "Regina Murphy", "United States"),
    ("Tokyo Traders", "Yoshi Nagase", "Japan"),
    ("Pavlova Ltd.", "Ian Devling", "Australia"),
    ("Formaggi Fortini s.r.l.", "Elio Rossi", "Italy"),
]

PRODUCTS = [
    # (name, supplier_id, category_id, unit_price, units_in_stock, discontinued)
    ("Chai Tea", 1, 8, 18.00, 39, False),
    ("Chang Beer", 1, 9, 19.00, 17, False),
    ("Aniseed Syrup", 1, 2, 10.00, 13, False),
    ("Chef Anton's Cajun Seasoning", 2, 2, 22.00, 53, False),
    ("Chef Anton's Gumbo Mix", 2, 2, 21.35, 0, True),
    ("Grandma's Boysenberry Spread", 3, 2, 25.00, 120, False),
    ("Uncle Bob's Organic Dried Pears", 3, 10, 30.00, 15, False),
    ("Northwoods Cranberry Sauce", 3, 2, 40.00, 6, False),
    ("Mishi Kobe Niku", 4, 6, 97.00, 29, True),
    ("Ikura Salmon Roe", 4, 6, 31.00, 31, False),
    ("Queso Cabrales", 6, 4, 21.00, 22, False),
    ("Queso Manchego La Pastora", 6, 4, 38.00, 86, False),
    ("Konbu Seaweed", 4, 7, 6.00, 24, False),
    ("Tofu", 4, 10, 23.25, 35, False),
    ("Genen Shouyu Soy Sauce", 4, 2, 15.50, 39, False),
    ("Pavlova Meringue", 5, 3, 17.45, 29, False),
    ("Alice Mutton", 5, 6, 39.00, 0, True),
    ("Carnarvon Tigers Prawns", 5, 6, 62.50, 42, False),
    ("Teatime Chocolate Biscuits", 5, 3, 9.20, 25, False),
    ("Sir Rodney's Marmalade", 5, 3, 81.00, 40, False),
    ("Sir Rodney's Scones", 5, 3, 10.00, 3, False),
    ("Gustaf's Knackebrod", 3, 5, 21.00, 104, False),
    ("Tunnbrod", 3, 5, 9.00, 61, False),
    ("Guarana Fantastica", 2, 9, 4.50, 20, False),
    ("NuNuCa Nuss-Nougat-Creme", 6, 3, 14.00, 76, False),
    ("Gumbar Gummibarchen", 6, 3, 31.23, 15, False),
    ("Schoggi Schokolade", 6, 3, 43.90, 49, False),
    ("Rossle Sauerkraut", 3, 7, 45.60, 26, False),
    ("Thuringer Rostbratwurst", 3, 6, 123.79, 0, True),
    ("Nord-Ost Matjeshering", 4, 6, 25.89, 10, False),
]

CUSTOMERS = [
    ("Alfreds Futterkiste", "Maria Anders", "Germany", "Berlin"),
    ("Ana Trujillo Emparedados", "Ana Trujillo", "Mexico", "Mexico City"),
    ("Antonio Moreno Taqueria", "Antonio Moreno", "Mexico", "Mexico City"),
    ("Around the Horn", "Thomas Hardy", "United Kingdom", "London"),
    ("Berglunds snabbkop", "Christina Berglund", "Sweden", "Lulea"),
    ("Blauer See Delikatessen", "Hanna Moos", "Germany", "Mannheim"),
    ("Bolido Comidas preparadas", "Martin Sommer", "Spain", "Madrid"),
    ("Bon app'", "Laurence Lebihan", "France", "Marseille"),
    ("Bottom-Dollar Markets", "Elizabeth Lincoln", "Canada", "Tsawwassen"),
    ("B's Beverages", "Victoria Ashworth", "United Kingdom", "London"),
    ("Cactus Comidas para llevar", "Patricio Simpson", "Argentina", "Buenos Aires"),
    ("Centro comercial Moctezuma", "Francisco Chang", "Mexico", "Mexico City"),
    ("Chop-suey Chinese", "Yang Wang", "Switzerland", "Bern"),
    ("Consolidated Holdings", "Elizabeth Brown", "United Kingdom", "London"),
    ("Drachenblut Delikatessen", "Sven Ottlieb", "Germany", "Aachen"),
    ("Eastern Connection", "Ann Devon", "United Kingdom", "London"),
    ("Ernst Handel", "Roland Mendel", "Austria", "Graz"),
    ("Familia Arquibaldo", "Aria Cruz", "Brazil", "Sao Paulo"),
    ("FISSA Fabrica Inter. Salchichas", "Diego Roel", "Spain", "Madrid"),
    ("Folk och fa HB", "Maria Larsson", "Sweden", "Brackne-Hoby"),
]


def _generate_orders(
    num_orders: int,
    num_customers: int,
    num_employees: int,
    num_products: int,
    product_prices: list[float],
) -> tuple[list[tuple], list[tuple]]:
    """Generate random orders and order_details rows.

    Returns (orders_rows, order_details_rows).
    """
    random.seed(42)  # reproducible data

    start_date = date(2023, 1, 1)
    end_date = date(2024, 12, 31)
    date_range = (end_date - start_date).days

    orders = []
    details = []
    detail_id = 1

    for order_id in range(1, num_orders + 1):
        customer_id = random.randint(1, num_customers)
        employee_id = random.randint(1, num_employees)
        order_date = start_date + timedelta(days=random.randint(0, date_range))
        shipped_date = order_date + timedelta(days=random.randint(1, 14))
        freight = round(random.uniform(5.0, 150.0), 2)

        orders.append((order_id, customer_id, employee_id, order_date, shipped_date, freight))

        # Each order has 1-5 line items
        num_items = random.randint(1, 5)
        chosen_products = random.sample(range(1, num_products + 1), min(num_items, num_products))

        for product_id in chosen_products:
            unit_price = product_prices[product_id - 1]
            quantity = random.randint(1, 30)
            discount = random.choice([0.00, 0.00, 0.00, 0.05, 0.10, 0.15, 0.20])

            details.append((detail_id, order_id, product_id, unit_price, quantity, discount))
            detail_id += 1

    return orders, details


# ---------------------------------------------------------------------------
# Main seed logic
# ---------------------------------------------------------------------------

def seed() -> None:
    """Create tables and insert all sample data."""
    product_prices = [p[3] for p in PRODUCTS]

    orders, order_details = _generate_orders(
        num_orders=200,
        num_customers=len(CUSTOMERS),
        num_employees=len(EMPLOYEES),
        num_products=len(PRODUCTS),
        product_prices=product_prices,
    )

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Drop and recreate tables
            print("  Dropping existing tables...")
            cur.execute(DROP_TABLES)
            conn.commit()

            print("  Creating tables...")
            cur.execute(CREATE_TABLES)
            conn.commit()

            # Categories
            print(f"  Inserting {len(CATEGORIES)} categories...")
            # Insert root categories first (parent_category_id IS NULL)
            for name, description, parent_id in CATEGORIES:
                if parent_id is None:
                    cur.execute(
                        "INSERT INTO categories (name, description, parent_category_id) "
                        "VALUES (%s, %s, NULL)",
                        (name, description),
                    )
            conn.commit()
            # Then subcategories
            for name, description, parent_id in CATEGORIES:
                if parent_id is not None:
                    cur.execute(
                        "INSERT INTO categories (name, description, parent_category_id) "
                        "VALUES (%s, %s, %s)",
                        (name, description, parent_id),
                    )
            conn.commit()

            # Employees (insert in order — parent must exist before child)
            print(f"  Inserting {len(EMPLOYEES)} employees...")
            for first_name, last_name, title, reports_to, hire_date in EMPLOYEES:
                cur.execute(
                    "INSERT INTO employees (first_name, last_name, title, reports_to, hire_date) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (first_name, last_name, title, reports_to, hire_date),
                )
            conn.commit()

            # Suppliers
            print(f"  Inserting {len(SUPPLIERS)} suppliers...")
            for company_name, contact_name, country in SUPPLIERS:
                cur.execute(
                    "INSERT INTO suppliers (company_name, contact_name, country) "
                    "VALUES (%s, %s, %s)",
                    (company_name, contact_name, country),
                )
            conn.commit()

            # Products
            print(f"  Inserting {len(PRODUCTS)} products...")
            for name, supplier_id, category_id, unit_price, units_in_stock, discontinued in PRODUCTS:
                cur.execute(
                    "INSERT INTO products (name, supplier_id, category_id, unit_price, "
                    "units_in_stock, discontinued) VALUES (%s, %s, %s, %s, %s, %s)",
                    (name, supplier_id, category_id, unit_price, units_in_stock, discontinued),
                )
            conn.commit()

            # Customers
            print(f"  Inserting {len(CUSTOMERS)} customers...")
            for company_name, contact_name, country, city in CUSTOMERS:
                cur.execute(
                    "INSERT INTO customers (company_name, contact_name, country, city) "
                    "VALUES (%s, %s, %s, %s)",
                    (company_name, contact_name, country, city),
                )
            conn.commit()

            # Orders
            print(f"  Inserting {len(orders)} orders...")
            for order_id, customer_id, employee_id, order_date, shipped_date, freight in orders:
                cur.execute(
                    "INSERT INTO orders (customer_id, employee_id, order_date, shipped_date, freight) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (customer_id, employee_id, order_date, shipped_date, freight),
                )
            conn.commit()

            # Order details
            print(f"  Inserting {len(order_details)} order detail lines...")
            for detail_id, order_id, product_id, unit_price, quantity, discount in order_details:
                cur.execute(
                    "INSERT INTO order_details (order_id, product_id, unit_price, quantity, discount) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (order_id, product_id, unit_price, quantity, discount),
                )
            conn.commit()

    _print_summary()


def _print_summary() -> None:
    """Print row counts for all tables as verification."""
    tables = ["categories", "employees", "suppliers", "products", "customers", "orders", "order_details"]

    print()
    print("  Seed complete. Row counts:")
    print("  " + "-" * 35)

    with get_connection() as conn:
        with conn.cursor() as cur:
            for table in tables:
                cur.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
                count = cur.fetchone()[0]
                print(f"  {table:<20s} {count:>6d}")

    print("  " + "-" * 35)
    print()


if __name__ == "__main__":
    print()
    print("=" * 50)
    print("  Seeding Northwind database...")
    print("=" * 50)
    print()
    seed()
