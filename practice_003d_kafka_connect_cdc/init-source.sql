-- ============================================================
-- Source database schema for CDC practice
-- ============================================================
-- PostgreSQL uses WAL (Write-Ahead Log) for crash recovery.
-- Setting wal_level=logical (done via docker-compose command)
-- enables the WAL to include enough information for Debezium
-- to reconstruct row-level changes (INSERT/UPDATE/DELETE).

-- ── Products table ──────────────────────────────────────────

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    category VARCHAR(100),
    stock INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ── Orders table ────────────────────────────────────────────

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    total_price DECIMAL(10,2) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    customer_email VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- ── REPLICA IDENTITY FULL ───────────────────────────────────
-- By default, PostgreSQL only includes the primary key in UPDATE
-- and DELETE WAL entries. REPLICA IDENTITY FULL tells PostgreSQL
-- to include ALL column values in both the "before" and "after"
-- images. This is essential for Debezium to provide complete
-- before/after snapshots in CDC events.

ALTER TABLE products REPLICA IDENTITY FULL;
ALTER TABLE orders REPLICA IDENTITY FULL;

-- ── Seed data ───────────────────────────────────────────────
-- These rows will appear in the initial snapshot when Debezium
-- first connects (op='r' = read/snapshot events).

INSERT INTO products (name, price, category, stock) VALUES
    ('Laptop', 999.99, 'Electronics', 50),
    ('Mouse', 29.99, 'Electronics', 200),
    ('Desk', 349.99, 'Furniture', 30),
    ('Chair', 249.99, 'Furniture', 45),
    ('Monitor', 449.99, 'Electronics', 80);
