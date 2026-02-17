//! Schema creation and seed data.
//!
//! This module handles the initial database setup: creating tables (DDL) and
//! inserting sample data. It demonstrates the key distinction between runtime
//! queries (for DDL) and compile-time macros (for DML after tables exist).

#[allow(unused_imports)] // Used when implementing seed_data — remove this after implementing
use rust_decimal::Decimal;
use sqlx::PgPool;

/// Create the `products` and `orders` tables in PostgreSQL.
///
/// # Exercise Context
/// This exercise teaches the fundamental distinction between compile-time and runtime
/// queries in SQLx. You MUST use `sqlx::query()` (runtime, no `!`) here because the
/// `query!()` macro verifies SQL against the database schema at compile time — but these
/// tables don't exist yet when the compiler runs for the first time. This is a common
/// pattern: DDL (CREATE TABLE, ALTER TABLE, DROP) always uses runtime queries.
///
/// The `sqlx::query()` function returns a `Query` struct. Call `.execute(pool)` on it
/// to run the SQL without expecting any rows back. Use `DROP TABLE IF EXISTS ... CASCADE`
/// to make the function idempotent (safe to run multiple times).
///
/// TODO(human): Implement this function.
///
/// Steps:
///   1. Use `sqlx::query("DROP TABLE IF EXISTS orders CASCADE").execute(pool).await?`
///      to clean up the orders table first (it has a FK to products)
///   2. Use `sqlx::query("DROP TABLE IF EXISTS products CASCADE").execute(pool).await?`
///      to clean up the products table
///   3. Create the `products` table with these columns:
///      - id: SERIAL PRIMARY KEY
///      - name: VARCHAR(255) NOT NULL UNIQUE
///      - price: NUMERIC(10,2) NOT NULL
///      - category: VARCHAR(100)  (nullable — not all products have categories)
///      - stock: INT NOT NULL DEFAULT 0
///      - created_at: TIMESTAMP NOT NULL DEFAULT NOW()
///   4. Create the `orders` table with these columns:
///      - id: SERIAL PRIMARY KEY
///      - product_id: INT NOT NULL REFERENCES products(id)
///      - quantity: INT NOT NULL
///      - total_price: NUMERIC(10,2) NOT NULL
///      - status: VARCHAR(50) NOT NULL DEFAULT 'pending'
///      - customer_email: VARCHAR(255)  (nullable)
///      - created_at: TIMESTAMP NOT NULL DEFAULT NOW()
///   5. Return Ok(())
///
/// Hint: Each `sqlx::query("SQL").execute(pool).await?` is one statement.
///       You need 4 calls total: 2 drops + 2 creates.
pub async fn create_schema(pool: &PgPool) -> Result<(), sqlx::Error> {
    todo!("TODO(human): implement create_schema — use sqlx::query() (runtime) for DDL")
}

/// Insert sample products into the `products` table.
///
/// # Exercise Context
/// Now that the tables exist (after `create_schema`), you can use the compile-time
/// `sqlx::query!()` macro for INSERT statements. This is your first hands-on experience
/// with the macro: when you run `cargo build`, the Rust compiler connects to PostgreSQL,
/// sends your SQL to the database for validation, and generates typed code. If you
/// misspell a column name, use the wrong type for a parameter, or reference a
/// non-existent table, the BUILD fails — not the runtime.
///
/// Use `$1`, `$2`, etc. for PostgreSQL-style positional bind parameters. The macro
/// verifies that each `$N` parameter matches the Rust type you provide. For example,
/// `$1` bound to a `&str` must correspond to a VARCHAR/TEXT column.
///
/// TODO(human): Implement this function.
///
/// Steps:
///   1. Insert 5 products using `sqlx::query!()` with bind parameters. Example:
///      ```
///      sqlx::query!(
///          "INSERT INTO products (name, price, category, stock) VALUES ($1, $2, $3, $4)",
///          "Widget Alpha",                              // $1: &str → VARCHAR
///          Decimal::new(1999, 2),                       // $2: Decimal → NUMERIC(10,2) = 19.99
///          Some("electronics") as Option<&str>,         // $3: Option<&str> → nullable VARCHAR
///          100i32                                       // $4: i32 → INT
///      )
///      .execute(pool)
///      .await?;
///      ```
///   2. Insert at least 5 products with varying categories (some None, some "electronics",
///      "books", "clothing", etc.) and different stock levels
///   3. Return Ok(())
///
/// Experiment: Try misspelling a column name (e.g., "proce" instead of "price") and
/// run `cargo build` — observe the compile-time error. This is SQLx's core value.
pub async fn seed_data(pool: &PgPool) -> Result<(), sqlx::Error> {
    todo!("TODO(human): implement seed_data — use sqlx::query!() macro for INSERT statements")
}
