//! Basic CRUD operations using SQLx compile-time checked queries.
//!
//! This module demonstrates the core SQLx workflow: `query_as!()` for typed
//! SELECT/INSERT/UPDATE statements that are verified against the live database
//! schema at compile time. Every function here uses the macro — the compiler
//! guarantees your SQL is valid before the code can run.

use rust_decimal::Decimal;
use sqlx::PgPool;

use crate::models::Product;

/// Fetch all products from the database, ordered by id.
///
/// # Exercise Context
/// This exercise introduces `query_as!()` — the macro that maps compile-time checked
/// SQL results directly to a named Rust struct. Unlike `query!()` which returns an
/// anonymous record (can't be used in function signatures), `query_as!(Product, ...)`
/// returns `Product` instances that you can pass around, store in collections, and
/// serialize. The compiler verifies that every column in the SELECT matches a field
/// in `Product` by name and type.
///
/// Use `.fetch_all(pool)` to get all rows as a `Vec<Product>`. SQLx also offers
/// `.fetch_one()` (exactly one row), `.fetch_optional()` (zero or one), and
/// `.fetch()` (streaming — returns an async Stream).
///
/// TODO(human): Implement this function.
///
/// Steps:
///   1. Use `sqlx::query_as!(Product, "SELECT * FROM products ORDER BY id")`
///   2. Chain `.fetch_all(pool).await?` to execute and collect all rows
///   3. Return the Vec<Product>
///
/// Note: `SELECT *` works with `query_as!()` because the macro resolves `*` against
/// the actual table schema at compile time and maps each column to the struct field.
pub async fn get_all_products(pool: &PgPool) -> Result<Vec<Product>, sqlx::Error> {
    todo!("TODO(human): implement get_all_products — use query_as! with fetch_all")
}

/// Fetch a single product by its id, returning None if not found.
///
/// # Exercise Context
/// This exercise teaches parameterized queries with `$1` bind parameters. PostgreSQL
/// uses positional parameters (`$1`, `$2`, `$3`...) instead of `?` like MySQL/SQLite.
/// The `query_as!()` macro verifies that the Rust type of each argument matches what
/// PostgreSQL expects for that column. Passing a `String` where the column is `INT`
/// produces a compile-time error.
///
/// Use `.fetch_optional(pool)` instead of `.fetch_one()` because the product might not
/// exist. `.fetch_one()` would return an error for zero rows; `.fetch_optional()` returns
/// `None` cleanly. This is the standard pattern for "find by ID" queries.
///
/// TODO(human): Implement this function.
///
/// Steps:
///   1. Use `sqlx::query_as!(Product, "SELECT * FROM products WHERE id = $1", id)`
///   2. Chain `.fetch_optional(pool).await?`
///   3. Return the Option<Product>
///
/// Experiment: Try passing a `&str` instead of `i32` for the id parameter and observe
/// the compile-time error — the macro knows `products.id` is INT.
pub async fn get_product_by_id(pool: &PgPool, id: i32) -> Result<Option<Product>, sqlx::Error> {
    todo!("TODO(human): implement get_product_by_id — use query_as! with fetch_optional")
}

/// Insert a new product and return the created row (with auto-generated id and timestamp).
///
/// # Exercise Context
/// This exercise teaches INSERT with RETURNING — a powerful PostgreSQL feature that
/// returns the inserted row in the same round-trip. Without RETURNING, you'd need a
/// separate SELECT to get the auto-generated `id` and `created_at` values. The
/// `query_as!(Product, "INSERT ... RETURNING *")` pattern compiles to a single prepared
/// statement that inserts AND returns the complete Product struct.
///
/// The `category` parameter is `Option<&str>` — when `None`, SQLx binds it as NULL.
/// The macro verifies that `Option<&str>` is compatible with the nullable VARCHAR column.
///
/// TODO(human): Implement this function.
///
/// Steps:
///   1. Use `sqlx::query_as!` with an INSERT ... RETURNING * statement:
///      ```
///      sqlx::query_as!(
///          Product,
///          "INSERT INTO products (name, price, category, stock) VALUES ($1, $2, $3, $4) RETURNING *",
///          name,       // $1: &str
///          price,      // $2: Decimal
///          category,   // $3: Option<&str>
///          stock       // $4: i32
///      )
///      ```
///   2. Chain `.fetch_one(pool).await?` — INSERT RETURNING always returns exactly one row
///   3. Return the Product
///
/// Note: `RETURNING *` sends back all columns including auto-generated ones (id, created_at).
pub async fn create_product(
    pool: &PgPool,
    name: &str,
    price: Decimal,
    category: Option<&str>,
    stock: i32,
) -> Result<Product, sqlx::Error> {
    todo!("TODO(human): implement create_product — use query_as! with INSERT RETURNING *")
}

/// Update a product's stock level and return the updated row.
///
/// # Exercise Context
/// This exercise teaches UPDATE with RETURNING — same PostgreSQL pattern as INSERT
/// RETURNING but for modifications. This is the idiomatic way to update a row and
/// get the new state in a single database round-trip. Without RETURNING, you'd need
/// UPDATE + SELECT (two queries, potential race condition between them).
///
/// If the product_id doesn't exist, the UPDATE affects 0 rows and `fetch_one()` will
/// return a `RowNotFound` error. In production, you might use `fetch_optional()` and
/// map `None` to a custom "not found" error.
///
/// TODO(human): Implement this function.
///
/// Steps:
///   1. Use `sqlx::query_as!` with an UPDATE ... RETURNING * statement:
///      ```
///      sqlx::query_as!(
///          Product,
///          "UPDATE products SET stock = $1 WHERE id = $2 RETURNING *",
///          new_stock,
///          product_id
///      )
///      ```
///   2. Chain `.fetch_one(pool).await?`
///   3. Return the updated Product
///
/// Bonus: After implementing, try changing the WHERE clause to use a non-existent id
/// and observe the runtime error from `.fetch_one()` when no row is returned.
pub async fn update_stock(
    pool: &PgPool,
    product_id: i32,
    new_stock: i32,
) -> Result<Product, sqlx::Error> {
    todo!("TODO(human): implement update_stock — use query_as! with UPDATE RETURNING *")
}
