//! Transaction handling: multi-step operations with atomicity guarantees.
//!
//! Transactions ensure that a group of SQL operations either ALL succeed or ALL
//! fail. SQLx leverages Rust's ownership model for safety: if a Transaction is
//! dropped without calling `.commit()`, it automatically rolls back. This means
//! early returns via `?` (the error propagation operator) trigger automatic rollback.

#[allow(unused_imports)] // Used when implementing the functions — remove after implementing
use rust_decimal::Decimal;
use sqlx::PgPool;

use crate::models::Order;

/// Place an order: check stock, create order, decrement stock — atomically.
///
/// # Exercise Context
/// This exercise teaches SQLx transactions — the most important pattern for data
/// integrity. A transaction groups multiple SQL operations into an atomic unit:
/// either ALL succeed (commit) or ALL are undone (rollback). Without a transaction,
/// you could decrement stock but fail to create the order, leaving the database
/// inconsistent.
///
/// SQLx transactions use Rust's ownership model for safety:
/// - `pool.begin().await?` starts a transaction and returns a `Transaction` object
/// - All queries execute on `&mut *tx` (dereference to the inner connection)
/// - `tx.commit().await?` finalizes the transaction
/// - If `tx` is dropped without `commit()` (e.g., via early `?` return), ROLLBACK
///   happens automatically — this is RAII (Resource Acquisition Is Initialization)
///
/// The business logic: look up the product's price and stock, verify sufficient stock,
/// calculate total_price, insert the order, and decrement the stock. If any step fails,
/// everything rolls back.
///
/// TODO(human): Implement this function.
///
/// Steps:
///   1. Start a transaction: `let mut tx = pool.begin().await?;`
///   2. Fetch the product (price and stock) inside the transaction:
///      ```
///      let product = sqlx::query!(
///          "SELECT price, stock FROM products WHERE id = $1 FOR UPDATE",
///          product_id
///      )
///      .fetch_one(&mut *tx)
///      .await?;
///      ```
///      Note: `FOR UPDATE` locks the row to prevent concurrent modifications
///      (pessimistic locking). Without it, two concurrent orders could both read
///      stock=1 and both succeed, overselling the product.
///   3. Check if `product.stock < quantity` — if so, return an error:
///      `return Err(sqlx::Error::Protocol("Insufficient stock".into()));`
///   4. Calculate total_price: `product.price * Decimal::from(quantity)`
///   5. Insert the order using `sqlx::query_as!(Order, "INSERT INTO orders ... RETURNING *", ...)`
///      executed on `&mut *tx` (not on `pool` — must use the transaction connection)
///   6. Decrement stock: `sqlx::query!("UPDATE products SET stock = stock - $1 WHERE id = $2", quantity, product_id)`
///      executed on `&mut *tx`
///   7. Commit: `tx.commit().await?;`
///   8. Return the created Order
///
/// Key insight: If step 5 or 6 fails (returns Err via `?`), the function returns early.
/// The `tx` variable is dropped, triggering automatic ROLLBACK. The stock is never
/// decremented and no order is created. This is Rust's RAII at its best.
pub async fn place_order(
    pool: &PgPool,
    product_id: i32,
    quantity: i32,
    email: &str,
) -> Result<Order, sqlx::Error> {
    todo!("TODO(human): implement place_order — transaction with stock check, order creation, and stock decrement")
}

/// Update prices for all products in a category by a percentage — atomically.
///
/// # Exercise Context
/// This exercise teaches batch updates within a transaction and working with the
/// `PgQueryResult` type. When you `.execute()` an UPDATE (instead of using RETURNING),
/// SQLx returns a `PgQueryResult` which has `.rows_affected()` — the count of rows
/// that were modified. This is useful for reporting ("updated 15 products") or
/// validating ("expected at least 1 row to change").
///
/// The percent_increase parameter is a f64 (e.g., 10.0 for 10% increase). You need
/// to convert it to a multiplier: `1.0 + (percent_increase / 100.0)`. PostgreSQL can
/// multiply NUMERIC by a float64 parameter — SQLx maps Rust's `f64` to PostgreSQL's
/// FLOAT8/DOUBLE PRECISION automatically.
///
/// TODO(human): Implement this function.
///
/// Steps:
///   1. Start a transaction: `let mut tx = pool.begin().await?;`
///   2. Calculate the multiplier: `let multiplier = 1.0 + (percent_increase / 100.0);`
///   3. Execute the UPDATE on all products matching the category:
///      ```
///      let result = sqlx::query!(
///          "UPDATE products SET price = price * $1 WHERE category = $2",
///          Decimal::try_from(multiplier).unwrap_or(Decimal::ONE),
///          category
///      )
///      .execute(&mut *tx)
///      .await?;
///      ```
///   4. Get the count: `let rows = result.rows_affected();`
///   5. Commit: `tx.commit().await?;`
///   6. Return rows as u64
///
/// Note: Using a transaction here isn't strictly necessary for a single UPDATE, but
/// it's good practice — it demonstrates the pattern and allows adding validation
/// steps (e.g., "don't update if rows_affected is 0") before committing.
pub async fn bulk_price_update(
    pool: &PgPool,
    category: &str,
    percent_increase: f64,
) -> Result<u64, sqlx::Error> {
    todo!("TODO(human): implement bulk_price_update — transaction with UPDATE and rows_affected")
}
