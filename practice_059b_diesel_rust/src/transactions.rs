use bigdecimal::BigDecimal;
use diesel::prelude::*;
use diesel::PgConnection;

use crate::models::{NewOrder, Order, Product};
use crate::schema::{orders, products};

// =============================================================================
// Transaction handling with Diesel.
//
// Diesel's transaction API wraps PostgreSQL's BEGIN / COMMIT / ROLLBACK.
// The closure-based API ensures that if ANY operation within the transaction
// fails (returns Err), the entire transaction is rolled back automatically.
// This is Rust's RAII pattern applied to database transactions — the
// transaction's lifetime is tied to the closure's scope.
// =============================================================================

/// TODO(human): Place an order inside a transaction.
///
/// This function must atomically:
///   1. Look up the product by ID (fail if not found)
///   2. Check that sufficient stock is available (fail if not enough)
///   3. Calculate total_price = product.price * quantity
///   4. Insert a new order row
///   5. Decrement the product's stock by the ordered quantity
///
/// All five steps must happen inside `conn.transaction(|conn| { ... })`.
/// If any step fails, ALL changes are rolled back — no partial orders,
/// no stock decremented without an order being created.
///
/// Diesel's transaction API:
///   `conn.transaction(|conn| {
///       // conn is a &mut PgConnection scoped to this transaction
///       // Return Ok(value) to commit, Err(e) to rollback
///       let product = products::table.find(product_id)
///           .select(Product::as_select())
///           .first(conn)?;    // ? propagates Err → rollback
///
///       // ... check stock, create order, update stock ...
///
///       Ok(order)             // Ok → commit
///   })`
///
/// The `?` operator is key: any `diesel::result::Error` returned via `?`
/// causes the closure to return `Err`, which triggers ROLLBACK. This is
/// why Diesel uses `QueryResult<T>` (which is `Result<T, diesel::result::Error>`)
/// throughout — it composes naturally with transactions.
///
/// For the "insufficient stock" case, return a custom error:
///   `Err(diesel::result::Error::RollbackTransaction)`
/// or use `diesel::result::Error::DatabaseError` with a custom message.
/// The simplest approach: `return Err(diesel::result::Error::RollbackTransaction);`
///
/// To calculate total_price, multiply product.price (BigDecimal) by quantity:
///   `let total = &product.price * BigDecimal::from(quantity);`
///
/// Hint: The closure receives `conn: &mut PgConnection` — use it for all
/// queries inside the transaction. Do NOT use the outer `conn`.
pub fn place_order_transaction(
    conn: &mut PgConnection,
    product_id: i32,
    quantity: i32,
    email: &str,
) -> QueryResult<Order> {
    todo!("TODO(human): Implement transactional order placement with stock validation")
}

// =============================================================================
// Demonstration helper (fully implemented)
// =============================================================================

/// Cancel an order and restore stock. Fully implemented as a reference example
/// showing the transaction pattern with multiple table updates.
pub fn cancel_order_transaction(
    conn: &mut PgConnection,
    order_id: i32,
) -> QueryResult<Order> {
    conn.transaction(|conn| {
        // 1. Find the order
        let order: Order = orders::table
            .find(order_id)
            .select(Order::as_select())
            .first(conn)?;

        // 2. Verify it's not already cancelled
        if order.status == "cancelled" {
            return Err(diesel::result::Error::RollbackTransaction);
        }

        // 3. Restore stock to the product
        diesel::update(products::table.find(order.product_id))
            .set(products::stock.eq(products::stock + order.quantity))
            .execute(conn)?;

        // 4. Mark the order as cancelled
        let cancelled_order = diesel::update(orders::table.find(order_id))
            .set(orders::status.eq("cancelled"))
            .get_result(conn)?;

        Ok(cancelled_order)
    })
}
