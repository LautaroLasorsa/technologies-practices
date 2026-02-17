use diesel::prelude::*;
use diesel::PgConnection;

use crate::models::{NewProduct, Order, Product};
use crate::schema::products;

// =============================================================================
// Basic CRUD operations using Diesel's type-safe query builder DSL.
//
// Each function demonstrates a fundamental Diesel pattern. The key insight:
// every function here is checked at COMPILE TIME — if you misspell a column
// name, use the wrong type, or reference a non-existent table, the compiler
// catches it. No database connection is needed for this validation.
// =============================================================================

/// TODO(human): Load all products from the database, ordered by creation date.
///
/// This is the simplest Diesel query. You have two options:
///
/// Option A (positional):
///   `products::table.load::<Product>(conn)`
///   This loads all columns and maps them to Product by position.
///   Fragile: if a column is added to the table, this breaks.
///
/// Option B (recommended, name-based):
///   `products::table.select(Product::as_select()).load(conn)`
///   Uses Selectable to construct a SELECT clause that matches struct fields
///   by name. Resilient to schema changes (new columns won't break this).
///
/// Both return `QueryResult<Vec<Product>>` — Diesel's alias for
/// `Result<Vec<Product>, diesel::result::Error>`.
///
/// Add `.order(products::created_at.desc())` to sort newest-first.
///
/// Hint: `products::table` is the entry point. Chain `.select()`, `.order()`,
/// then `.load(conn)`.
pub fn get_all_products(conn: &mut PgConnection) -> QueryResult<Vec<Product>> {
    todo!("TODO(human): Load all products ordered by created_at descending")
}

/// TODO(human): Find a single product by its primary key.
///
/// Diesel provides two idiomatic approaches:
///
/// Approach 1 — `.find()`:
///   `products::table.find(product_id).select(Product::as_select()).first(conn).optional()`
///   `.find(pk)` adds `WHERE id = pk` using the table's primary key.
///   `.first(conn)` returns the first row or an error.
///   `.optional()` converts "not found" errors into `Ok(None)` instead of
///   `Err(NotFound)`. Without `.optional()`, a missing row is an error.
///
/// Approach 2 — `.filter()`:
///   `products::table.filter(products::id.eq(product_id)).select(...).first(conn).optional()`
///   More verbose but shows the general `.filter()` pattern you'll use for
///   non-PK queries.
///
/// The return type is `QueryResult<Option<Product>>`:
///   - Ok(Some(product)) — found
///   - Ok(None) — no row with that ID
///   - Err(e) — database error (connection lost, etc.)
///
/// Hint: Use `.find()` for PK lookups, `.filter()` for everything else.
pub fn get_product_by_id(conn: &mut PgConnection, product_id: i32) -> QueryResult<Option<Product>> {
    todo!("TODO(human): Find product by ID, returning None if not found")
}

/// TODO(human): Insert a new product and return the created row.
///
/// Diesel's insert API:
///   `diesel::insert_into(products::table).values(new_product).get_result(conn)`
///
/// `insert_into(table)` starts the INSERT builder.
/// `.values(new_product)` adds the data — `new_product` implements `Insertable`
///   (from the `#[derive(Insertable)]` on NewProduct), so Diesel knows which
///   columns to fill and which to omit (id, created_at are DB-generated).
/// `.get_result::<Product>(conn)` executes INSERT ... RETURNING * and
///   deserializes the returned row into a Product. This is PostgreSQL-specific
///   (RETURNING clause) — SQLite/MySQL would need a separate SELECT.
///
/// For batch inserts, you can pass a `&[NewProduct]` to `.values()`.
///
/// Hint: The return type is `QueryResult<Product>` — the full row including
/// the DB-generated `id` and `created_at`.
pub fn create_product(conn: &mut PgConnection, new_product: &NewProduct) -> QueryResult<Product> {
    todo!("TODO(human): Insert new product and return the created row with RETURNING *")
}

/// TODO(human): Update a product's stock and return the updated row.
///
/// Diesel's update API:
///   `diesel::update(target).set(changes).get_result(conn)`
///
/// `target` identifies which rows to update. Common patterns:
///   - `products::table.find(id)` — single row by PK
///   - `products::table.filter(products::category.eq("Electronics"))` — multiple rows
///
/// `changes` specifies what to update:
///   - Single column: `products::stock.eq(new_stock)`
///   - Multiple columns: tuple `(products::stock.eq(42), products::name.eq("New Name"))`
///   - Struct with `#[derive(AsChangeset)]`: `&update_struct` (updates all non-None fields)
///
/// `.get_result::<Product>(conn)` returns the updated row (PostgreSQL RETURNING).
/// `.execute(conn)` returns the number of affected rows instead.
///
/// Hint: `diesel::update(products::table.find(product_id)).set(products::stock.eq(new_stock))`
pub fn update_product_stock(
    conn: &mut PgConnection,
    product_id: i32,
    new_stock: i32,
) -> QueryResult<Product> {
    todo!("TODO(human): Update the stock of a product by ID and return updated row")
}

/// TODO(human): Delete a product by ID and return the number of deleted rows.
///
/// Diesel's delete API:
///   `diesel::delete(target).execute(conn)`
///
/// `target` works the same as in `update` — `.find(id)` for PK,
/// `.filter(condition)` for bulk deletes.
///
/// `.execute(conn)` returns `QueryResult<usize>` — the number of rows deleted.
/// Typically 0 (not found) or 1 (deleted). If you need the deleted row back,
/// use `.get_result::<Product>(conn)` instead (PostgreSQL RETURNING).
///
/// IMPORTANT: Diesel does NOT cascade deletes by default — that's handled by
/// the database's foreign key ON DELETE clause. If orders reference this
/// product, the database will reject the delete (FK violation) unless the
/// FK has ON DELETE CASCADE.
///
/// Hint: `diesel::delete(products::table.find(product_id)).execute(conn)`
pub fn delete_product(conn: &mut PgConnection, product_id: i32) -> QueryResult<usize> {
    todo!("TODO(human): Delete a product by ID and return count of deleted rows")
}

// =============================================================================
// Order CRUD (fully implemented — reference for the Product TODOs above)
// =============================================================================

/// Create a new order. Fully implemented as a reference example.
pub fn create_order(conn: &mut PgConnection, new_order: &crate::models::NewOrder) -> QueryResult<Order> {
    use crate::schema::orders;

    diesel::insert_into(orders::table)
        .values(new_order)
        .get_result(conn)
}

/// Get all orders for a specific product. Fully implemented as reference.
pub fn get_orders_for_product(conn: &mut PgConnection, product_id: i32) -> QueryResult<Vec<Order>> {
    use crate::schema::orders;

    orders::table
        .filter(orders::product_id.eq(product_id))
        .select(Order::as_select())
        .order(orders::created_at.desc())
        .load(conn)
}
