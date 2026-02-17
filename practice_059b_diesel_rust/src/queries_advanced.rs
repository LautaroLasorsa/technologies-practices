use diesel::prelude::*;
use diesel::PgConnection;

use crate::models::{Order, Product};
use crate::schema::{orders, products};

// =============================================================================
// Advanced queries: joins, aggregations, and the belonging_to pattern.
//
// These demonstrate Diesel's composable query builder at its best.
// Each query is validated at compile time — the compiler checks that joined
// tables have a declared relationship (via joinable!), that grouped columns
// are valid, and that aggregation functions return correct types.
// =============================================================================

/// TODO(human): Load all products together with their orders using an INNER JOIN.
///
/// Diesel supports joins through the `.inner_join()` and `.left_join()` methods.
/// For this to work, you must have declared the relationship in schema.rs:
///   `diesel::joinable!(orders -> products (product_id));`
///   `diesel::allow_tables_to_appear_in_same_query!(products, orders);`
///
/// The query pattern:
///   `products::table
///       .inner_join(orders::table)
///       .select((Product::as_select(), Order::as_select()))
///       .load::<(Product, Order)>(conn)`
///
/// This returns `Vec<(Product, Order)>` — one tuple per joined row. A product
/// with 3 orders appears 3 times (once per order). Products with NO orders
/// are excluded (INNER JOIN). Use `.left_join()` if you want all products
/// regardless of orders.
///
/// To group orders by product (Vec<(Product, Vec<Order>)>), you need to
/// post-process the flat results. One approach:
///   1. Load the flat join results
///   2. Collect unique products
///   3. Group orders by product_id
///
/// Alternatively, load products first, then use belonging_to:
///   `let products = products::table.load::<Product>(conn)?;`
///   `let orders = Order::belonging_to(&products).load::<Order>(conn)?;`
///   `let grouped = orders.grouped_by(&products);`
///   (Requires #[derive(Associations)] on Order with #[diesel(belongs_to(Product))])
///
/// For this exercise, the simpler flat join is fine. Return the flat
/// Vec<(Product, Order)> and let the caller handle grouping if needed.
///
/// Hint: Start with `products::table.inner_join(orders::table)`.
pub fn products_with_orders(conn: &mut PgConnection) -> QueryResult<Vec<(Product, Order)>> {
    todo!("TODO(human): Join products with orders and return flat (Product, Order) tuples")
}

/// TODO(human): Count products per category using GROUP BY.
///
/// SQL equivalent:
///   SELECT category, COUNT(*) FROM products GROUP BY category;
///
/// Diesel's aggregation API:
///   `products::table
///       .group_by(products::category)
///       .select((products::category, diesel::dsl::count_star()))
///       .load::<(Option<String>, i64)>(conn)`
///
/// Key details:
/// - `group_by(column)` must come BEFORE `.select(...)` in the chain
/// - `diesel::dsl::count_star()` generates `COUNT(*)` — returns i64
/// - `diesel::dsl::count(column)` generates `COUNT(column)` — excludes NULLs
/// - The select tuple must only contain grouped columns or aggregates
///   (Diesel enforces this at compile time!)
/// - `category` is `Nullable<Varchar>` in schema.rs, so it maps to
///   `Option<String>` in Rust — NULL categories appear as `None`
///
/// This is one of Diesel's most impressive compile-time checks: if you try
/// to SELECT a non-grouped, non-aggregated column, you get a compile error.
/// In raw SQL, this is a runtime error (or silently wrong in MySQL).
///
/// Hint: Use `diesel::dsl::count_star()` for COUNT(*).
pub fn products_by_category_count(conn: &mut PgConnection) -> QueryResult<Vec<(Option<String>, i64)>> {
    todo!("TODO(human): GROUP BY category and count products per category")
}

// =============================================================================
// Filtering helpers (fully implemented — composability examples)
// =============================================================================

/// Find products with stock below a threshold. Demonstrates `.filter()` + `.lt()`.
pub fn low_stock_products(conn: &mut PgConnection, threshold: i32) -> QueryResult<Vec<Product>> {
    products::table
        .filter(products::stock.lt(threshold))
        .select(Product::as_select())
        .order(products::stock.asc())
        .load(conn)
}

/// Search products by name (case-insensitive LIKE). Demonstrates `.ilike()`.
pub fn search_products_by_name(conn: &mut PgConnection, search: &str) -> QueryResult<Vec<Product>> {
    let pattern = format!("%{}%", search);
    products::table
        .filter(products::name.ilike(pattern))
        .select(Product::as_select())
        .load(conn)
}

/// Get the top N most expensive products. Demonstrates `.order()` + `.limit()`.
pub fn top_expensive_products(conn: &mut PgConnection, limit: i64) -> QueryResult<Vec<Product>> {
    products::table
        .select(Product::as_select())
        .order(products::price.desc())
        .limit(limit)
        .load(conn)
}
