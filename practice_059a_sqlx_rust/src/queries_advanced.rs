//! Advanced query patterns: JOINs, aggregations, and custom result types.
//!
//! This module goes beyond single-table CRUD to demonstrate how SQLx handles
//! multi-table queries. When your SELECT doesn't map 1:1 to a table, you define
//! custom result structs whose fields match the query's output columns by name.

use sqlx::PgPool;

use crate::models::{CategoryRevenue, ProductWithOrders};

/// Fetch all products with their order count (LEFT JOIN + GROUP BY).
///
/// # Exercise Context
/// This exercise teaches how to handle JOIN queries in SQLx. When your SQL produces
/// columns from multiple tables (or computed columns like COUNT), you need a custom
/// struct whose field names match the SELECT column aliases. The `query_as!()` macro
/// checks this at compile time — if you SELECT `count(*) as order_count` but your
/// struct has `num_orders`, compilation fails.
///
/// The LEFT JOIN ensures products with zero orders still appear (with `order_count = NULL`).
/// This is why `ProductWithOrders.order_count` is `Option<i64>` — COUNT returns i64 in
/// PostgreSQL, and the LEFT JOIN can produce NULL for the count column. The `!` suffix
/// override (e.g., `"order_count!"`) could force it non-null if you know COUNT never
/// returns NULL, but leaving it as Option is safer for LEFT JOINs.
///
/// TODO(human): Implement this function.
///
/// Steps:
///   1. Write a SQL query that:
///      - SELECTs p.id, p.name, p.price, p.stock from products (aliased as p)
///      - LEFT JOINs orders (aliased as o) ON p.id = o.product_id
///      - GROUP BYs p.id, p.name, p.price, p.stock
///      - SELECTs COUNT(o.id) as order_count
///      - ORDER BYs p.id
///   2. Use `sqlx::query_as!(ProductWithOrders, "your SQL here")`
///   3. Chain `.fetch_all(pool).await?`
///   4. Return the Vec<ProductWithOrders>
///
/// Key detail: You must SELECT COUNT(o.id), not COUNT(*). COUNT(*) counts all rows
/// including those where the join produced NULLs, so it returns 1 for products with
/// no orders. COUNT(o.id) correctly returns 0 for unmatched LEFT JOIN rows.
pub async fn products_with_order_count(
    pool: &PgPool,
) -> Result<Vec<ProductWithOrders>, sqlx::Error> {
    todo!("TODO(human): implement products_with_order_count — LEFT JOIN with GROUP BY and COUNT")
}

/// Calculate total revenue and order count per product category.
///
/// # Exercise Context
/// This exercise teaches aggregate queries with nullable results. When you GROUP BY
/// a nullable column (category), NULL categories form their own group. The SUM of
/// total_price is also nullable — a category with no orders has NULL revenue, not 0.
/// Your `CategoryRevenue` struct reflects this: both `category` and `total_revenue`
/// are `Option<T>`.
///
/// This is a common analytics pattern: "show me revenue broken down by category."
/// In production, you'd add date ranges, filtering, and pagination. Here we focus
/// on the SQLx typing: how PostgreSQL's aggregate nullability maps to Rust's Option.
///
/// TODO(human): Implement this function.
///
/// Steps:
///   1. Write a SQL query that:
///      - SELECTs p.category from products (aliased as p)
///      - LEFT JOINs orders (aliased as o) ON p.id = o.product_id
///      - GROUP BYs p.category
///      - SELECTs SUM(o.total_price) as total_revenue
///      - SELECTs COUNT(o.id) as order_count
///      - ORDER BYs total_revenue DESC NULLS LAST
///   2. Use `sqlx::query_as!(CategoryRevenue, "your SQL here")`
///   3. Chain `.fetch_all(pool).await?`
///   4. Return the Vec<CategoryRevenue>
///
/// Note: `ORDER BY total_revenue DESC NULLS LAST` puts categories with no revenue
/// at the bottom. Without NULLS LAST, PostgreSQL puts NULLs first in DESC order.
pub async fn revenue_by_category(pool: &PgPool) -> Result<Vec<CategoryRevenue>, sqlx::Error> {
    todo!("TODO(human): implement revenue_by_category — aggregate SUM/COUNT with GROUP BY")
}
