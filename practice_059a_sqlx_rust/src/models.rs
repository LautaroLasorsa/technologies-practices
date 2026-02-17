//! Domain models for the product inventory system.
//!
//! These structs map directly to PostgreSQL table rows. The derives enable:
//! - `sqlx::FromRow`: automatic row-to-struct mapping for runtime queries
//! - `serde::Serialize`: JSON serialization for display/debugging
//! - `Debug`: println! formatting

use chrono::NaiveDateTime;
use rust_decimal::Decimal;

/// A product in the inventory.
///
/// Maps to the `products` table. All fields correspond to columns:
/// - `id` is auto-generated (SERIAL PRIMARY KEY)
/// - `created_at` defaults to NOW() in PostgreSQL
/// - `category` is nullable (not all products have categories)
#[derive(Debug, sqlx::FromRow, serde::Serialize)]
pub struct Product {
    pub id: i32,
    pub name: String,
    pub price: Decimal,
    pub category: Option<String>,
    pub stock: i32,
    pub created_at: NaiveDateTime,
}

/// An order placed against a product.
///
/// Maps to the `orders` table. The `product_id` field is a foreign key
/// referencing `products(id)`. The `status` field uses a simple VARCHAR
/// (not an enum) for flexibility — common values: "pending", "completed", "cancelled".
#[derive(Debug, sqlx::FromRow, serde::Serialize)]
pub struct Order {
    pub id: i32,
    pub product_id: i32,
    pub quantity: i32,
    pub total_price: Decimal,
    pub status: String,
    pub customer_email: Option<String>,
    pub created_at: NaiveDateTime,
}

/// Result struct for the products-with-order-count query.
///
/// This struct doesn't map to a single table — it represents a JOIN + aggregation.
/// The field names must match the column aliases in the SQL query exactly.
/// Used with `query_as!()` in advanced queries.
#[derive(Debug, sqlx::FromRow, serde::Serialize)]
pub struct ProductWithOrders {
    pub id: i32,
    pub name: String,
    pub price: Decimal,
    pub stock: i32,
    pub order_count: Option<i64>, // COUNT returns i64, nullable because of LEFT JOIN
}

/// Result struct for the revenue-by-category aggregation query.
///
/// `category` is Option because products may have NULL category.
/// `total_revenue` is Option because SUM over zero rows returns NULL.
#[derive(Debug, sqlx::FromRow, serde::Serialize)]
pub struct CategoryRevenue {
    pub category: Option<String>,
    pub total_revenue: Option<Decimal>,
    pub order_count: Option<i64>,
}
