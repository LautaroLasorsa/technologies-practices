use bigdecimal::BigDecimal;
use chrono::NaiveDateTime;
use diesel::prelude::*;
use serde::Serialize;

use crate::schema::{orders, products};

// =============================================================================
// Product models
// =============================================================================

/// Represents a row from the `products` table.
///
/// TODO(human): Study this struct and understand each Diesel derive.
///
/// `Queryable` generates an implementation that converts a database row into
/// this struct. CRITICAL: Queryable maps columns **by position** — the order
/// of fields in your struct MUST match the order of columns in `schema.rs`.
/// If you swap `name` and `price`, you'll get a compile error (type mismatch)
/// or — worse — silently wrong data if types happen to coincide.
///
/// `Selectable` adds a safer alternative: it generates an `as_select()` method
/// that constructs a SELECT clause matching fields **by name** (using
/// `#[diesel(table_name = products)]`). When you use
/// `products::table.select(Product::as_select()).load(conn)`, Diesel selects
/// exactly the columns your struct expects, in the right order, regardless of
/// how many columns the table has. This is the recommended pattern in Diesel
/// 2.x — it prevents breakage when new columns are added to the table.
///
/// `check_for_backend(diesel::pg::Pg)` adds a compile-time assertion that
/// this struct is compatible with PostgreSQL's type system. Without it, type
/// mismatches between Rust and PG types would produce confusing generic errors.
///
/// Field types must match the SQL types declared in `schema.rs`:
///   Int4      -> i32
///   Varchar   -> String
///   Numeric   -> BigDecimal
///   Nullable  -> Option<T>
///   Timestamp -> NaiveDateTime (via chrono feature)
#[derive(Queryable, Selectable, Debug, Serialize)]
#[diesel(table_name = products)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct Product {
    pub id: i32,
    pub name: String,
    pub price: BigDecimal,
    pub category: Option<String>,
    pub stock: i32,
    pub created_at: NaiveDateTime,
}

// TODO(human): Define the NewProduct struct for inserting new rows.
//
// `Insertable` generates the implementation for `diesel::insert_into().values()`.
// Unlike `Queryable`, Insertable uses `#[diesel(table_name = ...)]` to know
// which table to target. Fields that the database generates (like `id` via
// SERIAL and `created_at` via DEFAULT NOW()) should be **omitted** from this
// struct — Diesel will leave them out of the INSERT statement, letting
// PostgreSQL fill in the defaults.
//
// Using `&'a str` instead of `String` avoids an allocation when inserting
// from string literals or borrowed data. This is a common Diesel pattern
// for insert structs.
//
// Your task: Add the `Insertable` derive and the `#[diesel(table_name = products)]`
// attribute to the struct below. Without `Insertable`, the `create_product()`
// function in `queries_basic.rs` won't compile — Diesel needs the trait to
// know how to map struct fields to INSERT column values.
//
// The compiler error you'll see without it:
//   "the trait `Insertable<products::table>` is not implemented for `&NewProduct`"
//
// Compare with `NewOrder` below (fully implemented) to see the pattern.
//
// Required derives: Insertable, Debug
// Required attribute: #[diesel(table_name = products)]
#[derive(Debug)]
pub struct NewProduct<'a> {
    pub name: &'a str,
    pub price: BigDecimal,
    pub category: Option<&'a str>,
    pub stock: i32,
}

fn _new_product_todo() {
    todo!("TODO(human): Add #[derive(Insertable)] and #[diesel(table_name = products)] to NewProduct above")
}

// =============================================================================
// Order models (fully implemented — reference for the Product TODOs above)
// =============================================================================

/// Represents a row from the `orders` table.
///
/// Same pattern as Product: Queryable maps by column position,
/// Selectable enables safe `.select(Order::as_select())`.
#[derive(Queryable, Selectable, Debug, Serialize)]
#[diesel(table_name = orders)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct Order {
    pub id: i32,
    pub product_id: i32,
    pub quantity: i32,
    pub total_price: BigDecimal,
    pub status: String,
    pub customer_email: Option<String>,
    pub created_at: NaiveDateTime,
}

/// Insertable struct for creating new orders.
///
/// Omits `id` and `created_at` (database-generated).
/// Status defaults to 'pending' at the SQL level, but we include it here
/// to allow explicit status setting when needed.
#[derive(Insertable, Debug)]
#[diesel(table_name = orders)]
pub struct NewOrder<'a> {
    pub product_id: i32,
    pub quantity: i32,
    pub total_price: BigDecimal,
    pub status: &'a str,
    pub customer_email: Option<&'a str>,
}
