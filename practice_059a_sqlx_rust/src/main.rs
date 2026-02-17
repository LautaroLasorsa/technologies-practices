//! SQLx Practice — Compile-Time Checked SQL in Rust
//!
//! This program demonstrates SQLx's key features:
//! 1. Connection pooling with PgPool
//! 2. Compile-time checked queries (query!() and query_as!())
//! 3. Runtime queries for DDL
//! 4. Transactions with automatic rollback
//! 5. Embedded migrations
//!
//! Each phase runs sequentially, building on the previous one.

mod migrations_demo;
mod models;
mod queries_advanced;
mod queries_basic;
mod setup;
mod transactions;

use rust_decimal::Decimal;
use sqlx::postgres::PgPoolOptions;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load .env file (DATABASE_URL)
    dotenvy::dotenv().ok();

    let database_url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL must be set in .env or environment");

    // Create a connection pool with up to 5 connections.
    // connect_lazy() defers the actual TCP connection to the first query,
    // which is useful for fast startup. Use connect() if you want to fail
    // fast on bad credentials.
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    println!("Connected to PostgreSQL");
    println!("{}", "=".repeat(60));

    // ── Phase 1: Schema & Seeding ──────────────────────────────────
    run_phase_1(&pool).await?;

    // ── Phase 2: Basic CRUD ────────────────────────────────────────
    run_phase_2(&pool).await?;

    // ── Phase 3: Advanced Queries ──────────────────────────────────
    run_phase_3(&pool).await?;

    // ── Phase 4: Transactions ──────────────────────────────────────
    run_phase_4(&pool).await?;

    // ── Phase 5: Migrations ────────────────────────────────────────
    run_phase_5(&pool).await?;

    println!("\n{}", "=".repeat(60));
    println!("All phases completed successfully!");

    Ok(())
}

async fn run_phase_1(pool: &sqlx::PgPool) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Phase 1: Schema Creation & Seeding ---\n");

    setup::create_schema(pool).await?;
    println!("Schema created (products + orders tables)");

    setup::seed_data(pool).await?;
    println!("Seed data inserted (5 products)");

    Ok(())
}

async fn run_phase_2(pool: &sqlx::PgPool) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Phase 2: Basic CRUD with query!() ---\n");

    // Fetch all products
    let products = queries_basic::get_all_products(pool).await?;
    println!("All products ({} total):", products.len());
    for p in &products {
        println!(
            "  [{}] {} — ${} (stock: {}, category: {})",
            p.id,
            p.name,
            p.price,
            p.stock,
            p.category.as_deref().unwrap_or("none")
        );
    }

    // Fetch by id
    println!();
    if let Some(p) = queries_basic::get_product_by_id(pool, 1).await? {
        println!("Product #1: {} — ${}", p.name, p.price);
    }

    // Fetch non-existent product
    match queries_basic::get_product_by_id(pool, 999).await? {
        Some(_) => println!("Product #999: found (unexpected!)"),
        None => println!("Product #999: not found (expected)"),
    }

    // Create a new product
    println!();
    let new_product = queries_basic::create_product(
        pool,
        "Gadget Zeta",
        Decimal::new(4999, 2), // 49.99
        Some("electronics"),
        25,
    )
    .await?;
    println!(
        "Created: [{}] {} — ${} (created_at: {})",
        new_product.id, new_product.name, new_product.price, new_product.created_at
    );

    // Update stock
    let updated = queries_basic::update_stock(pool, new_product.id, 50).await?;
    println!(
        "Updated stock: [{}] {} — stock: {} → {}",
        updated.id, updated.name, 25, updated.stock
    );

    Ok(())
}

async fn run_phase_3(pool: &sqlx::PgPool) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Phase 3: Advanced Queries (JOINs & Aggregations) ---\n");

    // Products with order count
    let products_orders = queries_advanced::products_with_order_count(pool).await?;
    println!("Products with order counts:");
    for po in &products_orders {
        println!(
            "  [{}] {} — ${} (stock: {}, orders: {})",
            po.id,
            po.name,
            po.price,
            po.stock,
            po.order_count.unwrap_or(0)
        );
    }

    // Revenue by category
    println!();
    let revenue = queries_advanced::revenue_by_category(pool).await?;
    println!("Revenue by category:");
    for r in &revenue {
        println!(
            "  {} — revenue: ${}, orders: {}",
            r.category.as_deref().unwrap_or("(uncategorized)"),
            r.total_revenue
                .map(|d| d.to_string())
                .unwrap_or_else(|| "0.00".to_string()),
            r.order_count.unwrap_or(0)
        );
    }

    Ok(())
}

async fn run_phase_4(pool: &sqlx::PgPool) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Phase 4: Transactions ---\n");

    // Place a valid order
    let order = transactions::place_order(pool, 1, 2, "alice@example.com").await?;
    println!(
        "Order placed: [{}] product_id={}, qty={}, total=${}, status={}",
        order.id, order.product_id, order.quantity, order.total_price, order.status
    );

    // Place another order
    let order2 = transactions::place_order(pool, 2, 1, "bob@example.com").await?;
    println!(
        "Order placed: [{}] product_id={}, qty={}, total=${}, status={}",
        order2.id, order2.product_id, order2.quantity, order2.total_price, order2.status
    );

    // Try to order more than available stock (should fail)
    println!();
    match transactions::place_order(pool, 1, 99999, "greedy@example.com").await {
        Ok(_) => println!("Order placed (unexpected — should have failed!)"),
        Err(e) => println!("Order rejected (expected): {}", e),
    }

    // Bulk price update
    println!();
    let updated_count = transactions::bulk_price_update(pool, "electronics", 10.0).await?;
    println!(
        "Bulk price update: {} products in 'electronics' increased by 10%",
        updated_count
    );

    Ok(())
}

async fn run_phase_5(pool: &sqlx::PgPool) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Phase 5: Migrations ---\n");

    migrations_demo::run_migrations(pool).await?;

    // Verify the migration worked by checking if the 'tags' column exists
    let result = sqlx::query("SELECT column_name FROM information_schema.columns WHERE table_name = 'products' AND column_name = 'tags'")
        .fetch_optional(pool)
        .await?;

    match result {
        Some(_) => println!("Verified: 'tags' column exists in products table"),
        None => println!("Warning: 'tags' column not found — migration may not have run"),
    }

    Ok(())
}
