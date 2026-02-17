pub mod models;
pub mod queries_advanced;
pub mod queries_basic;
pub mod schema;
pub mod transactions;

use bigdecimal::BigDecimal;
use diesel::prelude::*;
use diesel::PgConnection;
use dotenvy::dotenv;
use std::env;
use std::str::FromStr;

use models::NewProduct;

/// Establish a connection to the PostgreSQL database using DATABASE_URL
/// from the .env file. Diesel's PgConnection is a synchronous, single-
/// threaded connection — no connection pooling. For production, you'd
/// wrap this with r2d2 (diesel's recommended connection pool).
fn establish_connection() -> PgConnection {
    dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set in .env");
    PgConnection::establish(&database_url)
        .unwrap_or_else(|_| panic!("Error connecting to {}", database_url))
}

fn main() {
    println!("=== Diesel ORM Practice ===\n");

    let conn = &mut establish_connection();
    println!("[OK] Connected to PostgreSQL\n");

    // Clean slate: delete all orders first (FK constraint), then products
    clean_tables(conn);

    // Phase 3: Basic CRUD
    demo_basic_crud(conn);

    // Phase 4: Advanced queries
    demo_advanced_queries(conn);

    // Phase 5: Transactions
    demo_transactions(conn);

    println!("\n=== All demos complete ===");
}

/// Remove all rows to start fresh. Orders must be deleted before products
/// due to the foreign key constraint.
fn clean_tables(conn: &mut PgConnection) {
    use schema::{orders, products};

    diesel::delete(orders::table)
        .execute(conn)
        .expect("Failed to clear orders");
    diesel::delete(products::table)
        .execute(conn)
        .expect("Failed to clear products");
    println!("[OK] Cleared existing data\n");
}

/// Demonstrates basic CRUD operations from queries_basic.rs.
fn demo_basic_crud(conn: &mut PgConnection) {
    println!("--- Phase 3: Basic CRUD ---\n");

    // Create products
    let products_data = vec![
        ("Mechanical Keyboard", "129.99", Some("Electronics"), 50),
        ("USB-C Hub", "49.99", Some("Electronics"), 120),
        ("Rust Programming Book", "39.95", Some("Books"), 200),
        ("Standing Desk", "599.00", Some("Furniture"), 15),
        ("Monitor Arm", "89.50", None, 75),
    ];

    println!("Creating {} products...", products_data.len());
    for (name, price, category, stock) in &products_data {
        let new_product = NewProduct {
            name,
            price: BigDecimal::from_str(price).unwrap(),
            category: *category,
            stock: *stock,
        };
        match queries_basic::create_product(conn, &new_product) {
            Ok(p) => println!("  Created: {} (id={}, price={}, stock={})", p.name, p.id, p.price, p.stock),
            Err(e) => println!("  Error creating {}: {:?}", name, e),
        }
    }

    // Read all products
    println!("\nAll products (sorted by created_at desc):");
    match queries_basic::get_all_products(conn) {
        Ok(products) => {
            for p in &products {
                println!(
                    "  [{}] {} - ${} | category: {} | stock: {}",
                    p.id,
                    p.name,
                    p.price,
                    p.category.as_deref().unwrap_or("(none)"),
                    p.stock
                );
            }
        }
        Err(e) => println!("  Error loading products: {:?}", e),
    }

    // Find by ID
    println!("\nLooking up product with id=1:");
    match queries_basic::get_product_by_id(conn, 1) {
        Ok(Some(p)) => println!("  Found: {} (price={})", p.name, p.price),
        Ok(None) => println!("  Not found (id=1)"),
        Err(e) => println!("  Error: {:?}", e),
    }

    // Update stock
    println!("\nUpdating stock for product id=1 to 42:");
    match queries_basic::update_product_stock(conn, 1, 42) {
        Ok(p) => println!("  Updated: {} now has stock={}", p.name, p.stock),
        Err(e) => println!("  Error: {:?}", e),
    }

    // Delete (the one with no category)
    println!("\nDeleting product with id=5 (Monitor Arm):");
    match queries_basic::delete_product(conn, 5) {
        Ok(count) => println!("  Deleted {} row(s)", count),
        Err(e) => println!("  Error: {:?}", e),
    }

    // Verify deletion
    println!("\nProducts after deletion:");
    if let Ok(products) = queries_basic::get_all_products(conn) {
        for p in &products {
            println!("  [{}] {}", p.id, p.name);
        }
        println!("  Total: {} products", products.len());
    }

    println!();
}

/// Demonstrates advanced queries from queries_advanced.rs.
fn demo_advanced_queries(conn: &mut PgConnection) {
    println!("--- Phase 4: Advanced Queries ---\n");

    // First, create some orders to join with
    seed_orders(conn);

    // Join: products with orders
    println!("Products with their orders (INNER JOIN):");
    match queries_advanced::products_with_orders(conn) {
        Ok(results) => {
            for (product, order) in &results {
                println!(
                    "  {} -> Order #{}: qty={}, total=${}, status={}",
                    product.name, order.id, order.quantity, order.total_price, order.status
                );
            }
            if results.is_empty() {
                println!("  (no results — products have no orders)");
            }
        }
        Err(e) => println!("  Error: {:?}", e),
    }

    // Group by category
    println!("\nProducts per category (GROUP BY):");
    match queries_advanced::products_by_category_count(conn) {
        Ok(results) => {
            for (category, count) in &results {
                println!(
                    "  {}: {} product(s)",
                    category.as_deref().unwrap_or("(uncategorized)"),
                    count
                );
            }
        }
        Err(e) => println!("  Error: {:?}", e),
    }

    // Bonus: filter helpers
    println!("\nLow stock products (threshold=50):");
    if let Ok(products) = queries_advanced::low_stock_products(conn, 50) {
        for p in &products {
            println!("  {} — stock: {}", p.name, p.stock);
        }
    }

    println!("\nSearch products by name ('Rust'):");
    if let Ok(products) = queries_advanced::search_products_by_name(conn, "Rust") {
        for p in &products {
            println!("  {} — ${}", p.name, p.price);
        }
    }

    println!("\nTop 2 most expensive products:");
    if let Ok(products) = queries_advanced::top_expensive_products(conn, 2) {
        for p in &products {
            println!("  {} — ${}", p.name, p.price);
        }
    }

    println!();
}

/// Demonstrates transactions from transactions.rs.
fn demo_transactions(conn: &mut PgConnection) {
    println!("--- Phase 5: Transactions ---\n");

    // Get a product to order
    let product = queries_basic::get_all_products(conn)
        .expect("Failed to load products")
        .into_iter()
        .next()
        .expect("No products available");

    println!(
        "Ordering 3 units of '{}' (current stock: {})...",
        product.name, product.stock
    );

    // Place an order (should succeed)
    match transactions::place_order_transaction(conn, product.id, 3, "buyer@example.com") {
        Ok(order) => {
            println!(
                "  Order placed! order_id={}, total=${}, status={}",
                order.id, order.total_price, order.status
            );
        }
        Err(e) => println!("  Error: {:?}", e),
    }

    // Verify stock was decremented
    if let Ok(Some(p)) = queries_basic::get_product_by_id(conn, product.id) {
        println!("  Stock after order: {}", p.stock);
    }

    // Try ordering more than available stock (should fail and rollback)
    println!("\nOrdering 9999 units (should fail — insufficient stock)...");
    match transactions::place_order_transaction(conn, product.id, 9999, "greedy@example.com") {
        Ok(_) => println!("  Unexpected success!"),
        Err(e) => println!("  Expected error: {:?}", e),
    }

    // Verify stock was NOT changed (rollback worked)
    if let Ok(Some(p)) = queries_basic::get_product_by_id(conn, product.id) {
        println!("  Stock after failed order (should be unchanged): {}", p.stock);
    }

    // Cancel order demo
    println!("\nCancelling the order...");
    let orders = queries_basic::get_orders_for_product(conn, product.id)
        .expect("Failed to load orders");
    if let Some(order) = orders.first() {
        match transactions::cancel_order_transaction(conn, order.id) {
            Ok(cancelled) => {
                println!("  Order {} cancelled (status={})", cancelled.id, cancelled.status);
            }
            Err(e) => println!("  Error: {:?}", e),
        }
        if let Ok(Some(p)) = queries_basic::get_product_by_id(conn, product.id) {
            println!("  Stock after cancellation (restored): {}", p.stock);
        }
    }
}

/// Seed some orders for the join demo. Uses the fully-implemented create_order.
fn seed_orders(conn: &mut PgConnection) {
    use crate::models::NewOrder;

    let products = queries_basic::get_all_products(conn).expect("Failed to load products");
    if products.is_empty() {
        println!("  (no products to create orders for)");
        return;
    }

    let order_data: Vec<(usize, i32, &str, Option<&str>)> = vec![
        (0, 2, "shipped", Some("alice@example.com")),
        (0, 1, "pending", Some("bob@example.com")),
        (1, 5, "delivered", Some("carol@example.com")),
    ];

    for (product_idx, qty, status, email) in &order_data {
        if let Some(product) = products.get(*product_idx) {
            let total = &product.price * BigDecimal::from(*qty);
            let new_order = NewOrder {
                product_id: product.id,
                quantity: *qty,
                total_price: total,
                status,
                customer_email: *email,
            };
            queries_basic::create_order(conn, &new_order).ok();
        }
    }
    println!("Seeded {} orders for join demos.\n", order_data.len());
}
