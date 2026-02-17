//! Migration system demonstration.
//!
//! SQLx provides two ways to run migrations:
//! 1. CLI: `sqlx migrate run` — applies pending `.sql` files from `migrations/`
//! 2. Embedded: `sqlx::migrate!("./migrations")` — compiles migrations INTO your binary
//!
//! The embedded approach is production-friendly: your binary carries its own migrations
//! and can apply them on startup, no external tooling needed.

use sqlx::PgPool;

/// Run all embedded migrations from the `migrations/` directory.
///
/// # Exercise Context
/// This exercise teaches SQLx's embedded migration system. The `sqlx::migrate!()`
/// macro reads all `.sql` files from the `migrations/` directory AT COMPILE TIME
/// and embeds them into your binary. At runtime, `.run(pool)` checks the
/// `_sqlx_migrations` table in PostgreSQL to determine which migrations have already
/// been applied, then runs only the pending ones — in order, each in its own transaction.
///
/// This pattern is common in production Rust services: on startup, the application
/// ensures its database schema is up to date before serving requests. No need for
/// external migration tooling or deployment scripts.
///
/// The `migrations/` directory contains a sample migration that adds a `tags` column
/// to the products table. After running this function, you'll see the new column in
/// the database. Each migration file is named `<timestamp>_<description>.sql` — the
/// timestamp ensures ordering.
///
/// TODO(human): Implement this function.
///
/// Steps:
///   1. Call `sqlx::migrate!("./migrations")` — this is a macro that reads migration
///      files at compile time and returns a `Migrator` instance
///   2. Chain `.run(pool).await?` to apply pending migrations
///   3. Print a message: `println!("Migrations applied successfully");`
///   4. Return Ok(())
///
/// Note: The path `"./migrations"` is relative to the Cargo.toml file (project root),
/// not relative to the source file. The macro resolves it at compile time.
pub async fn run_migrations(pool: &PgPool) -> Result<(), sqlx::Error> {
    todo!("TODO(human): implement run_migrations — use sqlx::migrate!() macro")
}
