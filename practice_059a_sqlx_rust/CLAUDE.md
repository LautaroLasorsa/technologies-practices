# Practice 059a: SQLx — Compile-Time Checked SQL in Rust

## Technologies

- **SQLx** — Async, pure Rust SQL toolkit with compile-time query verification
- **tokio** — Async runtime for Rust
- **PostgreSQL 16** — Relational database (Docker container)
- **sqlx-cli** — Command-line tool for database management and migrations
- **dotenvy** — Load `.env` files into environment variables
- **chrono** — Date/time types with serde and SQLx integration
- **rust_decimal** — Arbitrary-precision decimals with PostgreSQL `NUMERIC` mapping

## Stack

- Rust (cargo, edition 2021)
- Docker (PostgreSQL 16)

## Theoretical Context

### What is SQLx and What Problem Does It Solve?

[SQLx](https://github.com/launchbadge/sqlx) is a pure Rust, async-first SQL toolkit that provides **compile-time verified queries without a DSL**. You write raw SQL strings, but the Rust compiler verifies them against your actual database schema before your code can even run. This eliminates an entire class of bugs — misspelled column names, wrong parameter types, schema mismatches — that traditionally only surface at runtime (or worse, in production).

The problem SQLx solves is the **SQL correctness gap** in application development. Most languages offer two extremes: (1) raw SQL strings that are unchecked until runtime (Python's `psycopg2`, Go's `database/sql`), or (2) heavy ORMs that abstract SQL away entirely (Django ORM, Hibernate, SQLAlchemy). SQLx occupies a unique middle ground — raw SQL with compile-time type safety. You keep full control over your queries (no ORM-generated SQL surprises) while the compiler guarantees they're valid.

With [54.9M+ downloads on crates.io](https://crates.io/crates/sqlx), SQLx is the most popular Rust database library, used in production at companies building web services, data pipelines, and infrastructure tooling.

### How the query!() Macro Works

The `query!()` macro is SQLx's core innovation. Here's what happens at compile time:

1. **The macro reads `DATABASE_URL`** from the environment (or `.env` file)
2. **It connects to your development database** during `cargo build`
3. **It sends your SQL string to PostgreSQL** for parsing and validation
4. **PostgreSQL returns metadata**: column names, column types, nullability, parameter types
5. **The macro generates typed Rust code**: an anonymous struct with fields matching the query's output columns, and type-checked bind parameters

This means if you write `query!("SELECT naem FROM products")` and the column is actually `name`, **compilation fails** with a clear error. If you bind an `i32` where the column expects `TEXT`, compilation fails. If you add a `WHERE` clause referencing a non-existent column, compilation fails.

The generated anonymous struct has fields matching each output column. For `SELECT id, name, price FROM products`, the result has `.id: i32`, `.name: String`, `.price: Decimal` — all inferred from the database schema.

### query!() vs query_as!() vs query()

| Macro/Function | Compile-Time Check | Return Type | Use Case |
|---|---|---|---|
| `query!("SQL", ...)` | Yes | Anonymous struct (can't be named in signatures) | Quick queries where you consume results immediately |
| `query_as!(Type, "SQL", ...)` | Yes | Named struct `Type` | Returning results from functions, adding trait impls |
| `query("SQL")` | No (runtime only) | Dynamic rows | Schema not known at compile time (e.g., DDL, dynamic queries) |

`query_as!()` does **not** use `FromRow` — it directly maps columns to struct fields by name at compile time. The struct doesn't need any derives for `query_as!()` to work. `FromRow` is only needed for runtime `query().fetch()` with `.try_from_row()`.

### FromRow Derive

The `#[derive(sqlx::FromRow)]` attribute generates an implementation that maps database rows to Rust structs at runtime. It's used with the non-macro `query()` function (runtime queries) or with `query_as()` (runtime variant). Column names must match field names. Optional fields must be `Option<T>`. You can rename columns with `#[sqlx(rename = "col_name")]`.

### Connection Pool (PgPool)

SQLx uses connection pooling via [`PgPool`](https://docs.rs/sqlx/latest/sqlx/type.PgPool.html). The pool manages a set of database connections, reusing them across async tasks. Key configuration:

- `PgPoolOptions::new().max_connections(5)` — limit concurrent connections
- `PgPool::connect(&url)` — create pool and connect immediately
- `PgPool::connect_lazy(&url)` — create pool, defer actual connection to first use
- The pool is `Clone + Send + Sync` — share it freely across async tasks

### Prepared Statements

All `query!()` calls are automatically prepared statements. PostgreSQL parses the SQL once, then reuses the execution plan for subsequent calls with different parameters. This provides both performance (no re-parsing) and security (SQL injection is impossible because parameters are sent separately from the SQL).

### Transactions

Transactions in SQLx follow Rust's ownership model:

```rust
let mut tx = pool.begin().await?;           // BEGIN
sqlx::query!("INSERT ...").execute(&mut *tx).await?;  // use &mut *tx
tx.commit().await?;                          // COMMIT
// If tx is dropped without commit() → automatic ROLLBACK
```

The `&mut *tx` dereference is required because `Transaction` wraps an inner connection; you dereference to access it. If an error causes early return (via `?`), the transaction is dropped and automatically rolled back — Rust's RAII guarantees cleanup.

### Migrations

SQLx provides a migration system via `sqlx-cli`:

- `sqlx migrate add <name>` — creates `migrations/<timestamp>_<name>.sql`
- `sqlx migrate run` — applies pending migrations to the database
- `sqlx::migrate!("./migrations")` — embeds migrations in the binary, runs them programmatically

Migrations are tracked in a `_sqlx_migrations` table. Each migration runs in a transaction by default.

### Offline Mode (sqlx prepare)

For CI/CD without database access:

1. Run `cargo sqlx prepare` with the database running — caches query metadata in `.sqlx/` directory
2. Check `.sqlx/` into version control
3. Set `SQLX_OFFLINE=true` — the macros use cached metadata instead of connecting to the database

This is essential for production builds where the build server shouldn't have database access.

### Compile-Time vs Runtime Trade-offs

| Aspect | Compile-time (`query!()`) | Runtime (`query()`) |
|---|---|---|
| **Safety** | Schema verified at build time | Errors only at runtime |
| **Build requirement** | Needs `DATABASE_URL` or `.sqlx/` cache | No database needed |
| **Flexibility** | SQL must be string literals | Dynamic SQL, user-generated queries |
| **Refactoring** | Column rename → compile error everywhere | Column rename → runtime crash somewhere |
| **CI/CD** | Requires `cargo sqlx prepare` step | Works anywhere |

### SQLx vs Diesel

[Diesel](https://diesel.rs/) is the other major Rust database library. Key differences:

| Aspect | SQLx | Diesel |
|---|---|---|
| **Query style** | Raw SQL strings | Rust DSL (type-safe query builder) |
| **Async** | Native async (tokio/async-std) | Sync only (async via `deadpool-diesel` wrapper) |
| **Compile-time safety** | Checks against live DB | Checks against generated schema.rs |
| **SQL control** | Full control — you write the SQL | DSL may generate unexpected SQL |
| **Learning curve** | Know SQL → know SQLx | Must learn Diesel's DSL |
| **Complex queries** | Just write the SQL | DSL can get verbose for JOINs/subqueries |

SQLx is preferred when you want raw SQL control with type safety. Diesel is preferred when you want a fully type-safe query builder that doesn't need a database at compile time.

### Nullability Handling in query!()

The `query!()` macro infers nullability from PostgreSQL's `NOT NULL` constraints:

- `NOT NULL` column → `T` (e.g., `String`, `i32`)
- Nullable column → `Option<T>`
- Expressions (e.g., `COUNT(*)`) are assumed nullable unless overridden

Override with special column aliases: `"column_name!"` forces non-null, `"column_name?"` forces nullable.

### Fetch Methods

| Expected Rows | Method | Returns |
|---|---|---|
| None (INSERT/UPDATE/DELETE) | `.execute(pool)` | `PgQueryResult` (rows affected) |
| Zero or one | `.fetch_optional(pool)` | `Option<T>` |
| Exactly one | `.fetch_one(pool)` | `T` (error if 0 or 2+) |
| Multiple | `.fetch_all(pool)` | `Vec<T>` |
| Streaming | `.fetch(pool)` | `impl Stream<Item = Result<T>>` |

## Description

Build a **product inventory system** backed by PostgreSQL, exploring SQLx's full feature set: compile-time checked queries, typed result mapping, transactions with rollback, and embedded migrations. The focus is on understanding how SQLx bridges raw SQL and Rust's type system.

### What you'll learn

1. **Connection pooling** — creating and configuring PgPool
2. **query!() macro** — compile-time SQL verification returning anonymous records
3. **query_as!() macro** — mapping query results to named structs
4. **Runtime queries** — using query() for DDL and dynamic SQL
5. **Transactions** — begin/commit/rollback with Rust's ownership guarantees
6. **Migrations** — sqlx-cli and embedded migration system
7. **FromRow derive** — automatic row-to-struct mapping

## Instructions

### Phase 1: Setup & Infrastructure (~15 min)

1. Start PostgreSQL via Docker Compose: `docker compose up -d`
2. Install sqlx-cli: `cargo install sqlx-cli --no-default-features --features native-tls,postgres`
3. Verify database connection: `sqlx database create` (should succeed — DB already exists from Docker)
4. Review `Cargo.toml` dependencies and `.env` file
5. Key question: Why does `query!()` need `DATABASE_URL` at compile time, not just at runtime?

### Phase 2: Schema & Seeding (~20 min)

1. Open `src/setup.rs` — two functions to implement
2. **User implements:** `create_schema()` — Execute DDL to create `products` and `orders` tables using `sqlx::query()` (runtime, not macro). This teaches the distinction: DDL can't use `query!()` because the tables don't exist yet when the macro runs at compile time.
   - `products`: id (SERIAL PRIMARY KEY), name (VARCHAR NOT NULL UNIQUE), price (NUMERIC(10,2) NOT NULL), category (VARCHAR), stock (INT NOT NULL DEFAULT 0), created_at (TIMESTAMP NOT NULL DEFAULT NOW())
   - `orders`: id (SERIAL PRIMARY KEY), product_id (INT NOT NULL REFERENCES products(id)), quantity (INT NOT NULL), total_price (NUMERIC(10,2) NOT NULL), status (VARCHAR NOT NULL DEFAULT 'pending'), customer_email (VARCHAR), created_at (TIMESTAMP NOT NULL DEFAULT NOW())
3. **User implements:** `seed_data()` — Insert 5 sample products using `sqlx::query!()` macro. This is your first experience with the compile-time macro: the compiler connects to PostgreSQL and verifies your INSERT statement, column names, and parameter types.
4. Run the program: `cargo run` — watch the compile-time verification happen
5. Key question: What error do you get if you misspell a column name in `query!()`?

### Phase 3: Basic CRUD with query!() (~20 min)

1. Open `src/queries_basic.rs` — four functions to implement
2. **User implements:** `get_all_products()` — Use `query_as!(Product, "SELECT * ...")` to fetch all products. This demonstrates mapping query results to a named struct at compile time.
3. **User implements:** `get_product_by_id()` — Use `query_as!` with `WHERE id = $1` bind parameter. PostgreSQL-style positional parameters ($1, $2) replace values safely (no SQL injection).
4. **User implements:** `create_product()` — INSERT with RETURNING * using `query_as!`. The RETURNING clause makes PostgreSQL send back the inserted row, so you get the auto-generated `id` and `created_at`.
5. **User implements:** `update_stock()` — UPDATE with RETURNING. Practice modifying data and getting the updated row back in one round-trip.
6. Run and verify: `cargo run`
7. Key question: What's the difference between `query!()` and `query_as!()`? When would you use each?

### Phase 4: Advanced Queries & Joins (~15 min)

1. Open `src/queries_advanced.rs` — two functions with custom result structs
2. **User implements:** `products_with_order_count()` — LEFT JOIN products with orders, GROUP BY, COUNT. This teaches handling joined/aggregated results with a custom struct. The struct fields must match the SELECT column names exactly.
3. **User implements:** `revenue_by_category()` — Aggregate SUM(total_price) GROUP BY category. Demonstrates nullable aggregation results (categories with no orders).
4. Run and verify: `cargo run`
5. Key question: Why must the struct field names match the SQL column aliases exactly in `query_as!()`?

### Phase 5: Transactions & Migrations (~20 min)

1. Open `src/transactions.rs` — two functions to implement
2. **User implements:** `place_order()` — A multi-step transaction: check stock, create order, decrement stock. If stock is insufficient, return an error (the transaction auto-rollbacks on drop). This demonstrates SQLx's RAII-based transaction safety.
3. **User implements:** `bulk_price_update()` — Transaction that updates all products in a category. Return the count of affected rows.
4. Open `src/migrations_demo.rs` — one function to implement
5. **User implements:** `run_migrations()` — Use `sqlx::migrate!()` to run embedded migrations. Review the sample migration file in `migrations/`.
6. Run and verify: `cargo run`
7. Key question: What happens if you call `?` (early return) inside a transaction block without calling `commit()`?

## Motivation

- **Most popular Rust DB library**: 54.9M+ downloads, active development, production-proven
- **Unique compile-time safety**: No other language has this level of SQL verification at build time
- **Raw SQL + type safety**: Best of both worlds — full SQL control without ORM abstractions
- **Async-first**: Native tokio integration, essential for modern Rust web services
- **Complements current profile**: Adds Rust database skills alongside Python SQLModel/SQLAlchemy experience

## Commands

### Prerequisites

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start PostgreSQL 16 container on localhost:5432 |
| `docker compose down` | Stop and remove the PostgreSQL container |
| `docker compose down -v` | Stop container AND delete persistent volume (fresh database) |
| `cargo install sqlx-cli --no-default-features --features native-tls,postgres` | Install sqlx-cli with PostgreSQL support only |

### Database Management (sqlx-cli)

| Command | Description |
|---------|-------------|
| `sqlx database create` | Create the database specified in DATABASE_URL (if it doesn't exist) |
| `sqlx database drop` | Drop the database specified in DATABASE_URL |
| `sqlx database reset` | Drop, recreate, and run all migrations |
| `sqlx migrate add <name>` | Create a new migration file in `migrations/` |
| `sqlx migrate run` | Apply all pending migrations to the database |
| `sqlx migrate info` | Show status of all migrations (applied/pending) |

### Build & Run

| Command | Description |
|---------|-------------|
| `cargo build` | Compile the project (triggers query!() compile-time checks against DATABASE_URL) |
| `cargo run` | Build and run the program — executes all demo phases sequentially |
| `cargo sqlx prepare` | Cache query metadata in `.sqlx/` for offline builds (CI/CD) |
| `cargo sqlx prepare --check` | Verify `.sqlx/` cache is up-to-date (CI validation step) |

### Troubleshooting

| Command | Description |
|---------|-------------|
| `docker compose logs postgres` | View PostgreSQL container logs |
| `docker exec -it sqlx-postgres psql -U sqlx_user -d sqlx_db` | Open psql shell inside the running container |
| `SQLX_OFFLINE=true cargo build` | Build using cached metadata without database connection |

## References

- [SQLx GitHub Repository](https://github.com/launchbadge/sqlx)
- [SQLx docs.rs](https://docs.rs/sqlx/latest/sqlx/)
- [query!() macro documentation](https://docs.rs/sqlx/latest/sqlx/macro.query.html)
- [query_as!() macro documentation](https://docs.rs/sqlx/latest/sqlx/macro.query_as.html)
- [sqlx-cli README](https://github.com/launchbadge/sqlx/blob/main/sqlx-cli/README.md)
- [SQLx transaction example](https://github.com/launchbadge/sqlx/blob/main/examples/postgres/transaction/src/main.rs)
- [SQLx offline mode](https://github.com/launchbadge/sqlx#offline-mode-compile-time-verification)

## State

`not-started`
