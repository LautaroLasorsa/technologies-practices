# Practice 059b: Diesel -- Type-Safe ORM & Migrations in Rust

## Technologies

- **Diesel 2.2** -- Rust's most mature ORM and query builder with compile-time SQL validation
- **Diesel CLI** -- Command-line tool for managing migrations and schema generation
- **PostgreSQL 16** -- Relational database (via Docker)
- **chrono** -- Date/time types for Diesel's `Timestamp` mapping
- **bigdecimal** -- Arbitrary-precision decimal for Diesel's `Numeric` mapping
- **dotenvy** -- Loads `DATABASE_URL` from `.env` file

## Stack

- Rust (cargo, edition 2021)
- Docker / Docker Compose

## Theoretical Context

### What is Diesel and What Problem Does It Solve?

[Diesel](https://diesel.rs/) is a type-safe ORM (Object-Relational Mapper) and query builder for Rust. With over 21.9 million crates.io downloads, it is Rust's most established database library. Its core innovation: **a type-level DSL that catches query errors at compile time WITHOUT connecting to a database**. If your Diesel code compiles, your queries are structurally valid -- wrong column names, type mismatches, invalid joins, and illegal aggregations are all caught by the Rust compiler before your program ever runs.

The problem Diesel solves is the gap between SQL (a stringly-typed language) and Rust (a strongly-typed language). Traditional ORMs in dynamic languages (Python's SQLAlchemy, Ruby's ActiveRecord) defer query errors to runtime -- you discover typos and type mismatches when the query executes, often in production. Diesel eliminates this entire class of bugs by encoding the database schema into Rust's type system.

### How Diesel Works: The `table!` Macro and Type-Level DSL

Diesel's compile-time guarantees originate from a single source of truth: **`schema.rs`**. This file contains `table!` macro invocations that describe your database schema in Rust types. Here's what happens:

1. **Schema definition**: The `table!` macro generates a Rust module per table. For `products (id)`, it creates:
   - `products::table` -- a unit struct representing the SQL table
   - `products::id`, `products::name`, etc. -- unit structs for each column
   - Trait implementations connecting columns to their SQL types (`Int4`, `Varchar`, `Numeric`, etc.)
   - The primary key marker (`products::id` is the PK)

2. **Query construction**: When you write `products::table.filter(products::stock.lt(10)).select(Product::as_select())`, you're not building a string -- you're constructing a **type** that encodes the entire query structure. The Rust compiler verifies:
   - `products::stock` exists and is `Int4` (so `.lt(10)` is valid -- comparing int to int)
   - `Product::as_select()` matches columns available in the `products` table
   - The final `.load(conn)` returns the correct Rust type

3. **SQL generation**: At runtime, Diesel serializes the type-encoded query into actual SQL. This is fast (zero-overhead abstraction) because all validation happened at compile time.

The key insight: **Diesel never connects to the database to validate queries**. Unlike [SQLx](https://github.com/launchbadge/sqlx) (which runs `EXPLAIN` against a live database at compile time), Diesel validates against the `schema.rs` types. This means:
- You can compile Diesel code without a running database
- The `schema.rs` file must stay in sync with your actual schema (managed by migrations)
- If `schema.rs` is wrong, the code compiles but queries fail at runtime

### The Migration System

[Diesel CLI](https://crates.io/crates/diesel_cli) manages database schema evolution:

1. **`diesel setup`** -- Creates the database (if it doesn't exist) and an internal `__diesel_schema_migrations` table to track which migrations have been applied.

2. **`diesel migration generate <name>`** -- Creates a timestamped folder with `up.sql` (forward migration) and `down.sql` (rollback). You write raw SQL in these files -- Diesel doesn't abstract DDL.

3. **`diesel migration run`** -- Executes all pending `up.sql` files in order, then regenerates `schema.rs` by introspecting the actual database schema. This is how `schema.rs` stays in sync: it's always derived from the real database.

4. **`diesel migration redo`** -- Runs `down.sql` then `up.sql` for the last migration. Essential for testing that your `down.sql` actually works.

The migration workflow ensures `schema.rs` is never manually edited -- it's a generated file that reflects the true database state.

### Key Derive Macros

| Derive | Purpose |
|--------|---------|
| **`Queryable`** | Maps a database row to a Rust struct. Fields are matched **by position** (must match `schema.rs` column order). |
| **`Selectable`** | Adds `as_select()` method that constructs a `SELECT` clause matching fields **by name**. Safer than positional `Queryable` alone. Requires `#[diesel(table_name = ...)]`. |
| **`Insertable`** | Enables `insert_into(table).values(&struct)`. Omitted fields use DB defaults (e.g., `SERIAL`, `DEFAULT NOW()`). Requires `#[diesel(table_name = ...)]`. |
| **`AsChangeset`** | Enables `update(target).set(&struct)`. `Option<T>` fields are skipped when `None` (partial updates). Requires `#[diesel(table_name = ...)]`. |
| **`Identifiable`** | Marks a struct as having a primary key. Required for associations. Default PK field is `id`. |
| **`Associations`** | Declares parent-child relationships via `#[diesel(belongs_to(Parent))]`. Enables `belonging_to()` queries. |

### Connection Management

Diesel's `PgConnection` is a **synchronous, single-threaded** connection. For multi-threaded applications, Diesel integrates with [r2d2](https://docs.rs/diesel/latest/diesel/r2d2/index.html) for connection pooling. For async applications, the community [diesel-async](https://crates.io/crates/diesel-async) crate wraps Diesel's query builder with async execution.

### Diesel vs SQLx: Two Philosophies

| Aspect | Diesel | SQLx |
|--------|--------|------|
| **Query style** | Type-safe DSL (`table.filter(col.eq(val))`) | Raw SQL strings (`query_as!("SELECT ...")`) |
| **Validation** | Against `schema.rs` types (no DB needed) | Against live database via `EXPLAIN` at compile time |
| **Schema source** | `schema.rs` (generated by migrations) | The actual database (must be running at compile time) |
| **Sync/Async** | Sync-first (async via `diesel-async`) | Async-first (native `async/await`) |
| **Philosophy** | "Abstract SQL into Rust types" | "Write SQL, but type-check it" |
| **Trade-off** | More abstraction, longer compile times | Less abstraction, requires DB at compile time |

Both are production-grade. Diesel is preferred when you want maximum compile-time safety without database dependencies. SQLx is preferred when you want to write raw SQL with async support.

### Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **`schema.rs`** | Auto-generated file mapping database schema to Rust types. Source of truth for compile-time query validation. |
| **`table!` macro** | Generates column types, table struct, and trait impls from a schema definition. |
| **`QueryResult<T>`** | Alias for `Result<T, diesel::result::Error>`. Standard return type for all Diesel operations. |
| **`.find(pk)`** | Shortcut for `WHERE primary_key = pk`. Only works with the declared PK column. |
| **`.filter(expr)`** | General WHERE clause. Accepts column expressions (`.eq()`, `.lt()`, `.like()`, `.is_null()`, etc.). |
| **`.get_result(conn)`** | Executes query and returns one row (PostgreSQL `RETURNING`). |
| **`.execute(conn)`** | Executes query and returns affected row count (`usize`). |
| **`.optional()`** | Converts `Err(NotFound)` to `Ok(None)`. Essential for "find or nothing" queries. |
| **`joinable!`** | Declares a foreign key relationship between tables. Required for `.inner_join()` / `.left_join()`. |
| **`conn.transaction(\|c\| {...})`** | Executes a closure inside BEGIN/COMMIT. Returns `Err` triggers ROLLBACK. |

## Description

Build a **product-and-orders inventory system** using Diesel's type-safe query builder. The practice covers the full Diesel workflow: PostgreSQL setup, CLI-managed migrations, schema generation, model structs with derives, CRUD operations via the DSL, joins and aggregations, and transactional order placement with stock validation.

### What you'll learn

1. **Diesel CLI workflow** -- `diesel setup`, `diesel migration generate/run/redo`
2. **Schema DSL** -- How `table!` generates types and how `schema.rs` stays in sync
3. **Model structs** -- `Queryable`, `Selectable`, `Insertable` derives and their semantics
4. **Query builder DSL** -- `.filter()`, `.order()`, `.limit()`, `.find()`, `.optional()`
5. **CRUD operations** -- `insert_into`, `update`, `delete` with type-safe columns
6. **Joins & aggregations** -- `.inner_join()`, `.group_by()`, `count_star()`
7. **Transactions** -- `conn.transaction()` with automatic rollback on error

## Prerequisites

### Diesel CLI

Diesel's CLI tool manages migrations and schema generation. Install with PostgreSQL support only (avoids needing SQLite/MySQL dev libraries):

```
cargo install diesel_cli --no-default-features --features postgres
```

Verify:
```
diesel --version
```

### PostgreSQL client library (libpq)

The `diesel` crate with `features = ["postgres"]` links against `libpq` (PostgreSQL's C client library). On Windows, this is typically bundled with a PostgreSQL installation. If you get linker errors about `libpq`, install PostgreSQL or set `PQ_LIB_DIR` to the directory containing `libpq.lib`.

## Instructions

### Phase 1: Environment Setup (~10 min)

1. Start PostgreSQL: `docker compose up -d`
2. Verify it's healthy: `docker compose ps`
3. Run `diesel setup` -- this creates the database (if needed) and the internal migrations table
4. Inspect what `diesel setup` created: check for `__diesel_schema_migrations` table

### Phase 2: Migrations (~15 min)

1. Examine the pre-created migration SQL files in `migrations/`
2. Run `diesel migration run` -- this executes both `up.sql` files and regenerates `schema.rs`
3. Compare the regenerated `schema.rs` with the pre-created version (should match)
4. Test rollback: `diesel migration redo` -- runs `down.sql` then `up.sql` for the last migration
5. Key question: Why does Diesel regenerate `schema.rs` from the database instead of from the SQL files?

### Phase 3: Models & Basic CRUD (~30 min)

1. Open `src/models.rs` -- study the `Product` struct (fully derived, with detailed comments explaining each derive), then complete the `NewProduct` struct by adding the `Insertable` derive and `#[diesel(table_name = products)]` attribute
2. Build to verify: `cargo build` (compile errors here mean your struct fields don't match `schema.rs`)
3. Open `src/queries_basic.rs` -- implement the five TODO(human) functions:
   - `get_all_products` -- load all with `.select(Product::as_select()).load(conn)`
   - `get_product_by_id` -- `.find()` + `.optional()`
   - `create_product` -- `insert_into().values().get_result()`
   - `update_product_stock` -- `update().set().get_result()`
   - `delete_product` -- `delete().execute()`
4. Each function is 1-3 lines of Diesel DSL. The fully-implemented `create_order` and `get_orders_for_product` serve as reference
5. Run: `cargo run` to test Phase 3 output

### Phase 4: Advanced Queries (~20 min)

1. Open `src/queries_advanced.rs` -- implement the two TODO(human) functions:
   - `products_with_orders` -- `.inner_join()` with `.select()` tuple
   - `products_by_category_count` -- `.group_by()` with `count_star()`
2. Review the fully-implemented filter helpers (`low_stock_products`, `search_products_by_name`, `top_expensive_products`) to see DSL composability
3. Run: `cargo run` to test Phase 4 output
4. Key question: What happens if you try to `.select(products::name)` inside a `.group_by(products::category)` query? (Try it -- you'll get a compile error, which is the point.)

### Phase 5: Transactions (~15 min)

1. Open `src/transactions.rs` -- implement the TODO(human) function:
   - `place_order_transaction` -- `conn.transaction(|conn| { ... })` wrapping lookup + validation + insert + update
2. Review the fully-implemented `cancel_order_transaction` as reference
3. Run: `cargo run` to test -- verify that the failed order (9999 units) does NOT change stock
4. Key question: What happens if you use the outer `conn` inside the transaction closure instead of the closure's `conn` parameter?

### Phase 6: Reflection & Comparison (~10 min)

1. Try adding a column to the products table: write a new migration, run it, see `schema.rs` update
2. Observe: any code referencing the old schema breaks at compile time
3. Discussion: How does Diesel's "if it compiles, it works" compare to SQLx's "compile-time EXPLAIN against a live DB"? When would you choose each?

## Motivation

- **Most mature Rust ORM** -- Diesel has been production-ready since 2016, with 21.9M+ crates.io downloads. It's the standard choice for synchronous Rust database access.
- **"If it compiles, it works"** -- Diesel's type-level query validation is a unique selling point of Rust's type system applied to databases. Understanding how it works deepens your grasp of Rust's trait system and generics.
- **Migration-first workflow** -- Schema management via SQL migrations is a universal pattern across all ORMs (Django, Alembic, ActiveRecord). Diesel's implementation is minimal and explicit.
- **Complements current profile** -- Adds database layer expertise to Rust systems programming. Combined with gRPC (004b) and testing (024b), this builds toward full-stack Rust service development.

## Commands

### Prerequisites

| Command | Description |
|---------|-------------|
| `cargo install diesel_cli --no-default-features --features postgres` | Install Diesel CLI with PostgreSQL support only |
| `diesel --version` | Verify Diesel CLI is installed and on PATH |

### Phase 1: Environment Setup

All commands run from `practice_059b_diesel_rust/`.

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start PostgreSQL 16 container on port 5432 |
| `docker compose ps` | Verify the container is healthy |
| `diesel setup` | Create the database (if needed) and internal migrations tracking table |

### Phase 2: Migrations

| Command | Description |
|---------|-------------|
| `diesel migration run` | Execute all pending migrations (`up.sql`) and regenerate `schema.rs` |
| `diesel migration redo` | Rollback last migration (`down.sql`) then re-apply (`up.sql`) -- tests reversibility |
| `diesel migration list` | Show all migrations and their status (pending/applied) |
| `diesel print-schema` | Print the current database schema as Diesel `table!` macros (what goes into `schema.rs`) |

### Phase 3-5: Build & Run

| Command | Description |
|---------|-------------|
| `cargo build` | Compile the project -- Diesel validates all queries at compile time |
| `cargo run` | Build and run the demo program (all phases: CRUD, joins, transactions) |

### Cleanup

| Command | Description |
|---------|-------------|
| `docker compose down` | Stop PostgreSQL container (data persists in named volume) |
| `docker compose down -v` | Stop container AND delete the `pgdata` volume (full reset) |

## References

- [Diesel Official Site](https://diesel.rs/)
- [Diesel Getting Started Guide](https://diesel.rs/guides/getting-started)
- [Diesel All About Selects](https://diesel.rs/guides/all-about-selects.html)
- [Diesel Relations Guide](https://diesel.rs/guides/relations.html)
- [Diesel API Docs (2.2.x)](https://docs.diesel.rs/2.2.x/diesel/index.html)
- [Diesel CLI README](https://github.com/diesel-rs/diesel/blob/main/diesel_cli/README.md)
- [Diesel vs SQLx Deep Dive](https://leapcell.io/blog/diesel-and-sqlx-a-deep-dive-into-rust-orms)
- [Compare Diesel (official)](https://diesel.rs/compare_diesel.html)
- [A Guide to Rust ORMs (Shuttle, 2025)](https://www.shuttle.dev/blog/2024/01/16/best-orm-rust)

## State

`not-started`
