# Practice 058b -- SQLAlchemy 2.0: Async ORM & Advanced Queries

## Technologies

- **SQLAlchemy 2.0** (>=2.0.30) -- Python's industry-standard ORM and SQL toolkit
- **asyncpg** (>=0.29) -- High-performance async PostgreSQL driver (C-accelerated)
- **Alembic** (>=1.13) -- SQLAlchemy's database migration tool
- **PostgreSQL 16** -- Relational database via Docker
- **greenlet** (>=3.0) -- Lightweight coroutine library bridging sync ORM internals to async

## Stack

Python 3.12+ (uv), Docker / Docker Compose

## Theoretical Context

### What SQLAlchemy Is & The Problem It Solves

SQLAlchemy is a dual-layer SQL toolkit for Python: a **Core** layer for SQL expression construction and a **ORM** layer for object-relational mapping. Created by Mike Bayer in 2006, it has been the de facto standard Python ORM for 18+ years (11.5k+ GitHub stars, used by Dropbox, Reddit, Yelp, Mozilla, and most enterprise Python shops).

**The problem**: Python applications need to interact with relational databases. Raw SQL strings are error-prone (SQL injection, no IDE support, no type safety). Simple ORMs (Django ORM, ActiveRecord) hide SQL too aggressively -- they work for CRUD but break down for complex queries (window functions, CTEs, lateral joins). SQLAlchemy provides full SQL expressivity through Python constructs while also offering a complete ORM for the 80% of queries that are straightforward.

### Architecture: Core vs ORM

SQLAlchemy has two distinct layers that can be used independently or together:

```
  Application Code
       |
  ┌────┴────────────────────────────┐
  │           ORM Layer             │
  │  (Mapped classes, Session,      │
  │   relationships, Unit of Work)  │
  └────┬────────────────────────────┘
       |
  ┌────┴────────────────────────────┐
  │          Core Layer             │
  │  (Table, Column, select(),      │
  │   insert(), func.*, operators)  │
  └────┬────────────────────────────┘
       |
  ┌────┴────────────────────────────┐
  │     Engine / Connection Pool    │
  │  (Engine, QueuePool, dialect)   │
  └────┬────────────────────────────┘
       |
  ┌────┴────────────────────────────┐
  │       DBAPI Driver              │
  │  (asyncpg, psycopg2, sqlite3)  │
  └─────────────────────────────────┘
```

- **Core**: SQL expression language. You construct queries as Python objects (`select(users).where(users.c.name == "Alice")`). Core generates dialect-specific SQL. It does NOT track object state -- you work with rows and tuples.

- **ORM**: Maps Python classes to database tables. You define `class Employee(Base)` with `mapped_column()` and `relationship()`. The ORM adds: identity map (one Python object per DB row), unit of work (batched inserts/updates), change tracking (dirty detection), and relationship loading.

### SQLAlchemy 2.0: What Changed

Version 2.0 (released January 2023) was a major overhaul. Key changes:

| Feature | 1.x Style | 2.0 Style |
|---------|-----------|-----------|
| Querying | `session.query(User).filter(...)` | `session.execute(select(User).where(...))` |
| Column defs | `name = Column(String)` | `name: Mapped[str] = mapped_column(String)` |
| Base class | `Base = declarative_base()` | `class Base(DeclarativeBase): pass` |
| Type safety | No type annotations | Full `Mapped[T]` annotations |
| Async | Bolt-on (1.4 preview) | First-class `AsyncEngine`, `AsyncSession` |

The 2.0 `select()` style is more explicit and composable. It also works identically in sync and async contexts -- the same `select()` statement can be executed by both `Session` and `AsyncSession`.

### Async Support: How It Works Internally

SQLAlchemy's async support uses a **greenlet bridge** architecture:

1. **AsyncEngine** wraps a regular (sync) **Engine**.
2. When you call `await session.execute(stmt)`, the AsyncSession:
   a. Spawns a greenlet containing the sync ORM machinery
   b. The sync code calls `connection.execute()` which would normally block
   c. At the I/O point, the greenlet suspends and yields to the event loop
   d. **asyncpg** performs the actual async I/O via the event loop
   e. When data arrives, the greenlet resumes with the result
3. From your perspective, it's fully async -- `await` everywhere.

**Why this design?** The ORM has 18 years of sync code (Unit of Work, identity map, eager loading). Rewriting it as native async would take years and introduce bugs. The greenlet bridge lets the existing sync code work unmodified while providing a truly async I/O path.

**asyncpg**: The async PostgreSQL driver. Written in C and Cython, it is 2-5x faster than psycopg2 for common operations. It uses PostgreSQL's binary protocol (not text), which avoids parsing overhead for numeric/date types.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **DeclarativeBase** | Base class for all ORM models. Replaces `declarative_base()` factory. Holds `MetaData` registry. |
| **Mapped[T]** | Type annotation declaring that an attribute is a mapped column of type T. Enables mypy/pyright checking. |
| **mapped_column()** | Column definition with type integration. Replaces `Column()`. Infers SQL type from `Mapped[T]` when possible. |
| **relationship()** | Python-level link between models. NOT a database constraint -- it's an ORM concept for navigating between related objects. |
| **AsyncEngine** | Async wrapper around Engine. Manages the connection pool. Created via `create_async_engine()`. |
| **AsyncSession** | Async Unit of Work. Tracks object changes, flushes to DB, manages transactions. Created via `async_sessionmaker`. |
| **select()** | Core SQL construct. Builds SELECT statements functionally: `select(User).where(User.id == 1)`. |
| **Identity Map** | Cache within Session: each DB row maps to exactly one Python object. Prevents duplicate objects for the same row. |
| **Unit of Work** | Pattern: Session batches all pending changes (inserts, updates, deletes) and flushes them in dependency order. |
| **QueuePool** | Default connection pool. Maintains `pool_size` persistent connections + `max_overflow` temporary ones. |

### Loading Strategies

When you access a relationship (e.g., `department.employees`), SQLAlchemy must load the related data. The strategy determines HOW and WHEN:

| Strategy | SQL Pattern | Queries | Best For |
|----------|-------------|---------|----------|
| **lazy** (default) | `SELECT ... WHERE fk = ?` on access | N+1 | Sync code, rarely accessed relations |
| **selectinload** | `SELECT ... WHERE fk IN (...)` | 2 | One-to-many (recommended default for async) |
| **joinedload** | `LEFT JOIN ...` | 1 | Many-to-one, one-to-one |
| **subqueryload** | `SELECT ... WHERE fk IN (SELECT ...)` | 2 | Complex parent queries |

**Critical for async**: Lazy loading raises `MissingGreenlet` in async code because it tries to do synchronous I/O. You MUST use explicit eager loading (`selectinload`, `joinedload`) or `await session.run_sync()`.

### Alembic Migration Architecture

Alembic tracks schema changes as versioned Python scripts:

```
alembic/
  versions/
    a1b2c3_initial.py          # First migration
    d4e5f6_add_phone_column.py  # Second migration
  env.py                        # Migration environment config
alembic.ini                     # Connection URL, logging
```

Each migration has `upgrade()` and `downgrade()` functions. Alembic stores the current revision in an `alembic_version` table in the database. `alembic upgrade head` applies all pending migrations; `alembic downgrade -1` reverts the last one.

**Autogenerate**: `alembic revision --autogenerate` compares `Base.metadata` (Python models) against the live DB schema and generates a migration script with the diff. It detects: added/removed tables, added/removed columns, type changes, nullable changes, index changes. It does NOT detect: column renames (appears as drop+add), data migrations, or custom constraint changes.

### Where SQLAlchemy Fits in the Ecosystem

| Tool | Approach | Trade-off |
|------|----------|-----------|
| **Raw SQL** (asyncpg/psycopg) | Direct SQL strings | Maximum control, zero abstraction, injection risk |
| **SQLAlchemy Core** | Python SQL expressions | Type-safe SQL construction, no ORM overhead |
| **SQLAlchemy ORM** | Full object mapping | Productivity, relationships, change tracking |
| **SQLModel** | SQLAlchemy + Pydantic | FastAPI integration, simpler syntax, less flexibility |
| **Django ORM** | Tightly coupled to Django | Great for Django apps, limited outside Django |
| **Tortoise ORM** | Async-native ORM | Smaller ecosystem, less mature |

SQLAlchemy is the right choice when you need: async support, complex queries (CTEs, window functions, subqueries), fine-grained control over SQL generation, or when you're not using Django.

## Description

This practice builds proficiency with SQLAlchemy 2.0's async ORM through hands-on exercises covering: modern declarative models (`Mapped[]`, `mapped_column()`), async engine/session management, 2.0-style `select()` queries, advanced SQL patterns (aggregations, subqueries, recursive CTEs, window functions), eager loading strategies to solve N+1, and Alembic database migrations.

## Instructions

### Phase 1: Setup & Models (~15 min)

Start PostgreSQL and define the ORM models.

1. Start Docker: `docker compose up -d`
2. Verify PostgreSQL: `docker exec sqlalchemy2-postgres psql -U sa_user -d sa_db -c "\dt"`
3. Initialize the project: `uv sync`
4. **Exercise 1 (models.py)**: Implement the `Department` model -- define `__tablename__`, all `mapped_column()` fields (id, name, budget, created_at with `server_default=func.now()`), and the `relationship()` to employees. This teaches you the 2.0 declarative style: `Mapped[T]` annotations paired with `mapped_column()` for full type safety, and how `relationship(back_populates=...)` creates bidirectional Python-level links.

5. **Exercise 2 (models.py)**: Implement the `Employee` model -- all columns including the self-referential FK (`manager_id -> employees.id`), plus four relationships: `department` (many-to-one), `manager` (self-referential), `subordinates` (reverse of manager), and `projects` (many-to-many via association table). This is the most complex model -- it teaches self-referential FKs (`remote_side` to disambiguate direction), many-to-many via `secondary=`, and why lazy loading fails in async contexts.

### Phase 2: Engine & Basic CRUD (~20 min)

Create the async engine and write basic queries.

6. **Exercise 3 (engine.py)**: Implement `create_engine_factory()` -- create an `AsyncEngine` with `postgresql+asyncpg://` URL, pool_size=5, max_overflow=10, echo=True. This teaches the async engine architecture: AsyncEngine wraps a sync Engine, asyncpg handles I/O, QueuePool manages connections.

7. **Exercise 4 (engine.py)**: Implement `create_session_factory()` -- create an `async_sessionmaker` with `expire_on_commit=False`. This is critical for async: without it, accessing attributes after commit triggers lazy reloads that raise `MissingGreenlet`.

8. **Exercise 5 (queries_basic.py)**: Implement `get_employees_by_department()` -- build a `select(Employee)` with `.join()`, `.where()`, and `.options(selectinload(...))`. This teaches the 2.0 query pattern: `select()` -> `execute()` -> `scalars().all()`, and why eager loading is mandatory in async.

9. **Exercise 6 (queries_basic.py)**: Implement `create_employee()` -- instantiate, `session.add()`, `flush()`, `refresh()`. This teaches the Unit of Work insert pattern: add marks the object as pending, flush emits SQL and populates the auto-generated id, refresh reloads server-side defaults.

10. **Exercise 7 (queries_basic.py)**: Implement `update_salary()` -- fetch via `scalar_one_or_none()`, mutate, flush. This teaches the load-modify-flush update pattern: SQLAlchemy detects dirty attributes automatically and emits partial UPDATEs.

11. Run: `uv run python -m app.queries_basic`

### Phase 3: Advanced Queries (~25 min)

Write complex queries using SQLAlchemy's SQL expression language.

12. **Exercise 8 (queries_advanced.py)**: Implement `department_salary_stats()` -- GROUP BY with `func.avg()`, `func.min()`, `func.max()`, `func.count()`. This teaches aggregate functions in SQLAlchemy: `func.*` generates SQL functions, `.label()` assigns aliases, and the result contains named tuples (not ORM objects) when selecting specific columns.

13. **Exercise 9 (queries_advanced.py)**: Implement `employees_above_department_avg()` -- build a subquery with `.subquery()`, join it to the outer query, filter where salary > avg. This teaches subquery composition: `.subquery()` wraps a SELECT as a derived table, `.c` accesses its columns, and the outer query joins against it like a real table.

14. **Exercise 10 (queries_advanced.py)**: Implement `management_hierarchy()` -- recursive CTE with `.cte(recursive=True)`, anchor + recursive step + `union_all()`. This teaches recursive CTEs: the SQL standard for traversing trees/graphs in relational data. The anchor starts at one row, the recursive step joins to walk up the manager chain.

15. **Exercise 11 (queries_advanced.py)**: Implement `salary_rank_by_department()` -- `func.rank().over(partition_by=..., order_by=...)`. This teaches window functions: unlike GROUP BY (which collapses rows), window functions annotate each row with aggregate/ranking info while preserving all rows.

16. Run: `uv run python -m app.queries_advanced`

### Phase 4: Loading Strategies (~20 min)

Diagnose and fix the N+1 problem.

17. **Exercise 12 (loading_strategies.py)**: Implement `demonstrate_n_plus_one()` -- query departments without eager loading, then access `.employees` via `run_sync()` to trigger lazy loads. Observe the query count. This makes the N+1 problem concrete: 1 query for departments + N queries for employees (one per department).

18. **Exercise 13 (loading_strategies.py)**: Implement `fix_with_selectinload()` -- same query with `.options(selectinload(Department.employees))`. Verify 0 additional queries after execute. This teaches selectinload: the recommended strategy for one-to-many in async code (always 2 queries, no duplication).

19. **Exercise 14 (loading_strategies.py)**: Implement `compare_loading_strategies()` -- run the query with selectinload, joinedload, and subqueryload. Note that joinedload requires `.unique().scalars()`. This builds intuition for when to use each strategy: selectinload for one-to-many, joinedload for many-to-one, subqueryload for complex parent queries.

20. Run: `uv run python -m app.loading_strategies`

### Phase 5: Alembic Migrations (~15 min)

Add a column via Alembic's autogenerate workflow.

21. **Exercise 15 (migrations_demo.py + models.py)**: Add `phone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)` to the Employee model. Generate a migration with `uv run alembic revision --autogenerate -m "add phone column"`, inspect the generated script, then apply with `uv run alembic upgrade head`. Implement `verify_phone_column()` to check the column exists via `information_schema`. This teaches the complete Alembic workflow: model change -> autogenerate diff -> review script -> apply migration.

22. Run: `uv run python -m app.migrations_demo`

### Phase 6: Full Run (~5 min)

23. Run all phases: `uv run python -m app.main`
24. Clean up: `docker compose down -v`

## Motivation

SQLAlchemy is the industry-standard Python ORM -- required knowledge for any enterprise Python backend role. Version 2.0 brought first-class async support (essential for modern FastAPI/Starlette apps), strict typing with `Mapped[]` (aligns with the typing-everywhere trend in Python), and the new `select()` style that is cleaner and more explicit than the legacy Query API. Understanding async session management, eager loading strategies, and Alembic migrations is table-stakes for production Python applications. This practice covers the advanced patterns (CTEs, window functions, subqueries, N+1 diagnosis) that differentiate senior backend developers from juniors.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| Setup | `docker compose up -d` | Start PostgreSQL 16 container with seed data |
| Setup | `docker exec sqlalchemy2-postgres psql -U sa_user -d sa_db -c "\dt"` | Verify tables were created by init.sql |
| Setup | `uv sync` | Install Python dependencies |
| Phase 2 | `uv run python -m app.queries_basic` | Run basic CRUD demos (select, insert, update) |
| Phase 3 | `uv run python -m app.queries_advanced` | Run advanced query demos (aggregates, CTEs, windows) |
| Phase 4 | `uv run python -m app.loading_strategies` | Run N+1 and loading strategy demos |
| Phase 5 | `uv run alembic revision --autogenerate -m "add phone column"` | Generate migration from model diff |
| Phase 5 | `uv run alembic upgrade head` | Apply all pending migrations |
| Phase 5 | `uv run alembic history` | Show migration history |
| Phase 5 | `uv run alembic current` | Show current DB revision |
| Phase 5 | `uv run alembic downgrade -1` | Revert the last migration |
| Phase 5 | `uv run python -m app.migrations_demo` | Verify phone column migration |
| Phase 6 | `uv run python -m app.main` | Run all phases sequentially |
| Cleanup | `docker compose down -v` | Stop PostgreSQL and remove volume data |

**State:** `not-started`
