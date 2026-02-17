"""
Practice 058a — Database Engine & Session Setup

This module handles creating the SQLAlchemy engine (the connection pool to
PostgreSQL) and providing Session objects for each request via FastAPI's
dependency injection system.

Key concepts:
  - Engine: manages a pool of DB connections. Created once at app startup.
    The URL format is: postgresql://user:password@host:port/dbname
  - Session: a unit-of-work that tracks changes to objects and flushes them
    to the DB on commit. Each API request gets its own Session.
  - create_all: reads all SQLModel classes with table=True and issues
    CREATE TABLE IF NOT EXISTS for each one.
"""

from collections.abc import Generator
from typing import Any

from sqlmodel import Session, SQLModel, create_engine

# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------

DATABASE_URL = "postgresql://sqlmodel_user:sqlmodel_pass@localhost:5432/sqlmodel_db"


# ---------------------------------------------------------------------------
# Engine and table creation
# ---------------------------------------------------------------------------


def create_engine_and_tables() -> Any:
    # TODO(human): Create the SQLModel engine and initialize database tables.
    #
    # WHAT TO DO:
    #   1. Call `create_engine(DATABASE_URL, echo=True)` to create the engine.
    #      - The first argument is the PostgreSQL connection URL.
    #      - `echo=True` makes SQLAlchemy log every SQL statement it executes
    #        to stdout — invaluable for learning. You'll see the exact CREATE
    #        TABLE, INSERT, SELECT statements. Disable in production.
    #
    #   2. Call `SQLModel.metadata.create_all(engine)` to create all tables.
    #      - SQLModel.metadata is a shared MetaData registry that collects all
    #        classes defined with `table=True`. create_all() issues CREATE TABLE
    #        IF NOT EXISTS for each one, so it's safe to call multiple times.
    #      - IMPORTANT: All model modules must be imported BEFORE calling this,
    #        otherwise their tables won't be registered in the metadata.
    #
    #   3. Return the engine.
    #
    # WHY THIS MATTERS:
    #   The engine is the core of SQLAlchemy's connection management. It
    #   maintains a connection pool (default: 5 connections for PostgreSQL)
    #   and handles connection lifecycle (checkout, return, invalidation).
    #   create_all() is a development convenience — in production, you'd use
    #   Alembic migrations instead.
    #
    # EXPECTED RESULT:
    #   When the app starts, you should see SQL logs like:
    #     CREATE TABLE IF NOT EXISTS team (id SERIAL PRIMARY KEY, ...)
    #     CREATE TABLE IF NOT EXISTS hero (id SERIAL PRIMARY KEY, ...)
    #
    # HINT:
    #   engine = create_engine(DATABASE_URL, echo=True)
    #   SQLModel.metadata.create_all(engine)
    #   return engine

    raise NotImplementedError("TODO(human)")


# ---------------------------------------------------------------------------
# Session dependency for FastAPI
# ---------------------------------------------------------------------------


def get_session() -> Generator[Session, None, None]:
    # TODO(human): Yield a SQLModel Session for FastAPI dependency injection.
    #
    # WHAT TO DO:
    #   1. This function must be a generator (use `yield`, not `return`).
    #   2. Use the `engine` variable (module-level, set during app startup
    #      via `init_db()`).
    #   3. Create a Session using `with Session(engine) as session:` and
    #      yield it.
    #
    # HOW IT WORKS:
    #   FastAPI's Depends() system calls this generator for each request:
    #     - Everything before `yield` runs at request start
    #     - The yielded Session is injected into the endpoint function
    #     - Everything after `yield` runs at request end (cleanup)
    #   The `with` statement ensures the Session is properly closed even
    #   if the endpoint raises an exception.
    #
    # WHY THIS MATTERS:
    #   Each request gets its own Session (unit of work). This is critical:
    #   Sessions are NOT thread-safe. If two requests shared one Session,
    #   their changes would interfere. The generator pattern guarantees
    #   each request gets a fresh Session that's cleaned up afterward.
    #
    # EXPECTED RESULT:
    #   Endpoints can declare `session: Session = Depends(get_session)` and
    #   receive a live Session connected to PostgreSQL.
    #
    # HINT:
    #   with Session(engine) as session:
    #       yield session

    raise NotImplementedError("TODO(human)")


# ---------------------------------------------------------------------------
# Module-level engine (set by init_db, used by get_session)
# ---------------------------------------------------------------------------

engine = None


def init_db() -> None:
    """Initialize the database engine and create tables.

    Called once during app startup (in the lifespan handler).
    Sets the module-level `engine` variable so get_session() can use it.
    """
    global engine  # noqa: PLW0603
    engine = create_engine_and_tables()
