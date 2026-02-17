"""Async Engine & Session Factory.

SQLAlchemy's async support works via a layered architecture:
  1. AsyncEngine wraps a regular (sync) Engine
  2. The sync Engine uses a connection pool (QueuePool by default)
  3. asyncpg is the actual async PostgreSQL driver
  4. greenlet bridges sync/async: the ORM's internal sync code runs
     inside greenlet coroutines so it can be awaited from async code

The session factory (async_sessionmaker) creates AsyncSession instances.
Each AsyncSession is a Unit of Work that tracks object changes and
flushes them to the DB in a single transaction on commit().

Run: This module is imported by other modules; it has no standalone entry point.
"""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# ── Database Configuration ────────────────────────────────────────────

DATABASE_URL = "postgresql+asyncpg://sa_user:sa_pass@localhost:5432/sa_db"


# ── Engine Factory ────────────────────────────────────────────────────


def create_engine_factory() -> AsyncEngine:
    """Create and return an AsyncEngine connected to PostgreSQL via asyncpg.

    # TODO(human): Create the async engine with appropriate pool settings
    #
    # Use create_async_engine() from sqlalchemy.ext.asyncio.
    #
    # Parameters to configure:
    #   - url: DATABASE_URL (defined above)
    #       The URL scheme "postgresql+asyncpg://" tells SQLAlchemy to use
    #       the asyncpg driver. This is NOT the same as "postgresql://" which
    #       defaults to psycopg2 (sync). The "+asyncpg" part is called the
    #       "dialect+driver" specification.
    #
    #   - echo: True
    #       Logs all SQL statements to stdout. Essential for learning --
    #       you'll see exactly what queries SQLAlchemy generates. In
    #       production, set this to False (SQL logging is expensive).
    #
    #   - pool_size: 5
    #       Number of persistent connections in the pool. QueuePool (the
    #       default) maintains this many connections open and ready.
    #       When all 5 are in use, new requests wait or overflow.
    #
    #   - max_overflow: 10
    #       Additional connections allowed beyond pool_size. These are
    #       created on demand and destroyed when returned. Total max
    #       concurrent connections = pool_size + max_overflow = 15.
    #       This prevents connection exhaustion under burst load while
    #       keeping the steady-state pool small.
    #
    # Return the AsyncEngine instance.
    #
    # Architecture note:
    #   AsyncEngine is a thin wrapper around Engine. When you call
    #   async_engine.connect(), it internally does:
    #     1. Get a sync connection from the QueuePool
    #     2. Wrap it in an AsyncConnection (greenlet-based)
    #     3. Return it to your async code
    #   The actual I/O goes through asyncpg's event loop integration.
    """
    raise NotImplementedError("TODO(human)")


# ── Session Factory ───────────────────────────────────────────────────


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create a session factory bound to the given engine.

    # TODO(human): Create an async_sessionmaker with the right options
    #
    # Use async_sessionmaker from sqlalchemy.ext.asyncio.
    #
    # Parameters:
    #   - bind: engine
    #       Ties this session factory to a specific engine. Every session
    #       created by this factory will use this engine's connection pool.
    #
    #   - class_: AsyncSession
    #       Tells the factory to produce AsyncSession instances (not regular
    #       Session). This is required for the async API.
    #
    #   - expire_on_commit: False
    #       ** CRITICAL for async **. When True (the default), SQLAlchemy
    #       expires (invalidates) all loaded attributes after commit().
    #       The next attribute access triggers a lazy load to refresh data.
    #       In async code, this lazy load raises MissingGreenlet because
    #       it tries to do sync I/O in an async context.
    #
    #       Setting expire_on_commit=False means objects retain their
    #       attribute values after commit. You can safely access
    #       employee.name after committing without triggering a reload.
    #
    #       Trade-off: the data might be stale if another transaction
    #       modified it. For most web request patterns (read, modify,
    #       commit, return response), this is fine because the session
    #       is short-lived.
    #
    # Return the async_sessionmaker instance.
    #
    # Usage pattern:
    #   session_factory = create_session_factory(engine)
    #   async with session_factory() as session:
    #       result = await session.execute(select(Employee))
    #       ...
    """
    raise NotImplementedError("TODO(human)")
