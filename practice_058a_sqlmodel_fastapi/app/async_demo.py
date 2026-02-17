"""
Practice 058a — Async SQLModel Demonstration

This module demonstrates SQLModel's async support, which is built on
SQLAlchemy 2.0's AsyncEngine and AsyncSession. Async is important for
high-concurrency web servers: while one request waits for a DB query,
the event loop can handle other requests.

Key differences from sync:
  - Engine: create_async_engine() instead of create_engine()
  - Session: AsyncSession instead of Session
  - Queries: await session.exec() instead of session.exec()
  - Commit: await session.commit() instead of session.commit()
  - Driver: asyncpg (async PostgreSQL driver) instead of psycopg2 (sync)

The connection URL changes from postgresql:// to postgresql+asyncpg://
to tell SQLAlchemy to use the asyncpg driver.

NOTE: This is a standalone demo script, not part of the FastAPI app.
Run it directly: `uv run python -m app.async_demo`
"""

import asyncio

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlmodel import SQLModel, select

from app.models import Hero, HeroCreate

# ---------------------------------------------------------------------------
# Async database configuration
# ---------------------------------------------------------------------------

# NOTE: asyncpg requires the postgresql+asyncpg:// scheme.
# psycopg2-binary (sync) is NOT used here — asyncpg is the async driver.
# You'll need to install it: uv add asyncpg
ASYNC_DATABASE_URL = (
    "postgresql+asyncpg://sqlmodel_user:sqlmodel_pass@localhost:5432/sqlmodel_db"
)


# ---------------------------------------------------------------------------
# Async engine and session creation
# ---------------------------------------------------------------------------


async def create_async_engine_and_session() -> tuple:
    # TODO(human): Create an async engine and return it with an AsyncSession class.
    #
    # WHAT TO DO:
    #   1. Create the async engine:
    #        engine = create_async_engine(ASYNC_DATABASE_URL, echo=True)
    #      This is the async equivalent of create_engine(). Under the hood,
    #      it uses asyncpg's connection pool instead of psycopg2's.
    #
    #   2. Create the tables using the async engine:
    #        async with engine.begin() as conn:
    #            await conn.run_sync(SQLModel.metadata.create_all)
    #      create_all() is a SYNC function (it was designed before async
    #      existed). run_sync() runs it in a thread so it doesn't block
    #      the event loop. engine.begin() starts a transaction that auto-
    #      commits when the context manager exits.
    #
    #   3. Return the engine.
    #
    # WHY THIS MATTERS:
    #   Async DB access is critical for high-concurrency APIs. With sync
    #   psycopg2, each request blocks a thread during the DB query. With
    #   asyncpg, the event loop continues serving other requests while
    #   waiting for PostgreSQL. For I/O-bound workloads (most web APIs),
    #   async can handle 10-100x more concurrent requests per process.
    #
    # EXPECTED RESULT:
    #   An AsyncEngine connected to PostgreSQL via asyncpg, with all
    #   tables created.
    #
    # HINT:
    #   engine = create_async_engine(ASYNC_DATABASE_URL, echo=True)
    #   async with engine.begin() as conn:
    #       await conn.run_sync(SQLModel.metadata.create_all)
    #   return engine

    raise NotImplementedError("TODO(human)")


# ---------------------------------------------------------------------------
# Async CRUD demonstration
# ---------------------------------------------------------------------------


async def async_create_and_query() -> None:
    # TODO(human): Demonstrate async CRUD: create a hero, then query all heroes.
    #
    # WHAT TO DO:
    #   1. Get the async engine:
    #        engine = await create_async_engine_and_session()
    #
    #   2. Create a hero using an AsyncSession:
    #        async with AsyncSession(engine) as session:
    #            hero = Hero(name="Async Hero", secret_name="Eventloop", age=1)
    #            session.add(hero)
    #            await session.commit()
    #            await session.refresh(hero)
    #            print(f"Created: {hero}")
    #
    #   3. Query all heroes in a new session:
    #        async with AsyncSession(engine) as session:
    #            statement = select(Hero)
    #            results = await session.exec(statement)
    #            heroes = results.all()
    #            for h in heroes:
    #                print(f"  - {h.name} ({h.secret_name})")
    #
    #   4. Clean up the engine:
    #        await engine.dispose()
    #      dispose() closes all connections in the pool. Important for
    #      clean shutdown in async contexts.
    #
    # WHY THIS MATTERS:
    #   Notice the pattern is nearly identical to sync — the only differences
    #   are `await` keywords and `AsyncSession` instead of `Session`. SQLModel's
    #   design means switching from sync to async requires minimal code changes.
    #   The select() statement is IDENTICAL in both modes.
    #
    # EXPECTED RESULT:
    #   Prints the created hero and lists all heroes in the database.
    #   SQL logs show async queries via asyncpg.
    #
    # HINT:
    #   engine = await create_async_engine_and_session()
    #   async with AsyncSession(engine) as session:
    #       hero = Hero(name="Async Hero", secret_name="Eventloop", age=1)
    #       session.add(hero)
    #       await session.commit()
    #       await session.refresh(hero)
    #       print(f"Created: {hero}")
    #   async with AsyncSession(engine) as session:
    #       results = await session.exec(select(Hero))
    #       for h in results.all():
    #           print(f"  - {h.name} ({h.secret_name})")
    #   await engine.dispose()

    raise NotImplementedError("TODO(human)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=" * 60)
    print("Async SQLModel Demo")
    print("=" * 60)
    asyncio.run(async_create_and_query())
