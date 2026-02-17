"""Main entry point — runs all SQLAlchemy 2.0 Async practice demos in sequence.

Usage:
    docker compose up -d
    uv run python -m app.main

Each phase can also be run independently:
    uv run python -m app.queries_basic
    uv run python -m app.queries_advanced
    uv run python -m app.loading_strategies
    uv run python -m app.migrations_demo
"""

import asyncio

from app.engine import create_engine_factory, create_session_factory
from app.loading_strategies import run_loading_demo
from app.migrations_demo import run_migrations_demo
from app.queries_advanced import run_advanced_queries_demo
from app.queries_basic import run_basic_queries_demo


async def run_all() -> None:
    """Execute all practice phases sequentially."""
    engine = create_engine_factory()
    session_factory = create_session_factory(engine)

    print("=" * 70)
    print("SQLAlchemy 2.0 Async ORM & Advanced Queries")
    print("=" * 70)
    print(f"\nEngine: {engine.url}")
    print(f"Driver: {engine.dialect.name}+{engine.dialect.driver}")

    # Phase 2: Basic CRUD
    async with session_factory() as session:
        await run_basic_queries_demo(session)

    # Phase 3: Advanced queries (fresh session for clean state)
    async with session_factory() as session:
        await run_advanced_queries_demo(session)

    # Phase 4: Loading strategies (fresh session -- no cached objects)
    async with session_factory() as session:
        await run_loading_demo(session)

    # Phase 5: Alembic migrations
    async with session_factory() as session:
        await run_migrations_demo(session)

    # Cleanup: close all pooled connections
    await engine.dispose()
    print("\n" + "=" * 70)
    print("All demos complete. Engine disposed.")
    print("=" * 70)


def main() -> None:
    """Entry point for `python -m app.main`."""
    asyncio.run(run_all())


if __name__ == "__main__":
    main()
