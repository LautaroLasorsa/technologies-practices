"""Alembic migration environment for SQLAlchemy 2.0 Async practice.

This file is fully implemented boilerplate. Alembic uses it to:
1. Connect to the database (via sync psycopg2 driver -- Alembic doesn't need async)
2. Compare current DB schema against SQLAlchemy model metadata
3. Autogenerate migration scripts that add/remove/alter columns & tables

Key concept: Alembic runs migrations via a SYNC connection, even though the
application uses async. This is standard -- migrations are a one-time offline
operation, not a hot-path. The alembic.ini configures psycopg2 (sync), while
the app uses asyncpg (async).
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import Base so Alembic can see all models via Base.metadata.
# This import triggers model registration (Department, Employee, Project, etc.).
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.models import Base  # noqa: E402

# Alembic Config object -- provides access to alembic.ini values.
config = context.config

# Set up Python logging from alembic.ini [loggers] section.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate support.
# Alembic compares this metadata against the actual DB schema to produce diffs.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Generates SQL scripts without connecting to the database.
    Useful for review or applying in environments without direct DB access.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Connects to the database and applies migrations directly.
    Uses the sync psycopg2 driver (configured in alembic.ini).
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
