"""Dagster Resources -- dependency injection for external services.

PARADIGM COMPARISON WITH AIRFLOW:
  In Airflow, you configure "Connections" in the UI (Admin > Connections) and access
  them via hooks: `PostgresHook(postgres_conn_id="my_db").get_conn()`.

  In Dagster, you define Resources as Python classes with typed configuration.
  Resources are injected into assets via function parameters -- like FastAPI's Depends().

  Benefits of Dagster's approach:
  - Type-safe: IDE autocompletion, mypy/pyright can check resource usage
  - Testable: Swap real resources for mocks in tests without UI configuration
  - Explicit: Every asset declares which resources it needs in its signature
  - No global state: Resources are scoped to the Definitions object, not a global DB

This module is FULLY IMPLEMENTED -- no TODOs here.
Resources are infrastructure; the interesting learning is in the assets that use them.
"""

from contextlib import contextmanager
from typing import Generator

import psycopg2
from dagster import ConfigurableResource, EnvVar


class PostgresResource(ConfigurableResource):
    """A Dagster resource that provides PostgreSQL database connections.

    COMPARISON WITH AIRFLOW:
      Airflow: PostgresHook(postgres_conn_id="my_conn").get_conn()
      Dagster: def my_asset(postgres: PostgresResource): ...
               with postgres.get_connection() as conn: ...

    The resource is configured once in the Definitions object and injected
    everywhere it's needed. No connection IDs, no UI configuration, no
    runtime surprises from mistyped connection names.
    """

    host: str = "postgres"
    port: int = 5432
    user: str = "dagster"
    password: str = "dagster"
    dbname: str = "dagster"

    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Get a PostgreSQL connection as a context manager.

        Usage:
            with postgres.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM weather_readings")
                    rows = cur.fetchall()
        """
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.dbname,
        )
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute_query(self, query: str, params: tuple | None = None) -> list[tuple]:
        """Execute a SELECT query and return all rows."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()

    def execute_command(self, query: str, params: tuple | None = None) -> None:
        """Execute an INSERT/UPDATE/DELETE/DDL command."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)

    def ensure_table(self, table_name: str, schema: str) -> None:
        """Create a table if it doesn't exist.

        Args:
            table_name: Name of the table to create.
            schema: SQL column definitions, e.g. "id SERIAL PRIMARY KEY, name TEXT".
        """
        self.execute_command(
            f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})"
        )
