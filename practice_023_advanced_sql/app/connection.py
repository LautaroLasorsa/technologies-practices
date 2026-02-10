"""Database connection helpers for the Advanced SQL practice.

Provides:
- get_connection(): context manager yielding a psycopg connection
- run_query(sql, params): execute SQL and print results as a formatted table
- run_explain(sql, params): run EXPLAIN (ANALYZE, BUFFERS) and print the plan
"""

from contextlib import contextmanager
from typing import Any

import psycopg


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "practice",
    "password": "practice",
    "dbname": "northwind",
}


@contextmanager
def get_connection():
    """Yield a psycopg connection, auto-closing on exit."""
    conn = psycopg.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


def run_query(sql: str, params: dict[str, Any] | tuple | None = None) -> list[tuple]:
    """Execute a SQL query and print results as a formatted table.

    Returns the list of rows for programmatic use.
    If the SQL string is empty, prints a reminder and returns [].
    """
    if not sql or not sql.strip():
        print("  [empty query — implement the TODO(human) first]")
        return []

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)

            # DDL / DML without results
            if cur.description is None:
                conn.commit()
                print(f"  OK — {cur.statusmessage}")
                return []

            columns = [desc.name for desc in cur.description]
            rows = cur.fetchall()

            _print_table(columns, rows)
            return rows


def run_explain(sql: str, params: dict[str, Any] | tuple | None = None) -> list[str]:
    """Run EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) on a query and print the plan.

    Returns the plan lines for programmatic use.
    If the SQL string is empty, prints a reminder and returns [].
    """
    if not sql or not sql.strip():
        print("  [empty query — implement the TODO(human) first]")
        return []

    explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {sql}"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(explain_sql, params)
            plan_rows = cur.fetchall()
            conn.rollback()  # ANALYZE actually executes — rollback to avoid side effects

    plan_lines = [row[0] for row in plan_rows]

    print()
    print("  EXPLAIN ANALYZE")
    print("  " + "-" * 68)
    for line in plan_lines:
        print(f"  {line}")
    print("  " + "-" * 68)
    print()

    return plan_lines


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def _print_table(columns: list[str], rows: list[tuple], max_col_width: int = 40) -> None:
    """Print rows as an aligned text table with auto-detected column widths."""
    if not rows:
        print("  (0 rows)")
        return

    # Convert all values to strings, truncating long ones
    str_rows = []
    for row in rows:
        str_row = []
        for val in row:
            s = str(val) if val is not None else "NULL"
            if len(s) > max_col_width:
                s = s[: max_col_width - 3] + "..."
            str_row.append(s)
        str_rows.append(str_row)

    # Compute column widths (header vs data)
    col_widths = []
    for i, col_name in enumerate(columns):
        max_data = max((len(row[i]) for row in str_rows), default=0)
        col_widths.append(max(len(col_name), max_data))

    # Header
    header = " | ".join(name.ljust(col_widths[i]) for i, name in enumerate(columns))
    separator = "-+-".join("-" * col_widths[i] for i in range(len(columns)))

    print()
    print(f"  {header}")
    print(f"  {separator}")

    # Data rows
    for row in str_rows:
        line = " | ".join(row[i].ljust(col_widths[i]) for i in range(len(columns)))
        print(f"  {line}")

    print(f"\n  ({len(rows)} row{'s' if len(rows) != 1 else ''})")
    print()
