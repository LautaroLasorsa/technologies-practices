"""
Check ETL Results — Query PostgreSQL
======================================

Connects to the Docker PostgreSQL instance and queries the weather_summary table
to verify that the Phase 4 ETL pipeline loaded data correctly.

Usage:
    uv run python scripts/check_results.py

Prerequisites:
    - Docker Compose stack must be running (docker compose up -d)
    - Phase 4 ETL pipeline must have been triggered and completed
    - The weather_summary table must exist (created by the DAG)

Connection:
    PostgreSQL is exposed on localhost:5433 (mapped from container port 5432)
    User: airflow, Password: airflow, Database: airflow
"""

import sys

import psycopg


def check_table_exists(conn: psycopg.Connection) -> bool:
    """Check if the weather_summary table exists."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'weather_summary'
            )
        """)
        row = cur.fetchone()
        return row[0] if row else False


def print_summary(conn: psycopg.Connection) -> None:
    """Print a summary of all data in weather_summary."""
    with conn.cursor() as cur:
        # Row count
        cur.execute("SELECT COUNT(*) FROM weather_summary")
        row = cur.fetchone()
        count = row[0] if row else 0
        print(f"\nTotal rows in weather_summary: {count}")

        if count == 0:
            print("No data found. Has the Phase 4 DAG been triggered?")
            return

        # Distinct dates
        cur.execute("SELECT DISTINCT date FROM weather_summary ORDER BY date")
        dates = [row[0] for row in cur.fetchall()]
        print(f"Dates loaded: {', '.join(str(d) for d in dates)}")

        # Summary by city
        print(f"\n{'City':<20} {'Dates':<8} {'Avg Temp (C)':<14} {'Avg Humidity':<14} {'Readings':<10}")
        print("-" * 70)

        cur.execute("""
            SELECT
                city,
                COUNT(DISTINCT date) as date_count,
                ROUND(AVG(avg_temp_c)::numeric, 1) as overall_avg_temp,
                ROUND(AVG(avg_humidity)::numeric, 1) as overall_avg_humidity,
                SUM(reading_count) as total_readings
            FROM weather_summary
            GROUP BY city
            ORDER BY city
        """)

        for row in cur.fetchall():
            city, date_count, avg_temp, avg_humidity, readings = row
            print(f"{city:<20} {date_count:<8} {avg_temp:<14} {avg_humidity:<14} {readings:<10}")

        # Most recent load
        cur.execute("SELECT MAX(loaded_at) FROM weather_summary")
        row = cur.fetchone()
        last_load = row[0] if row else None
        print(f"\nMost recent load: {last_load}")


def print_recent_data(conn: psycopg.Connection) -> None:
    """Print the most recent day's data in detail."""
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(date) FROM weather_summary")
        row = cur.fetchone()
        if not row or not row[0]:
            return

        latest_date = row[0]
        print(f"\n--- Detailed data for {latest_date} ---")
        print(f"{'City':<20} {'Avg C':<8} {'Min C':<8} {'Max C':<8} {'Humidity':<10} {'Readings':<10}")
        print("-" * 64)

        cur.execute("""
            SELECT city, avg_temp_c, min_temp_c, max_temp_c, avg_humidity, reading_count
            FROM weather_summary
            WHERE date = %s
            ORDER BY city
        """, (latest_date,))

        for row in cur.fetchall():
            city, avg_c, min_c, max_c, humidity, readings = row
            print(f"{city:<20} {avg_c:<8.1f} {min_c:<8.1f} {max_c:<8.1f} {humidity:<10.1f} {readings:<10}")


def main() -> None:
    # Connect to PostgreSQL exposed on host port 5433
    conn_params = {
        "host": "localhost",
        "port": 5433,
        "dbname": "airflow",
        "user": "airflow",
        "password": "airflow",
    }

    print("Connecting to PostgreSQL (localhost:5433)...")

    try:
        conn = psycopg.connect(**conn_params)
    except psycopg.OperationalError as e:
        print(f"ERROR: Cannot connect to PostgreSQL: {e}")
        print("\nIs the Docker Compose stack running? Try: docker compose up -d")
        sys.exit(1)

    try:
        if not check_table_exists(conn):
            print("Table 'weather_summary' does not exist yet.")
            print("Trigger the Phase 4 DAG first to create it.")
            sys.exit(0)

        print_summary(conn)
        print_recent_data(conn)
    finally:
        conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
