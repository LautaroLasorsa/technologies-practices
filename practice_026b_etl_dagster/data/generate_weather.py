"""Mock Weather Data Generator for Dagster Practice.

Generates realistic weather readings and inserts them directly into PostgreSQL.
This script is run inside the Docker container to populate the raw_weather_readings
table that Dagster assets consume.

Usage:
    # Generate 7 days of data (default)
    python /app/data/generate_weather.py

    # Generate 30 days of historical data
    python /app/data/generate_weather.py --days 30

    # Generate data for specific cities
    python /app/data/generate_weather.py --days 14 --cities "New York,London,Tokyo"

The data includes:
    - Multiple cities with realistic temperature ranges
    - 24 readings per city per day (one per hour)
    - Occasional invalid readings (~5%) to test validation logic
    - Timestamps for each reading
"""

import argparse
import random
import sys
from datetime import datetime, timedelta
from typing import NamedTuple

import psycopg2


class CityProfile(NamedTuple):
    """Weather characteristics for a city."""

    name: str
    base_temp: float  # Average temperature in Celsius
    temp_variance: float  # Daily temperature swing
    base_humidity: float  # Average humidity percentage
    base_wind: float  # Average wind speed km/h


# Realistic city profiles with different climates
CITY_PROFILES = [
    CityProfile("New York", 12.0, 15.0, 65.0, 18.0),
    CityProfile("London", 10.0, 8.0, 78.0, 15.0),
    CityProfile("Tokyo", 16.0, 12.0, 60.0, 12.0),
    CityProfile("Sydney", 22.0, 10.0, 55.0, 20.0),
    CityProfile("Mumbai", 28.0, 8.0, 72.0, 14.0),
    CityProfile("Moscow", -2.0, 18.0, 70.0, 16.0),
    CityProfile("Cairo", 25.0, 15.0, 35.0, 10.0),
    CityProfile("Sao Paulo", 20.0, 10.0, 68.0, 11.0),
]

# Percentage of readings that are intentionally invalid (for testing validation)
INVALID_READING_RATE = 0.05


def connect_to_postgres() -> psycopg2.extensions.connection:
    """Connect to the PostgreSQL database."""
    return psycopg2.connect(
        host="postgres",
        port=5432,
        user="dagster",
        password="dagster",
        dbname="dagster",
    )


def create_table(conn: psycopg2.extensions.connection) -> None:
    """Create the raw_weather_readings table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_weather_readings (
                id SERIAL PRIMARY KEY,
                city TEXT NOT NULL,
                temperature_c REAL NOT NULL,
                humidity_pct REAL NOT NULL,
                wind_speed_kmh REAL NOT NULL,
                reading_date DATE NOT NULL,
                recorded_at TIMESTAMP DEFAULT NOW()
            )
        """)
    conn.commit()


def generate_reading(
    city: CityProfile,
    date: datetime,
    hour: int,
) -> tuple[str, float, float, float, str]:
    """Generate a single weather reading for a city at a specific hour.

    Returns (city_name, temperature, humidity, wind_speed, date_str).
    Approximately 5% of readings are intentionally invalid.
    """
    # Simulate daily temperature cycle: coldest at 4am, warmest at 2pm
    hour_factor = -0.5 * ((hour - 14) ** 2) / 50.0 + 0.5
    temperature = city.base_temp + city.temp_variance * hour_factor + random.gauss(0, 2.0)
    humidity = city.base_humidity + random.gauss(0, 8.0)
    wind = city.base_wind + random.gauss(0, 5.0)

    # Inject invalid readings (~5%) for testing validation logic
    if random.random() < INVALID_READING_RATE:
        fault_type = random.choice(["extreme_temp", "bad_humidity", "negative_wind"])
        if fault_type == "extreme_temp":
            temperature = random.choice([-80.0, 75.0, 100.0])
        elif fault_type == "bad_humidity":
            humidity = random.choice([-10.0, 120.0, 200.0])
        else:
            wind = random.uniform(-50.0, -1.0)

    return (
        city.name,
        round(temperature, 1),
        round(max(0, min(100, humidity)), 1),  # Clamp only non-fault humidity
        round(wind, 1),
        date.strftime("%Y-%m-%d"),
    )


def generate_and_insert(
    conn: psycopg2.extensions.connection,
    days: int,
    cities: list[CityProfile],
) -> int:
    """Generate weather data and insert into PostgreSQL.

    Returns the number of rows inserted.
    """
    today = datetime.now()
    start_date = today - timedelta(days=days)
    total_inserted = 0

    with conn.cursor() as cur:
        for day_offset in range(days):
            current_date = start_date + timedelta(days=day_offset)

            batch = []
            for city in cities:
                for hour in range(24):
                    reading = generate_reading(city, current_date, hour)
                    batch.append(reading)

            # Batch insert for performance
            args_str = ",".join(
                cur.mogrify("(%s, %s, %s, %s, %s)", row).decode("utf-8")
                for row in batch
            )
            cur.execute(
                f"INSERT INTO raw_weather_readings "
                f"(city, temperature_c, humidity_pct, wind_speed_kmh, reading_date) "
                f"VALUES {args_str}"
            )
            total_inserted += len(batch)

            if (day_offset + 1) % 7 == 0:
                print(f"  Generated {day_offset + 1}/{days} days ({total_inserted} readings)")

    conn.commit()
    return total_inserted


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate mock weather data for Dagster practice"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days of historical data to generate (default: 7)",
    )
    parser.add_argument(
        "--cities",
        type=str,
        default=None,
        help="Comma-separated city names to generate (default: all 8 cities)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before generating new data",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Select cities
    if args.cities:
        city_names = {name.strip() for name in args.cities.split(",")}
        cities = [c for c in CITY_PROFILES if c.name in city_names]
        if not cities:
            print(f"Error: No matching cities found. Available: {[c.name for c in CITY_PROFILES]}")
            sys.exit(1)
    else:
        cities = list(CITY_PROFILES)

    print(f"Connecting to PostgreSQL...")
    conn = connect_to_postgres()

    print(f"Creating table if needed...")
    create_table(conn)

    if args.clear:
        print("Clearing existing data...")
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE raw_weather_readings RESTART IDENTITY")
        conn.commit()

    print(f"Generating {args.days} days of weather data for {len(cities)} cities...")
    print(f"Cities: {', '.join(c.name for c in cities)}")
    print(f"Readings per day per city: 24 (one per hour)")
    print(f"Expected total: {args.days * len(cities) * 24} readings")
    print(f"Invalid reading rate: {INVALID_READING_RATE * 100:.0f}%")
    print()

    total = generate_and_insert(conn, args.days, cities)

    # Verify
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM raw_weather_readings")
        db_count = cur.fetchone()[0]

    print()
    print(f"Done! Inserted {total} readings.")
    print(f"Total rows in raw_weather_readings: {db_count}")
    conn.close()


if __name__ == "__main__":
    main()
