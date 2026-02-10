"""
Mock Weather Data Generator
============================

Generates JSON files with simulated weather readings for multiple cities.
These files are consumed by the Phase 4 ETL pipeline (FileSensor → extract → transform → load).

Usage:
    uv run python data/mock_weather.py              # Generate data for today
    uv run python data/mock_weather.py --days 7     # Generate 7 days of data
    uv run python data/mock_weather.py --date 2024-01-15  # Generate for a specific date

Output:
    data/weather_YYYY-MM-DD.json

Each file contains readings for 5 cities at 4 times of day (6am, 12pm, 6pm, 12am),
for a total of 20 readings per file.
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path


# ---- City profiles: base temp (°F) and humidity (%) for realistic-ish data ----
CITIES = {
    "New York": {"base_temp": 40, "base_humidity": 60, "temp_range": 25},
    "Los Angeles": {"base_temp": 68, "base_humidity": 35, "temp_range": 15},
    "Chicago": {"base_temp": 32, "base_humidity": 65, "temp_range": 30},
    "Miami": {"base_temp": 78, "base_humidity": 75, "temp_range": 10},
    "Seattle": {"base_temp": 48, "base_humidity": 70, "temp_range": 12},
}

# Reading times (hours of day)
READING_HOURS = [6, 12, 18, 0]


def generate_reading(city: str, profile: dict, date: datetime, hour: int) -> dict:
    """Generate a single weather reading with realistic-ish random variation."""
    # Temperature varies by time of day: cooler at 6am/12am, warmer at 12pm/6pm
    time_factor = {6: -0.3, 12: 0.4, 18: 0.2, 0: -0.5}[hour]
    base = profile["base_temp"]
    variation = profile["temp_range"]
    temp_f = round(base + time_factor * variation + random.gauss(0, 3), 1)

    # Humidity inversely correlated with temperature (roughly)
    humidity = round(
        profile["base_humidity"] + random.gauss(0, 5) - time_factor * 10, 1
    )
    humidity = max(10.0, min(100.0, humidity))

    # Wind speed: random, slightly higher in afternoon
    wind_mph = round(abs(random.gauss(8 + time_factor * 3, 4)), 1)

    timestamp = date.replace(hour=hour, minute=0, second=0)

    return {
        "city": city,
        "timestamp": timestamp.isoformat(),
        "temp_f": temp_f,
        "humidity": humidity,
        "wind_mph": wind_mph,
    }


def generate_day(date: datetime) -> dict:
    """Generate all weather readings for a single day."""
    readings = []
    for city, profile in CITIES.items():
        for hour in READING_HOURS:
            readings.append(generate_reading(city, profile, date, hour))

    return {
        "date": date.strftime("%Y-%m-%d"),
        "readings": readings,
        "metadata": {
            "source": "mock_weather_api",
            "generated_at": datetime.now().isoformat(),
            "cities": list(CITIES.keys()),
            "readings_per_city": len(READING_HOURS),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock weather data")
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of days to generate (default: 1, starting from today)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Specific date to generate (YYYY-MM-DD). Overrides --days.",
    )
    args = parser.parse_args()

    # Determine output directory (data/ in the practice root)
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.date:
        dates = [datetime.strptime(args.date, "%Y-%m-%d")]
    else:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        dates = [today - timedelta(days=i) for i in range(args.days)]

    for date in dates:
        data = generate_day(date)
        filename = f"weather_{date.strftime('%Y-%m-%d')}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        n_readings = len(data["readings"])
        n_cities = len(CITIES)
        print(f"Generated {filepath.name}: {n_readings} readings across {n_cities} cities")

    print(f"\nFiles written to: {output_dir.resolve()}")
    print("These files will be visible inside the Airflow container at /opt/airflow/data/")


if __name__ == "__main__":
    main()
