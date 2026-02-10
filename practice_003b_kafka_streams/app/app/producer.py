"""Sensor data producer: generates simulated sensor readings.

Sends SensorReading events to the sensor-readings topic at a
configurable rate. Includes both valid readings and occasional
malformed events (for testing dead-letter routing).

Usage (from the app/ directory):
    uv run python -m app.producer
    uv run python -m app.producer --count 50 --interval 0.5
    uv run python -m app.producer --include-bad

This is infrastructure/test tooling -- fully implemented (no TODOs).
"""

import argparse
import json
import random
import sys
import time

from aiokafka import AIOKafkaProducer
import asyncio

from app import config


# ── Sensor simulation data ───────────────────────────────────────────

SENSORS = [
    {"sensor_id": "sensor-01", "location": "warehouse-A", "base_temp": 22.0, "base_hum": 45.0},
    {"sensor_id": "sensor-02", "location": "warehouse-B", "base_temp": 18.0, "base_hum": 55.0},
    {"sensor_id": "sensor-03", "location": "cold-storage", "base_temp": 2.0,  "base_hum": 70.0},
    {"sensor_id": "sensor-04", "location": "server-room",  "base_temp": 28.0, "base_hum": 30.0},
    {"sensor_id": "sensor-05", "location": "rooftop",      "base_temp": 15.0, "base_hum": 60.0},
]

# Malformed events for testing dead-letter routing
BAD_EVENTS = [
    {"sensor_id": "", "temperature": 20.0, "humidity": 50.0, "timestamp": 0, "location": "bad-1"},
    {"sensor_id": "sensor-bad", "temperature": -100.0, "humidity": 50.0, "timestamp": 0, "location": "bad-2"},
    {"sensor_id": "sensor-bad", "temperature": 20.0, "humidity": 150.0, "timestamp": 0, "location": "bad-3"},
]


# ── Data generation ──────────────────────────────────────────────────


def generate_reading(sensor: dict) -> dict:
    """Generate a single plausible sensor reading with random jitter."""
    temp_jitter = random.gauss(0, 3.0)
    hum_jitter = random.gauss(0, 5.0)
    return {
        "sensor_id": sensor["sensor_id"],
        "temperature": round(sensor["base_temp"] + temp_jitter, 2),
        "humidity": round(max(0.0, min(100.0, sensor["base_hum"] + hum_jitter)), 2),
        "timestamp": time.time(),
        "location": sensor["location"],
    }


def generate_bad_reading() -> dict:
    """Pick a random malformed event and set its timestamp."""
    event = random.choice(BAD_EVENTS).copy()
    event["timestamp"] = time.time()
    return event


# ── Producer ─────────────────────────────────────────────────────────


async def produce_readings(count: int, interval: float, include_bad: bool) -> None:
    """Produce sensor readings to Kafka."""
    producer = AIOKafkaProducer(
        bootstrap_servers="localhost:9094",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    await producer.start()
    print(f"Connected to Kafka. Producing {count} readings (interval={interval}s)...")

    try:
        for i in range(count):
            if include_bad and random.random() < 0.1:
                reading = generate_bad_reading()
                label = "BAD"
            else:
                sensor = random.choice(SENSORS)
                reading = generate_reading(sensor)
                label = reading["sensor_id"]

            await producer.send_and_wait(config.SENSOR_READINGS_TOPIC, value=reading)
            print(f"  [{i + 1}/{count}] Sent {label}: temp={reading['temperature']}, hum={reading['humidity']}")

            if interval > 0 and i < count - 1:
                await asyncio.sleep(interval)
    finally:
        await producer.stop()

    print(f"\nDone. Produced {count} readings to '{config.SENSOR_READINGS_TOPIC}'.")


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce simulated sensor readings to Kafka.")
    parser.add_argument("--count", type=int, default=30, help="Number of readings to produce (default: 30)")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between readings (default: 1.0)")
    parser.add_argument("--include-bad", action="store_true", help="Include ~10%% malformed events for dead-letter testing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(produce_readings(args.count, args.interval, args.include_bad))


if __name__ == "__main__":
    main()
