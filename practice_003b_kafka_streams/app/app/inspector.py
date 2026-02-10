"""Topic inspector: reads messages from any topic for debugging.

Consumes and prints messages from a specified Kafka topic. Useful for
verifying that agents are producing the expected output.

Usage:
    uv run python -m app.inspector enriched-readings
    uv run python -m app.inspector window-aggregates --limit 5
    uv run python -m app.inspector sensor-dead-letter

This is debugging tooling -- fully implemented (no TODOs).
"""

import argparse
import asyncio
import json

from aiokafka import AIOKafkaConsumer


async def inspect_topic(topic: str, limit: int, timeout: float) -> None:
    """Consume and print messages from a Kafka topic."""
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers="localhost:9094",
        auto_offset_reset="earliest",
        group_id=f"inspector-{topic}",
        value_deserializer=lambda v: try_decode(v),
    )
    await consumer.start()
    print(f"Inspecting topic '{topic}' (limit={limit}, timeout={timeout}s)...")
    print("-" * 60)

    count = 0
    try:
        while count < limit:
            batch = await consumer.getmany(timeout_ms=int(timeout * 1000))
            if not batch:
                print("(no more messages)")
                break
            for tp, messages in batch.items():
                for msg in messages:
                    count += 1
                    print(f"[{count}] partition={tp.partition} offset={msg.offset}")
                    if isinstance(msg.value, dict):
                        print(f"    {json.dumps(msg.value, indent=2)}")
                    else:
                        print(f"    {msg.value}")
                    if count >= limit:
                        break
    finally:
        await consumer.stop()

    print("-" * 60)
    print(f"Total messages read: {count}")


def try_decode(raw: bytes):
    """Attempt JSON decode, fall back to string."""
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return raw.decode("utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect messages on a Kafka topic.")
    parser.add_argument("topic", help="Topic name to consume from")
    parser.add_argument("--limit", type=int, default=20, help="Max messages to read (default: 20)")
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for messages (default: 5.0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(inspect_topic(args.topic, args.limit, args.timeout))


if __name__ == "__main__":
    main()
