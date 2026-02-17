"""Benchmark Kafka compression codecs: none, snappy, lz4, zstd, gzip.

Demonstrates:
  - How compression.type affects producer throughput
  - Trade-offs between CPU usage, compression ratio, and latency
  - Codec selection guidance for different workloads

Run after admin.py has created the topics:
    uv run python compression_benchmark.py
"""

import json
import random
import string
import time

from confluent_kafka import KafkaError, Producer

import config


# -- Helpers (fully implemented) ----------------------------------------------

# Sample field values for generating realistic JSON messages
_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank"]
_ACTIONS = ["login", "logout", "purchase", "view", "click", "search", "scroll", "submit"]
_URLS = ["/home", "/products", "/cart", "/checkout", "/profile", "/settings", "/search", "/help"]
_STATUSES = ["active", "inactive", "pending", "completed", "cancelled"]


def generate_message(size: int) -> str:
    """Generate a realistic JSON-like message of approximately the given size.

    Produces a JSON object with typical event fields: user_id, action, url,
    timestamp, status, metadata. Pads with a 'payload' field to reach the
    target size. Real-world Kafka messages are typically JSON, Avro, or
    Protobuf -- JSON compresses well because of repeated field names and
    structural characters.

    Args:
        size: Approximate target size in bytes for the serialized JSON string.

    Returns:
        A JSON string of approximately `size` bytes.
    """
    base = {
        "user_id": f"user-{random.randint(1, 10000):05d}",
        "action": random.choice(_ACTIONS),
        "url": random.choice(_URLS),
        "timestamp": time.time(),
        "status": random.choice(_STATUSES),
        "name": random.choice(_NAMES),
        "session_id": "".join(random.choices(string.hexdigits, k=16)),
        "metadata": {
            "browser": random.choice(["Chrome", "Firefox", "Safari", "Edge"]),
            "os": random.choice(["Windows", "macOS", "Linux", "iOS", "Android"]),
            "version": f"{random.randint(1, 120)}.{random.randint(0, 9)}.{random.randint(0, 99)}",
        },
    }

    base_str = json.dumps(base)
    if len(base_str) >= size:
        return base_str[:size]

    # Pad with a 'payload' field containing repeated characters (compresses well,
    # simulating the redundancy in real JSON event streams)
    padding_needed = size - len(base_str) - len(', "payload": ""')
    if padding_needed > 0:
        # Mix of repeated patterns (compressible) and some randomness
        pattern = "abcdefghij" * (padding_needed // 10 + 1)
        base["payload"] = pattern[:padding_needed]

    return json.dumps(base)


def _silent_delivery(err, msg) -> None:
    """Silent delivery callback -- only logs errors, not successes."""
    if err is not None:
        print(f"  Delivery FAILED: {err}")


# -- TODO(human): Implement these functions -----------------------------------


def benchmark_codec(
    codec: str, topic: str, num_messages: int, message_size: int
) -> dict:
    """Benchmark a single compression codec by producing messages and measuring throughput.

    TODO(human): Implement this function.

    Background -- how Kafka compression works:
      The producer compresses messages in batches (not individually). When you
      set compression.type on the producer, each batch of messages sent to a
      partition is compressed as a unit before being sent over the network.
      The broker stores the compressed batch as-is (no re-compression by
      default). Consumers decompress when fetching. This means:
        - Larger batches = better compression ratio (more data to find patterns)
        - linger.ms > 0 helps batch more messages before compressing
        - The codec choice affects CPU on producer and consumer, network
          bandwidth, and broker disk usage

      Codec comparison:
        - none:   No compression. Baseline. Lowest CPU, highest bandwidth.
        - snappy: Fast compression/decompression, moderate ratio. Good default.
        - lz4:    Fastest compression, slightly less ratio than snappy. Best for latency.
        - zstd:   Best compression ratio, moderate speed. Best for bandwidth savings.
        - gzip:   Good ratio but SLOW. Avoid for latency-sensitive workloads.

    Steps:
      1. Create a Producer with:
         - "bootstrap.servers": config.BOOTSTRAP_SERVERS
         - "compression.type": codec
         - "linger.ms": 10  (allow batching for realistic compression)
         - "batch.size": 65536
         - "client.id": f"bench-{codec}"
      2. Generate messages using generate_message(message_size).
      3. Record start time.
      4. Produce num_messages to the topic. Use _silent_delivery as callback.
         Call producer.poll(0) periodically (e.g., every 1000 messages) to
         process callbacks without blocking.
      5. Call producer.flush(timeout=30).
      6. Record end time.
      7. Calculate elapsed seconds and throughput (messages per second).
      8. Return a dict: {"codec": codec, "messages": num_messages,
         "elapsed_s": float, "throughput_msg_s": float,
         "message_size": message_size}

    Args:
        codec: Compression type: "none", "snappy", "lz4", "zstd", or "gzip".
        topic: Topic to produce to.
        num_messages: Number of messages to produce.
        message_size: Approximate size of each message in bytes.

    Returns:
        Dict with benchmark results.
    """
    raise NotImplementedError("TODO(human)")


def run_compression_comparison(topic: str) -> None:
    """Run benchmark for all codecs and print a comparison table.

    TODO(human): Implement this function.

    Steps:
      1. Define the list of codecs: ["none", "snappy", "lz4", "zstd", "gzip"].
      2. Choose benchmark parameters:
         - num_messages = 50000 (enough for meaningful measurement)
         - message_size = 512 (bytes, realistic for event payloads)
      3. For each codec, call benchmark_codec() and collect results.
         Print progress: "Benchmarking {codec}..."
      4. Print a comparison table with columns:
         Codec | Messages | Msg Size | Elapsed (s) | Throughput (msg/s)
      5. Print which codec had the highest throughput and a brief recommendation.

    Note: Results will vary by machine. The relative ordering is what matters:
      - lz4/snappy should be fastest (near-zero CPU overhead)
      - zstd should be slightly slower but with best compression
      - gzip should be notably the slowest
      - "none" is the baseline -- any codec overhead shows as lower throughput

    Args:
        topic: Topic to produce to (should have multiple partitions for parallelism).
    """
    raise NotImplementedError("TODO(human)")


# -- Orchestration (boilerplate) ----------------------------------------------


def main() -> None:
    print("=== Compression Codec Benchmark ===\n")
    run_compression_comparison(config.COMPRESSION_TOPIC)


if __name__ == "__main__":
    main()
