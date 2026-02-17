"""Batch size and linger.ms tuning experiments.

Demonstrates:
  - How linger.ms and batch.size interact to control batching behavior
  - The throughput vs latency trade-off in producer configuration
  - Measuring the impact of different batch configurations

Run after admin.py has created the topics:
    uv run python batch_tuning.py
"""

import json
import time

from confluent_kafka import KafkaError, Producer

import config
from compression_benchmark import generate_message


# -- Helpers (boilerplate) ----------------------------------------------------


def _silent_delivery(err, msg) -> None:
    """Silent delivery callback -- only logs errors."""
    if err is not None:
        print(f"  Delivery FAILED: {err}")


# -- TODO(human): Implement these functions -----------------------------------


def benchmark_batch_config(
    linger_ms: int, batch_size: int, topic: str, num_messages: int
) -> dict:
    """Benchmark a specific linger.ms + batch.size combination.

    TODO(human): Implement this function.

    Background -- how linger.ms and batch.size control batching:
      The producer accumulates messages in an internal buffer, organized by
      partition. A batch is sent to the broker when EITHER:
        (a) The batch reaches batch.size bytes, OR
        (b) linger.ms milliseconds have elapsed since the first message
            was added to the batch
      whichever comes first.

      - linger.ms=0 (default): Send immediately. Lowest latency but smallest
        batches (often 1 message per request), which means more network
        round-trips and lower throughput.
      - linger.ms=10-100: Wait up to N ms to accumulate more messages per batch.
        Higher throughput (fewer requests, better compression, less overhead)
        at the cost of added latency.
      - batch.size: Maximum bytes per batch. Default is 16384 (16 KB). Larger
        batches amortize per-request overhead. If batch.size fills before
        linger.ms expires, the batch is sent immediately.

      The sweet spot depends on your workload:
        - High-throughput event streams: linger.ms=50-200, batch.size=131072+
        - Low-latency systems (e.g., HFT): linger.ms=0, batch.size=16384

    Steps:
      1. Create a Producer with:
         - "bootstrap.servers": config.BOOTSTRAP_SERVERS
         - "linger.ms": linger_ms
         - "batch.size": batch_size
         - "compression.type": "lz4"  (keep compression constant to isolate batching)
         - "client.id": f"batch-{linger_ms}-{batch_size}"
      2. Generate a pool of messages using generate_message(256) -- generate
         once and reuse to avoid message generation overhead affecting the benchmark.
      3. Record start time.
      4. Produce num_messages to the topic. Use _silent_delivery callback.
         Call producer.poll(0) every 1000 messages.
      5. Call producer.flush(timeout=30).
      6. Record end time and calculate elapsed/throughput.
      7. Return: {"linger_ms": linger_ms, "batch_size": batch_size,
         "messages": num_messages, "elapsed_s": float,
         "throughput_msg_s": float}

    Args:
        linger_ms: Producer linger.ms setting.
        batch_size: Producer batch.size setting (bytes).
        topic: Topic to produce to.
        num_messages: Number of messages to produce.

    Returns:
        Dict with benchmark results.
    """
    raise NotImplementedError("TODO(human)")


def run_batch_comparison(topic: str) -> None:
    """Test a matrix of linger.ms + batch.size configurations and compare.

    TODO(human): Implement this function.

    Steps:
      1. Define the test matrix as a list of (linger_ms, batch_size) tuples:
         [(0, 16384), (10, 32768), (50, 65536), (100, 131072), (200, 262144)]
         These progress from "send immediately, small batch" to "wait longer,
         big batch" to show the throughput/latency spectrum.
      2. Set num_messages = 50000 for each test.
      3. For each (linger_ms, batch_size) in the matrix:
         a. Print "Testing linger.ms={linger_ms}, batch.size={batch_size}..."
         b. Call benchmark_batch_config() and collect the result.
      4. Print a comparison table with columns:
         linger.ms | batch.size | Messages | Elapsed (s) | Throughput (msg/s)
      5. Identify and print the configuration with the highest throughput.
      6. Print a brief explanation of the results:
         - Why does increasing linger.ms improve throughput?
         - Why do diminishing returns set in at very high linger.ms?

    Args:
        topic: Topic to produce to (6 partitions for parallelism).
    """
    raise NotImplementedError("TODO(human)")


# -- Orchestration (boilerplate) ----------------------------------------------


def main() -> None:
    print("=== Batch Tuning Experiments ===\n")
    run_batch_comparison(config.CONSUMER_TUNING_TOPIC)


if __name__ == "__main__":
    main()
