"""Quick-start entry point -- runs setup then prints next steps.

Usage:
    uv run python main.py
"""

import setup_resources


def main() -> None:
    print("=" * 60)
    print("  Practice 002: Cloud Pub/Sub -- Order Processing System")
    print("=" * 60)

    setup_resources.main()

    print("\n" + "=" * 60)
    print("  Next steps:")
    print("  1. uv run python publisher.py          (publish orders)")
    print("  2. uv run python subscriber_pull.py     (sync pull)")
    print("  3. uv run python subscriber_streaming.py (streaming pull)")
    print("  4. uv run python test_fanout.py          (fan-out test)")
    print("  5. uv run python subscriber_deadletter.py (dead letter)")
    print("=" * 60)


if __name__ == "__main__":
    main()
