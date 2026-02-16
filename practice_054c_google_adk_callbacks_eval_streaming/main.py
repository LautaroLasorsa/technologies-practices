"""CLI entry point — choose which exercise to run.

Usage:
    uv run python main.py
"""

from __future__ import annotations

import asyncio


def print_menu() -> None:
    """Display the exercise selection menu."""
    print()
    print("=" * 60)
    print("  Practice 054c — Callbacks, Evaluation & Streaming")
    print("=" * 60)
    print()
    print("  1) Exercise 1  — Logging Before/After Tool Callbacks")
    print("  2) Exercise 2  — Caching & Permission Callbacks")
    print("  3) Exercise 3  — Golden Dataset Evaluation")
    print("  4) Exercise 4  — Streaming Responses (interactive)")
    print("  5) Exercise 5  — Memory Service Demo")
    print("  6) Run exercises 1 + 2 (all callback demos)")
    print("  q) Quit")
    print()


async def main() -> None:
    """Main event loop — prompt user for exercise choice and dispatch."""
    while True:
        print_menu()
        try:
            choice = input("Select exercise [1-6, q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if choice == "q":
            break

        if choice == "1":
            from callbacks import demo_logging_callbacks

            await demo_logging_callbacks()

        elif choice == "2":
            from callbacks import demo_caching_callbacks, demo_permission_callbacks

            await demo_caching_callbacks()
            await demo_permission_callbacks()

        elif choice == "3":
            from evaluate import GOLDEN_DATASET, run_evaluation

            if not GOLDEN_DATASET:
                print("\nERROR: GOLDEN_DATASET is empty.")
                print("Open evaluate.py and populate the GOLDEN_DATASET list first.")
            else:
                await run_evaluation()

        elif choice == "4":
            from streaming import run_streaming_demo

            await run_streaming_demo()

        elif choice == "5":
            from memory_demo import run_memory_demo

            await run_memory_demo()

        elif choice == "6":
            from callbacks import run_all_callback_demos

            await run_all_callback_demos()

        else:
            print(f"Unknown choice: {choice!r}")


if __name__ == "__main__":
    asyncio.run(main())
