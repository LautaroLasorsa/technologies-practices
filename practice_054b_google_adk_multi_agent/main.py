"""CLI runner for practice 054b — Google ADK Multi-Agent Orchestration.

Lets you choose which exercise to run (1-5), creates a session,
and runs the selected agent interactively in a REPL loop.

Usage:
    uv run python main.py
"""

import asyncio
import sys
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ---------------------------------------------------------------------------
# Exercise registry
# ---------------------------------------------------------------------------

EXERCISES: dict[str, dict[str, Any]] = {
    "1": {
        "title": "LLM-Based Delegation — Specialist Router",
        "module": "specialist_router",
        "description": "A parent LLM agent routes queries to math, trivia, or translator specialists.",
    },
    "2": {
        "title": "SequentialAgent — Research Pipeline",
        "module": "research_pipeline",
        "description": "A 3-stage pipeline: Gather -> Process -> Format using shared state.",
    },
    "3": {
        "title": "ParallelAgent — Concurrent Gathering",
        "module": "parallel_gather",
        "description": "Fan-out to 3 data sources in parallel, then summarize results.",
    },
    "4": {
        "title": "LoopAgent — Iterative Writer",
        "module": "iterative_writer",
        "description": "Write-critique-improve loop with quality threshold for exit.",
    },
    "5": {
        "title": "Nested Orchestration — Full Workflow",
        "module": "full_workflow",
        "description": "Combines Parallel + Loop + Sequential into a complete pipeline.",
    },
}


def load_agent(module_name: str) -> Any:
    """Dynamically import and return the root_agent from an exercise module."""
    import importlib
    module = importlib.import_module(module_name)
    return module.root_agent


def print_menu() -> None:
    """Display the exercise selection menu."""
    print("\n=== Practice 054b: Google ADK Multi-Agent Orchestration ===\n")
    for key, info in EXERCISES.items():
        print(f"  [{key}] {info['title']}")
        print(f"      {info['description']}")
    print(f"\n  [q] Quit")
    print()


async def run_agent_session(agent: Any, exercise_title: str) -> None:
    """Run an interactive REPL session with the given agent.

    Creates an InMemorySessionService and Runner, then loops:
    prompt -> run_async -> print events -> prompt again.
    """
    app_name = "practice_054b"
    user_id = "user"

    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
    )

    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service,
    )

    print(f"\n--- {exercise_title} ---")
    print("Type your message (or 'back' to return to menu, 'quit' to exit).\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("back", "b"):
            break
        if user_input.lower() in ("quit", "q", "exit"):
            sys.exit(0)

        message = types.Content(
            role="user",
            parts=[types.Part(text=user_input)],
        )

        print()
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=message,
        ):
            if event.content and event.content.parts:
                author = event.author or "agent"
                for part in event.content.parts:
                    if part.text:
                        print(f"[{author}]: {part.text}")
                    if part.function_call:
                        print(f"[{author}] -> tool call: {part.function_call.name}({part.function_call.args})")
                    if part.function_response:
                        print(f"[{author}] <- tool result: {part.function_response.response}")
        print()


async def main() -> None:
    """Main entry point: show menu, load agent, run session."""
    while True:
        print_menu()
        choice = input("Select exercise [1-5, q]: ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("Bye!")
            break

        if choice not in EXERCISES:
            print(f"Invalid choice: '{choice}'. Pick 1-5 or q.")
            continue

        exercise = EXERCISES[choice]
        print(f"\nLoading {exercise['title']}...")

        try:
            agent = load_agent(exercise["module"])
        except NotImplementedError as e:
            print(f"\n  Not yet implemented: {e}")
            print("  Complete the TODO(human) in the exercise module first.\n")
            continue
        except Exception as e:
            print(f"\n  Error loading module '{exercise['module']}': {e}\n")
            continue

        await run_agent_session(agent, exercise["title"])


if __name__ == "__main__":
    asyncio.run(main())
