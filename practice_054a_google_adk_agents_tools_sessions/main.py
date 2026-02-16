"""CLI runner for the Task Manager agent.

This script runs the agent programmatically (outside the ADK Web UI) using:
  - InMemorySessionService for session/state storage
  - Runner to orchestrate the agent execution loop
  - An interactive REPL that reads user input and streams agent responses

Run with:  uv run python main.py
"""

import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from task_manager.agent import root_agent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_NAME = "task_manager_app"
USER_ID = "user_01"
SESSION_ID = "session_01"


async def setup_session(session_service: InMemorySessionService) -> None:
    """Create the initial session for our user.

    ADK sessions track conversation history and state.  A session is
    identified by the combination of (app_name, user_id, session_id).
    """
    # TODO(human): Create a session using the session service.
    #
    # What this teaches:
    #   Session lifecycle management.  In ADK, you must explicitly create a
    #   session before running the agent.  The session stores:
    #     - Conversation history (list of events / messages)
    #     - State (key-value pairs with scope prefixes: user:, app:, or none)
    #
    # How InMemorySessionService works:
    #   It's a simple in-memory dict-based store — perfect for local dev.
    #   Data is lost on restart.  For production, you'd swap in
    #   DatabaseSessionService (SQLite/PostgreSQL) or VertexAISessionService.
    #
    # Steps:
    #   Call session_service.create_session() with these keyword arguments:
    #     - app_name=APP_NAME
    #     - user_id=USER_ID
    #     - session_id=SESSION_ID
    #
    #   IMPORTANT: create_session() is an async method — you must await it.
    #
    # Example:
    #   session = await session_service.create_session(
    #       app_name=..., user_id=..., session_id=...
    #   )
    #
    # After creating the session, print a confirmation like:
    #   print(f"Session created: app={APP_NAME}, user={USER_ID}, session={SESSION_ID}")
    raise NotImplementedError("Implement setup_session")


async def run_agent_turn(runner: Runner, user_message: str) -> str:
    """Send a single user message to the agent and return the text response.

    This is the core of the ADK execution model:
      1. Wrap the user's text in a types.Content message
      2. Call runner.run_async() which returns an async stream of Events
      3. Each Event represents a step: LLM text, tool call, tool result, etc.
      4. The final event (is_final_response() == True) contains the agent's answer

    Args:
        runner: The ADK Runner that orchestrates agent execution.
        user_message: The user's input text.

    Returns:
        The agent's text response.
    """
    # TODO(human): Implement the agent execution loop.
    #
    # What this teaches:
    #   The Runner is ADK's orchestration engine.  When you call run_async(),
    #   it starts a loop:
    #     1. Sends the user message + conversation history + tool schemas to the LLM
    #     2. If the LLM responds with a tool call → executes the tool → feeds
    #        the result back to the LLM → repeats
    #     3. If the LLM responds with text → that's the final answer → emits
    #        an event with is_final_response() == True
    #
    #   The events you receive from run_async() represent every step of this
    #   loop.  You can inspect them to see tool calls, tool results, and
    #   intermediate reasoning — this is what the ADK Web UI Trace tab shows.
    #
    # Steps:
    #   1. Create a Content object with the user's message:
    #        content = types.Content(
    #            role="user",
    #            parts=[types.Part(text=user_message)]
    #        )
    #
    #   2. Call runner.run_async() with:
    #        - user_id=USER_ID
    #        - session_id=SESSION_ID
    #        - new_message=content
    #      This returns an async iterable of Event objects.
    #
    #   3. Iterate with `async for event in runner.run_async(...):`
    #      For each event, check if it's the final response:
    #        if event.is_final_response():
    #            # Extract the text from the event
    #            if event.content and event.content.parts:
    #                return event.content.parts[0].text
    #            # Handle escalation (agent couldn't complete the task)
    #            elif event.actions and event.actions.escalate:
    #                return f"Agent escalated: {event.error_message}"
    #
    #   4. If no final response was found (shouldn't happen normally),
    #      return a fallback string like "No response from agent."
    #
    # Debugging tip:
    #   To see ALL events (not just the final one), add a print inside the
    #   loop:  print(f"  [event] author={event.author}, final={event.is_final_response()}")
    #   This shows you the full execution trace in the terminal.
    raise NotImplementedError("Implement run_agent_turn")


async def main() -> None:
    """Interactive REPL: read user input, run agent, print response."""
    # --- Session service ---
    session_service = InMemorySessionService()

    # --- Runner ---
    # The Runner connects the agent to the session service.  It handles:
    #   - Fetching the session (history + state) before each turn
    #   - Dispatching tool calls to the right Python functions
    #   - Appending events (including state deltas) back to the session
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # --- Create initial session ---
    await setup_session(session_service)

    # --- Interactive loop ---
    print("Task Manager Agent (type 'quit' to exit)")
    print("-" * 45)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        response = await run_agent_turn(runner, user_input)
        print(f"\nAgent: {response}")


if __name__ == "__main__":
    asyncio.run(main())
