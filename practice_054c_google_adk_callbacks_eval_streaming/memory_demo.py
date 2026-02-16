"""Exercise 5 — Memory Service: Cross-Session Knowledge.

ADK distinguishes between **State** and **Memory**:

State (covered in 054a):
  - Key-value pairs attached to sessions.
  - ``user:`` prefix persists across sessions for the same user.
  - Direct access: ``state["user:name"]`` → exact lookup by key.
  - Best for: structured data (preferences, config, counters).

Memory (this exercise):
  - Free-text semantic search over past conversation history.
  - When a session closes, its events are added to the memory store.
  - The agent can search memory: "What did the user mention about X?"
  - Best for: unstructured recall ("remember what we discussed last time").

How ``InMemoryMemoryService`` works:
  1. You pass it to ``Runner(memory_service=...)``.
  2. After a session's events are committed, they are indexed in memory.
  3. In future sessions, the agent can access this memory automatically —
     ADK injects relevant past context based on the current conversation.
  4. ``InMemoryMemoryService`` uses keyword matching (not embeddings) for
     search.  Production systems use ``VertexAIMemoryService`` with real
     vector search.

Memory lifecycle:
  Session starts → conversation happens → session closes →
  events are saved to memory → new session starts →
  agent searches memory for relevant context → uses it in responses.

Import:
  ``from google.adk.memory import InMemoryMemoryService``
"""

from __future__ import annotations

import asyncio

from google.adk.agents import Agent
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from task_agent.agent import add_task, complete_task, delete_task, list_tasks

MODEL = "ollama_chat/qwen2.5:7b"


async def _send_message(
    runner: Runner, user_id: str, session_id: str, text: str
) -> str:
    """Send a message and collect the final text response."""
    content = types.Content(role="user", parts=[types.Part(text=text)])
    final_text = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_text += part.text
    return final_text


async def demo_with_memory() -> None:
    """Demonstrate cross-session memory: agent recalls info from past sessions.

    This demo runs three phases:
    1. Phase 1 — Establish preferences in session A (e.g., "I prefer high
       priority tasks for work stuff").
    2. Phase 2 — Close session A so its events are committed to memory.
    3. Phase 3 — Open a NEW session B and ask the agent to recall preferences.
       With memory enabled, it should find and use context from session A.
    """

    # TODO(human): Implement the memory demo.
    #
    # Steps:
    #
    # === Setup ===
    #   1. Create the memory service:
    #      ``memory_service = InMemoryMemoryService()``
    #
    #   2. Create the session service:
    #      ``session_service = InMemorySessionService()``
    #
    #   3. Create the agent (same task agent):
    #      ``agent = Agent(
    #            name="task_agent", model=MODEL,
    #            instruction=(
    #                "You are a helpful task-management assistant. "
    #                "You help users create, list, complete, and delete tasks. "
    #                "Use memory to recall user preferences from past sessions. "
    #                "If the user has mentioned preferences before, apply them."
    #            ),
    #            tools=[add_task, list_tasks, complete_task, delete_task])``
    #
    #   4. Create the runner WITH memory:
    #      ``runner = Runner(
    #            agent=agent, app_name="memory_demo",
    #            session_service=session_service,
    #            memory_service=memory_service)``
    #
    # === Phase 1: Establish preferences in session A ===
    #   5. Create session A:
    #      ``session_a = await session_service.create_session(
    #            app_name="memory_demo", user_id="user1")``
    #
    #   6. Send messages that establish preferences:
    #      - "I always want my work tasks to be high priority"
    #      - "Add a task 'Finish quarterly report' with high priority"
    #      - "I prefer to see tasks sorted by priority"
    #      Print each prompt and response.
    #
    # === Phase 2: Close session A ===
    #   7. Close/finalize session A so events are committed to memory.
    #      In ADK, sessions are automatically committed when a new session
    #      is created.  You can also explicitly call:
    #      ``await session_service.close_session(
    #            app_name="memory_demo", user_id="user1",
    #            session_id=session_a.id)``
    #      (Note: ``close_session`` may not exist on InMemorySessionService.
    #       If not, just proceed — creating a new session should work.)
    #
    # === Phase 3: New session — recall preferences ===
    #   8. Create a NEW session B for the same user:
    #      ``session_b = await session_service.create_session(
    #            app_name="memory_demo", user_id="user1")``
    #
    #   9. Ask the agent to recall preferences:
    #      - "What are my task preferences?"
    #      - "Add a task 'Review PR' — what priority should it be?"
    #      Print each prompt and response.
    #
    #  10. The agent should use memory search to find the preferences
    #      established in session A and apply them in session B.
    #
    # Print a banner summarizing what happened.
    raise NotImplementedError


async def demo_without_memory() -> None:
    """Run the same scenario WITHOUT memory for comparison.

    This shows that without memory, the agent has NO context from session A
    when session B starts — it treats the user as a blank slate.
    """

    # TODO(human): Implement the no-memory comparison demo.
    #
    # Steps:
    #   1. Create session_service (no memory_service).
    #   2. Create runner WITHOUT memory_service.
    #   3. Run the same 3 phases as above.
    #   4. In Phase 3, the agent should NOT recall preferences from session A
    #      because there is no memory service.
    #   5. Print a comparison banner explaining the difference.
    #
    # The key difference is just the ``Runner`` constructor:
    #   WITH:    ``Runner(agent=..., session_service=..., memory_service=memory_service)``
    #   WITHOUT: ``Runner(agent=..., session_service=...)``
    #
    # Same prompts, same user, same agent — different behavior.
    raise NotImplementedError


async def run_memory_demo() -> None:
    """Run both demos side by side for comparison."""
    print("=" * 70)
    print("EXERCISE 5 — Memory Service: Cross-Session Knowledge")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("PART A: WITH Memory Service")
    print("=" * 70)
    await demo_with_memory()

    print("\n" + "=" * 70)
    print("PART B: WITHOUT Memory Service (comparison)")
    print("=" * 70)
    await demo_without_memory()

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print("With memory:    Agent recalls preferences from session A in session B.")
    print("Without memory: Agent has no knowledge of session A in session B.")
    print("Memory enables cross-session knowledge via semantic search over")
    print("past conversation history — no explicit state management needed.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_memory_demo())
