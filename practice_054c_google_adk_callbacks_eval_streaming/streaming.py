"""Exercise 4 — Streaming Responses.

By default, ``runner.run_async()`` waits for the LLM to finish its full
response before yielding an event.  With streaming, tokens arrive
incrementally as they're generated — the user sees output appear in real time.

How streaming works in ADK:
- ``runner.run_async()`` is an async generator that yields ``Event`` objects.
- When the LLM is streaming text, it yields events with ``event.partial = True``
  — these contain incomplete text chunks that will be assembled into the final
  response.
- When the full response is ready, ``event.is_final_response()`` returns True.
- Tool calls appear as events where ``event.content.parts`` contain
  ``function_call`` objects (the LLM requesting a tool) and later
  ``function_response`` objects (the tool's return value).

Event anatomy (what to look for):
- ``event.content.parts[i].text``          — text content (partial or final)
- ``event.content.parts[i].function_call`` — LLM requesting a tool call
- ``event.partial``                         — True if this is a partial chunk
- ``event.is_final_response()``             — True if this is the final text
- ``event.author``                          — who generated this event

The ``partial`` flag is key: when True, the text is an incremental chunk
that should be printed immediately (without a newline).  When False/None,
it's a complete piece of content.

Streaming + callbacks caveat:
  After-model callbacks may not receive fully populated grounding metadata
  in streaming mode.  Tool callbacks work correctly in both modes.
"""

from __future__ import annotations

import asyncio
import time

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from task_agent.agent import add_task, complete_task, delete_task, list_tasks

MODEL = "ollama_chat/qwen2.5:7b"


async def streaming_chat(
    runner: Runner, user_id: str, session_id: str, user_text: str
) -> None:
    """Send a message and display the response as a real-time stream.

    This function iterates over the event stream from ``runner.run_async()``
    and prints each chunk as it arrives.  The UX goal: the user sees text
    appearing character-by-character, with tool call indicators inline.

    Display format:
    - Partial text chunks: print immediately with ``end=""``, ``flush=True``
    - Tool call start: print ``\\n[Calling <tool_name>(<args>)]\\n``
    - Tool result (function_response): print ``\\n[Result: <summary>]\\n``
    - Final response boundary: print a separator line

    This is the same pattern used by ChatGPT, Claude, and Gemini UIs —
    stream tokens to the client as they're generated rather than waiting
    for the full response.
    """

    # TODO(human): Implement the streaming event loop.
    #
    # Steps:
    #   1. Build the user message:
    #      ``content = types.Content(
    #            role="user", parts=[types.Part(text=user_text)])``
    #
    #   2. Start timing: ``start = time.time()``
    #
    #   3. Iterate over the event stream:
    #      ``async for event in runner.run_async(
    #            user_id=user_id, session_id=session_id,
    #            new_message=content):``
    #
    #   4. For each event, check ``event.content`` and ``event.content.parts``:
    #
    #      a) **Function call** (LLM requesting a tool):
    #         Check each part for ``part.function_call``.
    #         If found, print:
    #           ``\n[Calling {part.function_call.name}({dict(part.function_call.args)})]``
    #
    #      b) **Function response** (tool returning a result):
    #         Check each part for ``part.function_response``.
    #         If found, print:
    #           ``\n[Result: {part.function_response.response}]``
    #
    #      c) **Partial text** (streaming tokens):
    #         If ``event.partial`` is True and the part has ``.text``:
    #           ``print(part.text, end="", flush=True)``
    #         This gives the real-time streaming effect.
    #
    #      d) **Final response** (complete text):
    #         If ``event.is_final_response()`` and the part has ``.text``:
    #           ``print(part.text, end="", flush=True)``
    #
    #   5. After the loop, print timing:
    #      ``elapsed = time.time() - start``
    #      ``print(f"\n--- {elapsed:.2f}s ---")``
    #
    # Note: Not all events have content or parts — always guard with
    # ``if event.content and event.content.parts``.
    #
    # Tip: To see the difference between streaming and non-streaming,
    # try adding a ``time.sleep(0.05)`` before each partial print —
    # it simulates a slower network and makes the streaming effect visible.
    raise NotImplementedError


async def non_streaming_chat(
    runner: Runner, user_id: str, session_id: str, user_text: str
) -> None:
    """Send a message and wait for the complete response before printing.

    This is the "traditional" mode — no real-time output, just the final
    result.  Included so you can compare the UX side-by-side with streaming.
    """
    content = types.Content(role="user", parts=[types.Part(text=user_text)])
    start = time.time()
    final_text = ""

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_text += part.text

    elapsed = time.time() - start
    print(final_text)
    print(f"\n--- {elapsed:.2f}s ---")


async def run_streaming_demo() -> None:
    """Interactive CLI that demonstrates streaming vs non-streaming responses.

    Creates the agent + runner, then enters a REPL loop where:
    - User types a prompt
    - Response streams in real-time
    - Type ``/compare <prompt>`` to see the same prompt in both modes
    - Type ``/quit`` to exit
    """
    print("=" * 70)
    print("EXERCISE 4 — Streaming Responses")
    print("=" * 70)
    print()
    print("Commands:")
    print("  <any text>             — send to agent (streaming mode)")
    print("  /compare <text>        — run in both streaming and non-streaming")
    print("  /quit                  — exit")
    print()

    agent = Agent(
        name="task_agent",
        model=MODEL,
        description="A task-management assistant.",
        instruction=(
            "You are a helpful task-management assistant. You help users "
            "create, list, complete, and delete tasks. Always confirm the "
            "action you took. Be concise."
        ),
        tools=[add_task, list_tasks, complete_task, delete_task],
    )

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent, app_name="streaming_demo", session_service=session_service
    )
    session = await session_service.create_session(
        app_name="streaming_demo", user_id="user1"
    )

    while True:
        try:
            user_input = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "/quit":
            break

        if user_input.lower().startswith("/compare "):
            prompt = user_input[len("/compare ") :]
            print("\n--- STREAMING MODE ---")
            await streaming_chat(runner, "user1", session.id, prompt)
            print("\n--- NON-STREAMING MODE ---")
            await non_streaming_chat(runner, "user1", session.id, prompt)
        else:
            await streaming_chat(runner, "user1", session.id, user_input)


if __name__ == "__main__":
    asyncio.run(run_streaming_demo())
