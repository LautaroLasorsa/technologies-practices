"""Exercise 1 & 2 — Before/After Tool Callbacks, Caching & Permission Control.

Google ADK callbacks let you hook into the agent execution lifecycle without
modifying agent or tool code. They are the framework's equivalent of middleware
in web frameworks — cross-cutting concerns (logging, auth, caching) live here.

Key API facts:
- ``before_tool_callback(tool, args, tool_context)`` fires before a tool runs.
    * Return ``None`` to proceed normally (tool executes).
    * Return a ``dict`` to **short-circuit** — the dict becomes the tool result
      and the actual tool is never called.
- ``after_tool_callback(tool, args, tool_context, tool_response)`` fires after.
    * Return ``None`` to keep the original result.
    * Return a ``dict`` to **replace** the result the LLM sees.
- Callbacks are assigned on the ``Agent`` via keyword arguments:
    ``Agent(..., before_tool_callback=my_fn, after_tool_callback=my_fn)``
- You can pass a *list* of callbacks; they run in order until one returns
  non-None (short-circuit semantics).

Imports you'll need:
    from google.adk.agents import Agent
    from google.adk.tools import ToolContext, BaseTool
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import BaseTool, ToolContext
from google.genai import types

from task_agent.agent import add_task, complete_task, delete_task, list_tasks

# ============================================================================
# Exercise 1 — Logging Callbacks
# ============================================================================
#
# Goal: add observability to every tool call without touching the tool code.
# You will implement two callbacks that together measure execution time and
# log inputs/outputs for every tool invocation.
#
# These callbacks form the backbone of production agent monitoring — you'd
# send these logs to OpenTelemetry, Datadog, etc. in a real system.
# ============================================================================


# We use a module-level dict to stash the start time so the after-callback
# can compute duration.  Keyed by (tool_name, invocation_id) to be safe
# with concurrent calls.
_call_start_times: dict[tuple[str, str], float] = {}


def logging_before_tool_callback(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> Optional[dict]:
    """Log tool name, arguments, and timestamp *before* the tool executes.

    This callback should:
    1. Record the current timestamp (``time.time()``) in ``_call_start_times``
       using the key ``(tool.name, tool_context.function_call_id)``.
    2. Print a formatted log line, for example:
       ``[BEFORE] tool=add_task args={'task_name': 'Buy milk', 'priority': 'high'}  ts=1718000000.123``
    3. Return ``None`` so that the tool proceeds normally.

    Why ``function_call_id``?  A single agent turn may invoke multiple tools
    (or the same tool twice with different args).  The function_call_id is a
    unique identifier ADK assigns to each individual tool invocation, so it
    disambiguates concurrent/sequential calls to the same tool.
    """

    # TODO(human): Implement the before-tool logging callback.
    #
    # Steps:
    #   1. Get the current time with ``time.time()``.
    #   2. Build the key tuple ``(tool.name, tool_context.function_call_id)``
    #      and store the timestamp in ``_call_start_times[key]``.
    #   3. Print a log line containing:
    #      - The tool name (``tool.name``)
    #      - The arguments dict (``args``)
    #      - The timestamp you just recorded
    #      Format suggestion: ``[BEFORE] tool=<name> args=<args>  ts=<timestamp>``
    #   4. Return ``None`` — this tells ADK "proceed with the tool execution".
    #
    # Returning anything other than None here would *skip* the tool entirely
    # and use the returned dict as the tool's result.  That's how caching and
    # permission callbacks work (Exercise 2), but for logging we always want
    # the tool to run.
    raise NotImplementedError


def logging_after_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict,
) -> Optional[dict]:
    """Log tool result and execution duration *after* the tool completes.

    This callback should:
    1. Look up the start time from ``_call_start_times`` using the same key
       as the before-callback.
    2. Compute the elapsed time.
    3. Print a formatted log line, for example:
       ``[AFTER]  tool=add_task duration=0.003s result={'status': 'ok', ...}``
    4. Clean up the entry from ``_call_start_times``.
    5. Return ``None`` so the original result is passed to the LLM unchanged.
    """

    # TODO(human): Implement the after-tool logging callback.
    #
    # Steps:
    #   1. Build the same key: ``(tool.name, tool_context.function_call_id)``.
    #   2. Pop the start time from ``_call_start_times`` (use ``.pop(key, None)``
    #      to handle the case where the before-callback didn't fire — defensive).
    #   3. Compute duration: ``time.time() - start_time`` (or "N/A" if missing).
    #   4. Print a log line with:
    #      - Tool name
    #      - Duration (formatted to 3 decimal places)
    #      - The tool_response dict (the actual return value of the tool)
    #   5. Return ``None`` to keep the original tool_response.
    #
    # Returning a dict here would *replace* the tool result the LLM sees.
    # That's useful for transforming results (Exercise 2) but not for logging.
    raise NotImplementedError


# ============================================================================
# Exercise 2 — Caching & Permission Callbacks
# ============================================================================
#
# Now we use the "return non-None to short-circuit" pattern for two
# production-critical use cases:
#
# A) **Caching**: If we've seen the same (tool, args) before, return the
#    cached result immediately — the tool never runs.
#
# B) **Permission control**: Check the user's role in session state and
#    block destructive operations for non-admin users.
#
# These patterns show why ADK callbacks are powerful: they intercept the
# execution flow *without modifying any tool code*.
# ============================================================================


# Simple in-memory cache: (tool_name, frozen_args_json) -> result dict
_tool_cache: dict[str, dict] = {}


def caching_before_tool_callback(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> Optional[dict]:
    """Cache tool results by (tool_name, args) key; skip execution on hit.

    Cache key strategy:
    - Serialize ``args`` to a deterministic JSON string (``sort_keys=True``).
    - Combine with ``tool.name`` to form the cache key.

    On cache **hit**: return the cached dict (this skips the tool).
    On cache **miss**: return ``None`` (tool runs normally; the after-callback
    will store the result).

    Why JSON serialization for the key?  Python dicts aren't hashable, so we
    need a stable string representation.  ``json.dumps(args, sort_keys=True)``
    ensures ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` produce the same
    key.  This is a common pattern for memoization of dict-keyed functions.
    """

    # TODO(human): Implement the caching before-tool callback.
    #
    # Steps:
    #   1. Build the cache key:
    #      ``cache_key = f"{tool.name}:{json.dumps(args, sort_keys=True)}"``
    #   2. Check if ``cache_key`` exists in ``_tool_cache``.
    #   3. If YES (cache hit):
    #      - Print a log: ``[CACHE HIT] tool=<name> args=<args>``
    #      - Return the cached result dict.
    #        Returning a dict from before_tool_callback *skips the tool* and
    #        uses this dict as the tool's result.  The LLM never knows the
    #        tool didn't actually run.
    #   4. If NO (cache miss):
    #      - Print a log: ``[CACHE MISS] tool=<name> args=<args>``
    #      - Return ``None`` so the tool executes normally.
    #
    # Note: We store the result in the *after* callback (below), not here.
    raise NotImplementedError


def caching_after_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict,
) -> Optional[dict]:
    """Store the tool result in the cache after successful execution.

    This pairs with ``caching_before_tool_callback``: on a cache miss the
    tool runs, then this callback saves the result so the next identical
    call is a cache hit.
    """

    # TODO(human): Implement the caching after-tool callback.
    #
    # Steps:
    #   1. Build the same cache key as in the before-callback:
    #      ``cache_key = f"{tool.name}:{json.dumps(args, sort_keys=True)}"``
    #   2. Store the result: ``_tool_cache[cache_key] = tool_response``
    #   3. Print a log: ``[CACHE STORE] tool=<name> args=<args>``
    #   4. Return ``None`` to keep the original tool_response unchanged.
    raise NotImplementedError


def permission_before_tool_callback(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> Optional[dict]:
    """Block destructive tools (delete, complete) for non-admin users.

    Reads ``user:role`` from session state:
    - If role is ``"admin"``: allow all tools (return None).
    - If role is missing or anything else: block ``delete_task`` and
      ``complete_task``, returning an error dict instead.

    This pattern is how you implement RBAC (Role-Based Access Control) in
    ADK agents.  The LLM receives the denial message as if it were a tool
    result, so it can explain to the user why the operation was blocked.

    State access in callbacks:
    - ``tool_context.state`` gives you the same state dict that tools use.
    - ``user:`` prefixed keys persist across sessions for the same user.
    - You can read *and write* state in callbacks.
    """

    # TODO(human): Implement the permission before-tool callback.
    #
    # Steps:
    #   1. Read the user's role: ``role = tool_context.state.get("user:role", "viewer")``
    #   2. Define a set of restricted tools: ``{"delete_task", "complete_task"}``
    #   3. If ``tool.name`` is in the restricted set AND ``role != "admin"``:
    #      - Print a log: ``[PERMISSION DENIED] tool=<name> role=<role>``
    #      - Return a dict with an error message, e.g.:
    #        ``{"status": "error", "message": "Permission denied: <role> cannot use <tool_name>"}``
    #        This dict becomes the tool result the LLM sees — the actual tool
    #        never executes.
    #   4. Otherwise:
    #      - Print a log: ``[PERMISSION OK] tool=<name> role=<role>``
    #      - Return ``None`` to let the tool proceed.
    raise NotImplementedError


# ============================================================================
# Wiring & Testing
# ============================================================================


def _create_agent_with_callbacks(
    use_logging: bool = False,
    use_caching: bool = False,
    use_permission: bool = False,
) -> Agent:
    """Create a task agent with the requested callbacks wired in.

    ADK's ``before_tool_callback`` and ``after_tool_callback`` accept either
    a single callable or a **list** of callables.  When a list is provided,
    they execute in order; the first one to return non-None wins (short-circuit).

    This means you can compose multiple before-callbacks:
    ``[permission_check, caching_check, logging]`` — permission runs first,
    then caching, then logging.  If permission denies, caching and logging
    never fire.
    """
    before_cbs: list = []
    after_cbs: list = []

    if use_permission:
        before_cbs.append(permission_before_tool_callback)
    if use_caching:
        before_cbs.append(caching_before_tool_callback)
        after_cbs.append(caching_after_tool_callback)
    if use_logging:
        before_cbs.append(logging_before_tool_callback)
        after_cbs.append(logging_after_tool_callback)

    return Agent(
        name="task_agent",
        model="ollama_chat/qwen2.5:7b",
        description="A task-management assistant with callback hooks.",
        instruction=(
            "You are a helpful task-management assistant. You help users "
            "create, list, complete, and delete tasks. Always confirm the "
            "action you took. Be concise."
        ),
        tools=[add_task, list_tasks, complete_task, delete_task],
        before_tool_callback=before_cbs if before_cbs else None,
        after_tool_callback=after_cbs if after_cbs else None,
    )


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


async def demo_logging_callbacks() -> None:
    """Exercise 1 demo: run the agent with logging callbacks and observe output."""
    print("=" * 70)
    print("EXERCISE 1 — Logging Before/After Tool Callbacks")
    print("=" * 70)

    agent = _create_agent_with_callbacks(use_logging=True)
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent, app_name="callbacks_demo", session_service=session_service
    )

    session = await session_service.create_session(
        app_name="callbacks_demo", user_id="user1"
    )

    prompts = [
        "Add a task called 'Buy groceries' with high priority",
        "Add a task 'Read ADK docs' with medium priority",
        "Show me all my tasks",
    ]

    for prompt in prompts:
        print(f"\n>>> User: {prompt}")
        response = await _send_message(runner, "user1", session.id, prompt)
        print(f"<<< Agent: {response}")

    print("\n" + "=" * 70)
    print("Look at the [BEFORE] / [AFTER] lines above — those are your callbacks!")
    print("=" * 70)


async def demo_caching_callbacks() -> None:
    """Exercise 2A demo: run the same query twice and verify the cache kicks in."""
    print("\n" + "=" * 70)
    print("EXERCISE 2A — Caching Callbacks")
    print("=" * 70)

    agent = _create_agent_with_callbacks(use_logging=True, use_caching=True)
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent, app_name="caching_demo", session_service=session_service
    )

    session = await session_service.create_session(
        app_name="caching_demo", user_id="user1"
    )

    print("\n--- First call (expect CACHE MISS) ---")
    prompt = "Show me all my tasks"
    print(f">>> User: {prompt}")
    resp1 = await _send_message(runner, "user1", session.id, prompt)
    print(f"<<< Agent: {resp1}")

    print("\n--- Second identical call (expect CACHE HIT) ---")
    print(f">>> User: {prompt}")
    resp2 = await _send_message(runner, "user1", session.id, prompt)
    print(f"<<< Agent: {resp2}")

    print("\n" + "=" * 70)
    print("The second call should show [CACHE HIT] — the tool never ran.")
    print("=" * 70)


async def demo_permission_callbacks() -> None:
    """Exercise 2B demo: test permission-based tool blocking."""
    print("\n" + "=" * 70)
    print("EXERCISE 2B — Permission Callbacks")
    print("=" * 70)

    agent = _create_agent_with_callbacks(
        use_logging=True, use_permission=True
    )
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent, app_name="permission_demo", session_service=session_service
    )

    # --- Viewer role (should be blocked from delete/complete) ---
    session_viewer = await session_service.create_session(
        app_name="permission_demo",
        user_id="viewer_user",
        state={"user:role": "viewer"},
    )

    print("\n--- As VIEWER: try to add a task (should work) ---")
    prompt = "Add a task 'Test permissions' with low priority"
    print(f">>> User: {prompt}")
    resp = await _send_message(runner, "viewer_user", session_viewer.id, prompt)
    print(f"<<< Agent: {resp}")

    print("\n--- As VIEWER: try to delete task 1 (should be BLOCKED) ---")
    prompt = "Delete task 1"
    print(f">>> User: {prompt}")
    resp = await _send_message(runner, "viewer_user", session_viewer.id, prompt)
    print(f"<<< Agent: {resp}")

    # --- Admin role (should be allowed everything) ---
    session_admin = await session_service.create_session(
        app_name="permission_demo",
        user_id="admin_user",
        state={"user:role": "admin"},
    )

    print("\n--- As ADMIN: add and then delete a task (should work) ---")
    prompt = "Add a task 'Admin task' with high priority"
    print(f">>> User: {prompt}")
    resp = await _send_message(runner, "admin_user", session_admin.id, prompt)
    print(f"<<< Agent: {resp}")

    prompt = "Delete task 1"
    print(f">>> User: {prompt}")
    resp = await _send_message(runner, "admin_user", session_admin.id, prompt)
    print(f"<<< Agent: {resp}")

    print("\n" + "=" * 70)
    print("Viewer should see [PERMISSION DENIED]; Admin should see [PERMISSION OK].")
    print("=" * 70)


async def run_all_callback_demos() -> None:
    """Run all callback exercises in sequence."""
    await demo_logging_callbacks()
    await demo_caching_callbacks()
    await demo_permission_callbacks()


if __name__ == "__main__":
    asyncio.run(run_all_callback_demos())
