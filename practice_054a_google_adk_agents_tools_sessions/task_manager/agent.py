"""Task Manager Agent — ADK agent with custom tools and session state.

This module defines:
  1. Tool functions that the LLM can invoke to manage tasks
  2. The root_agent that wires everything together

ADK auto-generates JSON schemas from the function signatures + docstrings,
so the LLM knows *when* and *how* to call each tool.
"""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# The "ollama_chat/" prefix routes through LiteLLM's Ollama chat-completion
# endpoint.  Do NOT use "ollama/" — that causes infinite tool-call loops
# because it hits the plain completion API which doesn't handle tool schemas.
MODEL = LiteLlm(model="ollama_chat/qwen2.5:7b")


# ---------------------------------------------------------------------------
# Tool: add_task
# ---------------------------------------------------------------------------
def add_task(name: str, description: str, tool_context: ToolContext) -> dict:
    """Add a new task to the user's task list.

    Creates a task with the given name and description and stores it in
    user-scoped state so it persists across sessions for the same user.

    Args:
        name: Short, unique name for the task (e.g. "buy-groceries").
        description: Longer explanation of what the task involves.
        tool_context: Automatically injected by ADK — provides access to
            session state and other runtime context.

    Returns:
        A dict with a status message confirming the task was added.
    """
    # TODO(human): Implement this tool.
    #
    # What this teaches:
    #   State persistence via the "user:" prefix.  Any key that starts with
    #   "user:" is scoped to the *user* — it survives across different
    #   sessions for the same user_id.  Keys without a prefix are
    #   session-only and vanish when the session ends.
    #
    # How ADK state works:
    #   tool_context.state is a dict-like object backed by the session's
    #   state store.  Reading and writing it is as simple as:
    #       tasks = tool_context.state.get("user:tasks", [])
    #       tool_context.state["user:tasks"] = tasks
    #   Mutations are automatically tracked and persisted by ADK's event
    #   system (they become part of EventActions.state_delta).
    #
    # Steps:
    #   1. Read the current task list from state using key "user:tasks".
    #      Default to an empty list if no tasks exist yet.
    #   2. Create a task dict with keys: "name", "description", "completed"
    #      (initially False).
    #   3. Append the new task to the list.
    #   4. Write the updated list back to state under "user:tasks".
    #   5. (Bonus) Also update an app-wide counter at "app:total_tasks_created"
    #      that tracks how many tasks have been created across ALL users.
    #   6. Return a dict like:
    #      {"status": "success", "message": f"Task '{name}' added."}
    #
    # Hints:
    #   - tool_context.state.get("user:tasks", [])  reads user-scoped state
    #   - tool_context.state["user:tasks"] = ...     writes user-scoped state
    #   - tool_context.state["app:total_tasks_created"] = ...  writes app-wide state
    raise NotImplementedError("Implement add_task")


# ---------------------------------------------------------------------------
# Tool: list_tasks
# ---------------------------------------------------------------------------
def list_tasks(tool_context: ToolContext) -> dict:
    """List all tasks in the user's task list.

    Returns the full task list from user-scoped state, including each task's
    name, description, and completion status.

    Args:
        tool_context: Automatically injected by ADK — provides access to
            session state and other runtime context.

    Returns:
        A dict containing the list of tasks and a count.
    """
    # TODO(human): Implement this tool.
    #
    # What this teaches:
    #   Reading from user-scoped state and returning structured data that
    #   the LLM can format into a human-friendly response.
    #
    # How the LLM uses this:
    #   When the user says "show my tasks" or "what's on my list?", the LLM
    #   matches that intent to this tool's docstring and calls it.  The dict
    #   you return is fed back into the LLM as the tool result, and the LLM
    #   then formats it into natural language for the user.
    #
    # Steps:
    #   1. Read the task list from "user:tasks" (default to []).
    #   2. Return a dict with:
    #      - "tasks": the list itself
    #      - "count": number of tasks
    #      Example: {"tasks": [...], "count": 3}
    #
    # Edge case to handle:
    #   If the list is empty, return something like:
    #   {"tasks": [], "count": 0, "message": "No tasks yet."}
    #   This helps the LLM give a useful response instead of just "[]".
    raise NotImplementedError("Implement list_tasks")


# ---------------------------------------------------------------------------
# Tool: complete_task
# ---------------------------------------------------------------------------
def complete_task(task_name: str, tool_context: ToolContext) -> dict:
    """Mark a task as completed by its name.

    Searches the user's task list for a task matching the given name and
    sets its completed flag to True.

    Args:
        task_name: The name of the task to mark as completed.  Must match
            an existing task name (case-insensitive comparison recommended).
        tool_context: Automatically injected by ADK — provides access to
            session state and other runtime context.

    Returns:
        A dict with a status message indicating success or failure.
    """
    # TODO(human): Implement this tool.
    #
    # What this teaches:
    #   Modifying existing state — finding an item in a list stored in state,
    #   mutating it, and writing the whole list back.  Also teaches error
    #   handling in tools: when the task isn't found, return an error dict
    #   instead of raising an exception.  The LLM uses this to inform the user.
    #
    # Important ADK detail — state is NOT a live reference:
    #   When you do `tasks = tool_context.state.get("user:tasks", [])`,
    #   you get a *copy* of the list.  Mutating that copy does NOT
    #   automatically update the state.  You MUST write it back:
    #       tool_context.state["user:tasks"] = tasks
    #   This is because ADK tracks changes via assignment, not via
    #   deep mutation detection.
    #
    # Steps:
    #   1. Read the task list from "user:tasks" (default to []).
    #   2. Search for a task whose "name" matches task_name
    #      (use case-insensitive comparison: task["name"].lower()).
    #   3. If found:
    #      a. Set task["completed"] = True
    #      b. Write the updated list back to "user:tasks"
    #      c. Return {"status": "success", "message": f"Task '{task_name}' completed."}
    #   4. If NOT found:
    #      Return {"status": "error", "message": f"Task '{task_name}' not found."}
    #
    # Why return dicts instead of raising exceptions:
    #   ADK feeds tool return values back to the LLM.  If you raise, the
    #   agent loop may fail or produce a confusing error.  Returning an
    #   error dict lets the LLM gracefully tell the user "I couldn't find
    #   that task" and ask for clarification.
    raise NotImplementedError("Implement complete_task")


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------
# ADK's Agent class wires together the model, instructions, and tools.
# The `name` must be a valid Python identifier (no spaces/hyphens).
# The `description` is used when this agent is part of a multi-agent system
# (practice 054b) — the orchestrator reads it to decide which sub-agent to route to.
root_agent = Agent(
    name="task_manager",
    model=MODEL,
    description="A task management assistant that helps users create, view, and complete tasks.",
    instruction="""You are a helpful task management assistant.  Your job is to help users
organize their work by managing a task list.

You have three tools available:
- add_task: Create a new task with a name and description
- list_tasks: Show all current tasks
- complete_task: Mark a task as done

Guidelines:
- When the user asks to add a task, extract a short name and a description from their message.
- When listing tasks, format the output clearly, indicating which tasks are completed.
- When completing a task, confirm which task was marked done.
- If the user's request is ambiguous, ask for clarification.
- Be concise and friendly in your responses.
""",
    tools=[add_task, list_tasks, complete_task],
)
