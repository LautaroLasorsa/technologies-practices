"""Base task-management agent — self-contained version for 054c.

The tools here are *fully implemented* (not TODO) because the learning focus
of this practice is callbacks, evaluation, and streaming — not tool logic.
The agent manages a simple in-session task list via ``user:`` scoped state.
"""

from typing import Any

from google.adk.agents import Agent
from google.adk.tools import ToolContext


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def add_task(task_name: str, priority: str, tool_context: ToolContext) -> dict[str, Any]:
    """Add a new task to the user's task list.

    Args:
        task_name: Short description of the task (e.g. "Buy groceries").
        priority: Priority level — "high", "medium", or "low".
    """
    tasks: list[dict[str, Any]] = tool_context.state.get("user:tasks", [])
    task = {
        "id": len(tasks) + 1,
        "name": task_name,
        "priority": priority,
        "done": False,
    }
    tasks.append(task)
    tool_context.state["user:tasks"] = tasks

    # Track app-wide stats
    total_created: int = tool_context.state.get("app:total_created", 0)
    tool_context.state["app:total_created"] = total_created + 1

    return {"status": "ok", "task": task}


def list_tasks(tool_context: ToolContext) -> dict[str, Any]:
    """List all tasks for the current user, grouped by status.

    Returns a dict with ``pending`` and ``completed`` lists.
    """
    tasks: list[dict[str, Any]] = tool_context.state.get("user:tasks", [])
    pending = [t for t in tasks if not t["done"]]
    completed = [t for t in tasks if t["done"]]
    return {"pending": pending, "completed": completed, "total": len(tasks)}


def complete_task(task_id: int, tool_context: ToolContext) -> dict[str, Any]:
    """Mark a task as completed by its numeric ID.

    Args:
        task_id: The ID of the task to mark as done.
    """
    tasks: list[dict[str, Any]] = tool_context.state.get("user:tasks", [])
    for task in tasks:
        if task["id"] == task_id:
            task["done"] = True
            tool_context.state["user:tasks"] = tasks

            total_completed: int = tool_context.state.get("app:total_completed", 0)
            tool_context.state["app:total_completed"] = total_completed + 1

            return {"status": "ok", "task": task}
    return {"status": "error", "message": f"Task {task_id} not found"}


def delete_task(task_id: int, tool_context: ToolContext) -> dict[str, Any]:
    """Delete a task permanently by its numeric ID.

    Args:
        task_id: The ID of the task to delete.
    """
    tasks: list[dict[str, Any]] = tool_context.state.get("user:tasks", [])
    original_len = len(tasks)
    tasks = [t for t in tasks if t["id"] != task_id]
    if len(tasks) == original_len:
        return {"status": "error", "message": f"Task {task_id} not found"}
    tool_context.state["user:tasks"] = tasks
    return {"status": "ok", "message": f"Task {task_id} deleted"}


# ---------------------------------------------------------------------------
# Agent definition (no callbacks — exercises wire them externally)
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="task_agent",
    model="ollama_chat/qwen2.5:7b",
    description="A task-management assistant that helps users track to-do items.",
    instruction=(
        "You are a helpful task-management assistant. You help users create, "
        "list, complete, and delete tasks. Always confirm the action you took. "
        "When listing tasks, format them clearly with ID, name, priority, and "
        "status. Be concise."
    ),
    tools=[add_task, list_tasks, complete_task, delete_task],
)
