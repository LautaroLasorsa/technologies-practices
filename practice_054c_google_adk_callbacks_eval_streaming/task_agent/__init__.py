"""Task management agent for ADK callbacks, evaluation & streaming practice.

This module exports ``root_agent`` — the single entry-point that the ADK web
UI and the exercise scripts use to interact with the agent.
"""

from task_agent.agent import root_agent

__all__ = ["root_agent"]
