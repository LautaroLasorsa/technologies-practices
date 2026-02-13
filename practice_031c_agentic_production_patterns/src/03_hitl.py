"""Phase 3 — Human-in-the-Loop: Approval Workflows with LangGraph.

Demonstrates the HITL pattern: agent proposes an action, a risk classifier
determines if human approval is needed, and LangGraph's interrupt()/Command(resume=)
mechanism pauses/resumes the graph.

Run:
    uv run python src/03_hitl.py
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Literal

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ── Configuration ────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"


# ── State & Models ───────────────────────────────────────────────────

class AgentState(TypedDict):
    """LangGraph state for the HITL workflow."""
    messages: Annotated[list, add_messages]
    proposed_action: str
    risk_level: str
    approved: bool
    result: str


class ProposedAction(BaseModel):
    """An action the agent proposes to take."""
    action: str = Field(description="Description of the proposed action")
    target: str = Field(description="What the action affects (e.g., file path, email recipient)")
    risk_justification: str = Field(description="Why this action might be risky")


# ── Risk classification rules ────────────────────────────────────────

# These keyword-based rules are intentionally simple for the exercise.
# Production systems use more sophisticated approaches: embeddings similarity
# to known-risky actions, policy engines, or even a separate classifier LLM.

HIGH_RISK_KEYWORDS = ["delete", "drop", "remove", "send email", "transfer", "payment", "execute"]
MEDIUM_RISK_KEYWORDS = ["update", "modify", "write", "create", "install"]
# Everything else is low risk


# ── TODO(human) #1: Risk Classifier ──────────────────────────────────

def classify_risk(action: str) -> Literal["low", "medium", "high"]:
    """Classify an agent's proposed action by risk level.

    TODO(human): Implement this function.

    Risk classification is the foundation of proportional trust. Not every agent
    action needs human approval — reading data is safe, deleting a database is not.
    By classifying risk, you can auto-approve safe actions (faster UX) while
    requiring human review for dangerous ones (safer system).

    Steps:
      1. Convert the action string to lowercase for case-insensitive matching
      2. Check if any keyword in HIGH_RISK_KEYWORDS appears in the action
         → return "high"
      3. Check if any keyword in MEDIUM_RISK_KEYWORDS appears in the action
         → return "medium"
      4. Otherwise → return "low"

    This keyword-based approach is a starting point. In production, you'd combine
    it with: (a) user role/permissions, (b) resource sensitivity labels, and
    (c) historical risk data from past agent actions.

    Returns:
        "low", "medium", or "high"
    """
    raise NotImplementedError


# ── TODO(human) #2: HITL LangGraph Workflow ──────────────────────────

# Build a LangGraph StateGraph with 3 nodes:
#
#   Node 1: "propose_action"
#     - Takes the user's request from state["messages"]
#     - Uses ChatOllama to generate a proposed action (what the agent wants to do)
#     - Stores the proposed action in state["proposed_action"]
#     - Returns updated state
#
#   Node 2: "check_risk"
#     - Reads state["proposed_action"]
#     - Calls classify_risk() to get the risk level
#     - Stores it in state["risk_level"]
#     - If risk is "high": call interrupt({"action": proposed_action, "risk": "high"})
#       This PAUSES the graph and returns control to the caller. The caller can
#       then resume with Command(resume=True/False) to approve or reject.
#     - After interrupt returns (when resumed), check the resume value:
#       If True → set state["approved"] = True
#       If False → set state["approved"] = False
#     - If risk is "medium" or "low": auto-approve (state["approved"] = True)
#     - Return updated state
#
#   Node 3: "execute_action"
#     - If state["approved"] is True: set state["result"] to a success message
#     - If state["approved"] is False: set state["result"] to a rejection message
#     - Return updated state
#
# Edges:
#   START → "propose_action" → "check_risk" → "execute_action" → END
#
# Key concept: interrupt() serializes the graph state to the checkpointer
# (MemorySaver) and raises a special exception that LangGraph catches. The
# graph is frozen at exactly that point. When you call graph.invoke(Command(resume=value),
# config=same_config), the graph deserializes state and continues from the
# interrupt with the provided value.
#
# Docs: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/

def build_hitl_graph() -> Any:
    """Build and compile the HITL LangGraph workflow.

    TODO(human): Implement this function.

    Returns:
        A compiled LangGraph (graph.compile(checkpointer=MemorySaver()))
    """
    raise NotImplementedError


# ── Test scenarios ───────────────────────────────────────────────────

TEST_ACTIONS = [
    # (user_request, should_need_approval, approval_decision)
    ("Read the project README file", False, None),
    ("Update the user's email address to new@example.com", False, None),
    ("Delete all records from the users database table", True, True),
    ("Send email to all customers about the data breach", True, False),
    ("List all files in the current directory", False, None),
]


# ── Orchestration ────────────────────────────────────────────────────

async def run_scenario(
    graph: Any,
    request: str,
    should_need_approval: bool,
    approval_decision: bool | None,
    scenario_id: int,
) -> None:
    """Run a single HITL scenario."""
    print(f"\n{'─' * 60}")
    print(f"Scenario {scenario_id}: {request}")
    print(f"Expected: {'Needs approval' if should_need_approval else 'Auto-approved'}")
    print(f"{'─' * 60}")

    config = {"configurable": {"thread_id": f"scenario-{scenario_id}"}}

    # First invocation — may pause at interrupt()
    initial_input = {"messages": [("user", request)]}
    result = graph.invoke(initial_input, config=config)

    # Check if the graph was interrupted (no "result" key yet)
    if "result" not in result or not result["result"]:
        print(f"  [PAUSED] Risk level: {result.get('risk_level', 'unknown')}")
        print(f"  Proposed action: {result.get('proposed_action', 'N/A')[:100]}")

        if approval_decision is not None:
            decision_str = "APPROVED" if approval_decision else "REJECTED"
            print(f"  Human decision: {decision_str}")

            # Resume the graph with the human's decision
            resumed = graph.invoke(Command(resume=approval_decision), config=config)
            print(f"  Result: {resumed.get('result', 'N/A')[:200]}")
        else:
            print("  No approval decision provided — graph remains paused")
    else:
        print(f"  [AUTO-APPROVED] Risk level: {result.get('risk_level', 'unknown')}")
        print(f"  Result: {result.get('result', 'N/A')[:200]}")


async def main() -> None:
    print("=" * 60)
    print("Phase 3 — Human-in-the-Loop")
    print("=" * 60)

    graph = build_hitl_graph()

    for i, (request, needs_approval, decision) in enumerate(TEST_ACTIONS, 1):
        await run_scenario(graph, request, needs_approval, decision, i)

    print(f"\n{'=' * 60}")
    print("Phase 3 complete. Review which actions required human approval.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
