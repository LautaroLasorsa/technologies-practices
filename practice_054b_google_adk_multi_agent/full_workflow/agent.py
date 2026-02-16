"""Exercise 5: Nested Orchestration — Full Workflow.

This exercise combines Sequential, Parallel, and Loop patterns into a
complete multi-stage workflow. This is how real-world ADK applications
are structured — nested workflow agents creating complex orchestration
from simple, composable pieces.

Architecture:
    SequentialAgent("full_workflow")
    ├── Stage 1: ParallelAgent("parallel_research")
    │   ├── ProAgent      — researches pros → state["pros"]
    │   └── ConAgent      — researches cons → state["cons"]
    ├── Stage 2: LoopAgent("draft_refinement", max_iterations=2)
    │   ├── DraftAgent    — writes/revises analysis → state["draft"]
    │   └── ReviewAgent   — evaluates, escalates if good
    └── Stage 3: FinalAgent — produces polished final output

Workflow in plain English:
    1. Two agents research pros and cons of a topic IN PARALLEL
    2. A writer agent drafts an analysis, reviewed iteratively IN A LOOP
    3. A final agent produces the polished output SEQUENTIALLY

Key concept — nesting workflow agents:
    ADK workflow agents are composable. Any sub_agent slot can contain
    another workflow agent, creating arbitrary depth:

        SequentialAgent(sub_agents=[
            ParallelAgent(sub_agents=[...]),   # Stage 1
            LoopAgent(sub_agents=[...]),        # Stage 2
            LlmAgent(...),                      # Stage 3
        ])

    Each workflow agent handles its own orchestration pattern:
    - SequentialAgent runs its children in order
    - ParallelAgent runs its children concurrently
    - LoopAgent repeats its children until escalate or max_iterations

Key concept — state flows through the entire tree:
    All agents in the tree share the SAME session state. This means:
    - Stage 1's parallel agents write to state["pros"] and state["cons"]
    - Stage 2's loop agents can read those keys
    - Stage 3's final agent can read everything written by Stages 1 and 2
    The state is the "glue" that connects all stages.
"""

from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext

MODEL = LiteLlm(model="ollama_chat/qwen2.5:7b", api_base="http://localhost:11434")


# ---------------------------------------------------------------------------
# Tools for Stage 1: Parallel research (pros and cons)
# ---------------------------------------------------------------------------


def research_pros(topic: str, tool_context: ToolContext) -> str:
    """Research and store the advantages/pros of a given topic.

    Simulates research by generating structured pro arguments.
    Stores results in state["pros"] for downstream stages.

    Args:
        topic: The subject to research pros for.
        tool_context: ADK-injected context for state access.

    Returns:
        Summary of pros found.
    """
    pros = [
        f"{topic} increases efficiency and productivity",
        f"{topic} reduces costs in the long term",
        f"{topic} enables innovation and new possibilities",
        f"{topic} improves scalability and flexibility",
    ]
    tool_context.state["pros"] = pros
    tool_context.state["topic"] = topic
    return f"Found {len(pros)} advantages of {topic}."


def research_cons(topic: str, tool_context: ToolContext) -> str:
    """Research and store the disadvantages/cons of a given topic.

    Simulates research by generating structured con arguments.
    Stores results in state["cons"] for downstream stages.

    Args:
        topic: The subject to research cons for.
        tool_context: ADK-injected context for state access.

    Returns:
        Summary of cons found.
    """
    cons = [
        f"{topic} has a steep learning curve",
        f"{topic} requires significant upfront investment",
        f"{topic} may introduce complexity and maintenance burden",
    ]
    tool_context.state["cons"] = cons
    return f"Found {len(cons)} disadvantages of {topic}."


# ---------------------------------------------------------------------------
# Tools for Stage 2: Iterative drafting
# ---------------------------------------------------------------------------


def write_analysis(draft_text: str, tool_context: ToolContext) -> str:
    """Save the current analysis draft to state.

    The draft agent calls this to store its work. It should incorporate
    the pros and cons from Stage 1 into a balanced analysis.

    Args:
        draft_text: The current version of the analysis.
        tool_context: ADK-injected context for state access.

    Returns:
        Confirmation with draft metadata.
    """
    draft_num = tool_context.state.get("draft_number", 0) + 1
    tool_context.state["draft_number"] = draft_num
    tool_context.state["draft"] = draft_text
    return f"Analysis draft v{draft_num} saved ({len(draft_text)} chars)."


def get_research_data(tool_context: ToolContext) -> str:
    """Read the pros and cons gathered during Stage 1 (parallel research).

    The draft agent calls this to access the research data before writing.

    Args:
        tool_context: ADK-injected context for state access.

    Returns:
        Formatted string of pros and cons.
    """
    topic = tool_context.state.get("topic", "unknown topic")
    pros = tool_context.state.get("pros", [])
    cons = tool_context.state.get("cons", [])

    if not pros and not cons:
        return "No research data found in state. Stage 1 may not have completed."

    pros_text = "\n".join(f"  + {p}" for p in pros) if pros else "  (none found)"
    cons_text = "\n".join(f"  - {c}" for c in cons) if cons else "  (none found)"

    return f"Research data for '{topic}':\nPros:\n{pros_text}\nCons:\n{cons_text}"


def review_analysis(tool_context: ToolContext) -> str:
    """Review the current analysis draft and decide if it's ready.

    Evaluates the draft on completeness and balance. If good enough,
    sets escalate=True to exit the loop. Otherwise returns feedback.

    The scoring is simulated — improves with each draft version.
    In production, this would use real evaluation criteria.

    Args:
        tool_context: ADK-injected context for state access.

    Returns:
        Review results with score and feedback.
    """
    draft = tool_context.state.get("draft", "")
    draft_num = tool_context.state.get("draft_number", 0)

    if not draft:
        return "No draft found to review."

    # Simulated scoring — second draft passes
    score = min(5 + draft_num * 3, 10)

    if score >= 8:
        tool_context.actions.escalate = True
        feedback = f"Score: {score}/10 — Analysis is thorough and balanced. Approved!"
        tool_context.state["review_feedback"] = feedback
        return f"APPROVED. {feedback}"

    feedback = (
        f"Score: {score}/10 — Needs improvement.\n"
        f"Suggestions: Ensure both pros and cons are addressed equally. "
        f"Add more nuance and specific examples."
    )
    tool_context.state["review_feedback"] = feedback
    return feedback


# ---------------------------------------------------------------------------
# Tools for Stage 3: Final output
# ---------------------------------------------------------------------------


def compile_final_output(tool_context: ToolContext) -> str:
    """Compile all data into a final structured output.

    Reads the approved draft, research data, and review feedback to
    produce the final polished result.

    Args:
        tool_context: ADK-injected context for state access.

    Returns:
        All compiled data for the final agent to polish.
    """
    topic = tool_context.state.get("topic", "unknown")
    draft = tool_context.state.get("draft", "No draft available.")
    pros = tool_context.state.get("pros", [])
    cons = tool_context.state.get("cons", [])
    draft_num = tool_context.state.get("draft_number", 0)

    return (
        f"=== Compilation for Final Output ===\n"
        f"Topic: {topic}\n"
        f"Drafts written: {draft_num}\n"
        f"Pro arguments: {len(pros)}\n"
        f"Con arguments: {len(cons)}\n"
        f"Approved draft:\n{draft}\n"
        f"===================================="
    )


# ---------------------------------------------------------------------------
# TODO(human): Build the complete nested orchestration workflow.
# ---------------------------------------------------------------------------
#
# Your task is to create agents for each stage and compose them into a
# nested workflow: Sequential → [Parallel, Loop, Final].
#
# Step 1 — Create Stage 1 agents (Parallel Research)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create two LlmAgents that research pros and cons concurrently:
#
#   pro_agent = LlmAgent(
#       name="pro_researcher",
#       model=MODEL,
#       instruction: Tell it to research the advantages of the user's topic
#           by calling `research_pros`. It should identify and articulate
#           the key benefits.
#       description: "Researches advantages and benefits of a topic."
#       tools: [research_pros],
#       output_key: "pro_output",
#   )
#
#   con_agent = LlmAgent(
#       name="con_researcher",
#       model=MODEL,
#       instruction: Tell it to research the disadvantages by calling
#           `research_cons`. It should identify and articulate the key risks
#           and drawbacks.
#       description: "Researches disadvantages and risks of a topic."
#       tools: [research_cons],
#       output_key: "con_output",
#   )
#
# Then wrap them in a ParallelAgent:
#
#   parallel_research = ParallelAgent(
#       name="parallel_research",
#       sub_agents=[pro_agent, con_agent],
#       description="Researches pros and cons of a topic concurrently.",
#   )
#
# Step 2 — Create Stage 2 agents (Loop: Draft + Review)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a draft writer and reviewer, wrapped in a LoopAgent:
#
#   draft_agent = LlmAgent(
#       name="draft_writer",
#       model=MODEL,
#       instruction: Tell it to read the research data (call
#           `get_research_data`), then write a balanced analysis
#           covering both pros and cons. Save it via `write_analysis`.
#           If there's review feedback from a previous iteration
#           (in state["review_feedback"]), address that feedback.
#       description: "Writes balanced analysis drafts from research data."
#       tools: [get_research_data, write_analysis],
#   )
#
#   review_agent = LlmAgent(
#       name="draft_reviewer",
#       model=MODEL,
#       instruction: Tell it to review the current draft by calling
#           `review_analysis`. Report the score and feedback.
#       description: "Reviews analysis drafts for quality and balance."
#       tools: [review_analysis],
#   )
#
#   draft_loop = LoopAgent(
#       name="draft_refinement",
#       sub_agents=[draft_agent, review_agent],
#       max_iterations=2,
#       description="Iteratively writes and reviews analysis until approved.",
#   )
#
# Step 3 — Create Stage 3 agent (Final Output)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create an LlmAgent that produces the polished final output:
#
#   final_agent = LlmAgent(
#       name="final_presenter",
#       model=MODEL,
#       instruction: Tell it to call `compile_final_output` to get all
#           the data, then produce a polished, well-structured final
#           analysis. It should present the topic, pros, cons, and a
#           balanced conclusion.
#       description: "Produces the final polished analysis output."
#       tools: [compile_final_output],
#       output_key: "final_output",
#   )
#
# Step 4 — Compose the full workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Combine all three stages in a SequentialAgent:
#
#   root_agent = SequentialAgent(
#       name="full_workflow",
#       sub_agents=[parallel_research, draft_loop, final_agent],
#       description="Full workflow: parallel research → iterative drafting → final output.",
#   )
#
# The execution flow:
#   1. parallel_research runs pro_agent and con_agent concurrently
#      → state["pros"], state["cons"] populated
#   2. draft_loop runs draft_writer → draft_reviewer in a loop
#      → state["draft"] refined until approved (max 2 iterations)
#   3. final_agent reads everything and produces polished output
#      → state["final_output"] contains the result
#
# State contract for the full workflow:
#   ┌──────────────────┬────────────────────┬────────────────────────┐
#   │ Agent            │ Reads              │ Writes                 │
#   ├──────────────────┼────────────────────┼────────────────────────┤
#   │ pro_researcher   │ (user topic)       │ pros, topic, pro_output│
#   │ con_researcher   │ (user topic)       │ cons, con_output       │
#   │ draft_writer     │ pros, cons, topic, │ draft, draft_number    │
#   │                  │ review_feedback    │                        │
#   │ draft_reviewer   │ draft, draft_number│ review_feedback        │
#   │ final_presenter  │ topic, pros, cons, │ final_output           │
#   │                  │ draft, draft_number│                        │
#   └──────────────────┴────────────────────┴────────────────────────┘
#
# Test prompts to try:
#   - "Analyze the pros and cons of microservices architecture"
#   - "Give me a balanced analysis of remote work"
# Watch the full pipeline: parallel research → iterative drafting → final output.
# ---------------------------------------------------------------------------

raise NotImplementedError(
    "TODO(human): Create pro_agent, con_agent, parallel_research, "
    "draft_agent, review_agent, draft_loop, final_agent, and "
    "the root_agent SequentialAgent composing them all."
)
