"""Exercise 4: LoopAgent — Iterative Refinement.

This exercise demonstrates ADK's LoopAgent, which repeatedly executes its
sub-agents in sequence until a termination condition is met. This models
"perfectionist" patterns: write, critique, improve, critique, improve...

Architecture:
    LoopAgent("refinement_loop", max_iterations=3)
    ├── WriterAgent  — writes/improves a draft → state["current_draft"]
    └── CriticAgent  — evaluates quality, decides if done → escalate if good

How LoopAgent works internally:
- It runs sub_agents in order (like SequentialAgent) inside a loop
- After each full cycle, it checks two termination conditions:
    1. Has `max_iterations` been reached? If yes, stop.
    2. Did any sub-agent yield an event with `actions.escalate = True`?
       If yes, stop the loop.
- If neither condition is met, it loops again from the first sub-agent
- All iterations share the same session state — the writer can read its
  own previous draft and the critic's feedback

Key concept — escalate to exit:
    LoopAgent does NOT have a `condition_key` or `exit_condition` parameter
    in the current API. Instead, termination is controlled by:

    1. `max_iterations` — hard limit on loop count
    2. `tool_context.actions.escalate = True` — a tool in any sub-agent
       can set this to signal that the loop should stop early

    The escalate pattern:
        def evaluate_quality(tool_context: ToolContext) -> str:
            score = tool_context.state.get("quality_score", 0)
            if score >= threshold:
                tool_context.actions.escalate = True  # Exits the loop
                return "Quality threshold met!"
            return "Needs improvement."

Key concept — iteration tracking:
    The LoopAgent internally tracks iteration count. Your agents can also
    maintain their own counter in state for more nuanced control:
        iteration = tool_context.state.get("iteration", 0) + 1
        tool_context.state["iteration"] = iteration
"""

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext

MODEL = LiteLlm(model="ollama_chat/qwen2.5:7b", api_base="http://localhost:11434")

# Quality threshold — the critic will signal loop exit when score >= this
QUALITY_THRESHOLD = 8


# ---------------------------------------------------------------------------
# Tools for the writer and critic agents
# ---------------------------------------------------------------------------


def save_draft(draft: str, tool_context: ToolContext) -> str:
    """Save or update the current draft in session state.

    The writer agent calls this to store its work. On subsequent iterations,
    the writer reads the previous draft and the critic's feedback to improve.

    Args:
        draft: The current version of the text.
        tool_context: ADK-injected context for state access.

    Returns:
        Confirmation with iteration number.
    """
    iteration = tool_context.state.get("iteration", 0) + 1
    tool_context.state["iteration"] = iteration
    tool_context.state["current_draft"] = draft

    # Keep history of all drafts for comparison
    history: list[str] = tool_context.state.get("draft_history", [])
    history.append(draft)
    tool_context.state["draft_history"] = history

    return f"Draft v{iteration} saved ({len(draft)} chars)."


def get_previous_feedback(tool_context: ToolContext) -> str:
    """Retrieve the critic's feedback from the previous iteration.

    The writer calls this at the start of each iteration (after the first)
    to understand what needs improvement.

    Args:
        tool_context: ADK-injected context for state access.

    Returns:
        The previous feedback, or a message if this is the first iteration.
    """
    feedback = tool_context.state.get("critic_feedback", None)
    iteration = tool_context.state.get("iteration", 0)
    if feedback and iteration > 0:
        return f"Feedback from previous review:\n{feedback}"
    return "This is the first iteration — no previous feedback. Write an initial draft."


def evaluate_draft(tool_context: ToolContext) -> str:
    """Evaluate the current draft's quality and decide whether to continue.

    Reads the current draft from state, assigns a quality score, and
    provides feedback. If the score meets the threshold, sets
    `tool_context.actions.escalate = True` to exit the loop.

    The scoring here is simulated — it increases with each iteration to
    demonstrate the loop pattern. In a real system, you'd use actual
    evaluation criteria (readability score, fact-checking, etc.).

    Args:
        tool_context: ADK-injected context for state access.

    Returns:
        Evaluation results with score and feedback.
    """
    draft = tool_context.state.get("current_draft", "")
    iteration = tool_context.state.get("iteration", 1)

    if not draft:
        tool_context.state["critic_feedback"] = "No draft found to evaluate."
        return "No draft found. The writer needs to produce a draft first."

    # Simulated scoring — increases with iterations and draft length
    # In a real system, this would be actual NLP evaluation
    base_score = min(iteration * 3, 7)  # Base score grows with iterations
    length_bonus = min(len(draft) / 200, 2)  # Bonus for longer, more detailed drafts
    score = min(round(base_score + length_bonus, 1), 10)

    tool_context.state["quality_score"] = score

    if score >= QUALITY_THRESHOLD:
        # Signal the LoopAgent to stop — quality threshold met!
        tool_context.actions.escalate = True
        feedback = f"Score: {score}/10 — Excellent! The draft meets quality standards."
        tool_context.state["critic_feedback"] = feedback
        return f"APPROVED. {feedback}"

    # Generate improvement suggestions based on iteration
    suggestions = []
    if iteration == 1:
        suggestions = ["Add more specific examples", "Expand on key points", "Improve structure"]
    elif iteration == 2:
        suggestions = ["Refine transitions between ideas", "Add a strong conclusion"]
    else:
        suggestions = ["Polish language and clarity", "Ensure all points are well-supported"]

    feedback = (
        f"Score: {score}/10 (threshold: {QUALITY_THRESHOLD}/10) — Needs improvement.\n"
        f"Suggestions:\n" + "\n".join(f"  - {s}" for s in suggestions)
    )
    tool_context.state["critic_feedback"] = feedback
    return feedback


# ---------------------------------------------------------------------------
# TODO(human): Create the writer agent, critic agent, and LoopAgent.
# ---------------------------------------------------------------------------
#
# Your task is to create a writer + critic loop that iteratively refines
# a piece of text until it meets a quality threshold.
#
# Step 1 — Create the Writer agent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create an LlmAgent named "writer_agent" that:
#   - model: MODEL
#   - instruction: Tell it that it's an iterative writer. On the FIRST
#     iteration, it should write an initial draft about whatever topic
#     the user requested. On SUBSEQUENT iterations, it should call
#     `get_previous_feedback` first to read the critic's feedback, then
#     revise its draft to address the suggestions. It should always call
#     `save_draft` with its improved text.
#
#     Suggested instruction structure:
#       "You are a skilled writer that iteratively improves text.
#        1. Call get_previous_feedback to check if there's feedback to address.
#        2. If there is feedback, revise the draft to address each suggestion.
#        3. If there is no feedback, write an initial draft on the topic.
#        4. Save your draft using save_draft.
#        Focus on clarity, detail, and engaging writing."
#
#   - description: "Writes and iteratively improves text drafts."
#   - tools: [save_draft, get_previous_feedback]
#
# Step 2 — Create the Critic agent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create an LlmAgent named "critic_agent" that:
#   - model: MODEL
#   - instruction: Tell it to evaluate the current draft by calling
#     `evaluate_draft`. It should report the score and feedback.
#     If the draft is approved (score >= threshold), it should confirm
#     that the writing process is complete.
#
#     Suggested instruction structure:
#       "You are a writing critic. Call evaluate_draft to assess the
#        current draft. Report the score and suggestions back. If the
#        draft is approved, confirm completion."
#
#   - description: "Evaluates draft quality and provides improvement feedback."
#   - tools: [evaluate_draft]
#
# Step 3 — Create the LoopAgent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Wrap the writer and critic in a LoopAgent:
#
#   root_agent = LoopAgent(
#       name="iterative_writer",
#       sub_agents=[writer_agent, critic_agent],
#       max_iterations=3,
#       description="Iteratively writes and refines text until quality threshold is met.",
#   )
#
# How the loop executes:
#   Iteration 1: writer writes draft → critic evaluates (score ~3-5, not enough)
#   Iteration 2: writer reads feedback + improves → critic evaluates (score ~6-8)
#   Iteration 3: writer reads feedback + improves → critic evaluates (score >= 8, escalates!)
#   Loop exits — the final approved draft is in state["current_draft"]
#
# The LoopAgent will stop when EITHER:
#   - max_iterations (3) is reached, OR
#   - evaluate_draft sets tool_context.actions.escalate = True
#
# Test prompts to try:
#   - "Write a short essay about the benefits of open source software"
#   - "Write a product description for a smart water bottle"
# Watch the writer improve across iterations based on the critic's feedback.
# ---------------------------------------------------------------------------

raise NotImplementedError(
    "TODO(human): Create writer_agent, critic_agent, and the "
    "root_agent LoopAgent wrapping them with max_iterations=3."
)
