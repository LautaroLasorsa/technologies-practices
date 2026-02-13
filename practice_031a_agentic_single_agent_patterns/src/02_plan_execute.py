"""Plan-and-Execute Agent — Upfront planning with step-by-step execution.

Pattern: Plan → Execute step 1 → Execute step 2 → ... → Replan if needed → Done

Unlike ReAct (which decides one action at a time), Plan-and-Execute creates a
full plan upfront, then executes each step. A replanner can adjust the plan
based on intermediate results. This pattern is better for complex, multi-step
tasks where having an overview helps the agent stay on track.

Run:
    uv run python src/02_plan_execute.py
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ── Configuration ────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"
MAX_REPLANS = 2

PLANNER_PROMPT = """You are a planning assistant. Given a question, create a step-by-step plan
to answer it. Each step should be a single, concrete action.

Output ONLY a numbered list of steps, nothing else. Example format:
1. Search for the population of France
2. Search for the population of Germany
3. Calculate the sum of both populations
4. Present the final answer

Keep plans short (3-6 steps). Each step should be independently executable."""

EXECUTOR_PROMPT = """You are an execution assistant. You are given a single step from a plan
and any results from previously completed steps.

Execute the step to the best of your ability. If the step asks to search for
something, provide the information you know. If it asks to calculate, show
the calculation and result.

Be concise and factual. Output ONLY the result of executing this step."""

REPLANNER_PROMPT = """You are a replanning assistant. You are given:
1. The original question
2. The original plan
3. Steps completed so far with their results

Decide if the remaining plan needs adjustment based on what we've learned.

If the plan is still good, output: PLAN_OK
If the plan needs changes, output a NEW numbered list of remaining steps only.
Do NOT repeat already-completed steps."""


# ── State ────────────────────────────────────────────────────────────


class PlanExecuteState(TypedDict):
    """State for the Plan-and-Execute agent.

    question: The original user question.
    plan: List of step descriptions (strings). Created by planner, may be
          modified by replanner.
    completed_steps: List of (step_description, result) tuples. Grows as
                     the executor processes each step.
    current_result: The result of the most recently executed step. Used by
                    replanner to decide if the plan needs adjustment.
    replan_count: Number of times the plan has been revised. Bounded by
                  MAX_REPLANS to prevent infinite replanning loops.
    final_answer: Set when the agent has enough information to answer.
    """

    question: str
    plan: list[str]
    completed_steps: list[tuple[str, str]]
    current_result: str
    replan_count: int
    final_answer: str


# ── LLM ──────────────────────────────────────────────────────────────

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)


# ── TODO(human): Implement these three functions ─────────────────────


def planner_node(state: PlanExecuteState) -> dict:
    """Create a step-by-step plan for answering the question.

    TODO(human): Implement this function.

    The planner is the first node in the graph. It receives the user's
    question and produces a numbered plan. The plan quality determines
    how well the executor can do its job — a vague plan leads to vague
    execution. This mirrors real-world project planning: decompose a
    complex task into concrete, independently-executable steps.

    Steps:
      1. Build messages: SystemMessage with PLANNER_PROMPT, then
         HumanMessage with state["question"].
      2. Invoke the LLM: llm.invoke(messages).
      3. Parse the response content into a list of step strings. The model
         outputs a numbered list like "1. Do X\n2. Do Y\n3. Do Z".
         Split by newlines, strip numbering prefixes (e.g., "1. "), and
         filter out empty lines.
      4. Return {"plan": parsed_steps, "completed_steps": [], "replan_count": 0}.

    Parsing hint:
      steps = []
      for line in response.content.strip().split("\\n"):
          line = line.strip()
          if line and line[0].isdigit():
              # Remove "1. ", "2. ", etc.
              step = line.split(". ", 1)[-1] if ". " in line else line
              steps.append(step)

    Why structured plans matter: Without a plan, the agent makes one-at-a-time
    decisions (ReAct). With a plan, it can see the full picture, avoid redundant
    steps, and the human can review/edit the plan before execution starts.
    """
    raise NotImplementedError("TODO(human): implement planner_node")


def executor_node(state: PlanExecuteState) -> dict:
    """Execute the next step in the plan.

    TODO(human): Implement this function.

    The executor takes the next unexecuted step from the plan and runs it.
    It has access to all previously completed steps and their results, which
    provides context for the current step (e.g., "calculate the sum" needs
    to know what numbers were found in previous steps).

    Steps:
      1. Determine the next step index: len(state["completed_steps"]).
         This is the index into state["plan"] for the next step to execute.
      2. If the index >= len(state["plan"]), all steps are done — return
         {"final_answer": "All steps completed. " + summary_of_results}.
      3. Get the current step string: state["plan"][step_index].
      4. Build context from completed steps. Format like:
         "Previously completed:\n- Step: X → Result: Y\n- Step: A → Result: B"
      5. Build messages: SystemMessage with EXECUTOR_PROMPT, then
         HumanMessage with the step to execute + the context.
      6. Invoke the LLM.
      7. Append (step_description, response.content) to completed_steps.
      8. Return {"completed_steps": updated_list, "current_result": response.content}.

    Key insight: The executor is "dumb" — it doesn't decide WHAT to do, only
    HOW to do the step it's given. This separation of planning and execution
    makes the system more auditable and debuggable than ReAct.
    """
    raise NotImplementedError("TODO(human): implement executor_node")


def replanner_node(state: PlanExecuteState) -> dict:
    """Review progress and optionally adjust the remaining plan.

    TODO(human): Implement this function.

    After each step execution, the replanner checks whether the plan still
    makes sense given what we've learned. Maybe a search returned unexpected
    results, or a step revealed the question needs a different approach.

    Steps:
      1. Build a summary of progress: the original question, the original plan,
         and all completed steps with their results.
      2. Build messages: SystemMessage with REPLANNER_PROMPT, then
         HumanMessage with the progress summary.
      3. Invoke the LLM.
      4. If the response contains "PLAN_OK" → return {"replan_count": state["replan_count"]}
         (no changes needed, keep the current plan).
      5. Otherwise, parse the response as a new list of remaining steps
         (same parsing as planner_node). Combine completed step descriptions
         with these new remaining steps to form the updated plan.
      6. Increment replan_count.
      7. Return {"plan": updated_plan, "replan_count": new_count}.

    Trade-off: Replanning adds an extra LLM call per step. For simple tasks,
    it's overhead. For complex tasks with uncertain intermediate results, it's
    essential. In production, you might only replan every N steps or when a
    step fails.
    """
    raise NotImplementedError("TODO(human): implement replanner_node")


# ── Routing logic (provided) ────────────────────────────────────────


def should_continue(state: PlanExecuteState) -> str:
    """Route after executor: replan, continue executing, or finish."""
    # If we have a final answer, we're done
    if state.get("final_answer"):
        return "end"

    # If all steps are completed, we're done
    if len(state["completed_steps"]) >= len(state["plan"]):
        return "end"

    # If we haven't exhausted replan attempts, check if replanning is needed
    if state["replan_count"] < MAX_REPLANS:
        return "replan"

    # Otherwise, keep executing
    return "execute"


def finalize_node(state: PlanExecuteState) -> dict:
    """Produce a final answer from all completed steps."""
    if state.get("final_answer"):
        return {}

    # Summarize all completed steps into a final answer
    summary_parts = [f"Question: {state['question']}\n"]
    for step, result in state["completed_steps"]:
        summary_parts.append(f"- {step}: {result}")

    messages = [
        SystemMessage(content="Summarize the following research into a concise final answer."),
        HumanMessage(content="\n".join(summary_parts)),
    ]
    response = llm.invoke(messages)
    return {"final_answer": response.content}


def build_plan_execute_graph():
    """Wire the Plan-and-Execute graph."""
    graph = StateGraph(PlanExecuteState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("replanner", replanner_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges(
        "executor",
        should_continue,
        {"replan": "replanner", "execute": "executor", "end": "finalize"},
    )
    graph.add_edge("replanner", "executor")
    graph.add_edge("finalize", END)

    return graph.compile()


# ── Orchestration ────────────────────────────────────────────────────


def run_plan_execute(question: str) -> None:
    """Run the Plan-and-Execute agent and print the trace."""
    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print("=" * 60)

    agent = build_plan_execute_graph()
    initial_state: PlanExecuteState = {
        "question": question,
        "plan": [],
        "completed_steps": [],
        "current_result": "",
        "replan_count": 0,
        "final_answer": "",
    }

    for event in agent.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n--- {node_name} ---")
            if node_name == "planner" and "plan" in node_output:
                print("  Plan:")
                for i, step in enumerate(node_output["plan"], 1):
                    print(f"    {i}. {step}")
            elif node_name == "executor" and "current_result" in node_output:
                step_idx = len(node_output.get("completed_steps", [])) - 1
                if step_idx >= 0 and node_output.get("completed_steps"):
                    step_desc, result = node_output["completed_steps"][-1]
                    print(f"  Step: {step_desc}")
                    print(f"  Result: {result[:200]}")
            elif node_name == "replanner":
                if "plan" in node_output:
                    print("  Revised plan:")
                    for i, step in enumerate(node_output["plan"], 1):
                        print(f"    {i}. {step}")
                else:
                    print("  Plan OK — no changes needed")
            elif node_name == "finalize" and "final_answer" in node_output:
                print(f"  Final answer: {node_output['final_answer'][:300]}")

    print(f"\n{'=' * 60}\n")


def main() -> None:
    print("Practice 031a — Phase 2: Plan-and-Execute Agent")
    print("Structured planning with step-by-step execution\n")

    # Test 1: Multi-step research question
    run_plan_execute(
        "Compare the populations of France and Japan, then calculate "
        "which country has more people and by how much."
    )

    # Test 2: Sequential reasoning
    run_plan_execute(
        "What is LangGraph, who created it, and how does it relate to the ReAct pattern?"
    )


if __name__ == "__main__":
    main()
