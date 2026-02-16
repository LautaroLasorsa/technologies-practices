"""Exercise 2: SequentialAgent — Research Pipeline.

This exercise demonstrates ADK's SequentialAgent, which executes child
agents in a fixed order — no LLM reasoning for routing. Each stage
processes the accumulated state from previous stages, creating a pipeline.

Architecture:
    SequentialAgent("research_pipeline")
    ├── Stage 1: GatherAgent   — collects raw information into state
    ├── Stage 2: ProcessAgent  — analyzes/transforms the gathered data
    └── Stage 3: FormatAgent   — produces final formatted output

How SequentialAgent works internally:
- It iterates through `sub_agents` in list order
- Each child runs to completion before the next starts
- All children share the SAME session state (InvocationContext)
- No LLM is used for orchestration — it's purely deterministic
- It's like a Unix pipeline: stage1 | stage2 | stage3

Key concept — `output_key`:
    When you set `output_key="some_key"` on an LlmAgent, ADK automatically
    saves the agent's final text response to `state["some_key"]`. This is
    the primary mechanism for passing data between pipeline stages:

        gather_agent = LlmAgent(..., output_key="raw_data")
        # After gather_agent runs, state["raw_data"] contains its response

    The next agent can then read state["raw_data"] via a tool that accesses
    tool_context.state, or its instruction can reference the key name
    (the LLM sees the state as part of its context).

Key concept — state contracts:
    In a SequentialAgent pipeline, each stage should document:
    - What state keys it READS (its inputs)
    - What state keys it WRITES (its outputs)
    This makes the pipeline's data flow explicit and debuggable.

    Stage 1 (Gather):  reads nothing     → writes "raw_data"
    Stage 2 (Process): reads "raw_data"  → writes "analysis"
    Stage 3 (Format):  reads "analysis"  → writes "final_report"
"""

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext

MODEL = LiteLlm(model="ollama_chat/qwen2.5:7b", api_base="http://localhost:11434")


# ---------------------------------------------------------------------------
# Tools for each pipeline stage
# ---------------------------------------------------------------------------
# These tools let agents read from and write to shared session state.
# In a real system, Stage 1 might call external APIs; here we simulate
# with local data to keep the practice focused on the orchestration pattern.
# ---------------------------------------------------------------------------


def gather_information(topic: str, tool_context: ToolContext) -> str:
    """Gather raw information about a topic and store it in session state.

    This simulates fetching data from multiple sources. In a real system,
    this could call search APIs, databases, or web scrapers.

    Args:
        topic: The topic to research.
        tool_context: ADK-injected context for state access.

    Returns:
        Confirmation that data was gathered.
    """
    # Simulated raw data — in production this would be real API calls
    raw_data = {
        "topic": topic,
        "sources": [
            f"Source A: {topic} is an important concept in modern technology.",
            f"Source B: Recent developments in {topic} show significant progress.",
            f"Source C: Experts predict {topic} will grow 40% by next year.",
        ],
        "timestamp": "2025-01-01T00:00:00Z",
    }
    tool_context.state["raw_data"] = raw_data
    return f"Gathered {len(raw_data['sources'])} sources about '{topic}'. Data stored in state['raw_data']."


def analyze_data(tool_context: ToolContext) -> str:
    """Read raw_data from state, perform analysis, and store results.

    This simulates processing/transforming raw data into structured insights.

    Args:
        tool_context: ADK-injected context for state access.

    Returns:
        Summary of analysis results.
    """
    raw_data = tool_context.state.get("raw_data")
    if not raw_data:
        return "Error: No raw_data found in state. Gather stage may not have run."

    topic = raw_data.get("topic", "unknown")
    sources = raw_data.get("sources", [])

    analysis = {
        "topic": topic,
        "source_count": len(sources),
        "key_themes": [
            f"{topic} is recognized as important in technology",
            f"Active development and progress in {topic}",
            f"Strong growth predicted for {topic}",
        ],
        "sentiment": "positive",
        "confidence": 0.85,
    }
    tool_context.state["analysis"] = analysis
    return (
        f"Analysis complete for '{topic}': found {len(analysis['key_themes'])} themes, "
        f"sentiment={analysis['sentiment']}, confidence={analysis['confidence']}."
    )


def format_report(style: str, tool_context: ToolContext) -> str:
    """Read analysis from state and produce a formatted report.

    Args:
        style: The report style — "brief", "detailed", or "bullet_points".
        tool_context: ADK-injected context for state access.

    Returns:
        The formatted report string.
    """
    analysis = tool_context.state.get("analysis")
    if not analysis:
        return "Error: No analysis found in state. Process stage may not have run."

    topic = analysis["topic"]
    themes = analysis["key_themes"]

    if style == "bullet_points":
        theme_text = "\n".join(f"  - {t}" for t in themes)
        report = f"Research Report: {topic}\n{theme_text}\nSentiment: {analysis['sentiment']}"
    elif style == "brief":
        report = f"{topic}: {'; '.join(themes)}. Overall sentiment: {analysis['sentiment']}."
    else:
        report = (
            f"=== Detailed Report: {topic} ===\n"
            f"Sources analyzed: {analysis['source_count']}\n"
            f"Key themes:\n" + "\n".join(f"  {i+1}. {t}" for i, t in enumerate(themes)) + "\n"
            f"Sentiment: {analysis['sentiment']} (confidence: {analysis['confidence']})\n"
            f"=== End of Report ==="
        )

    tool_context.state["final_report"] = report
    return report


# ---------------------------------------------------------------------------
# TODO(human): Create the three pipeline stage agents and the SequentialAgent.
# ---------------------------------------------------------------------------
#
# Your task is to create 3 LlmAgent instances (one per stage) and combine
# them into a SequentialAgent that runs them in order.
#
# Step 1 — Create the Gather agent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create an LlmAgent named "gather_agent" that:
#   - model: MODEL
#   - instruction: Tell it to gather information about whatever topic the
#     user asked about. It should call the `gather_information` tool with
#     the user's topic. After calling the tool, it should summarize what
#     was gathered.
#   - description: "Gathers raw data about a research topic."
#   - tools: [gather_information]
#   - output_key: "gather_output"
#     (This saves the agent's final text response to state["gather_output"].
#      The next stage can read this via its instruction context.)
#
# Step 2 — Create the Process agent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create an LlmAgent named "process_agent" that:
#   - model: MODEL
#   - instruction: Tell it to analyze the gathered data by calling the
#     `analyze_data` tool. It should then describe the key findings.
#     Mention that the raw data is already in the session state from the
#     previous stage.
#   - description: "Analyzes gathered data and extracts key themes."
#   - tools: [analyze_data]
#   - output_key: "process_output"
#
# Step 3 — Create the Format agent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create an LlmAgent named "format_agent" that:
#   - model: MODEL
#   - instruction: Tell it to format the analysis into a report by calling
#     `format_report` with a style (e.g., "detailed"). It should present
#     the formatted report as its response.
#   - description: "Formats analysis results into a readable report."
#   - tools: [format_report]
#   - output_key: "final_report_output"
#
# Step 4 — Create the SequentialAgent pipeline
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Combine the three agents into a SequentialAgent:
#
#   root_agent = SequentialAgent(
#       name="research_pipeline",
#       sub_agents=[gather_agent, process_agent, format_agent],
#       description="A 3-stage research pipeline: Gather -> Process -> Format.",
#   )
#
# Note: SequentialAgent does NOT take a `model` parameter — it's not
# an LLM agent. It's a pure orchestrator that runs children in order.
#
# State contract for this pipeline:
#   ┌──────────────┬────────────────────────┬────────────────────────┐
#   │ Stage        │ Reads from state       │ Writes to state        │
#   ├──────────────┼────────────────────────┼────────────────────────┤
#   │ gather_agent │ (none)                 │ raw_data, gather_output│
#   │ process_agent│ raw_data               │ analysis, process_output│
#   │ format_agent │ analysis               │ final_report,          │
#   │              │                        │ final_report_output    │
#   └──────────────┴────────────────────────┴────────────────────────┘
#
# Test prompts to try:
#   - "Research the topic of quantum computing"
#   - "Investigate microservices architecture"
# Watch how each stage runs in order and builds on the previous stage's state.
# ---------------------------------------------------------------------------

raise NotImplementedError(
    "TODO(human): Create gather_agent, process_agent, format_agent, "
    "and the root_agent SequentialAgent pipeline."
)
