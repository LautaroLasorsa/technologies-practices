"""
Phase 3: CrewAI — Role-Based Multi-Agent Teams

Architecture:
    [Researcher Agent] --output--> [Analyst Agent] --output--> [Writer Agent]
    (sequential process: each task feeds the next)

CrewAI takes a fundamentally different approach from LangGraph: instead of building
explicit graphs, you define agents by role/goal/backstory and tasks with dependencies.
The framework handles orchestration.

This file compares the same task using CrewAI's sequential process vs LangGraph's
supervisor/swarm from Phases 1-2.
"""

from __future__ import annotations

from crewai import Agent, Crew, LLM, Process, Task

# ---------------------------------------------------------------------------
# Configuration — CrewAI uses its own LLM class (backed by LiteLLM)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"

# CrewAI's LLM class wraps LiteLLM. For Ollama, prefix the model name with "ollama/".
crewai_llm = LLM(
    model=f"ollama/{MODEL_NAME}",
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
)

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


def create_agents() -> dict[str, Agent]:
    """Create the three CrewAI agents: researcher, analyst, writer.

    Returns a dict mapping role names to Agent objects.
    """
    # TODO(human) #1: Define three CrewAI Agents with role, goal, backstory, and LLM.
    #
    # CrewAI's Agent abstraction is built around the idea that giving an LLM a
    # "persona" (role + goal + backstory) produces better focused outputs than
    # a generic system prompt. This mirrors how real teams work: a data analyst
    # thinks differently from a copywriter.
    #
    # For each agent, create:
    #   Agent(
    #       role="<descriptive role title>",
    #       goal="<what this agent is trying to achieve>",
    #       backstory="<1-2 sentence persona that shapes reasoning style>",
    #       llm=crewai_llm,
    #       verbose=True,  # Shows the agent's thinking process
    #   )
    #
    # Create these three agents:
    #   - "researcher": role="Research Specialist", goal is to find accurate factual
    #     information, backstory mentions years of research experience
    #   - "analyst": role="Data Analyst", goal is to analyze and synthesize information
    #     into structured insights, backstory mentions analytical rigor
    #   - "writer": role="Content Writer", goal is to produce clear engaging text,
    #     backstory mentions writing expertise and audience awareness
    #
    # Return as: {"researcher": researcher_agent, "analyst": analyst_agent, "writer": writer_agent}
    #
    # Compare with LangGraph: In Phase 1 you wrote system prompts manually and wired
    # them into graph nodes. CrewAI abstracts this into role/goal/backstory, which
    # the framework converts into an optimized system prompt internally. Less control,
    # but faster to set up.
    raise NotImplementedError("TODO(human) #1: Define CrewAI agents")


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------


def create_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Create tasks with dependencies for a sequential pipeline.

    Returns an ordered list of Tasks.
    """
    # TODO(human) #2: Define Tasks with descriptions, expected outputs, and dependencies.
    #
    # CrewAI Tasks are the unit of work. Each task has:
    #   - description: what the agent should do (natural language instruction)
    #   - expected_output: what the result should look like (guides the agent)
    #   - agent: which Agent handles this task
    #   - context (optional): list of Tasks whose output feeds into this task
    #
    # Create three tasks in sequence:
    #
    #   1. research_task = Task(
    #          description="Research the following topic and provide key facts, "
    #                      "statistics, and important details: {topic}",
    #          expected_output="A structured list of 5-7 key facts with brief explanations",
    #          agent=agents["researcher"],
    #      )
    #
    #   2. analysis_task = Task(
    #          description="Analyze the research findings and identify the 3 most "
    #                      "important insights. Explain why each matters.",
    #          expected_output="3 numbered insights with 2-3 sentence explanations each",
    #          agent=agents["analyst"],
    #          context=[research_task],  # <-- dependency: receives research output
    #      )
    #
    #   3. writing_task = Task(
    #          description="Write a concise, engaging summary (150-200 words) based on "
    #                      "the analysis. Make it accessible to a general audience.",
    #          expected_output="A polished 150-200 word summary paragraph",
    #          agent=agents["writer"],
    #          context=[analysis_task],  # <-- dependency: receives analysis output
    #      )
    #
    # Return: [research_task, analysis_task, writing_task]
    #
    # The `context` parameter is how CrewAI handles agent communication. When
    # analysis_task runs, it automatically receives research_task's output as
    # additional context. This is cleaner than LangGraph's message-passing
    # (where all agents see all messages) but less flexible — you can only pass
    # output forward, not have arbitrary communication patterns.
    raise NotImplementedError("TODO(human) #2: Define CrewAI tasks with dependencies")


# ---------------------------------------------------------------------------
# Crew construction and execution
# ---------------------------------------------------------------------------


def build_and_run_crew(topic: str) -> str:
    """Build a Crew with sequential process and run it on a topic.

    Returns the final output string.
    """
    # TODO(human) #3: Create a Crew and execute it.
    #
    # The Crew is CrewAI's top-level orchestrator. It takes agents, tasks, and a
    # process type, then executes the workflow.
    #
    # Steps:
    #   1. Call create_agents() to get the agent dict.
    #   2. Call create_tasks(agents) to get the task list.
    #   3. Create the Crew:
    #      crew = Crew(
    #          agents=list(agents.values()),  # all agents
    #          tasks=tasks,                    # ordered task list
    #          process=Process.sequential,     # execute tasks in order
    #          verbose=True,                   # show execution details
    #      )
    #   4. Kick off execution with the topic as input:
    #      result = crew.kickoff(inputs={"topic": topic})
    #   5. Return str(result) — CrewAI's result object has a string representation
    #      containing the final task's output.
    #
    # Process.sequential means: research_task runs first, its output feeds into
    # analysis_task, which feeds into writing_task. This is the simplest process.
    # CrewAI also supports Process.hierarchical (a manager agent delegates) but
    # that requires a separate manager_llm configuration.
    #
    # Compare total code: This entire crew setup is ~30 lines of code vs ~80 lines
    # for the LangGraph supervisor. The trade-off: CrewAI makes many decisions for
    # you (prompt construction, output passing, agent selection), which is great for
    # prototyping but limits fine-grained control.
    raise NotImplementedError("TODO(human) #3: Build and run the CrewAI crew")


# ---------------------------------------------------------------------------
# Test topics (comparable to Phases 1-2 queries)
# ---------------------------------------------------------------------------

TEST_TOPICS = [
    "The current state of quantum computing",
    "The mathematical properties of prime numbers",
    "How distributed systems handle failures",
]


def main() -> None:
    print("=" * 70)
    print("Phase 3: CrewAI — Role-Based Multi-Agent Teams")
    print("=" * 70)

    for i, topic in enumerate(TEST_TOPICS, 1):
        print(f"\n{'─' * 70}")
        print(f"Topic {i}: {topic}")
        print("─" * 70)

        result = build_and_run_crew(topic)
        print(f"\nFinal output:\n{result}")


if __name__ == "__main__":
    main()
