"""Phase 1 — PydanticAI: Type-Safe Agent with Structured Output.

Demonstrates PydanticAI's core value: agents that return validated Pydantic models
instead of raw strings. The LLM's output is automatically parsed and validated against
your schema — if the LLM produces invalid JSON or missing fields, PydanticAI retries
transparently.

Run:
    uv run python src/01_pydantic_agent.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


# ── Configuration ────────────────────────────────────────────────────

OLLAMA_MODEL = "ollama:qwen2.5:7b"

# PydanticAI uses "ollama:<model_name>" format to route to a local Ollama instance.
# By default it connects to http://localhost:11434.


# ── Dependency Context ───────────────────────────────────────────────

# PydanticAI agents can receive a "dependencies" object — a typed context bag
# that tools and the system prompt can access. This avoids global state and
# makes the agent testable (you can inject mock dependencies).

@dataclass
class AgentDeps:
    """Dependencies injected into the agent at runtime."""
    user_name: str
    knowledge_base: dict[str, str]  # topic -> fact


# Sample knowledge base for the agent's tools to query
KNOWLEDGE_BASE: dict[str, str] = {
    "python": "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.",
    "rust": "Rust is a systems programming language focused on safety, speed, and concurrency, first released in 2015.",
    "pydantic": "Pydantic is the most downloaded Python package, providing data validation using Python type annotations.",
    "langchain": "LangChain is a framework for developing applications powered by large language models.",
    "docker": "Docker is a platform for building, shipping, and running applications in containers.",
}


# ── TODO(human) #1: Define the result model ──────────────────────────

# PydanticAI's Agent requires a `result_type` — a Pydantic model that defines
# the structure of the agent's response. When the agent runs, PydanticAI
# instructs the LLM to produce JSON matching this schema and validates the
# output automatically. If validation fails, PydanticAI retries the LLM call.
#
# Define a Pydantic model called `ResearchResult` with these fields:
#   - answer: str         — The agent's response to the query
#   - confidence: float   — How confident the agent is (0.0 to 1.0)
#   - sources: list[str]  — List of sources/topics the agent consulted
#
# Use Field() to add descriptions and constraints (e.g., ge=0.0, le=1.0 for
# confidence). These descriptions become part of the prompt sent to the LLM,
# helping it understand what each field means.
#
# Docs: https://ai.pydantic.dev/results/#structured-result-validation

class ResearchResult(BaseModel):
    raise NotImplementedError


# ── TODO(human) #2: Create the agent and tools ───────────────────────

# Create a PydanticAI Agent with:
#   1. model=OLLAMA_MODEL
#   2. result_type=ResearchResult (your model from TODO #1)
#   3. deps_type=AgentDeps (typed dependency injection)
#   4. system_prompt — a string instructing the agent to be a research assistant
#      that uses its tools to look up information before answering
#
# Then define two tools using @agent.tool:
#
#   Tool 1: lookup_topic(ctx: RunContext[AgentDeps], topic: str) -> str
#     - Search ctx.deps.knowledge_base for the topic (case-insensitive)
#     - Return the fact if found, or "No information found for '{topic}'"
#     - The RunContext gives tools access to the injected dependencies
#
#   Tool 2: get_user_name(ctx: RunContext[AgentDeps]) -> str
#     - Return ctx.deps.user_name
#     - Demonstrates a tool with no extra parameters — just context access
#
# The @agent.tool decorator registers the function as a tool the LLM can call.
# PydanticAI reads the function's type hints and docstring to generate the
# tool schema for the LLM. Good docstrings = better tool usage by the LLM.
#
# Docs: https://ai.pydantic.dev/tools/

# agent = Agent(...)
#
# @agent.tool
# async def lookup_topic(...) -> str:
#     ...
#
# @agent.tool
# async def get_user_name(...) -> str:
#     ...

raise NotImplementedError


# ── Test queries ─────────────────────────────────────────────────────

TEST_QUERIES = [
    "What is Pydantic and why is it popular?",
    "Compare Python and Rust for systems programming.",
    "What do you know about Docker?",
]


# ── Orchestration ────────────────────────────────────────────────────

async def run_query(agent: Agent, deps: AgentDeps, query: str) -> None:
    """Run a single query and display the structured result."""
    print(f"\n{'─' * 60}")
    print(f"Query: {query}")
    print(f"{'─' * 60}")

    result = await agent.run(query, deps=deps)
    data: ResearchResult = result.data

    print(f"  Answer:     {data.answer[:200]}...")
    print(f"  Confidence: {data.confidence:.2f}")
    print(f"  Sources:    {data.sources}")


async def main() -> None:
    print("=" * 60)
    print("Phase 1 — PydanticAI: Type-Safe Agent")
    print("=" * 60)

    deps = AgentDeps(
        user_name="Learner",
        knowledge_base=KNOWLEDGE_BASE,
    )

    for query in TEST_QUERIES:
        await run_query(agent, deps, query)

    print(f"\n{'=' * 60}")
    print("Phase 1 complete. All responses validated against ResearchResult schema.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
