"""Phase 5 — Observability: LLM Tracing with Langfuse.

Integrates Langfuse callback handler with a LangChain agent to capture
traces of every LLM call, tool invocation, and chain step. View traces
in the Langfuse UI at http://localhost:3000.

Run:
    uv run python src/05_observability.py

Prerequisites:
    1. Langfuse running (docker compose up -d)
    2. Create an account at http://localhost:3000 (first visit)
    3. Create a project and get API keys from Settings > API Keys
    4. Set environment variables or update the constants below
"""

from __future__ import annotations

import asyncio
import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler


# ── Configuration ────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"

# Langfuse connection — after signing up at http://localhost:3000, go to
# Settings > API Keys to create a key pair. You can either set env vars
# or hardcode them below for local development.
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")


# ── TODO(human) #1: Langfuse Integration ─────────────────────────────

async def create_traced_chain() -> tuple:
    """Create a LangChain chain with Langfuse tracing attached.

    TODO(human): Implement this function.

    Langfuse's CallbackHandler intercepts every LangChain operation (LLM call,
    chain step, tool call) and sends structured trace data to the Langfuse server.
    Each trace shows: the full prompt sent to the LLM, the response received,
    token counts, latency, and the chain of operations. This is essential for
    debugging agent behavior in production — when an agent gives a bad answer,
    you trace back through each step to find the root cause.

    Steps:
      1. Create a LangfuseCallbackHandler:
         handler = LangfuseCallbackHandler(
             public_key=LANGFUSE_PUBLIC_KEY,
             secret_key=LANGFUSE_SECRET_KEY,
             host=LANGFUSE_HOST,
         )
         If keys are empty strings, Langfuse will still work in "anonymous" mode
         for local development.
      2. Create a ChatPromptTemplate with a system message and a user message:
         prompt = ChatPromptTemplate.from_messages([
             ("system", "You are a helpful assistant. Answer concisely in 2-3 sentences."),
             ("user", "{question}"),
         ])
      3. Create a ChatOllama instance
      4. Build a chain using the pipe operator:
         chain = prompt | llm | StrOutputParser()
      5. Return (chain, handler) as a tuple

    The handler is passed to chain.invoke() via the config parameter:
    chain.invoke({"question": "..."}, config={"callbacks": [handler]})

    Returns:
        Tuple of (chain, langfuse_handler)
    """
    raise NotImplementedError


# ── TODO(human) #2: Run Traced Queries ────────────────────────────────

async def run_traced_queries(chain, handler) -> None:
    """Run test queries and send traces to Langfuse.

    TODO(human): Implement this function.

    Running multiple queries generates a rich set of traces to explore in the
    Langfuse UI. Each query becomes a separate trace, showing the full execution
    path. After running, you'll open the Langfuse dashboard to inspect traces,
    compare latencies, and understand token usage across queries.

    Steps:
      1. Define a list of 5 test queries (diverse topics to generate interesting traces):
         queries = [
             "What is the difference between REST and GraphQL?",
             "Explain the CAP theorem in distributed systems.",
             "What are the SOLID principles in software design?",
             "How does garbage collection work in Python?",
             "What is the actor model in concurrent programming?",
         ]
      2. For each query:
         a. Print the query
         b. Call chain.invoke({"question": query}, config={"callbacks": [handler]})
            Note: use invoke(), not ainvoke() — Langfuse's callback handler works
            more reliably with synchronous invocation.
         c. Print the response (truncated to 200 chars)
         d. Print a separator line
      3. After all queries, call handler.flush() to ensure all traces are sent
      4. Print instructions for viewing traces:
         "Open http://localhost:3000 to view traces in Langfuse"
         "Navigate to: Tracing > Traces to see all 5 query traces"
         "Click on a trace to see: prompt, response, latency, tokens used"

    Returns:
        None
    """
    raise NotImplementedError


# ── Orchestration ────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 60)
    print("Phase 5 — Observability with Langfuse")
    print("=" * 60)

    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        print()
        print("  NOTE: Langfuse API keys not set.")
        print("  To enable tracing:")
        print("    1. Open http://localhost:3000 and create an account")
        print("    2. Go to Settings > API Keys > Create New")
        print("    3. Set environment variables:")
        print("       export LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("       export LANGFUSE_SECRET_KEY=sk-lf-...")
        print("    Or update the constants in this file.")
        print()
        print("  Continuing without tracing (chain will still work)...")
        print()

    chain, handler = await create_traced_chain()
    await run_traced_queries(chain, handler)

    print(f"\n{'=' * 60}")
    print("Phase 5 complete.")
    print("Open http://localhost:3000 > Tracing > Traces to inspect results.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
