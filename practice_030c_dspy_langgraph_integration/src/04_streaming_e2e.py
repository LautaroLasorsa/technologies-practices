"""
Practice 030c — Phase 4: Streaming & End-to-End System

dspy.streamify() converts any DSPy module into an async generator that yields
tokens as they're produced. This enables real-time output for user-facing
applications.

This phase combines streaming with a full end-to-end query routing system.

Run: uv run python src/04_streaming_e2e.py
"""

import asyncio

import dspy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434"


def configure_dspy() -> None:
    """Configure DSPy to use the local Ollama model."""
    lm = dspy.LM(
        model=f"ollama_chat/{OLLAMA_MODEL}",
        api_base=f"{OLLAMA_BASE_URL}/v1",
        api_key="",
    )
    dspy.configure(lm=lm)


# ---------------------------------------------------------------------------
# TODO(human) #7: Stream DSPy module output
# ---------------------------------------------------------------------------
# dspy.streamify() takes a DSPy module and returns an async callable that
# yields tokens as they're generated. This is essential for user-facing
# applications where you want to show output incrementally rather than
# waiting for the full response.
#
# Steps:
#   1. Create a DSPy module, e.g.:
#      qa = dspy.ChainOfThought("question -> answer")
#
#   2. Wrap it with streamify:
#      streaming_qa = dspy.streamify(qa)
#
#   3. Call it asynchronously and iterate over the stream:
#      async for chunk in streaming_qa(question="Explain what a graph database is"):
#          # Each chunk is a dspy.streaming.StreamResponse
#          # When the chunk has a 'answer' attribute and it's a string, print it
#          if hasattr(chunk, 'answer') and isinstance(chunk.answer, str):
#              print(chunk.answer, end="", flush=True)
#
#   4. Handle the final prediction: the last item yielded is the complete
#      dspy.Prediction object (not a StreamResponse). You can detect this
#      by checking: isinstance(chunk, dspy.Prediction).
#
# Note: Streaming requires an async context. Wrap your code in an async
# function and call it with asyncio.run(). The streaming interface gives
# you token-level granularity for building responsive UIs.

async def demo_streaming() -> None:
    raise NotImplementedError("TODO(human): implement streaming DSPy output")


# ---------------------------------------------------------------------------
# TODO(human) #8: End-to-end query routing system with streaming
# ---------------------------------------------------------------------------
# Build a complete pipeline that combines classification, routing, and
# streaming into a cohesive system. This represents a production-like
# pattern where:
#   - A classifier determines query intent
#   - The query is routed to the appropriate expert
#   - The expert's response is streamed to the user
#
# Steps:
#   1. Reuse (or recreate) the classifier and specialist modules from Phase 3:
#      - classifier = dspy.ChainOfThought("question -> category")
#      - math_qa = dspy.ChainOfThought("question -> answer")
#      - fact_qa = dspy.ChainOfThought("question -> answer")
#      - creative_qa = dspy.ChainOfThought("question -> answer")
#
#   2. Create an async function `route_and_stream(question: str)` that:
#      a. Classifies the question (non-streaming, since classification is fast)
#      b. Selects the appropriate specialist module based on category
#      c. Wraps the specialist with dspy.streamify()
#      d. Streams the response token by token, printing as it goes
#
#   3. Test with a list of diverse questions:
#      - "Calculate 2^10 - 24"
#      - "Who invented the telephone?"
#      - "Write a short poem about recursion"
#
#   4. For each question, print:
#      - The classified category
#      - The streamed response (token by token)
#
# This exercise ties together everything: DSPy modules for LLM quality,
# classification for routing, and streaming for responsive output. In a
# real system, the routing logic would be a LangGraph (as in Phase 3),
# but here we do it manually to focus on the streaming aspect.

async def route_and_stream(question: str) -> None:
    raise NotImplementedError("TODO(human): implement end-to-end routing with streaming")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main() -> None:
    configure_dspy()

    print("=" * 60)
    print("Phase 4: Streaming & End-to-End System")
    print("=" * 60)

    print("\n--- Demo: Basic Streaming ---\n")
    await demo_streaming()

    print("\n\n--- Demo: End-to-End Query Routing with Streaming ---\n")
    test_questions = [
        "Calculate 2^10 - 24",
        "Who invented the telephone?",
        "Write a short poem about recursion",
    ]
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)
        await route_and_stream(question)
        print()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
