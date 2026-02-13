"""
Practice 029a — Phase 1: Your First LCEL Chain

This exercise teaches the fundamental LCEL pattern: prompt | llm | parser.
Everything in LangChain v0.3 builds on this — once you understand how three
Runnables compose into a pipeline, you can build arbitrarily complex chains.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------------
# Setup: model and output parser (boilerplate — already done for you)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"

llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.7,
)

output_parser = StrOutputParser()


# ---------------------------------------------------------------------------
# Exercise: Build your first LCEL chain
# ---------------------------------------------------------------------------

def build_and_run_first_chain() -> None:
    # TODO(human): Build a basic LCEL chain and test it with multiple topics.
    #
    # WHAT TO DO:
    #   1. Create a ChatPromptTemplate that takes a {topic} variable and asks
    #      the LLM to explain the topic in 2-3 sentences for a beginner.
    #      Use ChatPromptTemplate.from_messages() with a system message
    #      (e.g., "You are a helpful teacher") and a human message with {topic}.
    #
    #   2. Compose the chain using the pipe operator:
    #        chain = prompt_template | llm | output_parser
    #      This creates a RunnableSequence where:
    #        - prompt_template formats {"topic": "..."} into a list of messages
    #        - llm sends the messages to Ollama and returns an AIMessage
    #        - output_parser extracts the text string from the AIMessage
    #
    #   3. Call chain.invoke({"topic": "recursion"}) and print the result.
    #      Then test with 2 more topics of your choice (e.g., "hash tables",
    #      "dependency injection").
    #
    # WHY THIS MATTERS:
    #   The pipe operator is the foundation of LCEL. It doesn't just look clean —
    #   it gives you streaming, batching, and async for free. When you call
    #   chain.stream() instead of chain.invoke(), tokens flow through ALL three
    #   components incrementally. This is impossible with manual function calls.
    #
    # EXPECTED BEHAVIOR:
    #   For each topic, you should see a 2-3 sentence explanation printed.
    #   The chain should work identically for any topic string.
    #
    # HINT:
    #   ChatPromptTemplate.from_messages([
    #       ("system", "your system prompt here"),
    #       ("human", "{topic}"),
    #   ])
    raise NotImplementedError("Build your first LCEL chain here")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1: First LCEL Chain")
    print("=" * 60)
    build_and_run_first_chain()
