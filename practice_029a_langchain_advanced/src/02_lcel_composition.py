"""
Practice 029a — Phase 2: LCEL Composition

This exercise teaches the three core LCEL composition patterns:
  1. Sequential chains (multi-step pipelines)
  2. Parallel chains (fan-out with RunnableParallel)
  3. Custom transforms (RunnableLambda)

Together these let you build any data flow topology from simple Runnables.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# Setup: model, parser, and pre-built prompt templates
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:3b"

llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.7,
)

parser = StrOutputParser()

# Pre-built prompt templates for the multi-step chain exercise.
# Each template expects a specific input key and produces text output.

summarize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a concise summarizer. Summarize the given text in 2-3 sentences.",
        ),
        ("human", "{text}"),
    ]
)

translate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a translator. Translate the following text to Spanish. Output only the translation.",
        ),
        ("human", "{text}"),
    ]
)

format_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a formatter. Reformat the following text as a bullet-point list. Use '- ' for each bullet.",
        ),
        ("human", "{text}"),
    ]
)

# Pre-built prompt templates for the parallel chain exercise.

keywords_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract 3-5 keywords from the text. Output only the keywords, comma-separated.",
        ),
        ("human", "{text}"),
    ]
)

sentiment_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Analyze the sentiment of the text. Reply with exactly one word: positive, negative, or neutral.",
        ),
        ("human", "{text}"),
    ]
)

# Sample text for testing.
SAMPLE_TEXT = (
    "Artificial intelligence has transformed healthcare by enabling faster "
    "diagnosis through medical imaging analysis. Machine learning models can "
    "now detect certain cancers with accuracy comparable to experienced "
    "radiologists. However, concerns about data privacy, algorithmic bias, "
    "and the need for human oversight remain significant challenges that "
    "the industry must address before widespread clinical deployment."
)


# ---------------------------------------------------------------------------
# Exercise 1: Multi-step chain (sequential composition)
# ---------------------------------------------------------------------------


def exercise_multi_step_chain() -> None:
    # TODO(human): Build a multi-step chain: summarize → translate → format.
    #
    # WHAT TO DO:
    #   1. Create three sub-chains, one for each step:
    #        summarize_chain = summarize_prompt | llm | parser
    #        translate_chain = translate_prompt | llm | parser
    #        format_chain   = format_prompt | llm | parser
    #
    #   2. The challenge: each sub-chain expects {"text": ...} as input, but
    #      the previous sub-chain outputs a plain string. You need to bridge
    #      this gap. Use RunnableLambda to wrap the string into a dict:
    #        wrap = RunnableLambda(lambda s: {"text": s})
    #
    #   3. Compose the full pipeline:
    #        full_chain = summarize_chain | wrap | translate_chain | wrap | format_chain
    #
    #   4. Invoke with {"text": SAMPLE_TEXT} and print the result at each stage
    #      (or just the final result).
    #
    # WHY THIS MATTERS:
    #   Multi-step chains are the bread-and-butter of LLM applications. Each
    #   step is independently testable and reusable. The key insight is that
    #   LCEL chains are Runnables themselves — you can compose chains of chains.
    #   The RunnableLambda bridge pattern (string → dict) comes up constantly
    #   when connecting steps with different input/output shapes.
    #
    # EXPECTED BEHAVIOR:
    #   Input:  English paragraph about AI in healthcare
    #   Output: Spanish bullet-point list summarizing the key points
    #
    # HINT:
    #   You can also use RunnablePassthrough.assign() or a dict comprehension
    #   inside RunnableLambda to reshape data between steps.

    summarize_chain = summarize_prompt | llm | parser
    translate_chain = translate_prompt | llm | parser
    format_chain = format_prompt | llm | parser

    def wrap_step(s: str):
        print("Wrapping : \n", s)
        return {"text": s}

    wrap = RunnableLambda(wrap_step)

    full_chain = summarize_chain | wrap | translate_chain | wrap | format_chain

    print(full_chain.invoke({"text": SAMPLE_TEXT}))


# ---------------------------------------------------------------------------
# Exercise 2: RunnableParallel fan-out
# ---------------------------------------------------------------------------


def exercise_parallel_chain() -> None:
    # TODO(human): Build a RunnableParallel that processes text three ways simultaneously.
    #
    # WHAT TO DO:
    #   1. Create three independent sub-chains:
    #        summary_chain  = summarize_prompt | llm | parser
    #        keywords_chain = keywords_prompt  | llm | parser
    #        sentiment_chain = sentiment_prompt | llm | parser
    #
    #   2. Combine them into a RunnableParallel:
    #        analysis = RunnableParallel(
    #            summary=summary_chain,
    #            keywords=keywords_chain,
    #            sentiment=sentiment_chain,
    #        )
    #
    #   3. Invoke with {"text": SAMPLE_TEXT}. The result will be a dict:
    #        {"summary": "...", "keywords": "...", "sentiment": "..."}
    #      Print each field.
    #
    # WHY THIS MATTERS:
    #   RunnableParallel is how LCEL does fan-out — run multiple independent
    #   operations on the same input and merge results. Under the hood, it
    #   runs all branches concurrently (using asyncio or thread pool). This
    #   is critical for latency-sensitive apps: three sequential LLM calls
    #   take 3x the time, but three parallel calls take ~1x.
    #
    #   The output is a dict keyed by the names you gave each branch. This
    #   dict can be piped into the next step, making RunnableParallel the
    #   standard way to gather multiple pieces of information at once.
    #
    # EXPECTED BEHAVIOR:
    #   You should see three results printed — a summary paragraph, a comma-
    #   separated keyword list, and a single sentiment word. All three are
    #   derived from the same input text but computed independently.

    summary_chain = summarize_prompt | llm | parser
    keywords_chain = keywords_prompt | llm | parser
    sentiment_chain = sentiment_prompt | llm | parser

    parallel_chain = RunnableParallel(
        summary=summary_chain, keywords=keywords_chain, sentiment=sentiment_chain
    )

    print(parallel_chain.invoke({"text": SAMPLE_TEXT}))


# ---------------------------------------------------------------------------
# Exercise 3: RunnableLambda for custom transformation
# ---------------------------------------------------------------------------


def exercise_lambda_transform() -> None:
    # TODO(human): Use RunnableLambda to insert custom Python logic into a chain.
    #
    # WHAT TO DO:
    #   1. Define a Python function that takes a string and returns a
    #      transformed string. For example:
    #        def add_word_count(text: str) -> str:
    #            count = len(text.split())
    #            return f"[Word count: {count}]\n\n{text}"
    #
    #   2. Wrap it as a Runnable:
    #        word_counter = RunnableLambda(add_word_count)
    #
    #   3. Build a chain that:
    #      a) Takes {"text": ...} input
    #      b) Summarizes the text (summarize_prompt | llm | parser)
    #      c) Adds a word count header via your RunnableLambda
    #      d) Prints the final result
    #
    #      chain = summarize_prompt | llm | parser | word_counter
    #
    #   4. Invoke with {"text": SAMPLE_TEXT} and print the result.
    #
    # WHY THIS MATTERS:
    #   RunnableLambda is the "escape hatch" that lets you run arbitrary
    #   Python logic inside an LCEL chain. It's how you do data cleaning,
    #   validation, formatting, logging, or any custom transformation that
    #   doesn't need an LLM. The function must take one argument (the
    #   previous step's output) and return one value (the next step's input).
    #
    #   It also supports async: if you pass an async function, LCEL will
    #   await it when running in async mode. And it works with .stream()
    #   and .batch() like any other Runnable.
    #
    # EXPECTED BEHAVIOR:
    #   The output should be the summary text with a "[Word count: N]" header
    #   prepended to it.

    def signature_hash(text: str):
        P, B = int(1e9) + 7, 17771
        hash = 0
        for c in text:
            hash = (hash * B + ord(c)) % P
        return text + f"\n[Hash: {hex(hash)}]"

    hasher = RunnableLambda(signature_hash)
    chain = summarize_prompt | llm | parser | hasher
    print(chain.invoke({"text": SAMPLE_TEXT}))


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2: LCEL Composition")
    print("=" * 60)

    print("\n--- Exercise 1: Multi-step chain ---\n")
    exercise_multi_step_chain()

    print("\n--- Exercise 2: Parallel chain ---\n")
    exercise_parallel_chain()

    print("\n--- Exercise 3: Lambda transform ---\n")
    exercise_lambda_transform()
