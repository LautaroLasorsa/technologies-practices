"""
Practice 029a — Phase 4: Streaming & Callbacks

This exercise teaches two observability patterns:
  1. Streaming: token-by-token output for responsive UIs
  2. Callbacks: hooking into chain execution for logging and metrics
"""

import sys
import time
from typing import Any
from uuid import UUID

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

# ---------------------------------------------------------------------------
# Setup: model, prompt, and base chain
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"

llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.7,
)

parser = StrOutputParser()

story_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a creative storyteller. Write a short story (3-5 paragraphs) "
     "based on the given theme. Use vivid descriptions."),
    ("human", "{theme}"),
])

# Base chain for streaming exercises.
story_chain = story_prompt | llm | parser


# ---------------------------------------------------------------------------
# Exercise 1: Streaming output
# ---------------------------------------------------------------------------

def exercise_streaming() -> None:
    # TODO(human): Implement streaming that prints tokens as they arrive.
    #
    # WHAT TO DO:
    #   1. Use the story_chain defined above (it's already prompt | llm | parser).
    #
    #   2. Call chain.stream() instead of chain.invoke(). This returns an
    #      iterator that yields chunks as the LLM generates them:
    #
    #        for chunk in story_chain.stream({"theme": "a robot learning to paint"}):
    #            print(chunk, end="", flush=True)
    #        print()  # newline at the end
    #
    #      The flush=True is important — without it, Python buffers output
    #      and you won't see the streaming effect in the terminal.
    #
    #   3. Also try streaming with sys.stdout.write() for finer control:
    #        for chunk in story_chain.stream({"theme": "a cat in space"}):
    #            sys.stdout.write(chunk)
    #            sys.stdout.flush()
    #        sys.stdout.write("\n")
    #
    # WHY THIS MATTERS:
    #   Streaming is critical for user experience. A 7B model generating a
    #   story might take 10-15 seconds total — without streaming, the user
    #   stares at a blank screen. With streaming, they see text appear
    #   word-by-word (like ChatGPT) because LCEL propagates streaming
    #   through the entire chain automatically.
    #
    #   The key insight: because StrOutputParser is streaming-aware, it
    #   passes through each token as it arrives. If you used a non-streaming
    #   parser, it would buffer the entire response. LCEL's streaming
    #   propagation only works when ALL components in the chain support it.
    #
    # EXPECTED BEHAVIOR:
    #   You should see the story appear token by token in the terminal,
    #   not all at once after the LLM finishes generating.
    raise NotImplementedError("Implement streaming here")


# ---------------------------------------------------------------------------
# Exercise 2: Custom callback handler
# ---------------------------------------------------------------------------

def exercise_callback_handler() -> None:
    # TODO(human): Create a custom BaseCallbackHandler that logs chain execution.
    #
    # WHAT TO DO:
    #   1. Create a class that inherits from BaseCallbackHandler:
    #
    #        class MetricsCallbackHandler(BaseCallbackHandler):
    #            def __init__(self):
    #                self.start_time: float | None = None
    #                self.token_count: int = 0
    #
    #            def on_llm_start(
    #                self,
    #                serialized: dict[str, Any],
    #                prompts: list[str],
    #                *,
    #                run_id: UUID,
    #                **kwargs: Any,
    #            ) -> None:
    #                """Called when the LLM starts generating."""
    #                self.start_time = time.time()
    #                self.token_count = 0
    #                model_name = serialized.get("kwargs", {}).get("model", "unknown")
    #                print(f"\n[CALLBACK] LLM started — model: {model_name}")
    #
    #            def on_llm_new_token(
    #                self,
    #                token: str,
    #                *,
    #                run_id: UUID,
    #                **kwargs: Any,
    #            ) -> None:
    #                """Called for each new token during streaming."""
    #                self.token_count += 1
    #
    #            def on_llm_end(
    #                self,
    #                response: LLMResult,
    #                *,
    #                run_id: UUID,
    #                **kwargs: Any,
    #            ) -> None:
    #                """Called when the LLM finishes generating."""
    #                elapsed = time.time() - self.start_time if self.start_time else 0
    #                print(f"\n[CALLBACK] LLM finished — "
    #                      f"tokens: {self.token_count}, "
    #                      f"time: {elapsed:.2f}s, "
    #                      f"tokens/sec: {self.token_count / elapsed:.1f}" if elapsed > 0 else "")
    #
    #   2. Use the callback handler with a chain by passing it in the config:
    #
    #        handler = MetricsCallbackHandler()
    #        result = story_chain.invoke(
    #            {"theme": "a lighthouse keeper's last night"},
    #            config={"callbacks": [handler]},
    #        )
    #        print(result)
    #
    #   3. Also test with .stream() — the on_llm_new_token callback fires
    #      for each token during streaming, so you can track generation speed
    #      in real-time.
    #
    # WHY THIS MATTERS:
    #   Callbacks are the observability layer for LangChain. They let you:
    #     - Log every LLM call (model, prompts, responses) for debugging
    #     - Track token usage for cost estimation
    #     - Measure latency per component in the chain
    #     - Stream tokens to a WebSocket or SSE endpoint
    #     - Implement rate limiting or circuit breakers
    #
    #   The callback system is non-invasive — you don't modify your chain,
    #   just pass callbacks in the config. This follows the Open/Closed
    #   principle: chains are open for extension (via callbacks) but closed
    #   for modification.
    #
    # EXPECTED BEHAVIOR:
    #   You should see "[CALLBACK] LLM started" before the response,
    #   then the response text, then "[CALLBACK] LLM finished" with
    #   token count, elapsed time, and tokens/sec metrics.
    raise NotImplementedError("Create callback handler here")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4: Streaming & Callbacks")
    print("=" * 60)

    print("\n--- Exercise 1: Streaming output ---\n")
    exercise_streaming()

    print("\n--- Exercise 2: Callback handler ---\n")
    exercise_callback_handler()
