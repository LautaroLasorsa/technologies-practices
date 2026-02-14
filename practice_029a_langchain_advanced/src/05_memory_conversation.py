"""
Practice 029a — Phase 5: Memory & Conversation

This exercise teaches the modern LangChain memory pattern using
RunnableWithMessageHistory + InMemoryChatMessageHistory.

NOTE: ConversationBufferMemory and ConversationSummaryMemory were deprecated
in LangChain v0.3.1. The modern approach stores message history externally
and injects it via RunnableWithMessageHistory. This is more explicit, testable,
and compatible with LangGraph's persistence model.
"""

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# Setup: model, parser, and conversation prompt template
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:3b"

llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.7,
)

parser = StrOutputParser()

# This prompt template includes a MessagesPlaceholder for chat history.
# RunnableWithMessageHistory will inject the conversation history here
# automatically before each invocation.
conversation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Be concise and friendly. "
            "Remember what the user told you in previous messages.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Pre-built base chain (without memory).
base_chain = conversation_prompt | llm | parser


# ---------------------------------------------------------------------------
# Helper: Session-based message history store
# ---------------------------------------------------------------------------

# In-memory store mapping session_id → ChatMessageHistory.
# In production, this would be a database (Redis, PostgreSQL, etc.).
message_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Retrieve or create a message history for the given session."""
    if session_id not in message_store:
        message_store[session_id] = InMemoryChatMessageHistory()
    return message_store[session_id]


# ---------------------------------------------------------------------------
# Exercise 1: Conversation with buffer memory
# ---------------------------------------------------------------------------


def exercise_buffer_memory() -> None:
    # TODO(human): Build a conversational chain using RunnableWithMessageHistory.
    #
    # WHAT TO DO:
    #   1. Wrap the base_chain with RunnableWithMessageHistory:
    #
    #        chain_with_memory = RunnableWithMessageHistory(
    #            base_chain,
    #            get_session_history,
    #            input_messages_key="input",
    #            history_messages_key="history",
    #        )
    #
    #      This tells LangChain:
    #        - "input" is the key for the user's new message
    #        - "history" is the key where conversation history should be injected
    #        - get_session_history provides the storage backend per session
    #
    #   2. Send a sequence of messages using the same session_id to test memory:
    #
    #        config = {"configurable": {"session_id": "user-001"}}
    #
    #        response1 = chain_with_memory.invoke(
    #            {"input": "My name is Carlos and I'm a software engineer."},
    #            config=config,
    #        )
    #        print(f"Bot: {response1}\n")
    #
    #        response2 = chain_with_memory.invoke(
    #            {"input": "What's my name and what do I do?"},
    #            config=config,
    #        )
    #        print(f"Bot: {response2}\n")
    #
    #        response3 = chain_with_memory.invoke(
    #            {"input": "What programming languages should I learn next?"},
    #            config=config,
    #        )
    #        print(f"Bot: {response3}\n")
    #
    #   3. Verify that the assistant remembers your name and profession
    #      from message 1 when answering messages 2 and 3.
    #
    #   4. (Optional) Try a different session_id ("user-002") and verify
    #      it starts fresh — sessions are isolated.
    #
    # WHY THIS MATTERS:
    #   Without memory, every LLM call is stateless — the model has no idea
    #   what was said before. RunnableWithMessageHistory solves this by:
    #     1. Loading the conversation history before each call
    #     2. Injecting it into the prompt via MessagesPlaceholder
    #     3. Saving the new exchange (input + response) after each call
    #
    #   This is the modern replacement for the deprecated ConversationBufferMemory.
    #   The key difference: memory is managed OUTSIDE the chain (in message_store),
    #   not inside it. This makes it easier to test, persist, and share across
    #   different chain configurations.
    #
    # EXPECTED BEHAVIOR:
    #   Message 1: Bot acknowledges your name and profession
    #   Message 2: Bot correctly recalls "Carlos" and "software engineer"
    #   Message 3: Bot gives language recommendations personalized to your role

    chain_with_memory = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_message_key="input",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "user_001"}}

    response1 = chain_with_memory.invoke(
        {"input": "My name is Carlos and I'm a software engineer."},
        config=config,
    )
    print(f"Bot: {response1}\n")

    response2 = chain_with_memory.invoke(
        {"input": "What's my name and what do I do?"},
        config=config,
    )
    print(f"Bot: {response2}\n")

    response3 = chain_with_memory.invoke(
        {"input": "What programming languages should I learn next?"},
        config=config,
    )
    print(f"Bot: {response3}\n")

    response4 = chain_with_memory.invoke(
        {"input": "What's my name and what do I do?"},
        config={"configurable": {"session_id": "user_002"}},
    )
    print(f"Bot (to 2): {response4}\n")


# ---------------------------------------------------------------------------
# Exercise 2: Windowed memory (last N exchanges)
# ---------------------------------------------------------------------------


def exercise_windowed_memory() -> None:
    # TODO(human): Implement conversation memory that only keeps the last N exchanges.
    #
    # WHAT TO DO:
    #   1. Create a custom function that returns a trimmed message history.
    #      The idea: after each call, if the history exceeds 2*N messages
    #      (N human + N AI), trim the oldest ones.
    #
    #      One approach — create a wrapper that trims on retrieval:
    #
    #        WINDOW_SIZE = 3  # Keep last 3 exchanges (6 messages: 3 human + 3 AI)
    #
    #        windowed_store: dict[str, InMemoryChatMessageHistory] = {}
    #
    #        def get_windowed_history(session_id: str) -> InMemoryChatMessageHistory:
    #            if session_id not in windowed_store:
    #                windowed_store[session_id] = InMemoryChatMessageHistory()
    #            history = windowed_store[session_id]
    #            # Trim to last WINDOW_SIZE exchanges (each exchange = 2 messages)
    #            max_messages = WINDOW_SIZE * 2
    #            if len(history.messages) > max_messages:
    #                history.messages = history.messages[-max_messages:]
    #            return history
    #
    #   2. Build a new chain_with_windowed_memory using get_windowed_history:
    #
    #        chain_with_windowed_memory = RunnableWithMessageHistory(
    #            base_chain,
    #            get_windowed_history,
    #            input_messages_key="input",
    #            history_messages_key="history",
    #        )
    #
    #   3. Test with 5+ messages. In the first message, state a distinctive
    #      fact (e.g., "My favorite number is 42"). After 4+ more messages
    #      about different topics, ask "What's my favorite number?" — if the
    #      window is small enough, the bot should have forgotten it.
    #
    #        config = {"configurable": {"session_id": "windowed-001"}}
    #
    #        messages = [
    #            "My favorite number is 42.",
    #            "Tell me about Python.",
    #            "What are decorators?",
    #            "Explain generators.",
    #            "What are context managers?",
    #            "What's my favorite number?",
    #        ]
    #        for msg in messages:
    #            print(f"User: {msg}")
    #            resp = chain_with_windowed_memory.invoke(
    #                {"input": msg}, config=config,
    #            )
    #            print(f"Bot: {resp}\n")
    #
    # WHY THIS MATTERS:
    #   Unbounded conversation history is a real problem:
    #     - LLMs have finite context windows (4K-128K tokens)
    #     - Long histories increase latency and cost (more input tokens)
    #     - Old messages may confuse the model or contradict recent context
    #
    #   Windowed memory is the simplest solution: keep the last N exchanges,
    #   drop the rest. In production, you'd combine this with summary memory
    #   (summarize dropped messages into a system prompt) — but windowed
    #   memory alone handles most conversational use cases.
    #
    # EXPECTED BEHAVIOR:
    #   With WINDOW_SIZE=3, the bot should remember recent topics (decorators,
    #   generators, context managers) but forget the favorite number from
    #   message 1, since it falls outside the window.

    def get_session_window(window_size: int):
        def inner_function(session_id: str) -> InMemoryChatMessageHistory:
            memory = get_session_history(session_id)
            memory.messages = memory.messages[-2 * window_size :]
            return memory

        return inner_function

    chain_with_windowed_memory = RunnableWithMessageHistory(
        base_chain,
        get_session_window(4),
        input_messages_key="input",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "windowed-001"}}
    messages = [
        "My name is Lautaro",
        "A = 100",
        "B = 12",
        "C = 1000",
        "D = 0",
        "E = -1",
        "What is my name?",
    ]

    for msg in messages:
        answer = chain_with_windowed_memory.invoke({"input": msg}, config=config)
        print(f"USER:{msg}\n\nAgent:{answer}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 5: Memory & Conversation")
    print("=" * 60)

    print("\n--- Exercise 1: Buffer memory ---\n")
    exercise_buffer_memory()

    print("\n--- Exercise 2: Windowed memory ---\n")
    exercise_windowed_memory()
