"""Exercise 3 -- Structured Introspection with instructor.

Wires the `instructor` library over Ollama's OpenAI-compatible endpoint
to produce structured AgentTurn responses: {emotion, inner_thought, response}.

Only `response` is shown to the user. `emotion` and `inner_thought` are
logged for inspection and feed into the emotional state machine.

Architecture:
    instructor patches the OpenAI client to:
    1. Append a JSON schema instruction to the prompt
    2. Parse the model's output into a Pydantic model
    3. Retry with validation errors if the output doesn't match

    Ollama exposes an OpenAI-compatible API at /v1, so we create a
    standard OpenAI client pointed at localhost:11434/v1 and patch it
    with instructor.

Key concepts:
    - instructor.from_openai(): patches the client to support response_model
    - instructor.Mode.JSON: forces JSON mode (Ollama supports this)
    - response_model=AgentTurn: tells instructor to validate against this schema
    - max_retries: how many times to retry if validation fails (important for 3B models)
"""

from __future__ import annotations

import instructor

from src.llm_config import create_instructor_client, create_raw_openai_client
from src.models import AgentConfig, AgentTurn

# Re-export create_instructor_client so existing callers of
# `from src.introspection import create_instructor_client` keep working.
__all__ = ["create_instructor_client", "generate_agent_turn"]


def generate_agent_turn(
    client: instructor.Instructor,
    model: str,
    system_prompt: str,
    messages: list[dict[str, str]],
) -> AgentTurn:
    """Generate a structured agent response using instructor.

    # TODO(human): Implement the instructor call that returns an AgentTurn
    #
    # This is the core structured generation call. The instructor-patched client
    # works just like the OpenAI client, but accepts a `response_model` parameter
    # that constrains the output to a Pydantic model.
    #
    # What you need to do:
    #   1. Build the messages list for the API call:
    #      - First message: {"role": "system", "content": system_prompt}
    #      - Then: all messages from the `messages` parameter (conversation history)
    #
    #   2. Call client.chat.completions.create() with:
    #      - model=model
    #      - messages=<your message list>
    #      - response_model=AgentTurn
    #      - max_retries=3  (important! 3B models sometimes produce invalid JSON
    #        on the first try. instructor will include the validation error in the
    #        retry prompt, helping the model correct itself.)
    #      - temperature=0.8  (higher than default for more personality variance)
    #
    #   3. Return the AgentTurn directly (instructor returns a validated Pydantic
    #      model, not a raw completion)
    #
    # The magic: instructor automatically:
    #   - Appends a JSON schema instruction to the system prompt
    #   - Parses the model's output as JSON
    #   - Validates it against AgentTurn's schema
    #   - If validation fails, retries with the error message appended
    #
    # Design consideration:
    #   The temperature controls personality variance. Too low (0.0-0.3) and
    #   the agent sounds robotic and repetitive. Too high (>1.0) and it becomes
    #   incoherent. 0.7-0.9 is the sweet spot for persona agents.
    #
    # What if the model keeps failing validation?
    #   After max_retries, instructor raises an exception. The caller (main.py)
    #   should catch this and fall back to a generic response. With Qwen 2.5 3B
    #   and instructor.Mode.JSON, failures are rare (< 5% of calls).
    #
    # Example call pattern:
    #   result = client.chat.completions.create(
    #       model=model,
    #       messages=[...],
    #       response_model=AgentTurn,
    #       max_retries=3,
    #       temperature=0.8,
    #   )
    #   # result is already an AgentTurn instance
    """

    first_message = {
        "role": "system",
        "content": system_prompt
    }

    messages : list[ChatCompletionMessageParam] = [first_message, * messages]

    return client.chat.completions.create(
        model = model,
        messages = messages,
        response_model = AgentTurn,
        max_retries = 3,
        temperature = 0.8,
    )

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Test structured introspection with a single message."""
    from src.models import AgentConfig

    config = AgentConfig()
    print(f"Connecting to {config.provider} at {config.ollama_base_url}...")

    # Test LLM connectivity first
    try:
        base_client = create_raw_openai_client(config)
        models = base_client.models.list()
        available = [m.id for m in models.data]
        print(f"[OK] {config.provider} connected. Available models: {available}")
        if config.model not in available:
            print(f"[WARN] Model '{config.model}' not found. Pull it with:")
            print(f"  docker compose exec ollama ollama pull {config.model}")
            return
    except Exception as e:
        print(f"[FAIL] Cannot connect to {config.provider}: {e}")
        print("  Start Ollama with: docker compose up -d")
        return

    # Test instructor patching
    client = create_instructor_client(config)
    print("[OK] Instructor client created")

    # Test structured generation
    system_prompt = (
        "You are Mira, a 28-year-old former librarian turned software engineer. "
        "You have dry humor, use 'honestly' as a verbal tic, and get annoyed by "
        "vague questions. Right now you're feeling curious."
    )
    messages = [{"role": "user", "content": "Hey! What do you think about mechanical keyboards?"}]

    try:
        turn = generate_agent_turn(client, config.model, system_prompt, messages)
        print(f"\n[OK] Structured response received:")
        print(f"  Emotion:       {turn.emotion.value}")
        print(f"  Inner thought: {turn.inner_thought}")
        print(f"  Response:      {turn.response}")
    except NotImplementedError:
        print("[SKIP] generate_agent_turn not implemented yet (Exercise 3)")
    except Exception as e:
        print(f"[FAIL] Generation failed: {e}")

    print("\nIntrospection self-test complete.")


if __name__ == "__main__":
    _self_test()
