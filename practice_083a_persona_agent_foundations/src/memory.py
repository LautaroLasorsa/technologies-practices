"""Exercise 5 -- Two-Tier Memory (File-Based).

Implements per-turn fact extraction and persistent storage.

Two-tier memory architecture:
  Tier 1: In-context memory -- conversation messages in the LLM's context window.
          Dies when the process ends. Limited by context window (32k tokens).
  Tier 2: Persistent memory -- extracted facts stored in JSON files.
          Survives restarts. Loaded at session start into the system prompt.

The key operation is FACT EXTRACTION: after each user turn, the model is asked
"What new facts about the user can be extracted from this message?" This converts
unstructured conversation into structured knowledge.

Why not just extend the context window?
  1. Context windows are finite -- even 32k fills up in a long conversation
  2. Raw messages are noisy -- fact extraction distills the signal
  3. Facts are reusable -- they persist across sessions
  4. Facts are inspectable -- you can read the JSON file and see what the agent "knows"
"""

from __future__ import annotations

import instructor
from pydantic import BaseModel, Field

from src.models import AgentConfig, UserFact, UserFactStore


# ---------------------------------------------------------------------------
# Extraction schema (internal -- used only by extract_user_facts)
# ---------------------------------------------------------------------------

class ExtractedSingleFact(BaseModel):
    """A single extracted fact."""
    key: str = Field(description="Category of the fact: 'name', 'job', 'location', 'preference', 'opinion', 'context', etc.")
    value: str = Field(description="The actual fact, stated concisely. E.g., 'Works as an ML engineer at a startup'")
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="How confident you are this is a real fact (0.0-1.0). Lower for inferences, higher for explicit statements.",
    )


class ExtractedFacts(BaseModel):
    """Schema for the fact extraction LLM call.

    The model returns a list of facts extracted from a single user message.
    Each fact has a key (category) and value (content).
    """
    facts: list[ExtractedSingleFact] = Field(
        default_factory=list,
        description="Facts about the user extracted from their message. Empty list if no facts found.",
    )


# ---------------------------------------------------------------------------
# Fact Extraction
# ---------------------------------------------------------------------------

def extract_user_facts(
    client: instructor.Instructor,
    model: str,
    user_message: str,
) -> list[UserFact]:
    """Extract facts about the user from a single message.

    # TODO(human): Implement the fact extraction LLM call
    #
    # This function calls the instructor-patched client to extract structured
    # facts from a user message. Not every message contains facts -- "hello!"
    # has nothing to extract, while "I'm a data engineer in Chicago working
    # on Spark pipelines" has three facts (job, location, technology).
    #
    # What you need to do:
    #   1. Create a system prompt for the extraction call. This prompt should:
    #      - Tell the model it's a fact extractor, not a conversationalist
    #      - Define what counts as a "fact": name, job, location, preferences,
    #        opinions, context about their situation, relationships, etc.
    #      - Tell the model to return an EMPTY list if no facts are present
    #        (this is the common case -- most messages are conversation, not info)
    #      - Tell the model to be conservative: extract only what's explicitly
    #        stated or very clearly implied. No speculation.
    #      - Instruct low confidence (0.3-0.5) for inferences vs high (0.8-1.0)
    #        for explicit statements
    #
    #   2. Call client.chat.completions.create() with:
    #      - model=model
    #      - messages=[
    #          {"role": "system", "content": <your extraction prompt>},
    #          {"role": "user", "content": user_message},
    #        ]
    #      - response_model=ExtractedFacts
    #      - max_retries=2
    #      - temperature=0.1  (low temperature for factual extraction)
    #
    #   3. Convert ExtractedSingleFact objects to UserFact objects:
    #      - Copy key, value, confidence
    #      - Set source_message=user_message
    #      - extracted_at is auto-set by UserFact's default_factory
    #
    #   4. Return the list of UserFact objects (may be empty)
    #
    # Design considerations:
    #   - Temperature should be LOW (0.1) for extraction -- you want factual
    #     accuracy, not creative interpretation.
    #   - The extraction prompt is different from the persona prompt. The model
    #     is NOT in character here -- it's a utility call.
    #   - Some messages genuinely contain no facts ("lol", "interesting", "ok").
    #     The model should return an empty list, not hallucinate facts.
    #   - This runs AFTER every user message. If it's too slow, you could run
    #     it only every N turns or only for longer messages. But with a 3B model,
    #     the extraction call is typically <1 second.
    #
    # Example:
    #   Input:  "I'm Alex, I work on supply chain optimization in Austin"
    #   Output: [
    #     UserFact(key="name", value="Alex", confidence=1.0, ...),
    #     UserFact(key="job", value="Works on supply chain optimization", confidence=0.9, ...),
    #     UserFact(key="location", value="Based in Austin", confidence=0.9, ...),
    #   ]
    #
    #   Input:  "haha that's funny"
    #   Output: []
    """
    raise NotImplementedError("Exercise 5: implement extract_user_facts")


# ---------------------------------------------------------------------------
# Fact Merging
# ---------------------------------------------------------------------------

def merge_facts(existing: UserFactStore, new_facts: list[UserFact]) -> UserFactStore:
    """Merge new facts into an existing fact store.

    # TODO(human): Implement fact merging with deduplication and updates
    #
    # This is more nuanced than "just append." Facts can be:
    #   - NEW: a key that doesn't exist yet -> add it
    #   - UPDATED: a key that exists with a different value -> update it
    #     (e.g., user's job changed, or we got a more specific version)
    #   - DUPLICATE: same key and same/similar value -> skip it
    #
    # What you need to do:
    #   1. Build a dict of existing facts keyed by their `key` field.
    #      If there are multiple facts with the same key, keep the most recent.
    #
    #   2. For each new fact:
    #      - If the key doesn't exist in existing: add it
    #      - If the key exists but the value is different: replace it
    #        (the newer fact is more current -- e.g., "works at Google" replaces
    #         "works at startup")
    #      - If the key exists and the value is similar: skip it
    #        (simple string equality is fine -- don't over-engineer similarity)
    #
    #   3. Rebuild the facts list from the dict and return a new UserFactStore
    #      with updated last_updated timestamp
    #
    # Design considerations:
    #   - Should you keep a history of old values? For 083a, no -- just keep
    #     the latest. The Reflector in 083b might want history, but that's a
    #     future concern.
    #   - What about conflicting facts? E.g., "I live in Austin" followed later
    #     by "I moved to Seattle." The newer fact should win. Timestamp comparison
    #     on extracted_at handles this.
    #   - Fact keys are intentionally coarse ("job", "location", "name"). Multiple
    #     facts can have the same key if they represent different aspects:
    #     ("preference", "likes coffee") and ("preference", "hates mornings").
    #     Consider using (key, value) as the dedup key, not just key.
    #
    # Implementation hint:
    #   A simple approach: use a dict with (key, value) tuples as keys.
    #   For each fact (existing + new), the newest one wins.
    #   Then convert back to a list.
    """
    raise NotImplementedError("Exercise 5: implement merge_facts")


# ---------------------------------------------------------------------------
# Convenience: load facts for system prompt
# ---------------------------------------------------------------------------

def format_facts_for_prompt(facts: list[UserFact], min_confidence: float = 0.5) -> str:
    """Format user facts as a block for the system prompt.

    Filters by confidence threshold and formats as a concise list.
    Returns empty string if no facts meet the threshold.
    """
    relevant = [f for f in facts if f.confidence >= min_confidence]
    if not relevant:
        return ""

    lines = ["What you know about this person:"]
    for fact in relevant:
        lines.append(f"  - {fact.key}: {fact.value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Test fact extraction and merging."""
    from src.models import AgentConfig

    config = AgentConfig()

    # Test LLM connectivity
    from src.llm_config import create_instructor_client, create_raw_openai_client

    try:
        base_client = create_raw_openai_client(config)
        models = base_client.models.list()
        available = [m.id for m in models.data]
        if config.model not in available:
            print(f"[WARN] Model '{config.model}' not available. Pull it first.")
            return
    except Exception as e:
        print(f"[FAIL] Cannot connect to {config.provider}: {e}")
        return

    client = create_instructor_client(config)

    # Test fact extraction
    test_messages = [
        "Hi, I'm Alex and I work as an ML engineer in Austin.",
        "I've been really into Rust lately, trying to learn systems programming.",
        "haha yeah that's pretty funny",
        "My cat's name is Luna, she's a rescue.",
        "ok",
    ]

    print("Testing fact extraction:\n")
    all_facts: list[UserFact] = []

    for msg in test_messages:
        try:
            facts = extract_user_facts(client, config.model, msg)
            print(f"  '{msg[:50]}...' -> {len(facts)} facts")
            for f in facts:
                print(f"    [{f.key}] {f.value} (confidence: {f.confidence})")
            all_facts.extend(facts)
        except NotImplementedError:
            print("[SKIP] extract_user_facts not implemented yet (Exercise 5)")
            break

    # Test fact merging
    if all_facts:
        print("\nTesting fact merging:")
        store = UserFactStore(user_id="test")
        try:
            store = merge_facts(store, all_facts)
            print(f"  Merged {len(all_facts)} extracted facts into {len(store.facts)} stored facts")
            for f in store.facts:
                print(f"    [{f.key}] {f.value}")
        except NotImplementedError:
            print("[SKIP] merge_facts not implemented yet (Exercise 5)")

    # Test prompt formatting (already implemented -- no TODO)
    test_facts = [
        UserFact(key="name", value="Alex", confidence=0.9),
        UserFact(key="job", value="ML engineer", confidence=0.8),
        UserFact(key="low_confidence", value="might like jazz", confidence=0.3),
    ]
    formatted = format_facts_for_prompt(test_facts, min_confidence=0.5)
    print(f"\nFormatted facts for prompt (min_confidence=0.5):\n{formatted}")

    print("\nMemory self-test complete.")


if __name__ == "__main__":
    _self_test()
