"""Evaluator -- Automated human-likeness scoring.

Scores conversations on the human-likeness rubric from 083a, using a combination
of heuristic checks (bot tells, formatting patterns) and LLM-based assessment
(emotional continuity, persona consistency).

Scoring dimensions (0-10 each):
  1. Imperfection: Presence of human-like imperfection signals
  2. Emotional continuity: Natural emotional transitions across turns
  3. Persona consistency: Adherence to the persona's traits and speech patterns
  4. No bot tells: Absence of bullet points, over-explanation, customer-service phrases

The evaluator also computes entry_adherence: for each playbook entry, whether the
conversation's behavior aligned with the entry's advice. This is used by the Curator
to update helpful/harmful counters.

Scoring approach:
  - Heuristic checks are fast and deterministic (regex-based bot tell detection)
  - LLM scoring is slower but captures nuance (does the persona feel consistent?)
  - Combined score: 40% heuristic + 60% LLM
"""

from __future__ import annotations

import re

import instructor
from pydantic import BaseModel, Field

from src.llm_config import get_instructor_client
from src.models import ACEConfig, Conversation, ConversationScore
from src.playbook import Playbook


# ---------------------------------------------------------------------------
# Heuristic bot-tell detectors
# ---------------------------------------------------------------------------

# Patterns that indicate bot-like behavior
BOT_TELL_PATTERNS: list[tuple[str, re.Pattern[str], float]] = [
    # (description, regex, penalty per occurrence)
    ("bullet_points", re.compile(r"^\s*[-*]\s", re.MULTILINE), 1.5),
    ("numbered_lists", re.compile(r"^\s*\d+[.)]\s", re.MULTILINE), 1.5),
    ("great_question", re.compile(r"(?:that'?s a |what a )(?:great|good|excellent|wonderful) question", re.IGNORECASE), 2.0),
    ("appreciate_sharing", re.compile(r"(?:I appreciate|thank you for) (?:you )?sharing", re.IGNORECASE), 1.5),
    ("as_an_ai", re.compile(r"as an (?:AI|artificial intelligence|language model)", re.IGNORECASE), 3.0),
    ("certainly_absolutely", re.compile(r"\b(?:certainly|absolutely)!?\s", re.IGNORECASE), 0.5),
    ("would_you_like_me_to", re.compile(r"would you like me to (?:elaborate|explain|share|provide)", re.IGNORECASE), 1.0),
    ("here_are_some", re.compile(r"here are (?:some|a few|several)", re.IGNORECASE), 1.0),
    ("in_conclusion", re.compile(r"\b(?:in conclusion|to summarize|in summary)\b", re.IGNORECASE), 1.0),
]


def _score_bot_tells(conversation: Conversation) -> float:
    """Score absence of bot tells (higher = fewer bot tells = more human).

    Returns a score from 0-10 where 10 means no bot tells detected.
    """
    total_penalty = 0.0
    agent_turns = [t for t in conversation.turns if t.role == "agent"]

    for turn in agent_turns:
        for _name, pattern, penalty in BOT_TELL_PATTERNS:
            matches = len(pattern.findall(turn.content))
            total_penalty += matches * penalty

    # Scale: 0 penalty = 10, 10+ penalty = 0
    score = max(0.0, 10.0 - total_penalty)
    return round(score, 1)


def _score_imperfection(conversation: Conversation) -> float:
    """Score presence of human-like imperfection signals (higher = more human).

    Looks for: hedges, ellipsis, lowercase, varied sentence length, informal punctuation.
    """
    agent_turns = [t for t in conversation.turns if t.role == "agent"]
    if not agent_turns:
        return 0.0

    signals = 0.0
    total_checks = 0

    for turn in agent_turns:
        text = turn.content
        total_checks += 5  # 5 checks per turn

        # Hedging language
        if re.search(r"\b(?:honestly|like|I mean|hm|well|I think|probably|maybe)\b", text, re.IGNORECASE):
            signals += 1

        # Ellipsis or trailing off
        if "..." in text or "-- " in text:
            signals += 1

        # Some lowercase usage (not all-proper)
        sentences = re.split(r"[.!?]+", text)
        lowercase_starts = sum(1 for s in sentences if s.strip() and s.strip()[0].islower())
        if lowercase_starts > 0:
            signals += 1

        # Varied sentence length (not all similar length)
        words_per_sentence = [len(s.split()) for s in sentences if s.strip()]
        if len(words_per_sentence) >= 2:
            std = (sum((w - sum(words_per_sentence) / len(words_per_sentence)) ** 2 for w in words_per_sentence) / len(words_per_sentence)) ** 0.5
            if std > 3:  # Meaningful variance in sentence length
                signals += 1

        # Informal punctuation or self-correction
        if re.search(r"(?:lol|haha|omg|wait|okay so|oh)", text, re.IGNORECASE):
            signals += 1

    score = (signals / max(total_checks, 1)) * 10.0
    return round(min(10.0, score), 1)


# ---------------------------------------------------------------------------
# LLM-based scoring
# ---------------------------------------------------------------------------

class _LLMScore(BaseModel):
    """Structured output for LLM-based conversation scoring."""
    emotional_continuity: float = Field(
        ge=0.0, le=10.0,
        description="How naturally do emotions flow across turns? (0=jarring transitions, 10=seamless)",
    )
    persona_consistency: float = Field(
        ge=0.0, le=10.0,
        description="Does the agent stay in character? (0=breaks character constantly, 10=fully consistent)",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the scores",
    )


def _score_with_llm(conversation: Conversation, config: ACEConfig) -> _LLMScore:
    """Use the LLM to score emotional continuity and persona consistency."""
    client = get_instructor_client()

    turns_text = "\n".join(
        f"{'User' if t.role == 'user' else 'Agent'}: {t.content}"
        for t in conversation.turns
    )

    prompt = f"""Rate this conversation on two dimensions (0-10 each):

1. EMOTIONAL CONTINUITY: Do the agent's emotions flow naturally across turns?
   - 0: Emotions reset each turn, jarring transitions
   - 5: Some continuity but occasional abrupt shifts
   - 10: Emotions build naturally, transitions feel organic

2. PERSONA CONSISTENCY: Does the agent maintain a consistent personality?
   - 0: Generic, no personality, changes register constantly
   - 5: Some personality but inconsistent
   - 10: Clear, consistent personality with quirks and opinions

CONVERSATION:
{turns_text}"""

    try:
        return client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=_LLMScore,
            max_retries=2,
            temperature=0.3,
        )
    except Exception:
        # Fallback to neutral scores if LLM scoring fails
        return _LLMScore(emotional_continuity=5.0, persona_consistency=5.0)


# ---------------------------------------------------------------------------
# Entry adherence (for Curator counter updates)
# ---------------------------------------------------------------------------

def _check_entry_adherence(
    conversation: Conversation,
    playbook: Playbook,
) -> dict[str, bool]:
    """Check which playbook entries the conversation adhered to.

    Simple keyword-based heuristic: if the entry mentions avoiding something
    and the conversation doesn't contain that pattern, the entry was adhered to.

    This is intentionally simple -- a production system would use the LLM to
    assess adherence. For this practice, keyword matching provides a fast
    approximation that's sufficient for counter updates.
    """
    agent_text = " ".join(t.content for t in conversation.turns if t.role == "agent").lower()
    adherence: dict[str, bool] = {}

    for entry in playbook.all_entries():
        content_lower = entry.content.lower()

        # Check "never/don't/avoid" entries: adhered if the bad pattern is absent
        if any(word in content_lower for word in ("never", "don't", "avoid", "do not")):
            # Look for the forbidden pattern in agent text
            # Simple: check if key nouns from the entry appear in agent text
            forbidden_keywords = _extract_keywords(content_lower)
            violation = any(kw in agent_text for kw in forbidden_keywords)
            adherence[entry.entry_id] = not violation
        else:
            # Positive entries: assume adhered (conservative -- don't penalize)
            adherence[entry.entry_id] = True

    return adherence


def _extract_keywords(text: str) -> list[str]:
    """Extract likely-forbidden-pattern keywords from an avoidance entry."""
    # Simple extraction: look for words after "never/don't/avoid"
    keywords: list[str] = []
    patterns = [
        r"never (?:use |enumerate |start )?(\w+(?:\s+\w+)?)",
        r"(?:don't|do not) (?:use |start )?(\w+(?:\s+\w+)?)",
        r"avoid (?:using )?(\w+(?:\s+\w+)?)",
    ]
    for pat in patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            kw = match.group(1).strip().lower()
            if len(kw) > 3:  # Skip short words
                keywords.append(kw)
    return keywords


# ---------------------------------------------------------------------------
# Public scoring API
# ---------------------------------------------------------------------------

def score_conversation(
    conversation: Conversation,
    playbook: Playbook,
    config: ACEConfig,
    use_llm: bool = True,
) -> ConversationScore:
    """Score a conversation on all human-likeness dimensions.

    Combines heuristic checks (40% weight) with LLM scoring (60% weight).
    """
    # Heuristic scores
    bot_tell_score = _score_bot_tells(conversation)
    imperfection_score = _score_imperfection(conversation)

    # LLM scores
    if use_llm:
        llm_score = _score_with_llm(conversation, config)
        emotional_continuity = llm_score.emotional_continuity
        persona_consistency = llm_score.persona_consistency
    else:
        emotional_continuity = 5.0
        persona_consistency = 5.0

    # Combined overall score
    overall = (
        0.2 * imperfection_score
        + 0.2 * bot_tell_score
        + 0.3 * emotional_continuity
        + 0.3 * persona_consistency
    )

    # Entry adherence
    adherence = _check_entry_adherence(conversation, playbook)

    return ConversationScore(
        conversation_id=conversation.conversation_id,
        imperfection=imperfection_score,
        emotional_continuity=emotional_continuity,
        persona_consistency=persona_consistency,
        no_bot_tells=bot_tell_score,
        overall=round(overall, 1),
        entry_adherence=adherence,
    )


def score_batch(
    conversations: list[Conversation],
    playbook: Playbook,
    config: ACEConfig,
    use_llm: bool = True,
) -> list[ConversationScore]:
    """Score a batch of conversations."""
    return [
        score_conversation(conv, playbook, config, use_llm=use_llm)
        for conv in conversations
    ]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Score sample conversations."""
    import json
    import sys
    from pathlib import Path

    print("=" * 60)
    print("  Evaluator Self-Test")
    print("=" * 60)

    config = ACEConfig()
    sample_dir = Path(__file__).parent.parent / "data" / "sample_conversations"

    # Load sample conversations
    conversations: list[Conversation] = []
    for path in sorted(sample_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        conversations.append(Conversation(**data))

    if not conversations:
        print("[FAIL] No sample conversations found")
        sys.exit(1)

    # Load playbook
    from src.playbook import load_seed_playbook
    try:
        playbook = load_seed_playbook()
    except NotImplementedError:
        print("[SKIP] Playbook not implemented (Exercise 1 required)")
        sys.exit(0)

    # Score each conversation
    use_llm = "--no-llm" not in sys.argv
    if not use_llm:
        print("(LLM scoring disabled, using heuristics only)\n")

    for conv in conversations:
        print(f"\n--- {conv.conversation_id} ({conv.label}) ---")
        score = score_conversation(conv, playbook, config, use_llm=use_llm)
        print(f"  Imperfection:         {score.imperfection}/10")
        print(f"  Emotional continuity: {score.emotional_continuity}/10")
        print(f"  Persona consistency:  {score.persona_consistency}/10")
        print(f"  No bot tells:         {score.no_bot_tells}/10")
        print(f"  OVERALL:              {score.overall}/10")

        adhered = sum(1 for v in score.entry_adherence.values() if v)
        total = len(score.entry_adherence)
        print(f"  Entry adherence:      {adhered}/{total} entries")

    print("\n" + "=" * 60)
    print("  Evaluator self-test complete.")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
