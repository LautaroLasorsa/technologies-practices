"""Pydantic models for the ACE persona agent.

All data structures used across modules:
- Playbook entries and sections
- Delta lessons from the Reflector
- Conversation schemas (compatible with 083a format)
- Evaluation scores
- ACE loop state
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Playbook
# ---------------------------------------------------------------------------

class PlaybookSection(str, Enum):
    """The three sections of the persona playbook."""
    STRATEGIES = "STRATEGIES & INSIGHTS"
    EMOTIONAL = "EMOTIONAL PATTERNS"
    MISTAKES = "MISTAKES TO AVOID"


class PlaybookEntry(BaseModel):
    """A single entry in the playbook.

    Each entry has a unique ID (e.g., 'strat-00001'), counters tracking
    how often the entry was marked helpful or harmful, and the content
    text describing the strategy/pattern/mistake.
    """
    entry_id: str = Field(description="Unique identifier, e.g. 'strat-00001', 'emot-00003'")
    section: PlaybookSection
    helpful: int = Field(default=0, ge=0)
    harmful: int = Field(default=0, ge=0)
    content: str = Field(description="The actual strategy/pattern/mistake text")


# ---------------------------------------------------------------------------
# Delta Lessons (Reflector output)
# ---------------------------------------------------------------------------

class DeltaLesson(BaseModel):
    """A single lesson extracted by the Reflector.

    Delta lessons are the intermediate format between raw conversation
    analysis and playbook updates. The Reflector produces these; the
    Curator consumes them.
    """
    section: PlaybookSection = Field(
        description="Which playbook section this lesson belongs to"
    )
    content: str = Field(
        description=(
            "The concrete lesson learned. Must be specific and actionable, "
            "not vague (e.g., 'Use a neutral transition beat when shifting "
            "from humor to serious topics' NOT 'be better at transitions')"
        )
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How confident the Reflector is in this lesson (0-1)",
    )


class ReflectorOutput(BaseModel):
    """Structured output from the Reflector's conversation analysis."""
    lessons: list[DeltaLesson] = Field(
        description=(
            "Concrete lessons extracted from comparing natural vs robotic "
            "conversations. Each lesson should target a specific playbook "
            "section and be actionable."
        )
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the key differences identified",
    )


# ---------------------------------------------------------------------------
# Conversations (compatible with 083a format)
# ---------------------------------------------------------------------------

class ConversationTurn(BaseModel):
    """A single turn in a conversation."""
    role: Literal["user", "agent"]
    content: str
    emotion: str | None = None
    inner_thought: str | None = None


class Conversation(BaseModel):
    """A complete conversation log."""
    conversation_id: str
    label: Literal["natural", "robotic", "generated"] = "generated"
    human_likeness_score: float | None = None
    notes: str = ""
    turns: list[ConversationTurn] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class ConversationScore(BaseModel):
    """Automated human-likeness evaluation of a conversation.

    Scores are 0-10 on each dimension. The rubric follows 083a's
    human-likeness criteria: imperfection, timing variance, persona
    consistency, and memory recall.
    """
    conversation_id: str
    imperfection: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Presence of human-like imperfection signals (hedges, restarts, varied punctuation)",
    )
    emotional_continuity: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Emotional state consistency and natural transitions across turns",
    )
    persona_consistency: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Adherence to persona traits, speech patterns, and opinions",
    )
    no_bot_tells: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Absence of bot signals (bullet lists, over-explanation, customer-service phrases)",
    )
    overall: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Overall human-likeness score",
    )
    entry_adherence: dict[str, bool] = Field(
        default_factory=dict,
        description="Map of entry_id -> whether this conversation followed the entry's advice",
    )


# ---------------------------------------------------------------------------
# ACE Loop State
# ---------------------------------------------------------------------------

class IterationSnapshot(BaseModel):
    """Snapshot of one ACE iteration's results."""
    iteration: int
    playbook_token_count: int
    playbook_entry_count: int
    avg_score: float
    scores: list[ConversationScore] = Field(default_factory=list)
    lessons_extracted: int = 0
    entries_added: int = 0
    entries_pruned: int = 0
    entries_deduped: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ACEConfig(BaseModel):
    """Configuration for the ACE loop."""
    model: str = "qwen2.5:3b"
    ollama_base_url: str = "http://localhost:11434/v1"
    batch_size: int = 3
    dedup_threshold: float = 0.85
    prune_ratio: float = 2.0
    embedding_model: str = "all-MiniLM-L6-v2"
    max_retries: int = 3
    temperature: float = 0.8
