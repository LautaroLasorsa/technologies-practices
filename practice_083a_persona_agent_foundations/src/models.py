"""Shared Pydantic models for the persona agent.

All data structures used across modules are defined here:
- Emotion enum and emotional state
- Persona card schema
- User fact schema
- Conversation turn schema
- Agent introspection schema
- Configuration schema
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Emotion
# ---------------------------------------------------------------------------

class Emotion(str, Enum):
    """Discrete emotional states the agent can experience.

    These are intentionally coarse -- 7 states is enough for distinct behavioral
    routing without overwhelming a 3B model's ability to classify reliably.
    """
    NEUTRAL = "neutral"
    CURIOUS = "curious"
    ENGAGED = "engaged"
    AMUSED = "amused"
    ANNOYED = "annoyed"
    REFLECTIVE = "reflective"
    TIRED = "tired"


class EmotionalState(BaseModel):
    """Current emotional state with intensity and turn counter."""
    emotion: Emotion = Emotion.NEUTRAL
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    turns_in_state: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Persona Card
# ---------------------------------------------------------------------------

class SpeechPatterns(BaseModel):
    """How the persona speaks -- verbal habits, punctuation, style."""
    verbal_tics: list[str] = Field(default_factory=list)
    punctuation_style: str = ""
    capitalization: str = ""
    emoji_usage: str = ""
    response_length_tendency: str = ""


class PersonalityTraits(BaseModel):
    """Core personality: traits, contradictions, interests, pet peeves."""
    core: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    pet_peeves: list[str] = Field(default_factory=list)


class EmotionalDefaults(BaseModel):
    """Baseline emotional configuration for the persona."""
    baseline_emotion: str = "neutral"
    baseline_intensity: float = 0.5
    volatility: str = "moderate"
    recovery_speed: str = "returns to baseline after 3-4 turns"


class PersonaCard(BaseModel):
    """Complete persona definition loaded from data/persona.json."""
    name: str
    age: int
    backstory: str
    speech_patterns: SpeechPatterns = Field(default_factory=SpeechPatterns)
    personality_traits: PersonalityTraits = Field(default_factory=PersonalityTraits)
    emotional_defaults: EmotionalDefaults = Field(default_factory=EmotionalDefaults)
    forbidden_behaviors: list[str] = Field(default_factory=list)
    characteristic_phrases: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# User Facts (persistent memory)
# ---------------------------------------------------------------------------

class UserFact(BaseModel):
    """A single fact about the user, extracted from conversation."""
    key: str = Field(description="Category/topic of the fact, e.g. 'name', 'job', 'preference'")
    value: str = Field(description="The actual fact content")
    source_message: str = Field(default="", description="The user message this was extracted from")
    extracted_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class UserFactStore(BaseModel):
    """Collection of facts about a single user."""
    user_id: str
    facts: list[UserFact] = Field(default_factory=list)
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Conversation Log
# ---------------------------------------------------------------------------

class ConversationTurn(BaseModel):
    """A single turn in a conversation (user message + agent response)."""
    role: Literal["user", "agent"]
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    emotion: Emotion | None = None
    inner_thought: str | None = None
    intensity: float | None = None


class ConversationLog(BaseModel):
    """A complete conversation session."""
    conversation_id: str
    user_id: str
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    turns: list[ConversationTurn] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent Introspection (structured output from instructor)
# ---------------------------------------------------------------------------

class AgentTurn(BaseModel):
    """Structured output from the LLM via instructor.

    The model produces all three fields, but only `response` is shown to the
    user. `emotion` and `inner_thought` are logged for inspection and feed
    into the emotional state machine.
    """
    emotion: Emotion = Field(
        description=(
            "Your current emotional state as a reaction to the user's message. "
            "Choose from: neutral, curious, engaged, amused, annoyed, reflective, tired."
        )
    )
    inner_thought: str = Field(
        description=(
            "Your private inner thought about the user's message -- what you're "
            "really thinking. 1-2 sentences. This is NEVER shown to the user."
        )
    )
    response: str = Field(
        description=(
            "Your response to the user, written in character. Follow your persona's "
            "speech patterns, verbal tics, and emotional state. This IS shown to the user."
        )
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """Runtime configuration for the agent."""
    model: str = "qwen2.5:3b"
    ollama_base_url: str = "http://localhost:11434/v1"
    max_history_turns: int = 20
    fact_extraction_enabled: bool = True
    humanization_enabled: bool = True
    typing_speed_cps: float = 30.0  # characters per second base rate
    min_delay_seconds: float = 0.5
    max_delay_seconds: float = 5.0
