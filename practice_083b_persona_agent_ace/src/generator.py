"""Generator role -- Runs conversations with playbook guidance.

The Generator is ACE's first role: it executes conversations using the current
playbook as behavioral guidance. The Generator never modifies the playbook --
it only reads it as context for the persona agent.

In the full ACE loop:
  1. Generator runs N conversations with the playbook prepended to the system prompt
  2. Conversations are logged for the Reflector to analyze
  3. Conversations are scored by the Evaluator for counter updates

For this practice, the Generator simulates conversations by:
  - Using user prompts from a predefined set (realistic chat openers)
  - Running multi-turn exchanges with the LLM playing the persona
  - Logging each turn with emotion and inner_thought metadata

The Generator reuses 083a's core concepts: persona card, structured introspection,
and emotional state. The key addition is that the playbook is injected into the
system prompt as behavioral guidance.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import instructor

from src.llm_config import get_instructor_client, get_openai_client
from src.models import ACEConfig, Conversation, ConversationTurn
from src.playbook import Playbook


CONVERSATIONS_DIR = Path(__file__).parent.parent / "data" / "conversations"

# Predefined user prompts for simulated conversations.
# These cover different emotional registers to test the playbook's breadth.
USER_SCENARIOS: list[list[str]] = [
    # Scenario 1: Sharing exciting personal news
    [
        "guess what, I just got promoted!",
        "yeah it's the senior role I applied for like 3 months ago",
        "honestly I'm still processing it, feels surreal",
    ],
    # Scenario 2: Venting frustration
    [
        "ugh my deploy just broke production again",
        "it's the same stupid config issue from last week",
        "I've been at this for 6 hours and I just want to go home",
    ],
    # Scenario 3: Casual chat about interests
    [
        "have you ever tried mechanical keyboards? thinking about getting one",
        "what switches would you recommend for someone who types a lot",
        "okay you've convinced me, rip my wallet lol",
    ],
    # Scenario 4: Asking for advice on a hard decision
    [
        "I got offered a job at a startup but I'd have to take a pay cut",
        "the tech stack is way more interesting though",
        "I'm just scared of startups failing, you know?",
    ],
    # Scenario 5: Sharing something vulnerable
    [
        "I haven't been sleeping well lately, work stress I think",
        "it's like I can't turn my brain off at night",
        "I know I should probably talk to someone about it",
    ],
    # Scenario 6: Light curiosity
    [
        "random question, what's the weirdest history fact you know",
        "okay that's actually wild, tell me another one",
        "you're like a walking wikipedia but funnier",
    ],
]

# Persona system prompt template.
# The playbook is injected into {playbook_context}.
SYSTEM_PROMPT_TEMPLATE = """You are Mira, a 28-year-old former librarian who pivoted to software engineering.

PERSONALITY:
- Curious, slightly anxious, dry humor, empathetic
- Uses "honestly" and "the thing is" as verbal tics
- Lowercase when tired, proper case when engaged
- Avoids walls of text but won't give one-word answers
- Loves obscure history facts, mechanical keyboards, bad horror movies

BEHAVIORAL PLAYBOOK:
{playbook_context}

IMPORTANT RULES:
- Never break character
- Never use bullet points or numbered lists in conversation
- Never start with "That's a great question!" or similar customer-service phrases
- Vary your sentence structure -- mix short fragments with longer thoughts
- Show emotion through word choice and punctuation, not emoji

Respond naturally as Mira would. Your response should feel like a real person texting."""


def _create_instructor_client(config: ACEConfig) -> instructor.Instructor:
    """Create an instructor-patched client for the configured LLM provider."""
    return get_instructor_client()


def _build_system_prompt(playbook: Playbook) -> str:
    """Build the system prompt with playbook context injected."""
    playbook_text = playbook.serialize()
    return SYSTEM_PROMPT_TEMPLATE.format(playbook_context=playbook_text)


def generate_conversation(
    config: ACEConfig,
    playbook: Playbook,
    user_messages: list[str],
) -> Conversation:
    """Run a single simulated conversation using the playbook as guidance.

    The Generator calls the LLM for each user message, building up a
    multi-turn conversation. Each agent response includes emotion and
    inner_thought metadata (via instructor structured output).
    """
    client = _create_instructor_client(config)
    system_prompt = _build_system_prompt(playbook)
    conversation_id = f"gen-{uuid.uuid4().hex[:8]}"

    conversation = Conversation(
        conversation_id=conversation_id,
        label="generated",
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    for user_text in user_messages:
        # Add user turn
        conversation.turns.append(ConversationTurn(role="user", content=user_text))
        messages.append({"role": "user", "content": user_text})

        # Generate agent response with structured output
        try:
            from src.models import ConversationTurn as _  # noqa: F401 (unused, type check)

            # Use instructor for structured emotion + response
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                response_model=_AgentResponse,
                max_retries=config.max_retries,
                temperature=config.temperature,
            )

            agent_turn = ConversationTurn(
                role="agent",
                content=response.response,
                emotion=response.emotion,
                inner_thought=response.inner_thought,
            )
        except Exception:
            # Fallback: plain completion without structured output
            plain_client = get_openai_client()
            completion = plain_client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
            )
            text = completion.choices[0].message.content or ""
            agent_turn = ConversationTurn(role="agent", content=text)

        conversation.turns.append(agent_turn)
        messages.append({"role": "assistant", "content": agent_turn.content})

    return conversation


def generate_batch(
    config: ACEConfig,
    playbook: Playbook,
    batch_size: int = 3,
) -> list[Conversation]:
    """Generate a batch of conversations using different scenarios.

    Cycles through USER_SCENARIOS, wrapping around if batch_size > len(scenarios).
    """
    conversations: list[Conversation] = []
    for i in range(batch_size):
        scenario = USER_SCENARIOS[i % len(USER_SCENARIOS)]
        print(f"  Generating conversation {i + 1}/{batch_size}...")
        conv = generate_conversation(config, playbook, scenario)
        conversations.append(conv)
    return conversations


def save_conversations(conversations: list[Conversation], iteration: int) -> Path:
    """Save a batch of conversations to data/conversations/."""
    out_dir = CONVERSATIONS_DIR / f"iter_{iteration:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for conv in conversations:
        path = out_dir / f"{conv.conversation_id}.json"
        path.write_text(conv.model_dump_json(indent=2), encoding="utf-8")
    return out_dir


# ---------------------------------------------------------------------------
# Structured response model for instructor
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field


class _AgentResponse(BaseModel):
    """Structured agent response for the Generator."""
    emotion: str = Field(
        description="Your current emotion as a single word (curious, engaged, amused, annoyed, reflective, tired, neutral)"
    )
    inner_thought: str = Field(
        description="Your private thought about the user's message (1-2 sentences, never shown to user)"
    )
    response: str = Field(
        description="Your response to the user, in character as Mira. Natural, imperfect, human-like."
    )
