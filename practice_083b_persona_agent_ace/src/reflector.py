"""Exercise 3 -- Reflector: Conversation Analysis.

The Reflector is ACE's second role: it analyzes completed conversations to extract
concrete delta lessons about what makes the persona feel natural vs robotic.

Separation of concerns:
  - The Reflector ONLY extracts insights. It never modifies the playbook.
  - It receives conversation pairs (natural + robotic) and identifies behavioral
    differences as structured DeltaLesson objects.
  - The Curator (Exercise 4) is the only component that touches the playbook.

Why pairs matter:
  Analyzing a single conversation yields vague observations ("it was good").
  Comparing a natural conversation against a robotic one on the SAME topic
  surfaces specific differences:
    - "The natural version acknowledged the emotion before asking follow-ups"
    - "The robotic version used bullet points where the natural one rambled"
    - "The robotic version started with 'That's a great question!' -- a bot tell"

  These specific, contrastive observations are what make useful playbook entries.

Architecture:
  1. Format both conversations into a comparison prompt
  2. Ask the LLM (via instructor) to produce a ReflectorOutput:
     - reasoning: brief explanation of key differences
     - lessons: list of DeltaLesson objects, each targeting a playbook section
  3. Filter lessons by confidence (>= 0.3) to avoid noise
  4. Return the lessons for the Curator

The prompt engineering here is critical:
  - The model must produce SPECIFIC, ACTIONABLE lessons
  - Bad: "the natural one was better at emotions"
  - Good: "transition between humor and seriousness via a neutral beat, not abruptly"
  - The DeltaLesson schema's field descriptions guide the model toward specificity
"""

from __future__ import annotations

import json
from pathlib import Path

import instructor
from openai import OpenAI

from src.models import (
    ACEConfig,
    Conversation,
    ConversationTurn,
    DeltaLesson,
    PlaybookSection,
    ReflectorOutput,
)


REFLECTIONS_DIR = Path(__file__).parent.parent / "data" / "reflections"


def _create_instructor_client(config: ACEConfig) -> instructor.Instructor:
    """Create an instructor-patched client for structured Reflector output."""
    base_client = OpenAI(base_url=config.ollama_base_url, api_key="ollama")
    return instructor.from_openai(base_client, mode=instructor.Mode.JSON)


def _format_conversation(conversation: Conversation) -> str:
    """Format a conversation into readable text for the Reflector prompt."""
    lines: list[str] = []
    lines.append(f"[{conversation.label.upper()}] Conversation: {conversation.conversation_id}")
    if conversation.human_likeness_score is not None:
        lines.append(f"Human-likeness score: {conversation.human_likeness_score}/10")
    lines.append("")
    for turn in conversation.turns:
        prefix = "User" if turn.role == "user" else "Agent"
        lines.append(f"{prefix}: {turn.content}")
        if turn.emotion:
            lines.append(f"  [emotion: {turn.emotion}]")
    return "\n".join(lines)


def _build_reflector_prompt(
    natural: Conversation,
    robotic: Conversation,
) -> str:
    """Build the comparison prompt for the Reflector.

    The prompt presents both conversations side-by-side and asks the model to
    identify specific behavioral differences that explain the quality gap.
    """
    natural_text = _format_conversation(natural)
    robotic_text = _format_conversation(robotic)

    return f"""You are a conversation quality analyst. Your job is to compare two versions of a persona agent's conversations and extract specific, actionable lessons about what makes one feel human and the other feel robotic.

## NATURAL CONVERSATION (feels human)
{natural_text}

## ROBOTIC CONVERSATION (feels artificial)
{robotic_text}

## YOUR TASK

Compare these two conversations and extract SPECIFIC, ACTIONABLE lessons. Focus on:

1. **STRATEGIES & INSIGHTS**: What conversational techniques did the natural version use that the robotic version missed? (e.g., acknowledging emotions before asking questions, mirroring energy levels)

2. **EMOTIONAL PATTERNS**: How did the natural version handle emotional transitions differently? (e.g., smooth shifts vs abrupt changes, emotional memory across turns)

3. **MISTAKES TO AVOID**: What specific bot tells did the robotic version exhibit? (e.g., bullet points, "That's a great question!", over-explaining, customer-service tone)

IMPORTANT:
- Each lesson must be SPECIFIC and ACTIONABLE, not vague
- BAD: "be more natural" or "show more emotion"
- GOOD: "When user expresses fear about a career change, validate the fear as a positive signal rather than rushing to reassure"
- Rate your confidence in each lesson (0.0 to 1.0)
- Assign each lesson to the correct section"""


class Reflector:
    """Analyzes conversation pairs to extract delta lessons.

    The Reflector compares a natural-sounding conversation against a robotic one
    and extracts specific lessons about what behavioral differences explain the
    quality gap. Lessons are returned as DeltaLesson objects for the Curator.
    """

    def __init__(self, config: ACEConfig) -> None:
        self.config = config
        self.client = _create_instructor_client(config)

    def extract_lessons(
        self,
        natural: Conversation,
        robotic: Conversation,
        min_confidence: float = 0.3,
    ) -> list[DeltaLesson]:
        """Extract delta lessons from a natural vs robotic conversation pair.

        # TODO(human): Implement the Reflector's lesson extraction
        #
        # This is ACE's insight extraction step. The Reflector analyzes what makes
        # the natural conversation work and the robotic one fail, then packages
        # the differences as structured delta lessons.
        #
        # What you need to do:
        #
        # 1. Build the comparison prompt using _build_reflector_prompt(natural, robotic)
        #
        # 2. Call the instructor-patched client to get a ReflectorOutput:
        #    output = self.client.chat.completions.create(
        #        model=self.config.model,
        #        messages=[{"role": "user", "content": prompt}],
        #        response_model=ReflectorOutput,
        #        max_retries=self.config.max_retries,
        #        temperature=0.6,  # Lower than Generator -- we want analytical, not creative
        #    )
        #
        #    The instructor library will:
        #    a. Append the ReflectorOutput JSON schema to the prompt
        #    b. Parse the model's output into a validated ReflectorOutput
        #    c. Retry with validation errors if the output doesn't match
        #
        #    ReflectorOutput contains:
        #    - reasoning: str  (brief explanation of key differences)
        #    - lessons: list[DeltaLesson]  (the actual extracted lessons)
        #
        #    Each DeltaLesson has:
        #    - section: PlaybookSection  (STRATEGIES, EMOTIONAL, or MISTAKES)
        #    - content: str  (the specific, actionable lesson text)
        #    - confidence: float  (0.0 to 1.0)
        #
        # 3. Filter lessons by confidence:
        #    - Keep only lessons where confidence >= min_confidence
        #    - This filters out low-confidence noise from the model
        #    - The default threshold (0.3) is intentionally low -- we'd rather
        #      have false positives (which the Curator can prune) than miss
        #      genuine insights
        #
        # 4. Return the filtered list of DeltaLesson objects
        #
        # Why separation of concerns matters here:
        #   The Reflector NEVER sees the playbook. It only sees conversations.
        #   This means it can't accidentally overwrite or restructure existing
        #   knowledge. Its output is purely additive -- new lessons to consider.
        #   The Curator (Exercise 4) decides whether to add them, and handles
        #   dedup against existing entries.
        #
        #   If the Reflector could edit the playbook directly, it would face
        #   the same full-rewrite temptation as Exercise 2's baseline:
        #   "Let me just restructure these strategies to be cleaner..."
        #   -> context collapse.
        #
        # Prompt engineering notes:
        #   The quality of extracted lessons depends heavily on the comparison
        #   prompt. Key techniques:
        #   - Present BOTH conversations (contrastive analysis > single analysis)
        #   - Ask for SPECIFIC lessons (the DeltaLesson schema's description
        #     field guides the model toward specificity)
        #   - Use examples of good vs bad lessons in the prompt
        #   - Set temperature=0.6 (analytical, not creative)
        #
        # What to expect from a 3B model:
        #   Qwen 2.5 3B will produce reasonable but imperfect lessons. Some will
        #   be too vague, some will be duplicates of existing playbook entries,
        #   and some will be genuinely insightful. The confidence scores help
        #   filter, and the Curator's dedup step handles redundancy. Don't
        #   expect GPT-4-level analysis -- the value is in the ARCHITECTURE
        #   (incremental delta lessons), not the quality of any single lesson.
        """
        raise NotImplementedError("Exercise 3: implement Reflector.extract_lessons")


def save_reflections(
    lessons: list[DeltaLesson],
    iteration: int,
    pair_index: int,
) -> Path:
    """Save extracted lessons to data/reflections/."""
    out_dir = REFLECTIONS_DIR / f"iter_{iteration:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pair_{pair_index:03d}.json"
    data = [lesson.model_dump() for lesson in lessons]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Test the Reflector on sample conversation pairs."""
    import sys

    print("=" * 60)
    print("  Reflector Self-Test (Exercise 3)")
    print("=" * 60)

    config = ACEConfig()

    # Verify Ollama
    try:
        client = OpenAI(base_url=config.ollama_base_url, api_key="ollama")
        models = client.models.list()
        available = [m.id for m in models.data]
        if config.model not in available:
            print(f"[FAIL] Model '{config.model}' not found.")
            sys.exit(1)
        print(f"[OK] Ollama connected, model available")
    except Exception as e:
        print(f"[FAIL] Cannot connect to Ollama: {e}")
        sys.exit(1)

    # Load sample conversations
    sample_dir = Path(__file__).parent.parent / "data" / "sample_conversations"
    natural_path = sample_dir / "natural_01.json"
    robotic_path = sample_dir / "robotic_01.json"

    if not natural_path.exists() or not robotic_path.exists():
        print("[FAIL] Sample conversations not found in data/sample_conversations/")
        sys.exit(1)

    natural = Conversation(**json.loads(natural_path.read_text(encoding="utf-8")))
    robotic = Conversation(**json.loads(robotic_path.read_text(encoding="utf-8")))

    print(f"\nNatural conversation: {natural.conversation_id} (score: {natural.human_likeness_score})")
    print(f"Robotic conversation: {robotic.conversation_id} (score: {robotic.human_likeness_score})")

    # Run Reflector
    reflector = Reflector(config)
    print("\nExtracting lessons...")

    try:
        lessons = reflector.extract_lessons(natural, robotic)
    except NotImplementedError:
        print("[SKIP] Reflector.extract_lessons not implemented yet")
        sys.exit(0)

    print(f"\n[OK] Extracted {len(lessons)} lessons:\n")

    for i, lesson in enumerate(lessons, 1):
        print(f"  Lesson {i}:")
        print(f"    Section:    {lesson.section.value}")
        print(f"    Confidence: {lesson.confidence:.2f}")
        print(f"    Content:    {lesson.content}")
        print()

    # Save reflections
    save_reflections(lessons, iteration=0, pair_index=0)
    print(f"Reflections saved to {REFLECTIONS_DIR}/iter_000/")


if __name__ == "__main__":
    _self_test()
