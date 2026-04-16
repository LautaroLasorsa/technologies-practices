"""Exercise 6 -- Humanization Layer: Timing & Imperfection.

Post-processes the agent's response to add human-like behavioral signals:
  1. Typing delay proportional to message length, adjusted by emotion
  2. Truncation of over-explanation
  3. Probabilistic injection of hesitation markers

Why this matters:
  The "Human or Not" study (AI21 Labs, 2023) found that the top 3 bot tells
  are: (1) consistent response timing, (2) perfect grammar, (3) over-explanation.
  This module attacks all three.

  The delay is NOT real network latency -- it's a deliberate pause before
  displaying the response, simulating typing time. A human typing at 60 WPM
  takes ~3 seconds for a 30-word response, and longer for complex thoughts.
  An LLM response appears instantly -- the delay bridges this gap.

Design principle:
  Humanization is a POST-PROCESSING step, not a generation-time concern.
  The LLM generates the best response it can, and then we rough up the
  edges. This keeps the core logic clean and the humanization tunable.
"""

from __future__ import annotations

import time

from src.models import AgentConfig, Emotion
import random

def humanize_response(
    text: str,
    emotion: Emotion,
    intensity: float,
    config: AgentConfig,
) -> tuple[str, float]:
    """Post-process a response for human-likeness.

    Returns (modified_text, delay_seconds).
    The caller should sleep(delay_seconds) before displaying modified_text.

    # TODO(human): Implement the humanization pipeline
    #
    # This function applies three transformations in sequence:
    #
    # --- STEP 1: Calculate typing delay ---
    #
    # Base delay = len(text) / config.typing_speed_cps
    #   (typing_speed_cps = characters per second, default 30)
    #
    # Emotion multipliers:
    #   ENGAGED, AMUSED: 0.7x (excited people type faster)
    #   ANNOYED: 0.5x (short responses, fired off quickly)
    #   REFLECTIVE: 1.5x (thinking takes time)
    #   TIRED: 1.3x (slower, less energy)
    #   NEUTRAL, CURIOUS: 1.0x (baseline)
    #
    # Add random jitter: +/- 20% (use random.uniform(0.8, 1.2))
    # Clamp to [config.min_delay_seconds, config.max_delay_seconds]
    #
    # --- STEP 2: Truncation of over-explanation ---
    #
    # If the response is longer than a threshold (e.g., 500 characters for
    # ANNOYED or TIRED, 800 for others), truncate it:
    #   - Find the last sentence boundary (. or ! or ?) before the threshold
    #   - Cut there
    #   - Optionally append "..." for TIRED or "anyway." for ANNOYED
    #
    # Why: small models tend to over-explain. Humans don't -- especially
    # when annoyed or tired. Truncation is a cheap way to enforce this.
    #
    # --- STEP 3: Hesitation injection ---
    #
    # With some probability (based on emotion), prepend or inject a
    # hesitation marker:
    #   - REFLECTIVE (40% chance): prepend "hm, " or "let me think... "
    #   - CURIOUS (30% chance): prepend "oh, " or "wait -- "
    #   - TIRED (30% chance): prepend "ugh, " or lowercase the first character
    #   - NEUTRAL (15% chance): prepend "well, " or "I mean, "
    #   - ENGAGED (10% chance): prepend "oh! " or "honestly, "
    #   - ANNOYED (5% chance): no hesitation (annoyed people don't hedge)
    #   - AMUSED (20% chance): prepend "hah, " or "ok so "
    #
    # These probabilities should be tunable. Use random.random() < probability
    # to decide whether to inject, then random.choice() from the options.
    #
    # IMPORTANT: don't inject if the response already starts with a hesitation
    # marker (the LLM might have generated one from the persona's verbal_tics).
    # Check the first few characters before injecting.
    #
    # --- STEP 4 (optional): Lowercase start ---
    #
    # For TIRED emotion with intensity > 0.6, lowercase the first character
    # of the response. Small thing, but it reads as "low energy."
    #
    # Return (modified_text, delay_seconds)
    #
    # Implementation hint:
    #   Structure this as a pipeline: text -> truncate -> hesitate -> lowercase
    #   Calculate delay separately (it doesn't modify text).
    #   Keep the code simple -- this is a 10-minute exercise, not a thesis.
    """

    base_delay = len(text) / config.typing_speed_cps

    emotional_delay = base_delay * {
        Emotion.ENGAGED:0.7,
        Emotion.AMUSED:0.7,
        Emotion.ANNOYED:0.5,
        Emotion.REFLECTIVE:1.5,
        Emotion.TIRED:1.3,
        Emotion.NEUTRAL:1,
        Emotion.CURIOUS:1
        }[emotion]

    delay = emotional_delay * random.uniform(0.8,1.2)

    limit = 500 if emotion in (Emotion.ANNOYED, Emotion.TIRED) else 800

    if len(text) > limit:
        last_boundary = max( text.rfind(delimiter, 0, limit) for delimiter in ".?!")
        if last_boundary == -1:
            last_boundary = limit
        text = text[:last_boundary]
        match emotion:
            case Emotion.ANNOYED: text += " anyways."
            case Emotion.TIRED: text += "..."

    (prob, tic) = {
        Emotion.ENGAGED: (0.1, "honestly, "),
        Emotion.AMUSED:  (0.2, "ok so "),
        Emotion.ANNOYED: (0.05, "no hesitation\n"),
        Emotion.REFLECTIVE:(0.4, "hm "),
        Emotion.TIRED: (0.3, "ugh "),
        Emotion.NEUTRAL: (0.15, "well, "),
        Emotion.CURIOUS: (0.3, "wait --")
        }[emotion]

    if random.uniform(0,1) < prob:
        text = tic + text

    return (text, delay)

def apply_delay(delay_seconds: float) -> None:
    """Sleep for the specified duration. Separated for testability."""
    if delay_seconds > 0:
        time.sleep(delay_seconds)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Test humanization with various emotions and texts."""

    config = AgentConfig()

    test_cases = [
        ("This is a fairly normal response about something interesting.", Emotion.NEUTRAL, 0.5),
        ("Oh that's actually really fascinating! Let me tell you about the history of mechanical keyboards, which goes back to the 1970s when IBM developed the Model M with its buckling spring mechanism. The tactile feedback was revolutionary at the time and many enthusiasts still prefer it today. The modern mechanical keyboard renaissance started around 2010 with Cherry MX switches becoming more widely available to consumers.", Emotion.ENGAGED, 0.8),
        ("fine.", Emotion.ANNOYED, 0.9),
        ("I think there's something deeper going on here... like, the way we think about memory in distributed systems mirrors how human memory works. We don't store perfect copies of events, we store reconstructions that drift over time.", Emotion.REFLECTIVE, 0.7),
        ("yeah... idk, maybe.", Emotion.TIRED, 0.6),
        ("haha ok that's pretty good, I'll give you that one", Emotion.AMUSED, 0.7),
    ]

    print("Testing humanization:\n")
    for text, emotion, intensity in test_cases:
        try:
            modified, delay = humanize_response(text, emotion, intensity, config)
            print(f"  [{emotion.value:12s} {intensity:.1f}] delay={delay:.2f}s")
            print(f"    Original:  {text[:80]}{'...' if len(text) > 80 else ''}")
            print(f"    Modified:  {modified[:80]}{'...' if len(modified) > 80 else ''}")
            print()
        except NotImplementedError:
            print("[SKIP] humanize_response not implemented yet (Exercise 6)")
            break

    print("Humanizer self-test complete.")


if __name__ == "__main__":
    _self_test()
