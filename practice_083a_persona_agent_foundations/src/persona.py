"""Exercise 2 -- Persona Card Architecture.

Renders a PersonaCard + emotional state + user facts into a complete
system prompt for the LLM.

The system prompt is the single most important lever for persona consistency.
It must convey:
  1. WHO the agent is (identity, backstory, personality)
  2. HOW the agent speaks (verbal tics, punctuation, length)
  3. HOW the agent currently FEELS (emotional state)
  4. WHAT the agent KNOWS about this user (persistent facts)
  5. What the agent must NEVER do (forbidden behaviors)

The key insight: specificity beats generality. "Be friendly" produces generic
output. "Uses 'honestly' as a verbal tic, gets annoyed by vague questions,
types in lowercase when tired" produces memorable, consistent personality.
"""

from __future__ import annotations

from src.models import EmotionalState, PersonaCard, UserFact


def render_system_prompt(
    persona: PersonaCard,
    emotional_state: EmotionalState,
    user_facts: list[UserFact],
) -> str:
    """Assemble a complete system prompt from persona + emotion + memory.

    # TODO(human): Implement system prompt rendering
    #
    # This is CHARACTER WRITING, not just string formatting. The prompt must
    # feel like a coherent character brief that the model can embody, not a
    # mechanical list of constraints.
    #
    # The prompt should include these sections (in roughly this order):
    #
    # 1. IDENTITY BLOCK
    #    "You are {name}, a {age}-year-old {backstory summary}."
    #    Include personality traits, contradictions, and interests.
    #    This is the core -- the model reads it first and anchors on it.
    #
    # 2. SPEECH PATTERNS
    #    How to talk: verbal tics, punctuation style, capitalization rules,
    #    emoji policy, response length tendency.
    #    Be SPECIFIC: "You often start sentences with 'honestly'" is better
    #    than "You have casual speech patterns."
    #    Include characteristic_phrases as examples the model can draw from.
    #
    # 3. CURRENT EMOTIONAL STATE
    #    "Right now you're feeling {emotion} (intensity: {intensity})."
    #    Add behavioral guidance per emotion:
    #      - annoyed: shorter responses, skip pleasantries, flat punctuation
    #      - engaged: elaborate more, ask follow-up questions, expressive
    #      - curious: ask questions, show genuine interest
    #      - tired: lowercase, shorter, less energy, more ellipses
    #      - amused: lighter tone, might make jokes or references
    #      - reflective: thoughtful, slower pace, deeper responses
    #      - neutral: your default baseline
    #
    #    Consider: should intensity affect the behavioral guidance? A mildly
    #    annoyed agent behaves differently than a very annoyed one. You could
    #    use intensity thresholds (e.g., >0.7 = strong behavioral shift).
    #
    # 4. USER MEMORY (if any facts exist)
    #    "What you know about this person:"
    #    Format facts as a brief list. Only include high-confidence facts.
    #    This block is PREPENDED knowledge -- the model should reference
    #    these facts naturally in conversation when relevant, not parrot them.
    #
    # 5. FORBIDDEN BEHAVIORS
    #    "You must NEVER:"
    #    List forbidden_behaviors as hard constraints.
    #
    # 6. META-INSTRUCTION
    #    A closing instruction like: "Stay in character at all times. Your
    #    responses should feel natural and human -- not like an AI assistant.
    #    Vary your response length. Don't over-explain."
    #
    # Implementation hint:
    #   Build each section as a string, then join with double newlines.
    #   Use f-strings or .format() -- keep it readable.
    #   The total prompt should be ~300-600 tokens (well within the 32k context).
    #
    # Design question to consider:
    #   Should the emotional state guidance be prescriptive ("respond in short
    #   sentences") or descriptive ("you're feeling annoyed, which makes you
    #   less patient")? Descriptive tends to work better with small models
    #   because it lets the model interpret the emotion naturally rather than
    #   following rigid formatting rules.
    """

    prompt = f"You are {persona.name}, a {persona.age}-year-old {persona.backstory}.\n"
    prompt += "Your core:\n" + "\n".join(persona.personality_traits.core) + "\n"
    prompt += "Your contradictions:\n" + "\n".join(persona.personality_traits.contradictions) + "\n"
    prompt += "Your interests:\n" + "\n".join(persona.personality_traits.interests) + "\n"
    prompt += "Your pet_peeves:\n" + "\n".join(persona.personality_traits.pet_peeves) + "\n"
    prompt += "Speech Patterns\n"
    prompt += "Verbal tics:\n" + "\n".join(persona.speech_patterns.verbal_tics) + "\n"
    prompt += "Puntuation style: " + persona.speech_patterns.punctuation_style + "\n"
    prompt += "Capitalization: " + persona.speech_patterns.capitalization + "\n"
    prompt += "Emoji usage: " + persona.speech_patterns.emoji_usage + "\n"
    prompt += "Response length tendency: " + persona.speech_patterns.response_length_tendency + "\n"
    prompt += "Characteristic phrases:\n" + "\n".join(persona.characteristic_phrases) + "\n"
    prompt += f"Emotional status: {emotional_state.emotion} (intensity = {emotional_state.intensity})\n"
    prompt += "Knonw facts:\n" + '\n'.join(
        map(
            lambda uf : f"{uf.key} : {uf.value} with confidence : {uf.confidence}" , user_facts
        )
    ) + "\n"
    prompt += "You must NEVER:\n" + "\n".join(persona.forbidden_behaviors) + "\n"
    prompt += "Stay in character at all times. Your responses should feel natural and human -- not like an AI assistant. Vary your response length. Don't over-explain."

    return prompt
# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Test persona loading and prompt rendering."""
    from src.file_store import FileStore
    from src.models import Emotion, EmotionalState, UserFact

    store = FileStore()

    # Try loading persona -- will fail if persona.json doesn't exist
    try:
        persona = store.load_persona()
        print(f"[OK] Loaded persona: {persona.name}, age {persona.age}")
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        print("  Using persona_example.json for testing...")
        import json
        from pathlib import Path

        example_path = Path(__file__).parent.parent / "data" / "persona_example.json"
        with open(example_path, "r", encoding="utf-8") as f:
            persona = PersonaCard.model_validate(json.load(f))
        print(f"[OK] Loaded example persona: {persona.name}")

    # Test with different emotional states
    test_emotions = [
        EmotionalState(emotion=Emotion.NEUTRAL, intensity=0.5),
        EmotionalState(emotion=Emotion.ANNOYED, intensity=0.8),
        EmotionalState(emotion=Emotion.ENGAGED, intensity=0.9),
        EmotionalState(emotion=Emotion.TIRED, intensity=0.6),
    ]

    test_facts = [
        UserFact(key="name", value="Alex", source_message="I'm Alex"),
        UserFact(key="job", value="ML engineer", source_message="I work on ML systems"),
    ]

    for emo_state in test_emotions:
        try:
            prompt = render_system_prompt(persona, emo_state, test_facts)
            print(f"\n--- Emotion: {emo_state.emotion.value} (intensity: {emo_state.intensity}) ---")
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        except NotImplementedError:
            print("[SKIP] render_system_prompt not implemented yet (Exercise 2)")
            break

    print("\nPersona self-test complete.")


if __name__ == "__main__":
    _self_test()
