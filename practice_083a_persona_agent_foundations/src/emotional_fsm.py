"""Exercise 4 -- Emotional State Machine in LangGraph.

Implements emotion as a first-class state variable in a LangGraph StateGraph.
The graph processes each user turn through:

    START -> update_emotion -> [route by emotion] -> generate_response -> END

The emotion node examines the user's message and the agent's current emotional
state, then outputs a new emotion + intensity. Conditional edges route to
different response behaviors based on mood.

Why LangGraph for this?
    A simple if/else chain could handle emotion routing. But LangGraph gives us:
    1. Explicit state management -- emotion is in the TypedDict, not a global
    2. Visual graph -- you can print the graph structure for debugging
    3. Extensibility -- adding new emotional behaviors = adding nodes + edges
    4. Foundation for 083b -- the Reflector will be another node in this graph

Key concepts:
    - TypedDict State with emotion as a field
    - Conditional edges for routing based on emotional state
    - Nodes that read state and return partial updates
    - The graph compiles to a runnable that processes one user turn
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.models import AgentTurn, Emotion, EmotionalState

# ---------------------------------------------------------------------------
# Emotion transition rules (design decision: RULE-BASED, not LLM-based)
# ---------------------------------------------------------------------------
# Rationale:
#   - Deterministic, instant, debuggable (you can trace why the agent went
#     from NEUTRAL -> ANNOYED just by reading the input)
#   - The LLM already self-reports its felt emotion in AgentTurn (Ex 3),
#     so nuance is captured separately in logs.
#   - The FSM's job is STRUCTURAL routing (which response node to use),
#     not sentiment analysis of the user.
#
# Signal sets are keyword triggers. Order of priority matters: rudeness
# wins over curiosity (an angry question is still angry). Check ANNOYED
# triggers first, then other positive/active signals, then low-energy
# signals. Drift-to-neutral handles "nothing interesting happened" cases.

RUDENESS_SIGNALS = {"whatever", "that's dumb", "stupid", "shut up", "boring",
                    "who cares", "don't care", "useless"}
ENTHUSIASM_SIGNALS = {"amazing", "awesome", "love", "!!", "incredible",
                      "brilliant", "fantastic", "wow"}
CURIOSITY_SIGNALS = {"why", "how", "what if", "?", "how come", "explain",
                     "wonder"}
HUMOR_SIGNALS = {"lol", "haha", "lmao", "joke", "funny", ":)", "xd"}
LOW_EFFORT_SIGNALS = {"ok", "sure", "k", "yeah", "fine", "mhm", "uh huh"}

# Intensity bucket thresholds for routing
MILD = 0.4
STRONG = 0.7

# How many turns before drift-to-neutral kicks in
DRIFT_AFTER_TURNS = 4
DRIFT_PER_TURN = 0.1
NEUTRAL_THRESHOLD = 0.2


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class PersonaState(TypedDict):
    """Shared state flowing through the LangGraph graph.

    Every node reads from this state and returns a partial dict of updates.
    Fields without reducers use "last write wins" semantics.
    """
    # Input
    user_message: str
    system_prompt: str
    conversation_history: list[dict[str, str]]

    # Emotional state (persists across turns -- updated by update_emotion node)
    emotion: Emotion
    intensity: float
    turns_in_emotion: int

    # Output (set by response nodes)
    agent_turn: AgentTurn | None

    # Config (read-only, set at graph invocation)
    model: str
    instructor_client: Any  # instructor.Instructor (Any to avoid import in TypedDict)


# ---------------------------------------------------------------------------
# Helpers (scaffolded -- not part of TODO)
# ---------------------------------------------------------------------------

def _detect_signal(message: str) -> Emotion | None:
    """Scan the (lowercased) message for keyword triggers.

    Priority order: rudeness > enthusiasm > humor > curiosity > low-effort.
    Rudeness wins because an angry question is still angry. Low-effort is
    last because it's the weakest signal (might be misread).

    Returns the Emotion the signal shifts toward, or None if no signal hit.
    """
    msg = message.lower()
    if any(tok in msg for tok in RUDENESS_SIGNALS):
        return Emotion.ANNOYED
    if any(tok in msg for tok in ENTHUSIASM_SIGNALS):
        return Emotion.ENGAGED
    if any(tok in msg for tok in HUMOR_SIGNALS):
        return Emotion.AMUSED
    if any(tok in msg for tok in CURIOSITY_SIGNALS):
        return Emotion.CURIOUS
    # Low-effort detection: short message AND contains a low-effort token
    if len(msg.split()) <= 3 and any(tok == msg.strip(" .!?") or tok in msg.split()
                                     for tok in LOW_EFFORT_SIGNALS):
        return Emotion.TIRED
    # Reflective: long message, no question marks
    if len(msg.split()) > 30 and "?" not in msg:
        return Emotion.REFLECTIVE
    return None


# ---------------------------------------------------------------------------
# Emotion Update Node
# ---------------------------------------------------------------------------

def update_emotion(state: PersonaState) -> dict:
    """Determine the agent's new emotional state based on user message + current emotion.

    # TODO(human): Implement the emotion transition logic using the rule-based
    # approach. This is the CORE of the emotional state machine -- it decides
    # both WHICH response node the graph routes to and HOW intense the response is.
    #
    # Inputs (from state):
    #   - state["user_message"]: the latest user message
    #   - state["emotion"]: current Emotion enum value
    #   - state["intensity"]: current intensity (0.0 to 1.0)
    #   - state["turns_in_emotion"]: how many turns agent has been in this emotion
    #
    # Outputs (return dict with these keys):
    #   - "emotion": new Emotion value
    #   - "intensity": new intensity (0.0 to 1.0)
    #   - "turns_in_emotion": reset to 0 if emotion changed, else increment
    #
    # ALGORITHM (follow this recipe):
    #
    #   1. Call _detect_signal(state["user_message"]) to get a candidate Emotion
    #      (or None if no signal was detected).
    #
    #   2. If a signal was detected:
    #        a. If signal_emotion == current emotion: REINFORCE
    #             -> keep the emotion, boost intensity by +0.15 (capped at 1.0),
    #                increment turns_in_emotion
    #        b. If signal_emotion != current emotion: TRANSITION
    #             -> switch to the new emotion with intensity 0.6,
    #                reset turns_in_emotion to 0
    #
    #   3. If no signal was detected (the "nothing interesting happened" case):
    #        a. If already at NEUTRAL: stay NEUTRAL, keep intensity, increment turns.
    #        b. Otherwise, apply DRIFT-TO-NEUTRAL:
    #           - Increment turns_in_emotion
    #           - If turns_in_emotion > DRIFT_AFTER_TURNS (4):
    #               reduce intensity by DRIFT_PER_TURN (0.1)
    #           - If intensity drops below NEUTRAL_THRESHOLD (0.2):
    #               snap to Emotion.NEUTRAL with intensity 0.5,
    #               reset turns_in_emotion to 0
    #
    #   4. Return a dict with {"emotion", "intensity", "turns_in_emotion"}.
    #
    # WHY THIS DESIGN:
    #   - Signal detection is separated into _detect_signal (already written
    #     for you) so this function focuses on transition logic, not keyword
    #     matching. Single Level of Abstraction Principle.
    #   - Reinforcement (+0.15) vs transition (=0.6) creates distinct
    #     "getting more annoyed" vs "suddenly annoyed" behaviors.
    #   - Drift-to-neutral after 4 turns prevents the agent from getting
    #     stuck in an emotion forever when the user stops triggering signals.
    #
    # TEST YOUR IMPLEMENTATION by running:
    #   uv run python -m src.emotional_fsm
    # The _self_test at the bottom exercises all transition cases.
    """

    candidate_emotion = _detect_signal(state["user_message"])

    if candidate_emotion == state["emotion"]:
        state["intensity"] = (0.8 * state["intensity"] + 0.2)
        state["turns_in_emotion"] += 1
    elif candidate_emotion:
        state["emotion"] = candidate_emotion
        state["intensity"] = 0.6
        state["turns_in_emotion"] = 0
    else:
        state["turns_in_emotion"] += 1
        if not state["emotion"] == Emotion.NEUTRAL:
            if state["turns_in_emotion"] > DRIFT_AFTER_TURNS:
                state["intensity"] -= 0.1
                if state["intensity"] < NEUTRAL_THRESHOLD:
                    state["emotion"] = Emotion.NEUTRAL
                    state["intensity"] = 0.5
                    state["turns_in_emotion"] = 0

    return {"emotion":state["emotion"], "intensity":state["intensity"], "turns_in_emotion":state["turns_in_emotion"]}
# ---------------------------------------------------------------------------
# Emotion-based routing
# ---------------------------------------------------------------------------

def route_by_emotion(state: PersonaState) -> str:
    """Route to a response node based on current emotion + intensity bucket.

    Scaffolded: this is the mechanical half of Exercise 4. The interesting
    work is in update_emotion (above), which determines WHAT emotion state
    we land in. Routing is just a table lookup on that state.

    Bucketing rule: only STRONG (intensity > STRONG=0.7) or MODERATE
    (> MILD=0.4) feelings get a dedicated response node. Mild feelings
    fall through to the default node (a "hint" of emotion, not a
    structural change).
    """
    emotion = state["emotion"]
    intensity = state["intensity"]

    if emotion == Emotion.ANNOYED and intensity > MILD:
        return "response_annoyed"
    if emotion in (Emotion.ENGAGED, Emotion.CURIOUS, Emotion.AMUSED) and intensity > MILD:
        return "response_engaged"
    if emotion == Emotion.REFLECTIVE and intensity > MILD:
        return "response_reflective"
    return "response_default"


# ---------------------------------------------------------------------------
# Response Generation Nodes
# ---------------------------------------------------------------------------

def _generate_with_modifier(state: PersonaState, prompt_modifier: str) -> dict:
    """Shared response generation with an emotion-specific prompt modifier.

    This is the common implementation for all response_* nodes. The modifier
    is prepended to the system prompt to adjust the response style.

    NOT a TODO -- this is scaffolded because the interesting logic is in
    update_emotion and route_by_emotion, not in the response nodes themselves.
    """
    from src.introspection import generate_agent_turn

    modified_prompt = f"{prompt_modifier}\n\n{state['system_prompt']}"
    turn = generate_agent_turn(
        client=state["instructor_client"],
        model=state["model"],
        system_prompt=modified_prompt,
        messages=state["conversation_history"],
    )
    return {"agent_turn": turn}


def response_default(state: PersonaState) -> dict:
    """Default response -- no emotional modifier."""
    return _generate_with_modifier(state, "")


def response_engaged(state: PersonaState) -> dict:
    """Engaged/curious response -- expansive, asks follow-ups."""
    modifier = (
        "[EMOTIONAL CONTEXT: You're feeling genuinely engaged and curious right now. "
        "Elaborate more than usual, ask a follow-up question, and show enthusiasm. "
        "Use more expressive punctuation and longer responses.]"
    )
    return _generate_with_modifier(state, modifier)


def response_annoyed(state: PersonaState) -> dict:
    """Annoyed response -- short, flat, skips pleasantries."""
    modifier = (
        "[EMOTIONAL CONTEXT: You're feeling annoyed right now. Keep your response "
        "SHORT -- 1-2 sentences max. Skip pleasantries, use flat punctuation, "
        "don't ask follow-up questions. You're not being mean, just... not in the mood.]"
    )
    return _generate_with_modifier(state, modifier)


def response_reflective(state: PersonaState) -> dict:
    """Reflective response -- thoughtful, slower pace, deeper."""
    modifier = (
        "[EMOTIONAL CONTEXT: You're in a reflective mood. Take your time with the "
        "response. Go deeper than usual, share a genuine thought or perspective, "
        "use more ellipses and pauses. Thoughtful, not verbose.]"
    )
    return _generate_with_modifier(state, modifier)


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_emotional_graph() -> StateGraph:
    """Build the LangGraph StateGraph for emotional response generation.

    Graph structure:
        START -> update_emotion -> [conditional routing] -> response_* -> END

    The conditional edge after update_emotion uses route_by_emotion() to
    select which response node to invoke based on the current emotional state.
    """
    builder = StateGraph(PersonaState)

    # Add nodes
    builder.add_node("update_emotion", update_emotion)
    builder.add_node("response_default", response_default)
    builder.add_node("response_engaged", response_engaged)
    builder.add_node("response_annoyed", response_annoyed)
    builder.add_node("response_reflective", response_reflective)

    # START -> update_emotion
    builder.add_edge(START, "update_emotion")

    # update_emotion -> [conditional] -> response_*
    builder.add_conditional_edges(
        "update_emotion",
        route_by_emotion,
        {
            "response_default": "response_default",
            "response_engaged": "response_engaged",
            "response_annoyed": "response_annoyed",
            "response_reflective": "response_reflective",
        },
    )

    # All response nodes -> END
    builder.add_edge("response_default", END)
    builder.add_edge("response_engaged", END)
    builder.add_edge("response_annoyed", END)
    builder.add_edge("response_reflective", END)

    return builder


def compile_graph():
    """Compile the emotional graph into a runnable."""
    builder = build_emotional_graph()
    return builder.compile()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Test emotional state machine transitions."""
    print("Testing emotional state machine...\n")

    # Test update_emotion with various messages
    test_cases = [
        {"user_message": "Why does this work?", "emotion": Emotion.NEUTRAL, "intensity": 0.5, "turns_in_emotion": 0},
        {"user_message": "This is amazing!!", "emotion": Emotion.NEUTRAL, "intensity": 0.5, "turns_in_emotion": 0},
        {"user_message": "whatever, that's dumb", "emotion": Emotion.NEUTRAL, "intensity": 0.5, "turns_in_emotion": 0},
        {"user_message": "ok", "emotion": Emotion.ENGAGED, "intensity": 0.8, "turns_in_emotion": 3},
        {"user_message": "Tell me more about that", "emotion": Emotion.ANNOYED, "intensity": 0.6, "turns_in_emotion": 2},
    ]

    for tc in test_cases:
        state = {
            "user_message": tc["user_message"],
            "emotion": tc["emotion"],
            "intensity": tc["intensity"],
            "turns_in_emotion": tc["turns_in_emotion"],
            "system_prompt": "",
            "conversation_history": [],
            "agent_turn": None,
            "model": "",
            "instructor_client": None,
        }
        try:
            result = update_emotion(state)
            print(f"  '{tc['user_message'][:40]}...' | "
                  f"{tc['emotion'].value} -> {result['emotion'].value} "
                  f"(intensity: {result['intensity']:.1f})")
        except NotImplementedError:
            print("[SKIP] update_emotion not implemented yet (Exercise 4)")
            break

    # Test routing
    print()
    route_tests = [
        (Emotion.ANNOYED, 0.8),
        (Emotion.ENGAGED, 0.7),
        (Emotion.REFLECTIVE, 0.6),
        (Emotion.NEUTRAL, 0.5),
        (Emotion.CURIOUS, 0.3),
    ]

    for emo, intensity in route_tests:
        state = {
            "emotion": emo,
            "intensity": intensity,
            "user_message": "",
            "system_prompt": "",
            "conversation_history": [],
            "agent_turn": None,
            "model": "",
            "instructor_client": None,
            "turns_in_emotion": 0,
        }
        try:
            route = route_by_emotion(state)
            print(f"  {emo.value} (intensity {intensity}) -> {route}")
        except NotImplementedError:
            print("[SKIP] route_by_emotion not implemented yet (Exercise 4)")
            break

    # Test graph compilation (doesn't require TODO implementations)
    try:
        graph = compile_graph()
        print(f"\n[OK] Graph compiled successfully")
        print(f"  Nodes: {list(graph.get_graph().nodes.keys())}")
    except Exception as e:
        print(f"\n[FAIL] Graph compilation failed: {e}")

    print("\nEmotional FSM self-test complete.")


if __name__ == "__main__":
    _self_test()
