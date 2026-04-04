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
# Emotion Update Node
# ---------------------------------------------------------------------------

def update_emotion(state: PersonaState) -> dict:
    """Determine the agent's new emotional state based on user message + current emotion.

    # TODO(human): Implement emotion transition logic
    #
    # This is the CORE of the emotional state machine. Given the user's message
    # and the agent's current emotion, determine what the agent feels NOW.
    #
    # Inputs (from state):
    #   - state["user_message"]: the latest user message
    #   - state["emotion"]: current Emotion enum value
    #   - state["intensity"]: current intensity (0.0 to 1.0)
    #   - state["turns_in_emotion"]: how many turns the agent has been in this emotion
    #
    # Outputs (return dict with these keys):
    #   - "emotion": new Emotion value
    #   - "intensity": new intensity (0.0 to 1.0)
    #   - "turns_in_emotion": reset to 0 if emotion changed, else increment
    #
    # Strategy -- there are two approaches, and you should choose one:
    #
    # APPROACH A: Rule-based (simpler, more predictable)
    #   Use keyword/pattern matching on the user message to detect:
    #     - Questions ("?", "why", "how") -> shift toward CURIOUS
    #     - Enthusiasm ("!", "amazing", "love") -> shift toward ENGAGED
    #     - Rudeness/dismissiveness ("whatever", "that's dumb") -> shift toward ANNOYED
    #     - Deep/philosophical messages (long, no questions) -> shift toward REFLECTIVE
    #     - Short/low-effort messages ("ok", "sure", "k") -> shift toward TIRED
    #     - Humor/jokes ("lol", "haha") -> shift toward AMUSED
    #   Combine with current state:
    #     - Already ANNOYED + more rudeness -> increase intensity
    #     - Already ANNOYED + kindness -> decrease intensity, if low enough -> NEUTRAL
    #     - Any emotion held for too many turns -> drift back toward NEUTRAL
    #       (recovery_speed from persona's emotional_defaults)
    #
    # APPROACH B: LLM-based (more nuanced, less predictable)
    #   Call the instructor client to classify the emotion transition:
    #     - Include current emotion + user message in the prompt
    #     - Use a small Pydantic model: {new_emotion: Emotion, new_intensity: float}
    #     - This gives more nuanced transitions but adds an LLM call per turn
    #   Pros: handles subtlety (sarcasm, passive aggression, mixed signals)
    #   Cons: slower, more expensive, less deterministic
    #
    # RECOMMENDATION for a 3B model: use APPROACH A (rule-based) for the emotion
    # transition, then let the LLM's structured introspection (Exercise 3) handle
    # the nuance in the response. Rule-based transitions are fast, predictable, and
    # debuggable. The LLM already gets the emotion as input when generating the
    # response -- that's where nuance matters.
    #
    # Implementation hint:
    #   - Lowercase the message for keyword matching
    #   - Use sets for keyword groups: CURIOUS_SIGNALS = {"why", "how", "what if", "?"}
    #   - Check for signals in order of priority (rudeness > enthusiasm > curiosity)
    #   - Apply a "drift to neutral" rule: if turns_in_emotion > 4, reduce intensity
    #     by 0.1 per turn. If intensity drops below 0.2, reset to NEUTRAL.
    #   - Return the partial state update dict
    #
    # Design question: should intensity be a float (continuous) or discrete levels
    # (low/medium/high)? Float gives finer control but makes routing harder.
    # Discrete levels are easier to route on. Consider using float internally
    # but bucketing into "mild" (<0.4), "moderate" (0.4-0.7), "strong" (>0.7)
    # for routing decisions.
    """
    raise NotImplementedError("Exercise 4: implement update_emotion")


# ---------------------------------------------------------------------------
# Emotion-based routing
# ---------------------------------------------------------------------------

def route_by_emotion(state: PersonaState) -> str:
    """Route to a response node based on the current emotional state.

    # TODO(human): Implement conditional routing based on emotion
    #
    # This function is used as the routing_fn in add_conditional_edges().
    # It examines state["emotion"] and state["intensity"] and returns the
    # NAME of the next node to execute.
    #
    # Routing rules (suggested -- feel free to modify):
    #   - ANNOYED with intensity > 0.6 -> "response_annoyed"
    #   - ENGAGED or CURIOUS with intensity > 0.5 -> "response_engaged"
    #   - REFLECTIVE with intensity > 0.4 -> "response_reflective"
    #   - Everything else -> "response_default"
    #
    # The return value must be a string matching a node name in the graph.
    #
    # Design note: you could also route based on intensity thresholds within
    # the same emotion. E.g., mildly annoyed -> response_default (with an
    # annoyed tint), strongly annoyed -> response_annoyed (short, flat).
    # Start simple and add complexity if the behavior feels too uniform.
    """
    raise NotImplementedError("Exercise 4: implement route_by_emotion")


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
