"""Entry point -- Interactive CLI chat loop.

Wires together all exercises into a working persona agent:
  - FileStore (Exercise 1) for persistence
  - Persona rendering (Exercise 2) for system prompts
  - Structured introspection (Exercise 3) for think -> feel -> respond
  - Emotional FSM (Exercise 4) for state-based routing
  - Memory (Exercise 5) for cross-session fact persistence
  - Humanizer (Exercise 6) for response post-processing

Usage:
    uv run python -m src.main                  # Default interactive chat
    uv run python -m src.main --user-id alice  # Chat as a specific user
    uv run python -m src.main --show-internals # Show emotion + inner_thought
    uv run python -m src.main --verify         # Verify setup only
"""

from __future__ import annotations

import argparse
import sys

from src.emotional_fsm import compile_graph, PersonaState
from src.file_store import FileStore
from src.humanizer import apply_delay, humanize_response
from src.llm_config import create_instructor_client, create_raw_openai_client
from src.memory import extract_user_facts, format_facts_for_prompt, merge_facts
from src.models import (
    AgentConfig,
    ConversationTurn,
    Emotion,
    EmotionalState,
)
from src.persona import render_system_prompt
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_setup() -> bool:
    """Check that Ollama is running and the model is available."""
    print("Verifying setup...\n")

    # 1. File structure
    store = FileStore()
    print("[OK] Data directories exist")

    # 2. Persona
    try:
        persona = store.load_persona()
        print(f"[OK] Persona loaded: {persona.name}")
    except FileNotFoundError:
        print("[FAIL] No persona.json found. Run:")
        print("  cp data/persona_example.json data/persona.json")
        return False

    # 3. Config
    config = store.load_config()
    print(f"[OK] Config loaded: model={config.model}")

    # 4. LLM provider connectivity
    try:
        base_client = create_raw_openai_client(config)
        models = base_client.models.list()
        available = [m.id for m in models.data]
        print(f"[OK] {config.provider} connected at {config.ollama_base_url}. Models: {available}")
    except Exception as e:
        print(f"[FAIL] Cannot connect to {config.provider} at {config.ollama_base_url}: {e}")
        print("  Start Ollama: docker compose up -d")
        return False

    # 5. Model availability
    if config.model not in available:
        print(f"[FAIL] Model '{config.model}' not found. Pull it:")
        print(f"  docker compose exec ollama ollama pull {config.model}")
        return False
    print(f"[OK] Model '{config.model}' available")

    # 6. Instructor patching
    try:
        client = create_instructor_client(config)
        print("[OK] Instructor client created")
    except Exception as e:
        print(f"[FAIL] Instructor setup failed: {e}")
        return False

    # 7. LangGraph compilation
    try:
        graph = compile_graph()
        print("[OK] Emotional graph compiled")
    except Exception as e:
        print(f"[FAIL] Graph compilation failed: {e}")
        return False

    print("\n--- All checks passed! Ready to chat. ---")
    return True


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

def run_chat(user_id: str, show_internals: bool) -> None:
    """Main interactive chat loop."""
    # Initialize components
    store = FileStore()
    config = store.load_config()
    persona = store.load_persona()
    client = create_instructor_client(config)
    graph = compile_graph()

    # Load persistent memory
    fact_store = store.load_user_facts(user_id)
    print(f"  Loaded {len(fact_store.facts)} facts about '{user_id}'")

    # Initialize emotional state
    emo_state = EmotionalState()

    # Start conversation log
    conversation = store.start_new_conversation(user_id)
    print(f"  Conversation ID: {conversation.conversation_id}")

    # Conversation history (in-context messages for the LLM)
    messages: list[dict[str, str]] = []

    print(f"\n  Chatting with {persona.name}. Type 'quit' to exit.\n")
    print("-" * 60)

    while True:
        # Get user input
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print(f"\n{persona.name}: see you later!")
            break

        # Log user turn
        user_turn = ConversationTurn(role="user", content=user_input)
        store.append_conversation_turn(conversation, user_turn)

        # Add to in-context history
        messages.append({"role": "user", "content": user_input})

        # Trim history to max_history_turns
        if len(messages) > config.max_history_turns * 2:
            messages = messages[-(config.max_history_turns * 2):]

        # Extract facts (Exercise 5)
        if config.fact_extraction_enabled:
            try:
                new_facts = extract_user_facts(client, config.model, user_input)
                if new_facts:
                    fact_store = merge_facts(fact_store, new_facts)
                    store.save_user_facts(fact_store)
            except (NotImplementedError, Exception):
                pass  # Graceful degradation if memory isn't implemented

        # Render system prompt with current state
        user_facts_text = format_facts_for_prompt(fact_store.facts)
        system_prompt = render_system_prompt(
            persona=persona,
            emotional_state=emo_state,
            user_facts=fact_store.facts,
        )

        # Run emotional graph
        graph_input: PersonaState = {
            "user_message": user_input,
            "system_prompt": system_prompt,
            "conversation_history": messages.copy(),
            "emotion": emo_state.emotion,
            "intensity": emo_state.intensity,
            "turns_in_emotion": emo_state.turns_in_state,
            "agent_turn": None,
            "model": config.model,
            "instructor_client": client,
        }

        try:
            result = graph.invoke(graph_input)
        except NotImplementedError as e:
            # If any TODO isn't implemented, fall back to a placeholder
            print(f"\n  [Not yet implemented: {e}]")
            print("  Complete the exercises to enable this feature.")
            continue
        except Exception as e:
            print(f"\n  [Error: {e}]")
            continue

        agent_turn = result.get("agent_turn")
        if agent_turn is None:
            print(f"\n  [No response generated]")
            continue

        # Update emotional state from graph result
        emo_state = EmotionalState(
            emotion=result.get("emotion", emo_state.emotion),
            intensity=result.get("intensity", emo_state.intensity),
            turns_in_state=result.get("turns_in_emotion", emo_state.turns_in_state),
        )

        # Show internals if requested
        if show_internals:
            print(f"\n  [emotion: {agent_turn.emotion.value}, intensity: {emo_state.intensity:.1f}]")
            print(f"  [inner_thought: {agent_turn.inner_thought}]")

        # Humanize response (Exercise 6)
        response_text = agent_turn.response
        delay = 0.0
        if config.humanization_enabled:
            try:
                response_text, delay = humanize_response(
                    response_text, emo_state.emotion, emo_state.intensity, config
                )
            except (NotImplementedError, Exception):
                pass  # Graceful degradation

        # Apply typing delay
        apply_delay(delay)

        # Display response
        print(f"\n{persona.name}: {response_text}")

        # Log agent turn
        agent_log_turn = ConversationTurn(
            role="agent",
            content=response_text,
            emotion=agent_turn.emotion,
            inner_thought=agent_turn.inner_thought,
            intensity=emo_state.intensity,
        )
        store.append_conversation_turn(conversation, agent_log_turn)

        # Add to in-context history
        messages.append({"role": "assistant", "content": response_text})

    print(f"\nConversation saved to data/logs/conversation_{conversation.conversation_id}.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(description="Self-Evolving Persona Agent")
    parser.add_argument("--verify", action="store_true", help="Verify setup and exit")
    parser.add_argument("--user-id", default="default", help="User ID for memory isolation")
    parser.add_argument("--show-internals", action="store_true", help="Show emotion and inner_thought")
    args = parser.parse_args()

    if args.verify:
        success = verify_setup()
        sys.exit(0 if success else 1)

    print("=" * 60)
    print("  Self-Evolving Persona Agent v0.1")
    print("=" * 60)
    print(f"  User ID: {args.user_id}")
    print(f"  Show internals: {args.show_internals}")

    try:
        run_chat(user_id=args.user_id, show_internals=args.show_internals)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
