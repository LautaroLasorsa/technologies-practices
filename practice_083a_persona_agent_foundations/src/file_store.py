"""Exercise 1 -- File Architecture & Data Schemas.

FileStore: the single class responsible for ALL file I/O in the agent.
Every other module reads/writes state through FileStore -- no module touches
the filesystem directly.

# TODO(human): DESIGN THE FILE ARCHITECTURE
#
# Before writing any code, design the directory layout and file organization
# for the persona agent's persistent state. This is a real design exercise --
# the choices you make here affect debuggability, extensibility, and how
# easily 083b's Reflector can scan conversation logs.
#
# STEP 1 -- Define the directory tree.
#   Fill in the LAYOUT dict below. Each key is a purpose (e.g., "persona",
#   "user_memory", "conversation_logs", "config"), and the value is the
#   relative path under data/. Consider:
#
#   - FLAT vs NESTED: Should all files live in data/ or use subdirectories?
#     Flat is simpler but gets cluttered with many users/conversations.
#     Nested (data/memory/, data/logs/) groups related files but adds depth.
#
#   - BY-ENTITY vs BY-TYPE: Should user files be grouped by user
#     (data/users/alice/{facts,logs}/) or by type (data/memory/alice.json,
#     data/logs/conv_alice_*.json)? By-type is simpler for single-purpose
#     access (e.g., "load all conversations"); by-entity is better when you
#     need everything about one user.
#
#   - COMMITTED vs GITIGNORED: Which files are part of the scaffold (persona
#     templates, example configs) and which are runtime state (user facts,
#     conversation logs)? Runtime state should be gitignored.
#
# STEP 2 -- Define file naming conventions.
#   How will conversation log files be named? Options:
#     - UUID-based: `conv_a1b2c3d4.json` -- unique but opaque
#     - Timestamp-based: `conv_20260403_143000.json` -- sortable, readable
#     - Hybrid: `conv_20260403_143000_alice.json` -- sortable + identifiable
#   How about user fact files? One file per user? One global file?
#
# STEP 3 -- Define read/write strategy.
#   - MISSING FILES: Return defaults (ergonomic for new users) or raise
#     (forces explicit initialization)? Different files may warrant different
#     strategies -- a missing persona is an error, but missing user facts
#     just means a new user.
#   - WRITE PATTERN: Full JSON rewrite (simple, valid JSON, but re-writes
#     everything) or append-only JSONL (efficient for logs, but harder to
#     read/parse)? For <100 turns per conversation, full rewrite is fine.
#   - ATOMIC WRITES: Write to a temp file then rename (crash-safe) or write
#     directly (simpler)? For a single-user local agent, direct writes are
#     acceptable. For production, you'd want atomic writes.
#
# After designing, define the directory constants below and implement the
# FileStore methods. The _read_json/_write_json helpers and persona/config
# methods are provided as reference for the pattern -- implement the rest
# following the same conventions.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.models import (
    AgentConfig,
    ConversationLog,
    ConversationTurn,
    PersonaCard,
    UserFactStore,
)

# TODO(human): Define your directory layout constants.
# Replace these with your chosen paths based on the design above.
# Example (you may change this entirely):
DATA_DIR = Path(__file__).parent.parent / "data"
MEMORY_DIR = DATA_DIR / "memory"  # Where should user facts live?
LOGS_DIR = DATA_DIR / "logs"      # Where should conversation logs live?


class FileStore:
    """Handles all file-based persistence for the persona agent.

    Single Responsibility: this class owns file I/O. It knows about file
    paths, JSON serialization, and directory creation. It does NOT know
    about LLM calls, emotions, or conversation logic.

    All methods are synchronous -- file I/O on local disk is fast enough
    that async adds complexity without benefit for a single-user agent.
    """

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self._data_dir = data_dir
        self._memory_dir = data_dir / "memory"
        self._logs_dir = data_dir / "logs"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create data directories if they don't exist."""
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_json(self, path: Path) -> Any:
        """Read and parse a JSON file. Returns None if file doesn't exist."""
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: Path, data: Any) -> None:
        """Write data as formatted JSON. Creates parent directories."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Persona
    # ------------------------------------------------------------------

    def load_persona(self) -> PersonaCard:
        """Load the persona card from data/persona.json.

        Returns the parsed PersonaCard, or raises FileNotFoundError if
        no persona file exists (the user must copy persona_example.json).
        """
        path = self._data_dir / "persona.json"
        raw = self._read_json(path)
        if raw is None:
            raise FileNotFoundError(
                f"Persona file not found at {path}. "
                "Run: cp data/persona_example.json data/persona.json"
            )
        return PersonaCard.model_validate(raw)

    def save_persona(self, persona: PersonaCard) -> None:
        """Save a persona card to data/persona.json."""
        path = self._data_dir / "persona.json"
        self._write_json(path, persona.model_dump())

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def load_config(self) -> AgentConfig:
        """Load runtime config from data/config.json.

        Returns default config if file doesn't exist -- the agent should
        work out of the box without a config file.
        """
        path = self._data_dir / "config.json"
        raw = self._read_json(path)
        if raw is None:
            return AgentConfig()
        return AgentConfig.model_validate(raw)

    def save_config(self, config: AgentConfig) -> None:
        """Save runtime config to data/config.json."""
        path = self._data_dir / "config.json"
        self._write_json(path, config.model_dump())

    # ------------------------------------------------------------------
    # User Facts (Exercise 1 -- TODO(human))
    # ------------------------------------------------------------------

    def load_user_facts(self, user_id: str) -> UserFactStore:
        """Load the fact store for a given user.

        # TODO(human): Implement fact loading from data/memory/<user_id>.json
        #
        # Design decisions to make:
        #   1. File naming: should it be <user_id>.json? What if user_id has
        #      special characters? Consider sanitizing or hashing.
        #   2. Missing file: should this return an empty UserFactStore or raise?
        #      Think about the caller -- when a new user starts chatting, there
        #      are no facts yet. Returning empty is more ergonomic.
        #   3. Schema evolution: what happens if the JSON file has an older schema
        #      (e.g., missing the 'confidence' field)? Pydantic's model_validate
        #      handles this with defaults, but you should think about it.
        #
        # Implementation steps:
        #   - Build the path: self._memory_dir / f"{user_id}.json" (or sanitized)
        #   - Use self._read_json() to load the raw data
        #   - If None (file doesn't exist), return UserFactStore(user_id=user_id)
        #   - Otherwise, validate with UserFactStore.model_validate(raw)
        #
        # Hint: keep it simple. For a single-user local agent, user_id is likely
        # just "default" or a simple alphanumeric string. Over-engineering the
        # path sanitization isn't necessary here.
        """
        raise NotImplementedError("Exercise 1: implement load_user_facts")

    def save_user_facts(self, fact_store: UserFactStore) -> None:
        """Save the fact store for a user to data/memory/<user_id>.json.

        # TODO(human): Implement fact persistence
        #
        # This is the write counterpart to load_user_facts. Key considerations:
        #   1. Use the same path convention as load_user_facts
        #   2. Update the last_updated timestamp before saving -- the caller
        #      shouldn't have to remember this
        #   3. Use self._write_json() with fact_store.model_dump()
        #
        # The simplest correct implementation is ~4 lines:
        #   - Update last_updated on the fact_store
        #   - Build the path
        #   - Call self._write_json()
        #
        # Note: there's no locking here because this is a single-user agent.
        # If you were building a multi-user system, you'd need file locking
        # or a proper database.
        """
        raise NotImplementedError("Exercise 1: implement save_user_facts")

    # ------------------------------------------------------------------
    # Conversation Logs (Exercise 1 -- TODO(human))
    # ------------------------------------------------------------------

    def start_new_conversation(self, user_id: str) -> ConversationLog:
        """Create a new conversation log and return it.

        # TODO(human): Create a new ConversationLog with a unique ID
        #
        # Design decisions to make:
        #   1. Conversation ID format: UUID4? Timestamp-based? Something readable?
        #      Consider: the Reflector in 083b will scan these files. A timestamp-
        #      based ID (e.g., "2026-04-03T14-30-00_alice") makes it easy to sort
        #      chronologically and identify the user at a glance.
        #   2. Should you write the empty log file immediately, or wait until the
        #      first turn? Writing immediately ensures the file exists even if the
        #      conversation is abandoned. Waiting avoids empty files.
        #
        # Implementation steps:
        #   - Generate a conversation_id (e.g., using datetime + user_id, or uuid4)
        #   - Create a ConversationLog(conversation_id=..., user_id=user_id)
        #   - Optionally write the empty log to disk
        #   - Return the log
        #
        # Hint: a good convention is f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
        # This gives human-readable, sortable, unique-enough IDs.
        """
        raise NotImplementedError("Exercise 1: implement start_new_conversation")

    def append_conversation_turn(
        self, conversation: ConversationLog, turn: ConversationTurn
    ) -> None:
        """Append a turn to a conversation and persist to disk.

        # TODO(human): Append the turn and save the updated log
        #
        # Implementation steps:
        #   1. Append the turn to conversation.turns
        #   2. Build the log file path: self._logs_dir / f"conversation_{conversation.conversation_id}.json"
        #   3. Use self._write_json() with conversation.model_dump()
        #
        # Design note: this writes the ENTIRE conversation every time a turn is
        # appended. For a single-session conversation (typically <100 turns), this
        # is perfectly fine. For a long-running agent with thousands of turns per
        # session, you'd want append-only writes (JSONL format). But for this
        # practice, full-file writes keep the code simple and the files valid JSON.
        #
        # The write happens synchronously after each turn. This means if the process
        # crashes, you lose at most the current turn -- all previous turns are on disk.
        """
        raise NotImplementedError("Exercise 1: implement append_conversation_turn")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Quick smoke test for FileStore operations."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "test_data"
        store = FileStore(data_dir=tmp_path)

        # Config: should return defaults when no file exists
        config = store.load_config()
        print(f"[OK] Default config loaded: model={config.model}")

        # Save and reload config
        config.model = "test-model"
        store.save_config(config)
        reloaded = store.load_config()
        assert reloaded.model == "test-model", "Config roundtrip failed"
        print("[OK] Config save/load roundtrip")

        # Persona: should raise when no file exists
        try:
            store.load_persona()
            print("[FAIL] Expected FileNotFoundError for missing persona")
        except FileNotFoundError:
            print("[OK] Missing persona raises FileNotFoundError")

        # Save and load persona
        from src.models import PersonaCard
        persona = PersonaCard(name="Test", age=25, backstory="A test persona.")
        store.save_persona(persona)
        loaded = store.load_persona()
        assert loaded.name == "Test", "Persona roundtrip failed"
        print("[OK] Persona save/load roundtrip")

        # User facts (Exercise 1 -- will fail until implemented)
        try:
            facts = store.load_user_facts("test_user")
            print(f"[OK] load_user_facts returned: {len(facts.facts)} facts")

            from src.models import UserFact
            facts.facts.append(UserFact(key="name", value="Alice", source_message="I'm Alice"))
            store.save_user_facts(facts)
            reloaded_facts = store.load_user_facts("test_user")
            assert len(reloaded_facts.facts) == 1, "Facts roundtrip failed"
            print("[OK] User facts save/load roundtrip")
        except NotImplementedError:
            print("[SKIP] load_user_facts not implemented yet (Exercise 1)")

        # Conversation log (Exercise 1 -- will fail until implemented)
        try:
            conv = store.start_new_conversation("test_user")
            print(f"[OK] New conversation: {conv.conversation_id}")

            from src.models import ConversationTurn
            turn = ConversationTurn(role="user", content="Hello!")
            store.append_conversation_turn(conv, turn)
            print(f"[OK] Appended turn, total turns: {len(conv.turns)}")
        except NotImplementedError:
            print("[SKIP] start_new_conversation not implemented yet (Exercise 1)")

    print("\nFileStore self-test complete.")


if __name__ == "__main__":
    _self_test()
