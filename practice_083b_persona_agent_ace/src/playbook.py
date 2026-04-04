"""Exercise 1 -- Playbook Data Structure & Format.

The Playbook is ACE's central knowledge store: a structured markdown document
organized into sections, where each section contains uniquely-identified entries
with helpful/harmful counters.

Format:
    # PERSONA PLAYBOOK

    ## STRATEGIES & INSIGHTS
    [strat-00001] helpful=3 harmful=0 :: When the user shares personal news...
    [strat-00002] helpful=1 harmful=1 :: Mirror the user's energy level...

    ## EMOTIONAL PATTERNS
    [emot-00001] helpful=2 harmful=0 :: Transitioning from playful to serious...

    ## MISTAKES TO AVOID
    [avoid-00001] helpful=4 harmful=0 :: Never enumerate with bullet points...

Each entry line has the format:
    [<id>] helpful=<N> harmful=<M> :: <content text>

Where:
    - <id> is a unique identifier with a section prefix:
        - strat-NNNNN for STRATEGIES & INSIGHTS
        - emot-NNNNN for EMOTIONAL PATTERNS
        - avoid-NNNNN for MISTAKES TO AVOID
    - helpful/harmful are non-negative integer counters
    - <content text> is the natural-language strategy/pattern/mistake

Why markdown and not JSON:
    1. Human-readable: open in any editor, immediately understand what the agent knows
    2. Diffable: git diff shows exactly what changed between ACE iterations
    3. LLM-friendly: markdown is a natural format for model consumption/generation
    4. Editable: a human can manually add/remove entries without parsing code

Key properties:
    - Parse/serialize round-trip must be LOSSLESS (parse -> serialize = original text)
    - Entries are ordered by ID within each section
    - IDs auto-increment within their prefix (strat-00001, strat-00002, ...)
"""

from __future__ import annotations

import re
from pathlib import Path

from src.models import ACEConfig, PlaybookEntry, PlaybookSection

# Section prefix mapping for auto-ID generation
SECTION_PREFIXES: dict[PlaybookSection, str] = {
    PlaybookSection.STRATEGIES: "strat",
    PlaybookSection.EMOTIONAL: "emot",
    PlaybookSection.MISTAKES: "avoid",
}

# Regex for parsing a single entry line
# Matches: [strat-00001] helpful=3 harmful=0 :: content text here
ENTRY_PATTERN = re.compile(
    r"^\[(?P<id>[a-z]+-\d+)\]\s+"
    r"helpful=(?P<helpful>\d+)\s+"
    r"harmful=(?P<harmful>\d+)\s+"
    r"::\s+"
    r"(?P<content>.+)$"
)

# Default paths
DATA_DIR = Path(__file__).parent.parent / "data"
SEED_PATH = DATA_DIR / "playbook_seed.md"
PLAYBOOK_PATH = DATA_DIR / "playbook.md"


class Playbook:
    """Structured persona playbook with parse/serialize/query operations.

    The playbook stores entries grouped by section. Each entry has a unique ID,
    helpful/harmful counters, and content text. The class supports:
    - Parsing from markdown text
    - Serializing back to markdown (lossless round-trip)
    - Adding new entries with auto-generated IDs
    - Updating counters on existing entries
    - Querying entries by section
    - Counting tokens (approximate, for tracking playbook size)
    """

    # TODO(human): Implement the Playbook class
    #
    # This is ACE's central data structure. The playbook is a structured markdown
    # document that the agent uses as its "strategy manual." Every lesson learned
    # from conversations lives here as a uniquely-identified entry.
    #
    # You need to implement these methods:
    #
    # 1. __init__(self, entries: list[PlaybookEntry] | None = None)
    #    - Store entries internally, indexed for fast lookup by ID and section
    #    - Consider: what data structures give you O(1) lookup by ID AND
    #      efficient iteration by section? (hint: dict + grouping)
    #
    # 2. parse(cls, text: str) -> "Playbook"  (classmethod)
    #    - Parse the markdown playbook format into a Playbook instance
    #    - Strategy:
    #      a. Split the text into lines
    #      b. Track which section you're currently in by looking for "## SECTION NAME" headers
    #      c. For each non-empty, non-header line, try to match ENTRY_PATTERN
    #      d. For each match, create a PlaybookEntry with the extracted fields
    #      e. Map the entry ID prefix to the correct PlaybookSection:
    #         - "strat-" -> PlaybookSection.STRATEGIES
    #         - "emot-"  -> PlaybookSection.EMOTIONAL
    #         - "avoid-" -> PlaybookSection.MISTAKES
    #    - IMPORTANT: The round-trip property must hold:
    #         Playbook.parse(text).serialize() == text  (for well-formed input)
    #      This means you must preserve: section order, entry order within sections,
    #      exact spacing, and the "# PERSONA PLAYBOOK" header.
    #
    # 3. serialize(self) -> str
    #    - Render the playbook back to markdown format
    #    - Format:
    #         # PERSONA PLAYBOOK
    #         <blank line>
    #         ## STRATEGIES & INSIGHTS
    #         [strat-00001] helpful=3 harmful=0 :: Content text here
    #         [strat-00002] helpful=1 harmful=1 :: Another entry
    #         <blank line>
    #         ## EMOTIONAL PATTERNS
    #         ...
    #    - Section order is always: STRATEGIES, EMOTIONAL, MISTAKES
    #    - Entries within a section are sorted by ID (lexicographic on the ID string)
    #    - Each section separated by a blank line
    #    - Trailing newline at end of file
    #
    # 4. add_entry(self, section: PlaybookSection, content: str) -> PlaybookEntry
    #    - Create a new entry with an auto-generated ID
    #    - ID format: "<prefix>-<NNNNN>" where NNNNN is zero-padded to 5 digits
    #    - The number should be max(existing numbers in this section) + 1
    #    - New entries start with helpful=0, harmful=0
    #    - Return the created entry
    #    - Example: if section has [strat-00001] and [strat-00003], next ID is strat-00004
    #
    # 5. update_counters(self, entry_id: str, helpful_delta: int = 0, harmful_delta: int = 0) -> bool
    #    - Find the entry by ID and increment its counters
    #    - Return True if the entry was found, False otherwise
    #    - Counters should never go below 0 (clamp with max(0, ...))
    #
    # 6. query_section(self, section: PlaybookSection) -> list[PlaybookEntry]
    #    - Return all entries in the given section, sorted by ID
    #
    # 7. get_entry(self, entry_id: str) -> PlaybookEntry | None
    #    - Return the entry with the given ID, or None if not found
    #
    # 8. remove_entry(self, entry_id: str) -> bool
    #    - Remove the entry with the given ID
    #    - Return True if removed, False if not found
    #
    # 9. all_entries(self) -> list[PlaybookEntry]
    #    - Return all entries across all sections, sorted by ID
    #
    # 10. token_count(self) -> int
    #     - Approximate token count of the serialized playbook
    #     - Simple heuristic: len(self.serialize().split()) * 1.3
    #       (words * 1.3 approximates BPE tokens reasonably)
    #
    # Design note -- why text parsing instead of JSON:
    #   The playbook must be human-editable and git-diffable. A developer should
    #   be able to open data/playbook.md, read the strategies, and manually add
    #   or remove entries. JSON would require escaping, quoting, and careful
    #   bracket matching. Markdown with a simple line format gives us structured
    #   data with zero friction for humans.
    #
    #   The tradeoff: parsing is more fragile than JSON deserialization. Your
    #   parser must handle edge cases (blank lines, missing sections, entries
    #   with "::" in the content text). The ENTRY_PATTERN regex handles most
    #   of this, but consider: what happens if someone adds a comment line?
    #   Your parser should skip lines that don't match the entry pattern rather
    #   than raising an error.

    def __init__(self, entries: list[PlaybookEntry] | None = None) -> None:
        raise NotImplementedError("Exercise 1: implement Playbook.__init__")

    @classmethod
    def parse(cls, text: str) -> "Playbook":
        raise NotImplementedError("Exercise 1: implement Playbook.parse")

    def serialize(self) -> str:
        raise NotImplementedError("Exercise 1: implement Playbook.serialize")

    def add_entry(self, section: PlaybookSection, content: str) -> PlaybookEntry:
        raise NotImplementedError("Exercise 1: implement Playbook.add_entry")

    def update_counters(self, entry_id: str, helpful_delta: int = 0, harmful_delta: int = 0) -> bool:
        raise NotImplementedError("Exercise 1: implement Playbook.update_counters")

    def query_section(self, section: PlaybookSection) -> list[PlaybookEntry]:
        raise NotImplementedError("Exercise 1: implement Playbook.query_section")

    def get_entry(self, entry_id: str) -> PlaybookEntry | None:
        raise NotImplementedError("Exercise 1: implement Playbook.get_entry")

    def remove_entry(self, entry_id: str) -> bool:
        raise NotImplementedError("Exercise 1: implement Playbook.remove_entry")

    def all_entries(self) -> list[PlaybookEntry]:
        raise NotImplementedError("Exercise 1: implement Playbook.all_entries")

    def token_count(self) -> int:
        raise NotImplementedError("Exercise 1: implement Playbook.token_count")


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def load_playbook(path: Path = PLAYBOOK_PATH) -> Playbook:
    """Load a playbook from a markdown file."""
    if not path.exists():
        raise FileNotFoundError(f"Playbook not found at {path}")
    return Playbook.parse(path.read_text(encoding="utf-8"))


def save_playbook(playbook: Playbook, path: Path = PLAYBOOK_PATH) -> None:
    """Save a playbook to a markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(playbook.serialize(), encoding="utf-8")


def load_seed_playbook() -> Playbook:
    """Load the seed playbook from data/playbook_seed.md."""
    return load_playbook(SEED_PATH)


def reset_to_seed() -> Playbook:
    """Reset the working playbook to the seed state."""
    playbook = load_seed_playbook()
    save_playbook(playbook)
    return playbook


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Test playbook parse/serialize round-trip and operations."""
    import sys

    print("=" * 60)
    print("  Playbook Self-Test (Exercise 1)")
    print("=" * 60)

    # 1. Load seed playbook
    print("\n1. Loading seed playbook...")
    try:
        playbook = load_seed_playbook()
        entries = playbook.all_entries()
        print(f"   [OK] Loaded {len(entries)} entries")
    except NotImplementedError:
        print("   [SKIP] Playbook not implemented yet")
        sys.exit(0)
    except FileNotFoundError:
        print(f"   [FAIL] Seed playbook not found at {SEED_PATH}")
        sys.exit(1)

    # 2. Display entries by section
    print("\n2. Entries by section:")
    for section in PlaybookSection:
        section_entries = playbook.query_section(section)
        print(f"   {section.value}: {len(section_entries)} entries")
        for entry in section_entries:
            print(f"     [{entry.entry_id}] helpful={entry.helpful} harmful={entry.harmful}")
            print(f"       {entry.content[:80]}...")

    # 3. Round-trip test
    print("\n3. Round-trip test (parse -> serialize -> parse)...")
    original_text = SEED_PATH.read_text(encoding="utf-8")
    serialized = playbook.serialize()
    re_parsed = Playbook.parse(serialized)
    re_serialized = re_parsed.serialize()

    if serialized == re_serialized:
        print("   [OK] Round-trip is lossless")
    else:
        print("   [FAIL] Round-trip changed the output!")
        print(f"   First serialize:  {len(serialized)} chars")
        print(f"   Second serialize: {len(re_serialized)} chars")

    # 4. Add entry test
    print("\n4. Adding a new entry...")
    new_entry = playbook.add_entry(
        PlaybookSection.STRATEGIES,
        "Test entry: when debugging, explain the hypothesis before showing the fix.",
    )
    print(f"   [OK] Added: [{new_entry.entry_id}] {new_entry.content[:60]}...")

    # 5. Update counters test
    print("\n5. Updating counters...")
    success = playbook.update_counters(new_entry.entry_id, helpful_delta=3, harmful_delta=1)
    updated = playbook.get_entry(new_entry.entry_id)
    if success and updated and updated.helpful == 3 and updated.harmful == 1:
        print(f"   [OK] Counters updated: helpful={updated.helpful} harmful={updated.harmful}")
    else:
        print("   [FAIL] Counter update didn't work correctly")

    # 6. Remove entry test
    print("\n6. Removing the test entry...")
    removed = playbook.remove_entry(new_entry.entry_id)
    still_there = playbook.get_entry(new_entry.entry_id)
    if removed and still_there is None:
        print("   [OK] Entry removed")
    else:
        print("   [FAIL] Entry removal didn't work")

    # 7. Token count
    print(f"\n7. Approximate token count: {playbook.token_count()}")

    print("\n" + "=" * 60)
    print("  Playbook self-test complete.")
    print("=" * 60)

    # 8. If --seed flag, reset working playbook to seed
    if "--seed" in sys.argv:
        print("\nResetting working playbook to seed...")
        reset_to_seed()
        print(f"[OK] Playbook reset to {SEED_PATH}")


if __name__ == "__main__":
    _self_test()
