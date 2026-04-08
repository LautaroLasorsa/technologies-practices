"""Exercise 2 -- Witness Context Collapse.

Implements a naive "full rewrite" context updater: after each batch of
conversations, asks the model to rewrite the ENTIRE playbook from scratch.

This is the failure mode that ACE solves. The purpose of this exercise is to
experience context collapse firsthand:

1. Start with the seed playbook (~8 entries, ~500 tokens of specific strategies)
2. Run 3 iterations of: generate conversations -> rewrite entire playbook
3. Watch the playbook shrink from rich, specific strategies to generic platitudes
4. Measure: token count, entry count, and qualitative specificity

The ACE paper documents this exact failure at scale: at iteration 60 on AppWorld,
a monolithic rewrite collapsed 18,282 tokens (66.7% accuracy) to 122 tokens
(57.1% -- below the 63.7% no-adaptation baseline). You'll see a miniature version.

Why this happens:
  - LLMs are biased toward "clean," concise output when asked to "improve" text
  - Rewriting the full context gives the model permission to restructure everything
  - Specific, hard-won strategies (e.g., "insert a neutral beat between emotional
    registers") get generalized to platitudes (e.g., "transition smoothly")
  - Each iteration compounds the loss -- once specificity is gone, it can't be recovered

Architecture:
  - NaiveRewriter takes the current playbook text + conversation data
  - Prompts the LLM to "improve and rewrite" the playbook
  - Parses the output back into a Playbook (best-effort)
  - Repeats for N iterations, tracking metrics at each step
"""

from __future__ import annotations

import json
from pathlib import Path

import instructor

from src.llm_config import get_openai_client
from src.models import ACEConfig, Conversation, IterationSnapshot
from src.playbook import Playbook, load_seed_playbook, save_playbook, PLAYBOOK_PATH


SNAPSHOTS_DIR = Path(__file__).parent.parent / "data" / "snapshots"


def _create_client(config: ACEConfig) -> OpenAI:
    """Create a plain OpenAI client for the configured LLM provider."""
    return get_openai_client()


def _format_conversations_for_prompt(conversations: list[Conversation]) -> str:
    """Format conversation logs into a text block for the rewrite prompt."""
    parts: list[str] = []
    for conv in conversations:
        lines = [f"--- Conversation: {conv.conversation_id} (score: {conv.human_likeness_score}) ---"]
        for turn in conv.turns:
            prefix = "User" if turn.role == "user" else "Agent"
            lines.append(f"{prefix}: {turn.content}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


class NaiveRewriter:
    """Naive full-rewrite playbook updater.

    This class demonstrates the WRONG way to update a playbook: by asking the
    LLM to rewrite the entire thing from scratch after each batch of conversations.
    """

    def __init__(self, config: ACEConfig) -> None:
        self.config = config
        self.client = _create_client(config)

    def rewrite(self, current_playbook: Playbook, conversations: list[Conversation]) -> Playbook:
        """Ask the LLM to rewrite the entire playbook given new conversations.

        # TODO(human): Implement the naive full-rewrite updater
        #
        # This is deliberately the WRONG approach -- you're implementing it to
        # experience the failure mode that motivates ACE's design.
        #
        # What you need to do:
        #
        # 1. Build a prompt that includes:
        #    a. The current playbook text (current_playbook.serialize())
        #    b. The conversation data (use _format_conversations_for_prompt())
        #    c. Instructions to "analyze the conversations and rewrite the playbook
        #       to improve the persona agent's human-likeness"
        #
        #    The prompt should tell the model to:
        #    - Keep the same markdown format (# PERSONA PLAYBOOK, ## sections, [id] entries)
        #    - Add new strategies learned from conversations
        #    - Remove or update strategies that didn't work
        #    - "Improve clarity and conciseness" (this is the trap! This instruction
        #      triggers brevity bias -- the model will aggressively shorten entries)
        #
        # 2. Call self.client.chat.completions.create() with:
        #    - model=self.config.model
        #    - messages=[{"role": "user", "content": your_prompt}]
        #    - temperature=0.7
        #
        # 3. Extract the response text from the completion
        #    response = completion.choices[0].message.content
        #
        # 4. Parse the response back into a Playbook:
        #    - Try Playbook.parse(response)
        #    - If parsing fails (model output doesn't match format), return the
        #      current playbook unchanged and print a warning
        #
        # 5. Return the new playbook
        #
        # Key insight for the exercise:
        #   After implementing this, run it for 3 iterations (via src.baseline self-test
        #   or `uv run python -m src.main --baseline --iterations 3`).
        #   Watch what happens:
        #   - Iteration 1: Playbook may gain a few entries but starts generalizing
        #   - Iteration 2: Specific strategies collapse into shorter versions
        #   - Iteration 3: The playbook is noticeably shorter and more generic
        #
        #   Track these metrics at each iteration:
        #   - Token count (playbook.token_count())
        #   - Entry count (len(playbook.all_entries()))
        #   - Read the actual entries -- do they still have personality?
        #
        # Why the prompt causes collapse:
        #   The instruction to "improve clarity and conciseness" activates the
        #   model's instruction-following training. LLMs are trained to prefer
        #   shorter, cleaner outputs. When given permission to rewrite everything,
        #   they reliably strip out specifics:
        #     "When user shares personal news, acknowledge emotion before asking
        #      follow-up" becomes "Be empathetic in responses"
        #     "Never enumerate with bullet points -- humans don't speak in lists"
        #      becomes "Use natural language"
        #
        #   This is brevity bias + context collapse in action.
        """
        raise NotImplementedError("Exercise 2: implement NaiveRewriter.rewrite")


def _save_snapshot(iteration: int, playbook: Playbook, label: str = "baseline") -> None:
    """Save a playbook snapshot for later comparison."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = SNAPSHOTS_DIR / f"{label}_iter_{iteration:03d}.md"
    snapshot_path.write_text(playbook.serialize(), encoding="utf-8")


def run_baseline(config: ACEConfig, iterations: int = 3) -> list[IterationSnapshot]:
    """Run the naive baseline for N iterations and track metrics.

    Uses sample conversations from data/sample_conversations/ as input.
    """
    sample_dir = Path(__file__).parent.parent / "data" / "sample_conversations"
    conversations = _load_sample_conversations(sample_dir)

    rewriter = NaiveRewriter(config)
    playbook = load_seed_playbook()
    snapshots: list[IterationSnapshot] = []

    print(f"\nBaseline: starting with {len(playbook.all_entries())} entries, "
          f"~{playbook.token_count()} tokens\n")

    _save_snapshot(0, playbook, label="baseline")

    for i in range(1, iterations + 1):
        print(f"--- Baseline Iteration {i}/{iterations} ---")

        try:
            playbook = rewriter.rewrite(playbook, conversations)
        except NotImplementedError:
            print("[SKIP] NaiveRewriter.rewrite not implemented yet")
            return snapshots

        entry_count = len(playbook.all_entries())
        token_count = playbook.token_count()

        snapshot = IterationSnapshot(
            iteration=i,
            playbook_token_count=token_count,
            playbook_entry_count=entry_count,
            avg_score=0.0,
        )
        snapshots.append(snapshot)
        _save_snapshot(i, playbook, label="baseline")

        print(f"  Entries: {entry_count}, Tokens: ~{token_count}")
        print(f"  Playbook preview:")
        for line in playbook.serialize().split("\n")[:10]:
            print(f"    {line}")
        print()

    save_playbook(playbook)
    return snapshots


def _load_sample_conversations(sample_dir: Path) -> list[Conversation]:
    """Load sample conversations from JSON files."""
    conversations: list[Conversation] = []
    if not sample_dir.exists():
        return conversations
    for path in sorted(sample_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        conversations.append(Conversation(**data))
    return conversations


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Run the naive baseline and show context collapse."""
    import sys

    print("=" * 60)
    print("  Naive Baseline Self-Test (Exercise 2)")
    print("  Demonstrates context collapse from monolithic rewrites")
    print("=" * 60)

    config = ACEConfig()

    # Verify LLM provider connectivity
    try:
        client = _create_client(config)
        models = client.models.list()
        available = [m.id for m in models.data]
        if config.model not in available:
            print(f"[FAIL] Model '{config.model}' not found.")
            print(f"  If using Ollama: docker compose exec ollama ollama pull {config.model}")
            sys.exit(1)
    except Exception as e:
        print(f"[FAIL] Cannot connect to LLM provider: {e}")
        print("  If using Ollama: docker compose up -d")
        sys.exit(1)

    iterations = 3
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        iterations = int(sys.argv[1])

    snapshots = run_baseline(config, iterations=iterations)

    if snapshots:
        print("\n" + "=" * 60)
        print("  Baseline Results Summary")
        print("=" * 60)
        print(f"  {'Iteration':<12} {'Entries':<10} {'Tokens':<10}")
        print(f"  {'-' * 32}")
        for snap in snapshots:
            print(f"  {snap.iteration:<12} {snap.playbook_entry_count:<10} {snap.playbook_token_count:<10}")

        if len(snapshots) >= 2:
            first, last = snapshots[0], snapshots[-1]
            token_change = last.playbook_token_count - first.playbook_token_count
            pct = (token_change / max(first.playbook_token_count, 1)) * 100
            print(f"\n  Token change: {token_change:+d} ({pct:+.1f}%)")
            if token_change < 0:
                print("  --> Context is collapsing! This is exactly the problem ACE solves.")
            else:
                print("  --> Context grew (unusual for naive rewrite -- check if entries are generic)")


if __name__ == "__main__":
    _self_test()
