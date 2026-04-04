"""Exercise 5 -- Online Self-Improvement Loop.

Wires Generator + Reflector + Curator into a complete ACE iteration cycle:

    For each iteration:
      1. GENERATE: Run N conversations using the current playbook
      2. SCORE: Evaluate conversations with the human-likeness rubric
      3. REFLECT: Pair best/worst conversations, extract delta lessons
      4. CURATE: Merge lessons, dedup, update counters, prune
      5. SNAPSHOT: Save the playbook state and metrics

After all iterations, the learning curve can be plotted to show whether
ACE's incremental approach actually improves the agent over time.

The key insight: ACE's playbook should grow monotonically in quality
(with occasional dips from bad lessons, corrected by pruning), while the
baseline (Exercise 2) oscillates and eventually collapses.

Architecture:
    The ACELoop class orchestrates the three roles without any of them
    directly communicating. Data flows in one direction:

        Generator -> conversations -> Evaluator -> scores
                                  -> Reflector -> lessons
                                       scores + lessons -> Curator -> updated playbook

    No role has write access to another role's output. The Curator is the
    only component that modifies the playbook.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models import (
    ACEConfig,
    Conversation,
    ConversationScore,
    DeltaLesson,
    IterationSnapshot,
)
from src.playbook import Playbook, save_playbook, PLAYBOOK_PATH


SNAPSHOTS_DIR = Path(__file__).parent.parent / "data" / "snapshots"


def _save_snapshot(
    iteration: int,
    playbook: Playbook,
    snapshot: IterationSnapshot,
    label: str = "ace",
) -> None:
    """Save playbook and metrics snapshot for this iteration."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    # Save playbook text
    pb_path = SNAPSHOTS_DIR / f"{label}_iter_{iteration:03d}.md"
    pb_path.write_text(playbook.serialize(), encoding="utf-8")
    # Save metrics
    metrics_path = SNAPSHOTS_DIR / f"{label}_iter_{iteration:03d}_metrics.json"
    metrics_path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")


def _pair_conversations(
    conversations: list[Conversation],
    scores: list[ConversationScore],
) -> list[tuple[Conversation, Conversation]]:
    """Pair best and worst conversations for contrastive reflection.

    Strategy: sort by score, pair top-half with bottom-half.
    Returns list of (best, worst) pairs for the Reflector.
    """
    scored = sorted(
        zip(conversations, scores),
        key=lambda x: x[1].overall,
        reverse=True,
    )
    pairs: list[tuple[Conversation, Conversation]] = []
    n = len(scored)
    half = n // 2
    for i in range(half):
        best_conv = scored[i][0]
        worst_conv = scored[n - 1 - i][0]
        # Label them for the reflector
        best_conv.label = "natural"
        worst_conv.label = "robotic"
        pairs.append((best_conv, worst_conv))
    return pairs


class ACELoop:
    """Orchestrates the Generator + Reflector + Curator cycle.

    Each iteration:
    1. Generate conversations with the current playbook
    2. Score conversations with the evaluator
    3. Pair best/worst conversations for contrastive reflection
    4. Extract delta lessons with the Reflector
    5. Curate: merge + dedup + update counters + prune
    6. Snapshot the playbook and metrics
    """

    def __init__(self, config: ACEConfig) -> None:
        self.config = config
        # Lazy imports to avoid circular dependencies and allow Exercise isolation
        from src.curator import Curator
        from src.reflector import Reflector

        self.reflector = Reflector(config)
        self.curator = Curator(config)

    def run_iteration(
        self,
        playbook: Playbook,
        iteration: int,
    ) -> IterationSnapshot:
        """Execute one full ACE cycle.

        # TODO(human): Wire Generator + Reflector + Curator into one iteration
        #
        # This is where the three ACE roles come together. Each iteration is a
        # complete learning cycle: experience -> reflect -> improve.
        #
        # What you need to do:
        #
        # 1. GENERATE: Run a batch of conversations using the current playbook
        #    from src.generator import generate_batch, save_conversations
        #    conversations = generate_batch(self.config, playbook, self.config.batch_size)
        #    save_conversations(conversations, iteration)
        #
        #    The Generator uses the playbook as behavioral guidance in the system
        #    prompt. It produces conversations with emotion and inner_thought
        #    metadata for each agent turn.
        #
        # 2. SCORE: Evaluate each conversation with the human-likeness rubric
        #    from src.evaluator import score_batch
        #    scores = score_batch(conversations, playbook, self.config)
        #
        #    The Evaluator produces ConversationScore objects with:
        #    - Per-dimension scores (imperfection, emotional_continuity, etc.)
        #    - Overall score (weighted average)
        #    - Entry adherence map (which playbook entries were followed)
        #
        #    Compute avg_score = sum(s.overall for s in scores) / len(scores)
        #    Print it: "  Avg score: {avg_score:.1f}/10"
        #
        # 3. REFLECT: Extract delta lessons from conversation pairs
        #    pairs = _pair_conversations(conversations, scores)
        #
        #    For each pair (best, worst):
        #      pair_lessons = self.reflector.extract_lessons(best, worst)
        #      Accumulate into all_lessons list
        #      from src.reflector import save_reflections
        #      save_reflections(pair_lessons, iteration, pair_index)
        #
        #    Print: "  Extracted {len(all_lessons)} lessons from {len(pairs)} pairs"
        #
        #    If there are no pairs (e.g., batch_size < 2), skip reflection and
        #    use an empty lessons list.
        #
        # 4. CURATE: Merge lessons, dedup, update counters, prune
        #    stats = self.curator.curate(playbook, all_lessons, scores)
        #    Print: "  Curated: +{added} entries, -{deduped} deduped, -{pruned} pruned"
        #
        # 5. SAVE: Persist the updated playbook
        #    save_playbook(playbook)
        #
        # 6. SNAPSHOT: Create and return the iteration snapshot
        #    snapshot = IterationSnapshot(
        #        iteration=iteration,
        #        playbook_token_count=playbook.token_count(),
        #        playbook_entry_count=len(playbook.all_entries()),
        #        avg_score=avg_score,
        #        scores=scores,
        #        lessons_extracted=len(all_lessons),
        #        entries_added=stats["added"],
        #        entries_pruned=stats["pruned"],
        #        entries_deduped=stats["deduped"],
        #    )
        #    _save_snapshot(iteration, playbook, snapshot)
        #    return snapshot
        #
        # The flow ensures that:
        #   - The Generator never modifies the playbook (read-only context)
        #   - The Reflector never sees the playbook (only conversations)
        #   - The Curator is the ONLY component that touches the playbook
        #   - Data flows in one direction: generate -> score -> reflect -> curate
        #
        # Error handling:
        #   If the Reflector fails (model produces invalid output), catch the
        #   exception and continue with an empty lessons list. The iteration
        #   still produces useful counter updates from scores.
        #
        #   If the Generator fails on a conversation, skip that conversation
        #   and continue with the rest of the batch.
        #
        # Why this architecture prevents context collapse:
        #   No single step sees the full playbook AND has write access to it.
        #   The Generator sees the playbook but can't modify it.
        #   The Reflector produces delta lessons but doesn't see the playbook.
        #   The Curator modifies the playbook but only through small operations
        #   (add entry, update counter, remove entry) -- never a full rewrite.
        #
        #   Compare with Exercise 2's baseline where a single LLM call sees
        #   the full playbook + conversations and produces a complete replacement.
        """
        raise NotImplementedError("Exercise 5: implement ACELoop.run_iteration")

    def run(
        self,
        playbook: Playbook,
        num_iterations: int,
    ) -> list[IterationSnapshot]:
        """Run multiple ACE iterations and collect metrics.

        # TODO(human): Implement the multi-iteration loop
        #
        # What you need to do:
        #
        # 1. Save the initial playbook state (iteration 0 snapshot):
        #    initial_snapshot = IterationSnapshot(
        #        iteration=0,
        #        playbook_token_count=playbook.token_count(),
        #        playbook_entry_count=len(playbook.all_entries()),
        #        avg_score=0.0,
        #    )
        #    _save_snapshot(0, playbook, initial_snapshot)
        #    snapshots = [initial_snapshot]
        #
        # 2. For each iteration 1..num_iterations:
        #    print(f"\\n{'='*60}")
        #    print(f"  ACE Iteration {i}/{num_iterations}")
        #    print(f"{'='*60}")
        #    snapshot = self.run_iteration(playbook, iteration=i)
        #    snapshots.append(snapshot)
        #    print(f"  Playbook: {snapshot.playbook_entry_count} entries, "
        #          f"~{snapshot.playbook_token_count} tokens")
        #
        # 3. Return snapshots
        #
        # This is straightforward wiring -- the complexity is in run_iteration.
        # The multi-iteration loop just repeats the cycle and collects metrics.
        #
        # After running, the snapshots can be used to plot the learning curve
        # (see plot_learning_curve in main.py).
        """
        raise NotImplementedError("Exercise 5: implement ACELoop.run")


# ---------------------------------------------------------------------------
# Learning curve plotting
# ---------------------------------------------------------------------------

def plot_learning_curve(
    ace_snapshots: list[IterationSnapshot],
    baseline_snapshots: list[IterationSnapshot] | None = None,
    output_path: Path | None = None,
) -> None:
    """Plot the learning curve: score and playbook size over iterations.

    Produces a two-panel figure:
    - Top: Average human-likeness score per iteration
    - Bottom: Playbook token count per iteration

    If baseline_snapshots are provided, both are plotted for comparison.
    """
    import matplotlib.pyplot as plt

    fig, (ax_score, ax_tokens) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ACE data
    ace_iters = [s.iteration for s in ace_snapshots]
    ace_scores = [s.avg_score for s in ace_snapshots]
    ace_tokens = [s.playbook_token_count for s in ace_snapshots]

    ax_score.plot(ace_iters, ace_scores, "b-o", label="ACE", linewidth=2)
    ax_tokens.plot(ace_iters, ace_tokens, "b-o", label="ACE", linewidth=2)

    # Baseline data
    if baseline_snapshots:
        bl_iters = [s.iteration for s in baseline_snapshots]
        bl_scores = [s.avg_score for s in baseline_snapshots]
        bl_tokens = [s.playbook_token_count for s in baseline_snapshots]

        ax_score.plot(bl_iters, bl_scores, "r--s", label="Baseline (full rewrite)", linewidth=2)
        ax_tokens.plot(bl_iters, bl_tokens, "r--s", label="Baseline (full rewrite)", linewidth=2)

    # Labels and formatting
    ax_score.set_ylabel("Avg Human-Likeness Score (0-10)")
    ax_score.set_title("ACE Self-Improvement: Learning Curve")
    ax_score.legend()
    ax_score.grid(True, alpha=0.3)
    ax_score.set_ylim(0, 10)

    ax_tokens.set_xlabel("Iteration")
    ax_tokens.set_ylabel("Playbook Token Count")
    ax_tokens.legend()
    ax_tokens.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Learning curve saved to {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Run a single ACE iteration for testing."""
    import sys

    print("=" * 60)
    print("  ACE Loop Self-Test (Exercise 5)")
    print("=" * 60)

    config = ACEConfig()

    # Load or create playbook
    from src.playbook import load_seed_playbook, load_playbook, PLAYBOOK_PATH
    try:
        try:
            playbook = load_playbook()
            print(f"[OK] Loaded working playbook: {len(playbook.all_entries())} entries")
        except FileNotFoundError:
            playbook = load_seed_playbook()
            save_playbook(playbook)
            print(f"[OK] Initialized from seed: {len(playbook.all_entries())} entries")
    except NotImplementedError:
        print("[SKIP] Playbook not implemented (Exercise 1 required)")
        sys.exit(0)

    # Run one iteration
    loop = ACELoop(config)
    try:
        snapshot = loop.run_iteration(playbook, iteration=1)
        print(f"\n[OK] Iteration complete:")
        print(f"  Avg score:        {snapshot.avg_score:.1f}/10")
        print(f"  Lessons extracted: {snapshot.lessons_extracted}")
        print(f"  Entries added:     {snapshot.entries_added}")
        print(f"  Entries deduped:   {snapshot.entries_deduped}")
        print(f"  Entries pruned:    {snapshot.entries_pruned}")
        print(f"  Final playbook:    {snapshot.playbook_entry_count} entries, "
              f"~{snapshot.playbook_token_count} tokens")
    except NotImplementedError as e:
        print(f"[SKIP] {e}")

    print("\n" + "=" * 60)
    print("  ACE Loop self-test complete.")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
