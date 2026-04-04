"""Exercise 4 -- Curator: Merge, Dedup & Prune.

The Curator is ACE's third role and the ONLY component that modifies the playbook.
It performs four deterministic operations:

1. MERGE: Add new delta lessons as playbook entries (helpful=0, harmful=0)
2. DEDUP: Find semantically similar entries using sentence-transformer embeddings,
   merge them (combine counters, keep the more specific wording)
3. UPDATE COUNTERS: Increment helpful/harmful based on conversation quality scores
4. PRUNE: Remove entries where harmful > helpful * prune_ratio

Why the Curator is separate from the Reflector:
  The Reflector extracts insights but never touches the playbook. The Curator
  modifies the playbook but never generates new insights. This separation
  prevents context collapse: no single LLM call ever sees and rewrites the
  full playbook. The Curator's operations are mostly DETERMINISTIC (merge,
  counter update, prune) with one embedding-based step (dedup).

Semantic deduplication:
  As lessons accumulate, the playbook will contain near-duplicates:
    - "Don't use bullet points in conversation"
    - "Avoid enumerated lists when speaking"
  These are semantically identical but lexically different. String matching
  won't catch them. Embedding similarity (cosine > threshold) will.

  Model: all-MiniLM-L6-v2 (22M params, 384-dim embeddings, fast on CPU)
  Threshold: 0.85 cosine similarity (balances aggressive vs conservative dedup)

Counter-based fitness:
  Each entry tracks helpful/harmful counters:
  - helpful++: The entry's advice was followed AND the conversation scored well
  - harmful++: The entry's advice was followed AND the conversation scored poorly
  - Prune rule: remove if harmful > helpful * 2 (deliberately conservative)

  This is a lightweight fitness signal -- no gradient computation, no reward model.
  The counters accumulate over iterations, giving statistically meaningful signal
  after 5-10 conversations.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.models import (
    ACEConfig,
    ConversationScore,
    DeltaLesson,
    PlaybookEntry,
    PlaybookSection,
)
from src.playbook import Playbook


def _load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load the sentence-transformer model for semantic similarity.

    all-MiniLM-L6-v2 produces 384-dimensional embeddings and runs fast on CPU.
    The model is cached after first download (~80MB).
    """
    return SentenceTransformer(model_name)


class Curator:
    """Manages playbook evolution through merge, dedup, counter update, and prune.

    The Curator is the only component that modifies the playbook. All operations
    are deterministic except dedup (which uses embeddings for similarity).
    """

    def __init__(self, config: ACEConfig) -> None:
        self.config = config
        self._embedding_model: SentenceTransformer | None = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model on first use."""
        if self._embedding_model is None:
            self._embedding_model = _load_embedding_model(self.config.embedding_model)
        return self._embedding_model

    def merge_lessons(
        self,
        playbook: Playbook,
        lessons: list[DeltaLesson],
    ) -> int:
        """Add delta lessons as new playbook entries.

        Each DeltaLesson becomes a new entry in its target section with
        helpful=0, harmful=0. Returns the number of entries added.
        """
        added = 0
        for lesson in lessons:
            playbook.add_entry(section=lesson.section, content=lesson.content)
            added += 1
        return added

    def deduplicate(
        self,
        playbook: Playbook,
        threshold: float | None = None,
    ) -> int:
        """Find and merge semantically similar entries within each section.

        # TODO(human): Implement semantic deduplication
        #
        # This is the Curator's most sophisticated operation. It uses sentence-
        # transformer embeddings to detect near-duplicate entries that string
        # matching would miss.
        #
        # What you need to do:
        #
        # 1. For EACH section in the playbook (iterate over PlaybookSection):
        #    a. Get all entries in the section: playbook.query_section(section)
        #    b. If the section has <= 1 entry, skip (nothing to dedup)
        #
        # 2. Compute embeddings for all entries in the section:
        #    texts = [entry.content for entry in entries]
        #    embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        #
        #    This returns a numpy array of shape (N, 384) where N is the number
        #    of entries and 384 is the embedding dimension.
        #
        # 3. Compute pairwise cosine similarity:
        #    For each pair (i, j) where i < j:
        #      similarity = dot(embeddings[i], embeddings[j]) / (norm(i) * norm(j))
        #
        #    Tip: normalize the embeddings first (divide each by its L2 norm),
        #    then cosine similarity = simple dot product.
        #
        #    numpy operations:
        #      norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        #      normalized = embeddings / norms
        #      similarity_matrix = normalized @ normalized.T
        #
        # 4. Find pairs above the threshold (default: self.config.dedup_threshold = 0.85):
        #    For each pair (i, j) where similarity > threshold and i < j:
        #      - These entries are near-duplicates and should be merged
        #
        # 5. Merge near-duplicate pairs:
        #    For each duplicate pair (entry_i, entry_j):
        #    a. Choose which wording to keep:
        #       - Keep the LONGER content (more specific = more useful)
        #       - If same length, keep the one with higher helpful count
        #    b. Combine counters:
        #       - helpful = entry_i.helpful + entry_j.helpful
        #       - harmful = entry_i.harmful + entry_j.harmful
        #    c. Update the kept entry's counters via playbook.update_counters()
        #    d. Remove the other entry via playbook.remove_entry()
        #
        #    IMPORTANT: Process pairs in order and track which entries have already
        #    been removed (a set of removed IDs). Skip pairs where either entry
        #    has already been removed. This prevents chain-merges where A merges
        #    with B, then the merged A tries to merge with C.
        #
        # 6. Return the total number of entries removed (deduplicated)
        #
        # Why 0.85 as the threshold:
        #   - 0.80: Too aggressive -- merges entries that are related but distinct
        #     (e.g., "acknowledge emotion before follow-ups" vs "mirror energy level")
        #   - 0.90: Too conservative -- only catches near-identical phrasings
        #   - 0.85: Sweet spot for catching semantic duplicates while preserving
        #     genuinely different strategies
        #
        #   You may want to experiment with this threshold. Log the actual similarity
        #   scores during development to calibrate.
        #
        # Why per-section dedup (not global):
        #   Entries in different sections serve different purposes. A strategy
        #   ("acknowledge emotion before follow-ups") and a mistake ("don't skip
        #   emotional acknowledgment") are semantically similar but functionally
        #   different -- one says what to DO, the other says what to AVOID.
        #   Deduplicating within sections preserves this distinction.
        #
        # Example:
        #   Before dedup:
        #     [avoid-00001] helpful=4 harmful=0 :: Never use bullet points in conversation
        #     [avoid-00005] helpful=1 harmful=0 :: Avoid enumerated lists when responding
        #
        #   Cosine similarity: 0.89 (above 0.85 threshold)
        #
        #   After dedup (keep longer, combine counters):
        #     [avoid-00001] helpful=5 harmful=0 :: Never use bullet points in conversation
        #     (avoid-00005 removed)
        """
        raise NotImplementedError("Exercise 4: implement Curator.deduplicate")

    def update_counters(
        self,
        playbook: Playbook,
        scores: list[ConversationScore],
    ) -> None:
        """Update playbook entry counters based on conversation scores.

        For each scored conversation:
        - If overall score >= 6.0 (good conversation):
            Increment helpful for every entry the conversation adhered to
        - If overall score < 4.0 (bad conversation):
            Increment harmful for every entry the conversation adhered to
        - Scores between 4.0 and 6.0 are neutral (no counter change)

        The entry_adherence dict in ConversationScore maps entry_id -> bool,
        indicating whether the conversation followed that entry's advice.
        """
        for score in scores:
            for entry_id, adhered in score.entry_adherence.items():
                if not adhered:
                    continue
                if score.overall >= 6.0:
                    playbook.update_counters(entry_id, helpful_delta=1)
                elif score.overall < 4.0:
                    playbook.update_counters(entry_id, harmful_delta=1)

    def prune(self, playbook: Playbook) -> int:
        """Remove entries where harmful > helpful * prune_ratio.

        # TODO(human): Implement counter-based pruning
        #
        # This is the Curator's quality control step. Entries that have been
        # consistently harmful (their advice leads to low-scoring conversations)
        # are removed from the playbook.
        #
        # What you need to do:
        #
        # 1. Get all entries: playbook.all_entries()
        #
        # 2. Identify entries to prune:
        #    An entry should be pruned if:
        #      entry.harmful > entry.helpful * self.config.prune_ratio
        #
        #    With prune_ratio=2.0 (default), this means an entry needs MORE
        #    THAN TWICE as many harmful marks as helpful marks to be pruned.
        #
        #    IMPORTANT: Only prune entries that have been used enough to have
        #    meaningful signal. An entry with helpful=0 harmful=1 technically
        #    satisfies the condition (1 > 0 * 2), but that's just one data point.
        #    Add a minimum usage threshold: only prune if (helpful + harmful) >= 3
        #    This ensures we have at least 3 data points before making a decision.
        #
        # 3. Remove pruned entries: playbook.remove_entry(entry_id)
        #
        # 4. Return the number of entries pruned
        #
        # Why conservative pruning:
        #   ACE's philosophy is "grow, refine, prune" -- in that order. Pruning
        #   should be rare and only catch genuinely harmful entries. The prune_ratio
        #   of 2.0 plus the minimum usage threshold means an entry needs to be
        #   consistently bad across multiple conversations to be removed.
        #
        #   Over-aggressive pruning (e.g., prune anything with harmful > 0) would
        #   remove entries that are sometimes unhelpful but sometimes valuable.
        #   Under-aggressive pruning (never prune) would let the playbook accumulate
        #   noise. The ratio + threshold balance lets genuinely bad advice be
        #   identified and removed while preserving entries with mixed signal.
        #
        # Example:
        #   [emot-00003] helpful=1 harmful=5 :: total=6 >= 3, 5 > 1*2 -> PRUNE
        #   [strat-00002] helpful=0 harmful=1 :: total=1 < 3             -> KEEP (not enough data)
        #   [avoid-00001] helpful=4 harmful=2 :: total=6 >= 3, 2 < 4*2  -> KEEP (mostly helpful)
        """
        raise NotImplementedError("Exercise 4: implement Curator.prune")

    def curate(
        self,
        playbook: Playbook,
        lessons: list[DeltaLesson],
        scores: list[ConversationScore],
    ) -> dict[str, int]:
        """Run the full curation pipeline: merge -> dedup -> update -> prune.

        Returns a dict with operation counts for logging.
        """
        added = self.merge_lessons(playbook, lessons)
        deduped = self.deduplicate(playbook)
        self.update_counters(playbook, scores)
        pruned = self.prune(playbook)
        return {
            "added": added,
            "deduped": deduped,
            "pruned": pruned,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Test Curator operations on the seed playbook."""
    import sys

    print("=" * 60)
    print("  Curator Self-Test (Exercise 4)")
    print("=" * 60)

    config = ACEConfig()

    # Load seed playbook
    from src.playbook import load_seed_playbook
    try:
        playbook = load_seed_playbook()
        print(f"\n[OK] Loaded seed playbook: {len(playbook.all_entries())} entries")
    except NotImplementedError:
        print("[SKIP] Playbook not implemented (Exercise 1 required)")
        sys.exit(0)

    curator = Curator(config)

    # 1. Test merge
    print("\n--- Merge Test ---")
    test_lessons = [
        DeltaLesson(
            section=PlaybookSection.STRATEGIES,
            content="When the user makes a self-deprecating joke, play along briefly before checking if they're actually okay.",
            confidence=0.7,
        ),
        DeltaLesson(
            section=PlaybookSection.MISTAKES,
            content="Don't use semicolons in casual conversation -- they read as overly formal and robotic.",
            confidence=0.6,
        ),
    ]
    added = curator.merge_lessons(playbook, test_lessons)
    print(f"  Added {added} entries. Total: {len(playbook.all_entries())}")

    # 2. Test dedup
    print("\n--- Dedup Test ---")
    # Add a near-duplicate to test dedup
    playbook.add_entry(
        PlaybookSection.MISTAKES,
        "Never format responses as numbered or bulleted lists -- humans don't talk in organized points.",
    )
    print(f"  Before dedup: {len(playbook.all_entries())} entries")

    try:
        deduped = curator.deduplicate(playbook)
        print(f"  Deduplicated: {deduped} entries removed")
        print(f"  After dedup: {len(playbook.all_entries())} entries")
    except NotImplementedError:
        print("  [SKIP] Dedup not implemented yet")

    # 3. Test prune
    print("\n--- Prune Test ---")
    # Create an entry that should be pruned
    bad_entry = playbook.add_entry(
        PlaybookSection.EMOTIONAL,
        "Always match the user's exact emotional intensity -- if they're angry, be angry too.",
    )
    # Simulate bad signal
    playbook.update_counters(bad_entry.entry_id, harmful_delta=5, helpful_delta=1)
    print(f"  Added bad entry [{bad_entry.entry_id}] with harmful=5 helpful=1")
    print(f"  Before prune: {len(playbook.all_entries())} entries")

    try:
        pruned = curator.prune(playbook)
        print(f"  Pruned: {pruned} entries")
        print(f"  After prune: {len(playbook.all_entries())} entries")

        if playbook.get_entry(bad_entry.entry_id) is None:
            print("  [OK] Bad entry was correctly pruned")
        else:
            print("  [FAIL] Bad entry should have been pruned (harmful=5 > helpful=1 * 2)")
    except NotImplementedError:
        print("  [SKIP] Prune not implemented yet")

    # 4. Show final playbook
    print("\n--- Final Playbook ---")
    print(playbook.serialize()[:500])
    print("...")

    print("\n" + "=" * 60)
    print("  Curator self-test complete.")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
