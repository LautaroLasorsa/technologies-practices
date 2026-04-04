"""Entry point -- CLI for ACE loop, baseline, and utilities.

Usage:
    uv run python -m src.main --verify                     # Verify setup
    uv run python -m src.main --iterations 3               # Run 3 ACE iterations
    uv run python -m src.main --iterations 5 --batch-size 4  # 5 iters, 4 convos/batch
    uv run python -m src.main --baseline --iterations 3    # Naive baseline (Exercise 2)
    uv run python -m src.main --plot                       # Plot learning curve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

from src.models import ACEConfig, IterationSnapshot


DATA_DIR = Path(__file__).parent.parent / "data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_setup() -> bool:
    """Check that Ollama is running, model available, and embedding model loadable."""
    print("Verifying setup...\n")

    config = ACEConfig()

    # 1. Data directories
    for d in ("conversations", "reflections", "snapshots"):
        (DATA_DIR / d).mkdir(parents=True, exist_ok=True)
    print("[OK] Data directories exist")

    # 2. Seed playbook
    seed_path = DATA_DIR / "playbook_seed.md"
    if not seed_path.exists():
        print(f"[FAIL] Seed playbook not found at {seed_path}")
        return False
    print("[OK] Seed playbook found")

    # 3. Playbook parsing
    try:
        from src.playbook import load_seed_playbook
        playbook = load_seed_playbook()
        entries = playbook.all_entries()
        print(f"[OK] Playbook parsed: {len(entries)} entries, ~{playbook.token_count()} tokens")
    except NotImplementedError:
        print("[WARN] Playbook not implemented yet (Exercise 1)")
    except Exception as e:
        print(f"[FAIL] Playbook parsing failed: {e}")
        return False

    # 4. Ollama connectivity
    try:
        client = OpenAI(base_url=config.ollama_base_url, api_key="ollama")
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"[OK] Ollama connected. Models: {available}")
    except Exception as e:
        print(f"[FAIL] Cannot connect to Ollama at {config.ollama_base_url}: {e}")
        print("  Start Ollama: docker compose up -d")
        return False

    # 5. Model availability
    if config.model not in available:
        print(f"[FAIL] Model '{config.model}' not found. Pull it:")
        print(f"  docker compose exec ollama ollama pull {config.model}")
        return False
    print(f"[OK] Model '{config.model}' available")

    # 6. Embedding model
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model '{config.embedding_model}'...")
        model = SentenceTransformer(config.embedding_model)
        test_emb = model.encode(["test"])
        print(f"[OK] Embedding model loaded ({test_emb.shape[1]}d)")
    except Exception as e:
        print(f"[FAIL] Embedding model failed: {e}")
        return False

    # 7. Sample conversations
    sample_dir = DATA_DIR / "sample_conversations"
    samples = list(sample_dir.glob("*.json")) if sample_dir.exists() else []
    print(f"[OK] {len(samples)} sample conversations found")

    print("\n--- All checks passed! Ready to run. ---")
    return True


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_from_snapshots() -> None:
    """Load saved snapshots and plot the learning curve."""
    from src.ace_loop import plot_learning_curve

    ace_snapshots = _load_snapshots("ace")
    baseline_snapshots = _load_snapshots("baseline")

    if not ace_snapshots and not baseline_snapshots:
        print("[FAIL] No snapshots found. Run --iterations first.")
        sys.exit(1)

    if ace_snapshots:
        print(f"ACE snapshots: {len(ace_snapshots)} iterations")
    if baseline_snapshots:
        print(f"Baseline snapshots: {len(baseline_snapshots)} iterations")

    output_path = DATA_DIR / "learning_curve.png"
    plot_learning_curve(
        ace_snapshots=ace_snapshots or [],
        baseline_snapshots=baseline_snapshots or None,
        output_path=output_path,
    )


def _load_snapshots(label: str) -> list[IterationSnapshot]:
    """Load all metric snapshots for a given label (ace or baseline)."""
    snapshots: list[IterationSnapshot] = []
    if not SNAPSHOTS_DIR.exists():
        return snapshots
    for path in sorted(SNAPSHOTS_DIR.glob(f"{label}_iter_*_metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        snapshots.append(IterationSnapshot(**data))
    return snapshots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-Evolving Persona Agent: ACE Loop"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify setup (Ollama, model, embeddings) and exit",
    )
    parser.add_argument(
        "--iterations", type=int, default=0,
        help="Number of ACE (or baseline) iterations to run",
    )
    parser.add_argument(
        "--batch-size", type=int, default=3,
        help="Conversations per iteration batch (default: 3)",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Run naive full-rewrite baseline instead of ACE",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Plot learning curve from saved snapshots",
    )
    args = parser.parse_args()

    if args.verify:
        success = verify_setup()
        sys.exit(0 if success else 1)

    if args.plot:
        plot_from_snapshots()
        return

    if args.iterations <= 0:
        parser.print_help()
        return

    config = ACEConfig(batch_size=args.batch_size)

    print("=" * 60)
    print("  Self-Evolving Persona Agent: ACE Loop")
    print("=" * 60)
    print(f"  Mode: {'Baseline (full rewrite)' if args.baseline else 'ACE (delta updates)'}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Model: {config.model}")

    if args.baseline:
        # Run naive baseline (Exercise 2)
        from src.baseline import run_baseline
        snapshots = run_baseline(config, iterations=args.iterations)
    else:
        # Run ACE loop (Exercise 5)
        from src.playbook import load_playbook, load_seed_playbook, save_playbook, PLAYBOOK_PATH

        try:
            try:
                playbook = load_playbook()
                print(f"\n  Loaded working playbook: {len(playbook.all_entries())} entries")
            except FileNotFoundError:
                playbook = load_seed_playbook()
                save_playbook(playbook)
                print(f"\n  Initialized from seed: {len(playbook.all_entries())} entries")
        except NotImplementedError:
            print("\n[FAIL] Playbook not implemented (Exercise 1 required)")
            sys.exit(1)

        from src.ace_loop import ACELoop
        loop = ACELoop(config)

        try:
            snapshots = loop.run(playbook, num_iterations=args.iterations)
        except NotImplementedError as e:
            print(f"\n[FAIL] {e}")
            sys.exit(1)

    # Summary
    if snapshots:
        print("\n" + "=" * 60)
        print("  Results Summary")
        print("=" * 60)
        print(f"  {'Iter':<6} {'Score':<8} {'Entries':<9} {'Tokens':<9} {'Added':<8} {'Deduped':<9} {'Pruned':<8}")
        print(f"  {'-' * 55}")
        for s in snapshots:
            print(
                f"  {s.iteration:<6} {s.avg_score:<8.1f} {s.playbook_entry_count:<9} "
                f"{s.playbook_token_count:<9} {s.entries_added:<8} {s.entries_deduped:<9} "
                f"{s.entries_pruned:<8}"
            )


if __name__ == "__main__":
    main()
