"""Benchmark: LMDB reads vs Python dict lookups."""

import json
import random
import time
from pathlib import Path

import lmdb
import numpy as np

EMBEDDING_DIM = 384
DB_DIR = Path(__file__).parent.parent / "shared_db"
JSON_FILE = Path(__file__).parent / "embeddings.json"


def prepare_json_baseline(env: lmdb.Environment) -> dict[str, list[float]]:
    """Export LMDB contents to a JSON file, then load it back as a dict.

    This simulates the common alternative: "just load everything into memory."
    """
    emb_db = env.open_db(b"embeddings")
    data = {}
    with env.begin(db=emb_db) as txn:
        cursor = txn.cursor()
        for key, value in cursor.iternext():
            word = key.decode()
            vec = np.frombuffer(value, dtype=np.float32).tolist()
            data[word] = vec

    with open(JSON_FILE, "w") as f:
        json.dump(data, f)

    # Simulate "cold load from disk"
    with open(JSON_FILE) as f:
        return json.load(f)


def benchmark_lmdb_reads(
    env: lmdb.Environment, words: list[str], n_reads: int
) -> float:
    """TODO(human): Benchmark random LMDB reads.

    Steps:
    1. Open the 'embeddings' named database
    2. Pick n_reads random words from the words list (use random.choices)
    3. Start timing (time.perf_counter)
    4. For each word, open a read txn, get the value, deserialize with np.frombuffer
    5. Stop timing
    6. Return total elapsed seconds

    Design question: Should you open one transaction for all reads,
    or one transaction per read? Think about what's more realistic
    for a production cache (hint: each request = separate txn).

    Then try both approaches and compare!
    """
    # TODO(human): Implement the LMDB read benchmark here.
    #
    # 1. Use random.choices(words, k=n_reads) to pick random words to look up.
    # 2. Open the 'embeddings' named database.
    # 3. Start a timer with time.perf_counter().
    # 4. For each word, open a read txn, call txn.get(word.encode()),
    #    and deserialize with np.frombuffer(raw, dtype=np.float32).
    #    Using one txn per read simulates real production access patterns.
    # 5. Return elapsed time in seconds.
    pass


def benchmark_dict_reads(
    data: dict[str, list[float]], words: list[str], n_reads: int
) -> float:
    """Benchmark random dict lookups for comparison."""
    sample = random.choices(words, k=n_reads)
    start = time.perf_counter()
    for word in sample:
        _ = data[word]
    return time.perf_counter() - start


def main() -> None:
    env = lmdb.open(str(DB_DIR), max_dbs=2, readonly=True)
    try:
        # Get word list from LMDB
        emb_db = env.open_db(b"embeddings")
        with env.begin(db=emb_db) as txn:
            words = [key.decode() for key, _ in txn.cursor().iternext()]
        print(f"Loaded {len(words)} words from LMDB")

        # Prepare dict baseline
        print("Preparing JSON baseline...")
        dict_data = prepare_json_baseline(env)
        print(f"Dict loaded: {len(dict_data)} entries\n")

        n_reads = 10_000

        # Benchmark LMDB
        lmdb_time = benchmark_lmdb_reads(env, words, n_reads)
        if lmdb_time is None:
            print("benchmark_lmdb_reads() not implemented yet — look for TODO(human)")
            return

        # Benchmark dict
        dict_time = benchmark_dict_reads(dict_data, words, n_reads)

        # Results
        print(f"--- {n_reads} random reads ---")
        print(f"LMDB:  {lmdb_time:.4f}s  ({n_reads / lmdb_time:,.0f} reads/sec)")
        print(f"Dict:  {dict_time:.4f}s  ({n_reads / dict_time:,.0f} reads/sec)")
        print(f"Ratio: dict is {lmdb_time / dict_time:.1f}x faster")
        print()
        print("But consider: the dict required loading ALL data into RAM first.")
        print(
            f"  Dict memory: ~{len(words) * EMBEDDING_DIM * 8 / 1024:.0f} KB (float64 in Python)"
        )
        print(
            f"  LMDB on disk: {sum(f.stat().st_size for f in (DB_DIR).iterdir()) / 1024:.0f} KB"
        )

    finally:
        env.close()
        if JSON_FILE.exists():
            JSON_FILE.unlink()


if __name__ == "__main__":
    main()
