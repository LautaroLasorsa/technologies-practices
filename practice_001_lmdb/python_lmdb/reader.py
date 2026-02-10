"""LMDB Embedding Reader — queries, prefix search, and cosine similarity."""

from pathlib import Path

import lmdb
import numpy as np

EMBEDDING_DIM = 384
DB_DIR = Path(__file__).parent.parent / "shared_db"
EMBEDDINGS = b"embeddings"


def open_env_readonly() -> lmdb.Environment:
    """Open the LMDB environment in read-only mode.

    read-only mode is safer (no accidental writes) and allows
    multiple processes to read simultaneously without coordination.
    """
    return lmdb.open(str(DB_DIR), max_dbs=2, readonly=True)


def deserialize_embedding(raw: bytes) -> np.ndarray:
    """Convert raw bytes back to a float32 numpy array."""
    return np.frombuffer(raw, dtype=np.float32)


def get_embedding(env: lmdb.Environment, word: str) -> np.ndarray | None:
    """Fetch a single word's embedding from LMDB.

    TODO(human): Implement this function.

    Steps:
    1. Open the 'embeddings' named database
    2. Begin a read-only transaction (just env.begin(), no write=True)
    3. Use txn.get() with the word encoded as bytes
    4. If found, deserialize and return; if not, return None

    Hint: txn.get() returns None if key doesn't exist — no exception.
    """
    # TODO(human): Implement single-key lookup here.
    #
    # Open the named database with env.open_db(EMBEDDINGS).
    # Start a read-only transaction with env.begin(db=...) — no write=True needed.
    # Use txn.get(word.encode()) to fetch the raw bytes.
    # If the key exists, pass the bytes through deserialize_embedding() and return.
    # If the key doesn't exist, txn.get() returns None — propagate that.
    pass


def prefix_search(env: lmdb.Environment, prefix: str) -> list[str]:
    """Find all words that start with the given prefix.

    TODO(human): Implement this function.

    Steps:
    1. Open the 'embeddings' named database
    2. Begin a read-only transaction
    3. Open a cursor: txn.cursor()
    4. Position the cursor at the first key >= prefix using cursor.set_range(prefix.encode())
    5. Iterate with cursor.iternext() — for each (key, value), decode the key
    6. Stop when the key no longer starts with the prefix
    7. Return the list of matching words (just the words, not the embeddings)

    Hint: LMDB stores keys in sorted (lexicographic) order — that's what
    makes prefix search efficient. set_range() jumps directly to the right
    position in the B+ tree, no scanning from the beginning.
    """
    # TODO(human): Implement cursor-based prefix search here.
    #
    # Open the named database and start a read-only transaction.
    # Create a cursor with txn.cursor().
    # Use cursor.set_range(prefix.encode()) to jump to the first key >= prefix.
    # Then iterate with cursor.iternext(), decoding each key from bytes to str.
    # Collect keys that start with `prefix`; stop as soon as one doesn't
    # (LMDB keys are sorted, so all matches are contiguous in the B+ tree).
    # Return the list of matching word strings.
    pass


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    TODO(human): Implement this function.

    cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)

    Since our embeddings are already unit-normalized (norm=1),
    this simplifies to just the dot product. But implement the
    full formula anyway — it's more robust if vectors aren't normalized.
    """
    # TODO(human): Implement cosine similarity here.
    #
    # Formula: cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
    # Use np.dot() for the dot product and np.linalg.norm() for magnitudes.
    # Even though our embeddings are pre-normalized (norm ~= 1.0), implement
    # the full formula — it's more robust and teaches the concept properly.
    # Return a single float.
    pass


def main() -> None:
    env = open_env_readonly()
    try:
        # Test 1: Single lookup
        vec = get_embedding(env, "algorithm")
        if vec is None:
            print("get_embedding() not implemented yet — look for TODO(human)")
            return
        print(
            f"get_embedding('algorithm'): shape={vec.shape}, norm={np.linalg.norm(vec):.4f}"
        )

        # Test 2: Lookup a word that doesn't exist
        missing = get_embedding(env, "zzz_nonexistent")
        print(f"get_embedding('zzz_nonexistent'): {missing}")

        # Test 3: Prefix search
        matches = prefix_search(env, "pro")
        if matches is None:
            print("prefix_search() not implemented yet — look for TODO(human)")
            return
        print(f"prefix_search('pro'): {matches}")

        matches_co = prefix_search(env, "co")
        print(f"prefix_search('co'): {matches_co}")

        # Test 4: Cosine similarity
        vec_a = get_embedding(env, "algorithm")
        vec_b = get_embedding(env, "function")
        vec_c = get_embedding(env, "algorithm")  # same word
        if vec_a is not None and vec_b is not None and vec_c is not None:
            sim_ab = cosine_similarity(vec_a, vec_b)
            sim_ac = cosine_similarity(vec_a, vec_c)
            if sim_ab is None:
                print("cosine_similarity() not implemented yet — look for TODO(human)")
                return
            print(f"cosine_similarity('algorithm', 'function'): {sim_ab:.6f}")
            print(
                f"cosine_similarity('algorithm', 'algorithm'): {sim_ac:.6f}"
            )  # should be ~1.0
    finally:
        env.close()


if __name__ == "__main__":
    main()
