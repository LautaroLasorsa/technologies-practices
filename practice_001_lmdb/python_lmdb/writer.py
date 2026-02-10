"""LMDB Embedding Writer — stores word embeddings in LMDB."""

import time
from pathlib import Path

import lmdb
import numpy as np

# Shared constants — both Python and Rust must agree on these
EMBEDDING_DIM = 384
DB_DIR = Path(__file__).parent.parent / "shared_db"

# Words to generate embeddings for (subset of common English words)
WORD_LIST = [
    "algorithm",
    "binary",
    "cache",
    "database",
    "embedding",
    "function",
    "graph",
    "hash",
    "index",
    "join",
    "kernel",
    "latency",
    "memory",
    "network",
    "optimize",
    "parallel",
    "query",
    "runtime",
    "stack",
    "thread",
    "update",
    "vector",
    "worker",
    "yield",
    "zero",
    "process",
    "protocol",
    "proxy",
    "publish",
    "queue",
    "reduce",
    "register",
    "render",
    "request",
    "resolve",
    "response",
    "restore",
    "route",
    "schema",
    "scope",
    "segment",
    "sequence",
    "server",
    "session",
    "signal",
    "socket",
    "source",
    "state",
    "stream",
    "struct",
    "subnet",
    "symbol",
    "syntax",
    "target",
    "tensor",
    "token",
    "trace",
    "transform",
    "traverse",
    "trigger",
    "tuple",
    "type",
    "union",
    "unit",
    "value",
    "variable",
    "version",
    "virtual",
    "volume",
    "weight",
    "abstract",
    "adapter",
    "allocate",
    "analyze",
    "archive",
    "assert",
    "async",
    "atomic",
    "authenticate",
    "authorize",
    "balance",
    "benchmark",
    "bitwise",
    "block",
    "boolean",
    "branch",
    "breakpoint",
    "broadcast",
    "buffer",
    "build",
    "callback",
    "capture",
    "channel",
    "checkpoint",
    "cipher",
    "classify",
    "client",
    "clone",
    "cluster",
    "codec",
    "collect",
    "commit",
    "compile",
    "compress",
    "compute",
    "concatenate",
    "concurrent",
    "configure",
    "connect",
    "consensus",
    "container",
    "context",
    "controller",
    "convert",
    "coordinate",
]


def create_environment() -> lmdb.Environment:
    """Create and return an LMDB environment with named databases enabled.

    Key settings:
    - map_size: Maximum database size (must be set upfront, can be grown later).
      We use 100MB — generous for ~1000 embeddings of 384 floats each.
    - max_dbs: Must be set > 0 to use named databases (default is 0!).
    - subdir: When True (default), path is a directory containing data.mdb + lock.mdb.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return lmdb.open(
        str(DB_DIR),
        map_size=100 * 1024 * 1024,  # 100 MB
        max_dbs=2,
    )


def generate_embedding(word: str) -> np.ndarray:
    """Generate a deterministic pseudo-embedding for a word.

    Uses the word as a seed so both Python and Rust get the same vectors.
    In production, you'd call a model (e.g., sentence-transformers).
    """
    seed = sum(ord(c) for c in word)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    # Normalize to unit vector (useful for cosine similarity later)
    vec /= np.linalg.norm(vec)
    return vec


def serialize_embedding(vec: np.ndarray) -> bytes:
    """Serialize a float32 numpy array to raw bytes.

    Format: just the raw float32 values, no header.
    This is the simplest cross-language format — Rust can read these
    directly as &[f32] via bytemuck or manual deserialization.

    384 floats * 4 bytes = 1536 bytes per embedding.
    """
    return vec.tobytes()


def store_embeddings(env: lmdb.Environment, words: list[str]) -> int:
    """Store embeddings for all words in a single write transaction.

    TODO(human): Implement this function.

    Steps:
    1. Open the 'embeddings' named database within the environment
    2. Begin a WRITE transaction (env.begin with write=True)
    3. For each word, generate its embedding, serialize it, and put(key, value)
       - Key: word encoded as UTF-8 bytes
       - Value: serialized embedding (raw float32 bytes)
    4. The transaction commits automatically when the `with` block exits
    5. Return the number of words stored

    Hints:
    - Use `env.open_db(b'embeddings')` to get/create a named database
    - The db handle must be passed to `env.begin(db=...)` OR to `txn.put(..., db=db)`
    - Remember: keys and values must be `bytes`, not `str`
    """
    # TODO(human): Implement the batch write logic here.
    #
    # Use env.open_db(b"embeddings") to get the named database handle.
    # Then open a WRITE transaction with env.begin(write=True, db=...).
    # Inside the transaction, loop over `words`:
    #   - Call generate_embedding(word) to get the numpy vector
    #   - Call serialize_embedding(vec) to get raw bytes
    #   - Use txn.put(key_bytes, value_bytes) to store the pair
    # The `with` block auto-commits on success, auto-aborts on exception.
    # Return the count of words successfully stored.
    pass


def store_metadata(env: lmdb.Environment, word_count: int) -> None:
    """Store metadata about the embedding collection."""
    meta_db = env.open_db(b"metadata")
    with env.begin(write=True, db=meta_db) as txn:
        txn.put(b"word_count", str(word_count).encode())
        txn.put(b"embedding_dim", str(EMBEDDING_DIM).encode())
        txn.put(b"created_at", time.strftime("%Y-%m-%dT%H:%M:%S").encode())
        txn.put(b"format", b"float32_raw")


def main() -> None:
    env = create_environment()
    try:
        count = store_embeddings(env, WORD_LIST)
        if count is None:
            print("store_embeddings() not implemented yet — look for TODO(human)")
            return
        store_metadata(env, count)
        print(f"Stored {count} embeddings ({EMBEDDING_DIM}d) in {DB_DIR}")

        # Quick verification: read back one embedding
        emb_db = env.open_db(b"embeddings")
        with env.begin(db=emb_db) as txn:
            raw = txn.get(b"algorithm")
            if raw:
                vec = np.frombuffer(raw, dtype=np.float32)
                print(
                    f"  'algorithm' -> shape={vec.shape}, norm={np.linalg.norm(vec):.4f}"
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
