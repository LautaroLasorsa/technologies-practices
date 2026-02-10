# Practice 001: LMDB (Lightning Memory-Mapped Database)

## Technologies

- **LMDB** — Memory-mapped B+ tree key-value store with MVCC
- **py-lmdb** — Python bindings (`lmdb` package)
- **heed** — Typed Rust LMDB wrapper (by Meilisearch)
- **msgpack** — Cross-language binary serialization

## Stack

- Python 3.12+ (uv)
- Rust (cargo)

## Description

Build a **cross-language embedding cache**: a Python service writes word embeddings into LMDB, and a Rust CLI reads and queries them. This teaches LMDB's core mechanics — transactions, cursors, named databases, zero-copy reads — while demonstrating cross-language data sharing via a common binary format.

### What you'll learn

1. **LMDB architecture** — memory-mapped files, B+ trees, copy-on-write, single-writer/multi-reader
2. **Transaction discipline** — read vs write transactions, MVCC snapshots, context managers
3. **Named databases** — partitioning data within a single environment
4. **Cursor navigation** — iteration, prefix search, range queries
5. **Cross-language interop** — sharing LMDB files between Python and Rust with compatible serialization
6. **Performance characteristics** — why LMDB excels at read-heavy workloads vs SQLite/RocksDB/Redis

## Instructions

### Phase 1: Setup & Concepts (~10 min)

1. Initialize Python project with `uv` and Rust project with `cargo`
2. Understand LMDB's architecture: mmap, B+ tree, MVCC, copy-on-write
3. Key question: How does single-writer/multi-reader differ from traditional locking?

### Phase 2: Python Writer (~25 min)

1. Create an LMDB environment with proper `map_size` and `max_dbs`
2. Define two named databases: `embeddings` (word → vector) and `metadata` (stats)
3. **User implements:** Write function that stores word embeddings (384-dim float32 vectors) serialized as raw bytes
4. **User implements:** Batch write with a single transaction for ~1000 words
5. Write metadata: word count, vector dimension, creation timestamp
6. Key question: Why use `with env.begin(write=True)` instead of manual commit/abort?

### Phase 3: Python Reader & Queries (~15 min)

1. **User implements:** Read function with read-only transaction to fetch a single embedding by word
2. **User implements:** Cursor-based prefix search (find all words starting with "pro")
3. **User implements:** Cosine similarity between two word vectors
4. Key question: What happens if you try to write inside a read-only transaction?

### Phase 4: Rust Reader (~30 min)

1. Set up `heed` crate with proper dependencies
2. Open the same LMDB environment created by Python (read-only)
3. **User implements:** Query a word's embedding using heed's typed API
4. **User implements:** Deserialize the raw f32 bytes into a `Vec<f32>`
5. **User implements:** Cosine similarity in Rust, compare result with Python
6. Key question: How does heed's type system prevent transaction misuse at compile time?

### Phase 5: Exploration & Benchmarks (~15 min)

1. **User implements:** Simple benchmark — read 1000 random words, measure latency
2. Compare: LMDB read vs Python dict lookup (from JSON file)
3. Observe: memory usage with `buffers=True` vs copying bytes
4. Discussion: When would you choose LMDB over SQLite? Over Redis?

## Motivation

- **ML infrastructure**: LMDB is the standard for fast embedding/dataset storage in ML pipelines (used by Caffe, DALI, many embedding libraries)
- **Cross-language data sharing**: Understanding memory-mapped I/O and binary serialization is essential for polyglot systems
- **Performance intuition**: Knowing when B+ trees beat LSM trees (RocksDB) or in-memory stores (Redis) is a key architectural skill
- **Rust interop**: Demonstrates practical Rust + Python collaboration on shared data

## References

- [LMDB Official Docs](http://www.lmdb.tech/doc/)
- [py-lmdb Documentation](https://lmdb.readthedocs.io/)
- [heed (Rust LMDB wrapper)](https://github.com/meilisearch/heed)
- [LMDB Architecture Overview](https://dbdb.io/db/lmdb)

## Commands

### Phase 1-3: Python (writer & reader)

Run from `practice_001_lmdb/python_lmdb/`.

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies (`lmdb`, `numpy`) into the virtual environment |
| `uv run python main.py` | Run the Python script: writes embeddings to `../shared_db/`, then reads and queries them |

### Phase 4-5: Rust (reader & benchmarks)

Run from `practice_001_lmdb/rust_lmdb/`.

| Command | Description |
|---------|-------------|
| `cargo build --release` | Compile the Rust LMDB reader (links `advapi32` on Windows via `build.rs`) |
| `cargo run --release` | Run the Rust reader against `../shared_db/` — lookups, prefix search, cosine similarity |

## State

`not-started`
