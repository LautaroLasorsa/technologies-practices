use std::path::PathBuf;

use heed::types::Bytes;
use heed::{Database, Env, EnvOpenOptions};

const EMBEDDING_DIM: usize = 384;

fn db_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("shared_db")
}

fn open_env() -> Env {
    unsafe {
        EnvOpenOptions::new()
            .max_dbs(2)
            .open(db_path())
            .expect("Failed to open LMDB environment")
    }
}

fn open_embeddings_db(env: &Env) -> Database<Bytes, Bytes> {
    let rtxn = env.read_txn().expect("Failed to start read txn");
    let db = env
        .open_database(&rtxn, Some("embeddings"))
        .expect("Failed to open database")
        .expect("Database 'embeddings' not found");
    rtxn.commit().expect("Failed to commit read txn");
    db
}

fn bytes_to_f32_vec(raw: &[u8]) -> Vec<f32> {
    assert_eq!(
        raw.len(),
        EMBEDDING_DIM * 4,
        "Expected {} bytes, got {}",
        EMBEDDING_DIM * 4,
        raw.len()
    );
    raw.chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn get_embedding(env: &Env, db: &Database<Bytes, Bytes>, word: &str) -> Option<Vec<f32>> {
    // TODO(human): Implement this function.
    //
    // 1. Start a read transaction with env.read_txn()
    // 2. Use db.get(&txn, word.as_bytes()) to look up the key
    //    - The DB type is Database<Bytes, Bytes>, so keys are &[u8]
    //    - get() returns Result<Option<&[u8]>>
    // 3. If found, pass the raw bytes to bytes_to_f32_vec() and wrap in Some()
    // 4. If not found (None) or error, return None
    todo!()
}

fn prefix_search(env: &Env, db: &Database<Bytes, Bytes>, prefix: &str) -> Vec<String> {
    // TODO(human): Implement this function.
    //
    // 1. Start a read transaction with env.read_txn()
    // 2. Use db.prefix_iter(&txn, prefix.as_bytes()) to get an iterator
    //    over all keys that start with the prefix bytes
    //    - heed handles the B+ tree range scan internally
    // 3. For each Ok((key, _value)) in the iterator, decode key with
    //    std::str::from_utf8(key) and collect into a Vec<String>
    // 4. Return the collected words; return vec![] on any error
    todo!()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // TODO(human): Implement cosine similarity for f32 slices.
    //
    // Formula: dot(a, b) / (||a|| * ||b||)
    // 1. Compute ||a|| = sqrt(sum of a[i]^2) and ||b|| similarly
    // 2. Compute dot product = sum of a[i]*b[i]
    // 3. Return dot / (norm_a * norm_b)
    // Rust hint: use .iter().zip().map().sum::<f32>() for dot product,
    // and f32::sqrt() for the square root.
    todo!()
}

fn main() {
    let env = open_env();
    let db = open_embeddings_db(&env);

    // Test 1: Single lookup
    match get_embedding(&env, &db, "algorithm") {
        Some(vec) => {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!(
                "get_embedding('algorithm'): len={}, norm={:.4}",
                vec.len(),
                norm
            );
        }
        None => {
            println!("get_embedding() not implemented yet — look for TODO(human)");
            return;
        }
    }

    // Test 2: Missing key
    let missing = get_embedding(&env, &db, "zzz_nonexistent");
    println!("get_embedding('zzz_nonexistent'): {:?}", missing.is_none());

    // Test 3: Prefix search
    let matches = prefix_search(&env, &db, "pro");
    println!("prefix_search('pro'): {:?}", matches);

    // Test 4: Cosine similarity (compare with Python output)
    if let (Some(va), Some(vb)) = (
        get_embedding(&env, &db, "algorithm"),
        get_embedding(&env, &db, "function"),
    ) {
        let sim = cosine_similarity(&va, &vb);
        println!("cosine_similarity('algorithm', 'function'): {:.6}", sim);

        let self_sim = cosine_similarity(&va, &va);
        println!(
            "cosine_similarity('algorithm', 'algorithm'): {:.6}",
            self_sim
        );
    }
}
