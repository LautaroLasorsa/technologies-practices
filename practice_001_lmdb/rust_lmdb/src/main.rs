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
    // Same as before, but now the DB uses Bytes for keys too.
    // Pass word.as_bytes() instead of word directly.
    let txn = env.read_txn().ok()?;
    match db.get(&txn, word.as_bytes()) {
        Ok(Some(e)) => Some(bytes_to_f32_vec(e)),
        _ => None,
    }
}

fn prefix_search(env: &Env, db: &Database<Bytes, Bytes>, prefix: &str) -> Vec<String> {
    // TODO(human): Implement this function.
    //
    // With Bytes keys, prefix_iter takes &[u8].
    // Decode keys with std::str::from_utf8().
    let txn = env.read_txn();
    if let Ok(txn) = txn {
        if let Ok(prefix_iter) = db.prefix_iter(&txn, prefix.as_bytes()) {
            return prefix_iter
                .filter_map(|item| {
                    item.ok().and_then(|(k, _)| {
                        std::str::from_utf8(k).ok().map(String::from)
                    })
                })
                .collect();
        }
    }
    vec![]
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let a_norm = f32::sqrt(a.iter().map(|x| x * x).sum::<f32>());
    let b_norm = f32::sqrt(b.iter().map(|x| x * x).sum::<f32>());
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai / a_norm) * (bi / b_norm))
        .sum::<f32>()
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
