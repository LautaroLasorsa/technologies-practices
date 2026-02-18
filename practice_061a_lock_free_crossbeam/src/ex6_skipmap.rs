//! Exercise 6: Concurrent SkipMap
//!
//! crossbeam-skiplist provides `SkipMap<K, V>` and `SkipSet<K>` — lock-free
//! concurrent sorted containers. They provide the same interface as BTreeMap/BTreeSet
//! but support safe concurrent access from multiple threads.
//!
//! A skip list is a probabilistic data structure: a linked list with multiple "express
//! lane" levels. Each element is randomly assigned a height (1 to ~log N levels).
//! Searching starts at the top level and drops down, achieving O(log n) expected time
//! for search, insert, and delete — like a balanced BST but much easier to make concurrent.
//!
//! Why skip lists for concurrency:
//! - BSTs require rotations for balancing, which touch multiple nodes atomically → hard to
//!   make lock-free. Skip lists only modify local pointers on insert/remove.
//! - Each insert affects O(log n) pointers at different levels, and each level can be
//!   updated independently with CAS.
//! - crossbeam's SkipMap uses epoch-based reclamation (crossbeam-epoch internally),
//!   so removed entries are garbage-collected safely.
//!
//! SkipMap API:
//! - `insert(key, value)` → Entry (takes &self, not &mut self!)
//! - `get(key)` → Option<Entry>
//! - `remove(key)` → Option<Entry>
//! - `range(start..end)` → iterator over entries in sorted order
//! - `iter()` → iterator over all entries in sorted order
//!
//! The Entry type is a reference-counted handle to the key-value pair in the skip list.
//! It keeps the pair alive as long as the Entry exists, even if another thread removes
//! the key from the map.

use crossbeam_skiplist::SkipMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Concurrent sorted insertions from multiple threads.
///
/// This exercise demonstrates that SkipMap maintains sorted order even when
/// multiple threads insert concurrently. Unlike BTreeMap (which requires &mut self
/// for insert), SkipMap's insert takes &self — truly concurrent mutation.
pub fn concurrent_ordered_insert() {
    let num_threads = 4;
    let items_per_thread = 500;

    // TODO(human): Implement concurrent insertions into a SkipMap and verify sorted order.
    //
    // SkipMap is the lock-free alternative to `RwLock<BTreeMap>`. The key advantage:
    // writers don't block readers, and multiple writers can insert simultaneously.
    // This is possible because skip list insertions only modify local pointers at
    // each level, and each modification uses CAS.
    //
    // Steps:
    //
    // 1. Create a shared SkipMap: `let map = Arc::new(SkipMap::<u64, String>::new());`
    //    Note: SkipMap::new() doesn't require specifying capacity — it grows dynamically.
    //
    // 2. Spawn `num_threads` threads. Each thread inserts `items_per_thread` entries:
    //    ```
    //    for i in 0..items_per_thread {
    //        let key = (thread_id * items_per_thread + i) as u64;
    //        let value = format!("thread-{}-item-{}", thread_id, i);
    //        map.insert(key, value);
    //    }
    //    ```
    //    - `insert()` takes `&self` — no mutex needed! Multiple threads call insert
    //      simultaneously, and the skip list handles synchronization internally via CAS.
    //    - If a key already exists, the old value is replaced (like BTreeMap).
    //    - The returned Entry is a handle to the inserted pair. We don't need it here.
    //
    // 3. After all threads join, verify the map:
    //    a) Check total count: `map.len() == num_threads * items_per_thread`
    //    b) Check sorted order by iterating:
    //       ```
    //       let mut prev_key = None;
    //       for entry in map.iter() {
    //           let key = *entry.key();
    //           if let Some(prev) = prev_key {
    //               assert!(key > prev, "Keys not sorted: {} after {}", key, prev);
    //           }
    //           prev_key = Some(key);
    //       }
    //       ```
    //       iter() returns entries in ascending key order — guaranteed by the skip list.
    //
    // 4. Print: "  Inserted {} entries across {} threads, sorted order verified"
    //    Also print the first and last 3 entries to show the sorted structure.
    //
    // Key insight: The entries are sorted DESPITE being inserted concurrently from
    // different threads in arbitrary order. The skip list's CAS-based insertion
    // ensures each entry lands in its correct sorted position atomically.
    //
    // Performance note: SkipMap's O(log n) operations have higher constant factors
    // than BTreeMap due to atomics and random level generation. For single-threaded
    // code, BTreeMap is faster. SkipMap's advantage appears under concurrent access
    // where BTreeMap would require a Mutex or RwLock.

    todo!("Exercise 6a: Implement concurrent SkipMap insertions with sorted order verification")
}

/// Range queries under concurrent modification.
///
/// This exercise demonstrates a powerful property of crossbeam's SkipMap: range
/// iterators provide a consistent view even while other threads are modifying the map.
/// This is possible because entries are reference-counted (the Entry type keeps the
/// key-value pair alive), and the epoch system ensures removed entries aren't freed
/// while an iterator holds a reference.
pub fn range_queries_under_contention() {
    let num_writers = 2;
    let num_readers = 2;
    let write_ops = 2000;
    let read_ops = 100;

    // TODO(human): Implement concurrent readers (range queries) and writers on a SkipMap.
    //
    // This exercise tests SkipMap's concurrent safety guarantee: readers always see
    // a consistent snapshot of entries within their range, even while writers are
    // actively inserting and removing entries. This is the lock-free equivalent of
    // snapshot isolation in databases.
    //
    // Steps:
    //
    // 1. Create a shared SkipMap: `let map = Arc::new(SkipMap::<u64, u64>::new());`
    //
    // 2. Pre-populate with initial data:
    //    ```
    //    for i in 0..1000u64 {
    //        map.insert(i, i * 10);
    //    }
    //    ```
    //
    // 3. Create a `running` flag: `let running = Arc::new(AtomicBool::new(true));`
    //
    // 4. Spawn `num_writers` writer threads. Each writer:
    //    - Performs `write_ops` operations, alternating insert and remove:
    //      ```
    //      for i in 0..write_ops {
    //          let key = (1000 + thread_id * write_ops + i) as u64;
    //          if i % 2 == 0 {
    //              map.insert(key, key * 10);
    //          } else {
    //              map.remove(&(key - 1));  // remove the one we just inserted
    //          }
    //      }
    //      ```
    //    - After finishing, continue (don't set running=false yet).
    //
    // 5. Spawn `num_readers` reader threads. Each reader:
    //    - Performs `read_ops` range queries while `running` is true:
    //      ```
    //      let mut query_count = 0;
    //      while running.load(Ordering::Relaxed) && query_count < read_ops {
    //          // Query a range of 50 keys
    //          let start = (query_count * 10) as u64;
    //          let end = start + 50;
    //
    //          let entries: Vec<_> = map.range(start..end)
    //              .map(|entry| (*entry.key(), *entry.value()))
    //              .collect();
    //
    //          // Verify sorted order within the range result
    //          for window in entries.windows(2) {
    //              assert!(
    //                  window[0].0 < window[1].0,
    //                  "Range query returned unsorted keys: {:?}",
    //                  window
    //              );
    //          }
    //
    //          query_count += 1;
    //      }
    //      query_count  // return count of queries performed
    //      ```
    //
    //    The key assertion: `windows(2)` checks that consecutive keys are strictly
    //    increasing. If the SkipMap had a concurrency bug, a range iterator could
    //    return entries out of order or include partially-initialized entries.
    //
    // 6. Join writer threads, then set `running = false`, then join reader threads.
    //    Collect the total number of range queries performed.
    //
    // 7. Print:
    //    - "  Writers: {} threads x {} ops = {} total writes"
    //    - "  Readers: {} range queries, all returned sorted results"
    //    - "  Final map size: {}"
    //
    // Key insight: In a `RwLock<BTreeMap>`, readers block writers and writers block
    // readers. With SkipMap, readers and writers operate simultaneously without blocking.
    // The range() iterator doesn't take a snapshot — it traverses the live data structure
    // — but the epoch system and reference-counted entries ensure it sees a consistent
    // linearizable view.
    //
    // Note on Entry lifetime: When you call `entry.key()` or `entry.value()`, the
    // returned references are tied to the Entry. The Entry keeps the key-value pair
    // alive even if another thread removes the key from the map. This is safe because
    // the epoch GC won't reclaim the underlying memory while any Entry exists.

    todo!("Exercise 6b: Implement range queries under concurrent write contention")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concurrent_ordered_insert() {
        concurrent_ordered_insert();
    }

    #[test]
    fn test_range_queries_under_contention() {
        range_queries_under_contention();
    }
}
