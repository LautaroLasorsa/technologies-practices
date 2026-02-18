//! Exercise 3: bumpalo Arena Basics
//!
//! A bump allocator (arena allocator) is the polar opposite of a general-purpose allocator.
//! Instead of managing individual allocations with free lists, metadata headers, and
//! coalescing logic, a bump allocator does one thing:
//!
//!   ptr = bump_pointer;
//!   bump_pointer += align_up(size, alignment);
//!   return ptr;
//!
//! That's it. ONE pointer increment per allocation. No free lists. No metadata.
//! No fragmentation. No locking (in single-threaded use).
//!
//! The trade-off: you CANNOT free individual objects. The only "free" operation is
//! resetting the entire arena — moving the bump pointer back to the start. This makes
//! arenas perfect for batch/phase-based workloads where many objects share a lifetime.
//!
//! **Memory layout inside a bumpalo arena:**
//!
//! ```text
//! Chunk 1 (e.g., 256 bytes):
//! [AAAA][padding][BBBBBBBB][CCC][padding][DDDDDDDDDDDD][....free....]
//! ^                                                     ^             ^
//! chunk_start                                           bump_ptr      chunk_end
//! ```
//!
//! When the current chunk is exhausted, bumpalo allocates a NEW chunk (roughly double
//! the size of the previous one) from the system allocator. The old chunk remains valid
//! (existing references still work). This means the arena is not one contiguous block
//! but a linked list of chunks — but within each chunk, allocations are contiguous.
//!
//! bumpalo allocates "downward" within each chunk (from high addresses to low), which
//! simplifies alignment handling — but this is an implementation detail you don't need
//! to worry about as a user.

#[allow(unused_imports)]
use bumpalo::Bump;
#[allow(unused_imports)]
use std::time::Instant;

/// A simple struct to demonstrate arena allocation of complex types.
#[derive(Debug)]
#[allow(dead_code)]
pub struct Record {
    pub id: u64,
    pub name: &'static str,
    pub values: [f64; 4],
}

/// Demonstrate the arena lifecycle: create, allocate, use, reset.
///
/// This function shows how bumpalo's `Bump` arena works at the most basic level.
/// Every value allocated from the arena lives as long as the arena itself — Rust's
/// lifetime system enforces this at compile time.
pub fn arena_lifecycle() {
    // TODO(human): Implement the arena lifecycle demonstration.
    //
    // This exercise makes arena allocation tangible by showing pointer addresses,
    // demonstrating contiguity, and illustrating the reset operation.
    //
    // Steps:
    //
    // 1. Create a new arena: `let bump = Bump::new();`
    //    Bump::new() allocates an initial chunk (default ~512 bytes) from the system
    //    allocator. The bump pointer starts at the end of this chunk.
    //
    // 2. Allocate several different types into the arena:
    //
    //    a) An integer:
    //       `let x: &mut i32 = bump.alloc(42i32);`
    //       `bump.alloc(value)` moves `value` into the arena and returns `&mut T`.
    //       The reference is valid for the lifetime of `bump`.
    //
    //    b) A struct:
    //       `let rec: &mut Record = bump.alloc(Record { id: 1, name: "test", values: [1.0, 2.0, 3.0, 4.0] });`
    //
    //    c) A string slice:
    //       `let s: &mut str = bump.alloc_str("hello from the arena");`
    //       `alloc_str` copies the string data into the arena. The returned `&mut str`
    //       lives as long as the arena — unlike `String::from()`, which allocates on the heap.
    //
    //    d) A slice copy:
    //       `let data: &mut [u8] = bump.alloc_slice_copy(&[1, 2, 3, 4, 5]);`
    //       `alloc_slice_copy` copies the slice contents into the arena.
    //
    //    e) A filled slice:
    //       `let zeros: &mut [u8] = bump.alloc_slice_fill_copy(100, 0u8);`
    //       Creates a 100-byte zero-filled slice in the arena.
    //
    // 3. Print the pointer address of each allocation:
    //    `println!("  x      @ {:p}", x as *const i32);`
    //    `println!("  rec    @ {:p}", rec as *const Record);`
    //    `println!("  s      @ {:p}", s as *const str);`
    //    `println!("  data   @ {:p}", data.as_ptr());`
    //    `println!("  zeros  @ {:p}", zeros.as_ptr());`
    //
    //    Observe that addresses are CLOSE together (within the same chunk) and typically
    //    DECREASING (bumpalo allocates downward). This contiguity is what gives arenas
    //    their cache locality advantage — all allocated objects are packed together,
    //    so iterating over them hits the CPU cache perfectly.
    //
    // 4. Print arena memory usage:
    //    `println!("  Arena allocated: {} bytes in {} chunks",
    //        bump.allocated_bytes(), bump.allocated_bytes_including_metadata());`
    //    `allocated_bytes()` reports only the user data bytes.
    //    The difference from `allocated_bytes_including_metadata()` is bumpalo's own
    //    chunk management overhead (a few pointers per chunk).
    //
    // 5. Reset the arena:
    //    `bump.reset();`
    //    This moves the bump pointer back to the start, effectively freeing ALL
    //    allocations. The chunk memory is retained (not returned to the OS), so
    //    subsequent allocations reuse the same memory without syscalls.
    //    After reset, `allocated_bytes()` returns 0.
    //
    //    IMPORTANT: After reset(), all previous references (x, rec, s, data, zeros)
    //    are INVALID. Using them would be use-after-free. Rust's borrow checker prevents
    //    this at compile time — you cannot call `bump.reset()` while any references
    //    from the arena are still alive. Try it and see the compiler error!
    //    (For the demo, let the references go out of scope before calling reset.)
    //
    // 6. Print arena usage after reset:
    //    `println!("  After reset: {} bytes", bump.allocated_bytes());`
    //    Shows 0 bytes — all memory is available for reuse.
    //
    // 7. Allocate again after reset to show memory reuse:
    //    `let y: &mut i32 = bump.alloc(99);`
    //    `println!("  New allocation after reset: y = {} @ {:p}", y, y as *const i32);`
    //    The address may be similar to the original allocations — bumpalo reuses chunks.

    todo!("Exercise 3a: Implement arena lifecycle — create, allocate, observe, reset, reuse")
}

/// Benchmark: arena allocation vs heap allocation.
///
/// This benchmark allocates many small objects with both strategies and measures
/// the wall-clock time. The arena should be dramatically faster because:
///
/// 1. Each arena allocation is a pointer bump (O(1), ~1-2 ns)
/// 2. No free-list search, no metadata, no coalescing
/// 3. No per-object deallocation — arena reset is O(num_chunks), not O(num_objects)
/// 4. Cache-friendly: sequential allocations are adjacent in memory
///
/// For heap allocation, each Box::new() must:
/// 1. Search the free list for a suitable block (~10-50 ns for small allocations)
/// 2. Possibly split the block and update free-list metadata
/// 3. Write allocation metadata (size, alignment) adjacent to the pointer
/// And each Box drop must:
/// 4. Look up the metadata to determine the block size
/// 5. Add the block back to the free list, possibly coalescing with neighbors
pub fn arena_vs_heap_benchmark() {
    // TODO(human): Implement a comparative benchmark of arena vs heap allocation.
    //
    // This benchmark makes the cost difference between allocation strategies visceral.
    // On typical hardware, you should see arena allocation 5-20x faster than Box::new()
    // for small objects, and the gap increases with object count (arena's O(1) reset
    // vs heap's O(n) individual frees).
    //
    // Steps:
    //
    // 1. Define the number of objects: `let n = 100_000;`
    //
    // 2. HEAP benchmark:
    //    - Start timer: `let start = Instant::now();`
    //    - Allocate n objects with Box::new():
    //      ```
    //      let mut boxes: Vec<Box<Record>> = Vec::with_capacity(n);
    //      for i in 0..n {
    //          boxes.push(Box::new(Record {
    //              id: i as u64,
    //              name: "heap-allocated",
    //              values: [i as f64; 4],
    //          }));
    //      }
    //      ```
    //      Use `with_capacity` so Vec growth doesn't pollute the measurement.
    //    - Read all objects to prevent optimization:
    //      `let _sum: f64 = boxes.iter().map(|r| r.values[0]).sum();`
    //    - Drop all boxes (deallocation):
    //      `drop(boxes);`
    //    - Stop timer and record elapsed.
    //
    // 3. ARENA benchmark:
    //    - Create a new `Bump` arena.
    //    - Start timer.
    //    - Allocate n objects into the arena:
    //      ```
    //      let mut refs: Vec<&Record> = Vec::with_capacity(n);
    //      for i in 0..n {
    //          refs.push(bump.alloc(Record {
    //              id: i as u64,
    //              name: "arena-allocated",
    //              values: [i as f64; 4],
    //          }));
    //      }
    //      ```
    //      Note: `bump.alloc()` returns `&mut T`. We collect `&Record` (immutable borrow
    //      after allocation) because we only need to read them.
    //    - Read all objects to prevent optimization:
    //      `let _sum: f64 = refs.iter().map(|r| r.values[0]).sum();`
    //    - Drop refs and reset arena (deallocation):
    //      `drop(refs);`
    //      `bump.reset();`
    //    - Stop timer and record elapsed.
    //
    // 4. Print comparison:
    //    ```
    //    println!("  Heap:  {:>8.2}ms ({} objects)", heap_elapsed.as_secs_f64() * 1000.0, n);
    //    println!("  Arena: {:>8.2}ms ({} objects)", arena_elapsed.as_secs_f64() * 1000.0, n);
    //    println!("  Arena speedup: {:.1}x", heap_elapsed.as_secs_f64() / arena_elapsed.as_secs_f64());
    //    ```
    //
    // IMPORTANT: Run with `cargo run --release` for meaningful results.
    // Debug builds add bounds checks and disable optimizations, distorting the comparison.
    // In debug, the arena might only be ~2x faster. In release, expect 5-20x.
    //
    // Why the arena is so much faster:
    // - Allocation: bump.alloc() = 1 pointer comparison + 1 pointer decrement (O(1))
    //   Box::new() = free-list search + possible sbrk/mmap + metadata write (O(1) amortized but high constant)
    // - Deallocation: bump.reset() = reset 1 pointer per chunk (O(chunks), typically O(1))
    //   drop(boxes) = n individual dealloc calls, each updating the free list
    // - Cache: arena objects are contiguous → prefetcher works perfectly
    //   Box objects are scattered → cache misses on iteration

    todo!("Exercise 3b: Benchmark arena allocation vs Box::new() for 100k objects")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_lifecycle() {
        arena_lifecycle();
    }

    #[test]
    fn test_arena_vs_heap_benchmark() {
        arena_vs_heap_benchmark();
    }
}
