//! Exercise 4: bumpalo Collections — Vec and String in the Arena
//!
//! Standard `Vec<T>` and `String` allocate from the global allocator. When they grow
//! (capacity doubling), the old buffer is freed and a new larger one is allocated.
//! This is correct but generates many alloc/dealloc pairs in growth-heavy code.
//!
//! bumpalo provides arena-backed versions: `bumpalo::collections::Vec<'bump, T>` and
//! `bumpalo::collections::String<'bump>`. These have the SAME API as their std
//! counterparts but allocate from a `Bump` arena instead of the global allocator.
//!
//! Key differences:
//!
//! 1. **No deallocation on growth.** When a bumpalo Vec outgrows its capacity, it
//!    allocates a new larger buffer from the arena — but the OLD buffer is NOT freed.
//!    It remains in the arena as wasted space (internal waste). This is the arena
//!    trade-off: faster allocation at the cost of higher peak memory usage.
//!
//! 2. **Lifetime-bound.** `Vec<'bump, T>` carries the arena's lifetime `'bump`.
//!    The compiler ensures the Vec cannot outlive the arena. This is a compile-time
//!    guarantee against use-after-free — no runtime checks needed.
//!
//! 3. **No Drop overhead.** When the arena is dropped/reset, ALL arena memory is freed
//!    at once. The individual Vec/String destructors don't need to call dealloc.
//!    (Note: if T itself has a Drop impl — e.g., contains a `File` handle — bumpalo
//!    does NOT call Drop for arena-allocated T values by default. Use
//!    `bumpalo::boxed::Box` if you need Drop to run.)
//!
//! 4. **Optimized extensions.** `bumpalo::collections::Vec` has `extend_from_slice_copy()`
//!    which uses a single `memcpy` for Copy types — up to 80x faster than the standard
//!    `extend_from_slice` which must go through Iterator.

#[allow(unused_imports)]
use bumpalo::Bump;
#[allow(unused_imports)]
use bumpalo::collections::Vec as BumpVec;
#[allow(unused_imports)]
use bumpalo::collections::String as BumpString;

/// Demonstrate bumpalo Vec operations and compare allocation behavior with std Vec.
pub fn arena_vec_operations() {
    // TODO(human): Implement arena Vec operations.
    //
    // This exercise shows how bumpalo's Vec behaves identically to std's Vec from the
    // user's perspective, while allocating entirely within an arena. You'll also see
    // the allocation count difference — arena Vecs generate ZERO global allocations
    // (everything stays in the arena).
    //
    // Steps:
    //
    // 1. Import `AllocSnapshot` from `crate::ex1_tracking_allocator`.
    //
    // 2. Create a `Bump` arena.
    //
    // 3. Measure std Vec allocations:
    //    - Snapshot before.
    //    - Create a `Vec::<u64>::new()` and push 1000 elements.
    //    - Snapshot after. Print delta with label "std::Vec push 1000".
    //    - Drop the Vec to free its memory.
    //
    // 4. Measure bumpalo Vec allocations:
    //    - Snapshot before.
    //    - Create a `BumpVec::<u64>::new_in(&bump)` and push 1000 elements.
    //      `new_in(&bump)` creates a Vec that allocates from the given arena.
    //      The API is identical: `v.push(42)`, `v.len()`, `v[i]`, etc.
    //    - Snapshot after. Print delta with label "bumpalo::Vec push 1000".
    //    - The delta should show ZERO global allocations — all allocation happened
    //      inside the arena, which the tracking allocator doesn't see (the arena's
    //      internal chunk allocations happened when `Bump::new()` was called or when
    //      the arena needed a new chunk, but the per-element pushes don't hit the
    //      global allocator).
    //
    // 5. Demonstrate `extend_from_slice_copy()`:
    //    - Create a source slice: `let source = [1u64, 2, 3, 4, 5];`
    //    - Create a bumpalo Vec and extend it 200 times from the source:
    //      ```
    //      let mut v = BumpVec::new_in(&bump);
    //      for _ in 0..200 {
    //          v.extend_from_slice_copy(&source);
    //      }
    //      ```
    //      `extend_from_slice_copy` is special to bumpalo — it uses `memcpy` internally
    //      for Copy types. Standard `extend_from_slice` goes through Iterator trait methods,
    //      which adds overhead from bounds checking and capacity management per element.
    //    - Print the Vec length (should be 1000) and the arena usage.
    //
    // 6. Demonstrate that arena memory is NOT freed when the Vec is dropped:
    //    - Record `bump.allocated_bytes()` before dropping the bumpalo Vec.
    //    - Drop the Vec (it goes out of scope or use `drop(v)`).
    //    - Record `bump.allocated_bytes()` after dropping.
    //    - Print both values — they should be IDENTICAL. The arena does not free
    //      individual allocations. The memory is only reclaimed when the arena is
    //      reset or dropped.
    //    - This is the fundamental arena trade-off: fast allocation, batch-only free.
    //
    // 7. Print summary comparing std Vec vs bumpalo Vec:
    //    - std Vec: N global allocations (growth doublings + final buffer)
    //    - bumpalo Vec: 0 global allocations (or just the chunk allocations)
    //    - In arena-heavy workloads, this eliminates thousands of global alloc/dealloc
    //      calls per batch, reducing allocator contention and improving throughput.

    todo!("Exercise 4a: Implement arena Vec operations and compare with std Vec")
}

/// Demonstrate bumpalo String operations and the `bumpalo::format!` macro.
pub fn arena_string_building() {
    // TODO(human): Implement arena String building.
    //
    // String formatting (`format!()`, `write!()`, `println!()`) is the #1 source of
    // hidden allocations in most Rust programs. Each `format!()` call allocates a new
    // `String` on the heap. In hot loops processing thousands of records, this creates
    // thousands of small allocations that fragment the heap and waste time.
    //
    // bumpalo's `String` and `format!` macro route all allocations to the arena,
    // eliminating global allocator pressure entirely.
    //
    // Steps:
    //
    // 1. Create a `Bump` arena.
    //
    // 2. Build a string using bumpalo's String type:
    //    ```
    //    let mut s = BumpString::new_in(&bump);
    //    s.push_str("Hello");
    //    s.push_str(", ");
    //    s.push_str("arena world!");
    //    println!("  Arena string: {}", s.as_str());
    //    ```
    //    `BumpString::new_in(&bump)` creates a string that grows within the arena.
    //    `as_str()` returns `&str` for reading — same as std String.
    //
    // 3. Use `bumpalo::format!` for formatted arena strings:
    //    ```
    //    let formatted = bumpalo::format!(in &bump, "Record #{}: value = {:.2}", 42, 3.14159);
    //    println!("  Formatted: {}", formatted.as_str());
    //    ```
    //    `bumpalo::format!(in &bump, ...)` is the arena equivalent of `format!(...)`.
    //    It returns `BumpString<'bump>` instead of `std::string::String`.
    //
    // 4. Benchmark: create 10,000 formatted strings with std vs bumpalo.
    //    - Measure global allocations for 10,000 `format!("item_{}", i)` calls.
    //    - Measure global allocations for 10,000 `bumpalo::format!(in &bump, "item_{}", i)` calls.
    //    - Print comparison. The bumpalo version should show ~0 global allocations.
    //
    // 5. Demonstrate `alloc_str` for string literals and borrowed data:
    //    ```
    //    let owned_string = String::from("this is a heap string");
    //    let arena_copy: &str = bump.alloc_str(&owned_string);
    //    println!("  Arena copy of heap string: {}", arena_copy);
    //    ```
    //    `alloc_str` copies the string bytes into the arena, returning `&mut str`.
    //    This is useful when you want to "intern" strings — copy them once into the
    //    arena and then reference them cheaply for the rest of the phase.
    //
    // Key insight: In batch processing, you often build thousands of temporary strings
    // (log messages, serialized records, formatted output). Using bumpalo, these all
    // go into the phase's arena and are freed in one shot — no per-string deallocation.

    todo!("Exercise 4b: Implement arena String building with bumpalo::format!")
}

/// Demonstrate arena allocation of nested/tree-like structures.
///
/// One of the most powerful uses of arenas is allocating graph/tree structures.
/// In standard Rust, a tree where nodes reference other nodes requires either:
/// - `Box<Node>` — works for trees but not DAGs (no shared ownership)
/// - `Rc<Node>` — adds reference counting overhead per access
/// - `Arc<Node>` — even more overhead (atomic refcount)
/// - Index-based (nodes stored in a Vec, referenced by index) — fast but loses type safety
///
/// With an arena, all nodes live in the same allocation region. References between
/// nodes are plain `&Node` borrows — zero overhead, no refcounting, no indirection.
/// The arena's lifetime ensures all nodes are valid for the same duration.
pub fn arena_nested_structures() {
    // TODO(human): Implement a tree structure allocated entirely in a bump arena.
    //
    // This exercise demonstrates how arenas simplify ownership for complex data structures.
    // Instead of fighting the borrow checker with Rc<RefCell<Node>>, you allocate all
    // nodes in an arena and use plain references. The arena owns everything; references
    // are just borrowed views.
    //
    // Steps:
    //
    // 1. Define a tree node type. Because nodes reference other nodes in the arena,
    //    they need the arena's lifetime parameter:
    //
    //    ```
    //    #[derive(Debug)]
    //    struct TreeNode<'bump> {
    //        value: i32,
    //        children: BumpVec<'bump, &'bump TreeNode<'bump>>,
    //    }
    //    ```
    //
    //    `children` is a `BumpVec` (arena-allocated Vec) of references to other nodes
    //    in the same arena. The lifetime `'bump` ties everything to the arena.
    //
    // 2. Create a `Bump` arena and build a tree:
    //
    //    ```
    //    let bump = Bump::new();
    //
    //    // Leaf nodes (no children)
    //    let leaf1 = bump.alloc(TreeNode {
    //        value: 10,
    //        children: BumpVec::new_in(&bump),
    //    });
    //    let leaf2 = bump.alloc(TreeNode {
    //        value: 20,
    //        children: BumpVec::new_in(&bump),
    //    });
    //    let leaf3 = bump.alloc(TreeNode {
    //        value: 30,
    //        children: BumpVec::new_in(&bump),
    //    });
    //    ```
    //
    //    `bump.alloc()` returns `&mut TreeNode<'bump>`. We can immediately use it as
    //    `&TreeNode<'bump>` (shared reference) when adding to another node's children.
    //
    // 3. Build internal nodes that reference the leaves:
    //
    //    ```
    //    let mut internal_children = BumpVec::new_in(&bump);
    //    internal_children.push(leaf1 as &TreeNode);  // reborrow as shared
    //    internal_children.push(leaf2 as &TreeNode);
    //
    //    let internal = bump.alloc(TreeNode {
    //        value: 5,
    //        children: internal_children,
    //    });
    //    ```
    //
    // 4. Build the root:
    //
    //    ```
    //    let mut root_children = BumpVec::new_in(&bump);
    //    root_children.push(internal as &TreeNode);
    //    root_children.push(leaf3 as &TreeNode);
    //
    //    let root = bump.alloc(TreeNode {
    //        value: 1,
    //        children: root_children,
    //    });
    //    ```
    //
    // 5. Implement a recursive function to print the tree:
    //
    //    ```
    //    fn print_tree(node: &TreeNode, depth: usize) {
    //        let indent = "  ".repeat(depth + 1);
    //        println!("{}value: {}", indent, node.value);
    //        for child in &node.children {
    //            print_tree(child, depth + 1);
    //        }
    //    }
    //    print_tree(root, 0);
    //    ```
    //
    // 6. Implement a recursive function to sum all values in the tree:
    //    ```
    //    fn sum_tree(node: &TreeNode) -> i32 {
    //        node.value + node.children.iter().map(|c| sum_tree(c)).sum::<i32>()
    //    }
    //    let total = sum_tree(root);
    //    println!("  Sum of all nodes: {} (expected: {})", total, 1 + 5 + 10 + 20 + 30);
    //    assert_eq!(total, 66);
    //    ```
    //
    // 7. Print arena usage:
    //    `println!("  Arena bytes used: {}", bump.allocated_bytes());`
    //    All nodes, all child vectors — everything in one arena. No Rc, no Arc,
    //    no reference counting overhead. When `bump` drops, everything is freed.
    //
    // Key insight: Arenas are how compilers (rustc, V8, LLVM) manage AST nodes.
    // The entire AST is allocated in a phase arena, traversed during analysis,
    // and freed in bulk when the phase completes. No per-node refcounting, no GC pauses,
    // just batch alloc → batch process → batch free.

    todo!("Exercise 4c: Build a tree structure in a bump arena with shared references")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_vec_operations() {
        arena_vec_operations();
    }

    #[test]
    fn test_arena_string_building() {
        arena_string_building();
    }

    #[test]
    fn test_arena_nested_structures() {
        arena_nested_structures();
    }
}
