//! Exercise 6: Writing Miri-Clean Unsafe Code
//!
//! This exercise focuses on writing CORRECT unsafe code that passes Miri
//! with maximum strictness:
//!   MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check"
//!
//! `-Zmiri-strict-provenance` enforces that:
//! - Pointers are never created from integers (ptr::from_exposed_addr is UB)
//! - Pointer provenance is tracked precisely
//! - Integer-to-pointer casts are rejected
//!
//! `-Zmiri-symbolic-alignment-check` enforces that:
//! - Alignment is checked symbolically, not just numerically
//! - A pointer at a "lucky" aligned address still fails if its provenance
//!   doesn't guarantee alignment
//!
//! These flags represent the strictest interpretation of Rust's memory model.
//! Code that passes under these flags is very likely to be correct.

use std::alloc::{self, Layout};
use std::ptr;

// =============================================================================
// TODO(human): Implement manual_vec_push
// =============================================================================

/// Implement a simplified Vec::push using raw pointer allocation.
///
/// This is the core operation of Vec<T> — the most used unsafe abstraction
/// in Rust's standard library. Understanding how it works is essential for
/// writing any custom collection.
///
/// TODO(human): Implement this function that:
///
/// 1. Takes a `*mut u32` buffer pointer, current length, and current capacity.
/// 2. If len == capacity, reallocates to double the capacity (or 4 if capacity is 0).
/// 3. Writes the new value at position `len`.
/// 4. Returns the (possibly new) buffer pointer and the new capacity.
///
/// Steps:
///
/// a) Handle the reallocation case (len == capacity):
///    ```rust
///    let new_cap = if capacity == 0 { 4 } else { capacity * 2 };
///    let new_layout = Layout::array::<u32>(new_cap).unwrap();
///    ```
///
///    If buf is null (first allocation):
///    ```rust
///    let new_buf = unsafe { alloc::alloc(new_layout) as *mut u32 };
///    ```
///
///    If buf is non-null (reallocation — must copy existing data):
///    ```rust
///    let old_layout = Layout::array::<u32>(capacity).unwrap();
///    let new_buf = unsafe {
///        alloc::realloc(buf as *mut u8, old_layout, new_layout.size()) as *mut u32
///    };
///    ```
///
///    Check that the returned pointer is non-null (allocation failure → abort):
///    ```rust
///    if new_buf.is_null() {
///        alloc::handle_alloc_error(new_layout);
///    }
///    ```
///
/// b) Write the value at position `len`:
///    ```rust
///    unsafe { ptr::write(buf.add(len), value); }
///    ```
///
///    Why `ptr::write` instead of `*buf.add(len) = value`?
///    - `*ptr = value` runs Drop on the old value at that location
///    - But the memory at `buf.add(len)` is UNINITIALIZED — there is no old value
///    - Running Drop on uninitialized memory is UB (reading uninitialized data)
///    - `ptr::write` writes without reading/dropping the old value
///    - This is the critical difference that Miri catches!
///
/// c) Return `(buf, new_cap)` where buf is the (possibly reallocated) pointer.
///
/// Why this matters for Miri:
/// - Miri verifies that `alloc::alloc` returns properly aligned memory
/// - Miri tracks the provenance of the returned pointer (it belongs to the new allocation)
/// - Miri verifies that `ptr::write` doesn't access out-of-bounds memory
/// - Miri verifies that `realloc` properly transfers provenance
/// - With `-Zmiri-strict-provenance`, pointer arithmetic must stay within bounds
///
/// SAFETY requirements you must uphold:
/// - `buf` must be either null or a pointer previously returned by alloc/realloc
///   with a Layout compatible with `Layout::array::<u32>(capacity)`
/// - `len` must be <= `capacity`
/// - The caller is responsible for eventually deallocating the buffer
pub unsafe fn manual_vec_push(
    buf: *mut u32,
    len: usize,
    capacity: usize,
    value: u32,
) -> (*mut u32, usize) {
    // TODO(human): Implement as described above.
    // Handle reallocation when len == capacity, then write the value.
    // Return (possibly_new_buf, possibly_new_capacity).
    todo!()
}

// =============================================================================
// TODO(human): Implement linked_list_node_alloc
// =============================================================================

/// Allocate and link nodes for a simple singly-linked list using raw pointers.
///
/// This exercise demonstrates the Box::into_raw / Box::from_raw pattern for
/// manually managing heap-allocated nodes — the foundation of all pointer-based
/// data structures (linked lists, trees, graphs).
///
/// TODO(human): Implement this function that:
///
/// 1. Defines a Node struct (provided below).
///
/// 2. Allocates 3 nodes using Box and links them:
///    ```rust
///    let node3 = Box::into_raw(Box::new(Node { value: 3, next: ptr::null_mut() }));
///    let node2 = Box::into_raw(Box::new(Node { value: 2, next: node3 }));
///    let node1 = Box::into_raw(Box::new(Node { value: 1, next: node2 }));
///    ```
///
///    `Box::into_raw` converts a Box<T> into a raw *mut T without running Drop.
///    This "leaks" the allocation — YOU are now responsible for freeing it.
///
/// 3. Traverse the list, collecting values:
///    ```rust
///    let mut current = node1;
///    let mut values = Vec::new();
///    while !current.is_null() {
///        unsafe {
///            values.push((*current).value);
///            current = (*current).next;
///        }
///    }
///    ```
///
/// 4. Free all nodes by reclaiming them into Box:
///    ```rust
///    unsafe {
///        let _ = Box::from_raw(node1);
///        let _ = Box::from_raw(node2);
///        let _ = Box::from_raw(node3);
///    }
///    ```
///
///    `Box::from_raw` reclaims ownership — the Box's destructor will free the
///    memory when it goes out of scope. The `let _ =` pattern drops immediately.
///
///    IMPORTANT: Each node must be freed EXACTLY ONCE. Double-free is UB.
///    Also, don't access a node after freeing it (use-after-free).
///
/// 5. Return the collected values: should be `[1, 2, 3]`.
///
/// Why this matters for Miri:
/// - Miri tracks each Box allocation separately with its own provenance tag
/// - Miri verifies that `Box::from_raw` receives a pointer that was originally
///   created by `Box::into_raw` (or Box::new) with the correct type and layout
/// - Miri detects memory leaks: if you forget to free a node, Miri reports it
/// - Miri detects double-free: freeing the same node twice is UB
/// - Miri detects use-after-free: accessing a node after `Box::from_raw`
///
/// The critical invariant: every `Box::into_raw` must be matched by exactly
/// one `Box::from_raw` before the program ends. This is the manual equivalent
/// of RAII — without it, you have leaks or double-frees.
pub fn linked_list_node_alloc() -> Vec<i32> {
    // TODO(human): Implement as described above.
    // Allocate 3 nodes, link them, traverse, free, return values.
    todo!()
}

/// Node type for the linked list exercise.
///
/// Using raw pointers for the next field because ownership semantics of linked
/// lists don't map cleanly to Rust's single-owner model. Each node is
/// heap-allocated via Box and manually managed.
pub struct Node {
    pub value: i32,
    pub next: *mut Node,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex6_manual_vec_push_basic() {
        let mut buf: *mut u32 = ptr::null_mut();
        let mut len: usize = 0;
        let mut cap: usize = 0;

        // Push 5 values
        for i in 0..5 {
            let (new_buf, new_cap) = unsafe { manual_vec_push(buf, len, cap, i * 10) };
            buf = new_buf;
            cap = new_cap;
            len += 1;
        }

        // Verify values
        let mut values = Vec::new();
        for i in 0..len {
            values.push(unsafe { *buf.add(i) });
        }
        assert_eq!(values, vec![0, 10, 20, 30, 40]);

        // Clean up — must free the buffer to avoid Miri's leak detector
        if !buf.is_null() {
            let layout = Layout::array::<u32>(cap).unwrap();
            unsafe { alloc::dealloc(buf as *mut u8, layout); }
        }
    }

    #[test]
    fn ex6_manual_vec_push_triggers_realloc() {
        let mut buf: *mut u32 = ptr::null_mut();
        let mut len: usize = 0;
        let mut cap: usize = 0;

        // Push enough values to trigger multiple reallocations
        // cap: 0 → 4 → 8 → 16
        for i in 0..12 {
            let (new_buf, new_cap) = unsafe { manual_vec_push(buf, len, cap, i) };
            buf = new_buf;
            cap = new_cap;
            len += 1;
        }

        assert_eq!(len, 12);
        assert!(cap >= 12); // Should be 16 (after 0→4→8→16)

        // Verify all values survived reallocation
        for i in 0..len {
            assert_eq!(unsafe { *buf.add(i) }, i as u32);
        }

        // Clean up
        if !buf.is_null() {
            let layout = Layout::array::<u32>(cap).unwrap();
            unsafe { alloc::dealloc(buf as *mut u8, layout); }
        }
    }

    #[test]
    fn ex6_linked_list_node_alloc_works() {
        let values = linked_list_node_alloc();
        assert_eq!(values, vec![1, 2, 3]);
    }

    /// Verify that our Node struct has reasonable layout.
    #[test]
    fn ex6_node_layout() {
        let layout = Layout::new::<Node>();
        // Node contains an i32 (4 bytes) and a *mut Node (8 bytes on 64-bit)
        // With alignment padding, size should be 16 on 64-bit
        println!("Node size: {}, align: {}", layout.size(), layout.align());
        assert!(layout.size() > 0);
        assert!(layout.align() > 0);
    }

    /// Demonstrate that forgetting to free memory triggers Miri's leak detector.
    /// This test is ignored because it INTENTIONALLY leaks.
    #[test]
    #[ignore] // Miri will report a memory leak if you run this
    fn ex6_intentional_leak_detected_by_miri() {
        let _leaked = Box::into_raw(Box::new(42_i32));
        // We never call Box::from_raw — this leaks 4 bytes.
        // Miri's leak detector (enabled by default) will report this.
    }
}
