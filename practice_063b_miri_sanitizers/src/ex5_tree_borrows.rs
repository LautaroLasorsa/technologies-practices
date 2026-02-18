//! Exercise 5: Tree Borrows & Aliasing Exploration
//!
//! Tree Borrows (Neven Villani, PLDI 2025) is the next-generation aliasing model
//! for Rust, designed to replace Stacked Borrows. Key differences:
//!
//! - Replaces the "stack" per allocation with a "tree" of permissions
//! - Each pointer has a node in the tree; permissions propagate parent→child
//! - More permissive than Stacked Borrows (rejects 54% fewer real-world cases)
//! - Better support for two-phase borrows and interior mutability
//! - Same optimization guarantees as Stacked Borrows
//!
//! Run this exercise twice:
//!   1. `cargo miri test ex5` — default Stacked Borrows
//!   2. `MIRIFLAGS="-Zmiri-tree-borrows" cargo miri test ex5` — Tree Borrows
//!
//! Compare which tests pass under each model and annotate why.

// =============================================================================
// TODO(human): Implement read_through_parent_pointer
// =============================================================================

/// Read through a "parent" pointer after creating a child borrow.
///
/// This pattern is VALID under Tree Borrows but UB under Stacked Borrows.
///
/// TODO(human): Implement this function that:
///
/// 1. Creates a mutable variable: `let mut x: i32 = 10;`
///
/// 2. Creates a raw pointer (the "parent"): `let parent_ptr = &mut x as *mut i32;`
///
/// 3. Creates a child reference through the raw pointer:
///    `let child_ref = unsafe { &*parent_ptr };`
///    This creates a shared reference derived from parent_ptr.
///
/// 4. Reads through the PARENT pointer:
///    `let val = unsafe { *parent_ptr };`
///
///    Under Stacked Borrows: This is UB! Creating `child_ref` pushed SharedReadOnly
///    onto the stack. Reading through `parent_ptr` (which is below `child_ref` on
///    the stack) requires that nothing above it is SharedReadOnly with a different
///    tag... actually, re-reading through the parent raw pointer that `child_ref`
///    was derived from IS allowed in Stacked Borrows too if the pointer has
///    SharedReadWrite. The exact behavior depends on how the pointer was created.
///
///    Under Tree Borrows: Reading through a parent node is always fine as long as
///    only reads have happened — the tree structure naturally allows parent reads.
///
/// 5. Also reads through child_ref: `let child_val = *child_ref;`
///
/// 6. Returns `(val, child_val)` — both should be 10.
///
/// The purpose of this exercise is to EXPLORE the difference. Run under both
/// models and annotate the results.
///
/// NOTE: The exact behavior here depends on subtle details of how pointers are
/// derived. The point is to experiment and see Miri's output, not to memorize
/// specific rules. In real code, prefer the conservative approach that works
/// under BOTH models.
pub fn read_through_parent_pointer() -> (i32, i32) {
    // TODO(human): Implement as described above.
    // Run under both Stacked Borrows and Tree Borrows.
    // Annotate in a comment which model accepts this and why.
    todo!()
}

// =============================================================================
// Tests — run under both Stacked Borrows and Tree Borrows
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test the parent-pointer read pattern.
    ///
    /// TODO(human): After implementing read_through_parent_pointer, run this
    /// test under both aliasing models and add a comment here documenting:
    /// - Does it pass under Stacked Borrows? (cargo miri test ex5)
    /// - Does it pass under Tree Borrows? (MIRIFLAGS="-Zmiri-tree-borrows" cargo miri test ex5)
    /// - Why the difference (if any)?
    #[test]
    fn ex5_read_through_parent() {
        let (parent_val, child_val) = read_through_parent_pointer();
        assert_eq!(parent_val, 10);
        assert_eq!(child_val, 10);
    }

    /// Write through a parent pointer after creating a SHARED child borrow.
    ///
    /// Under Stacked Borrows: UB — writing through the parent pops the child's
    /// SharedReadOnly tag, and subsequent reads through child would find their
    /// tag missing.
    ///
    /// Under Tree Borrows: Also UB — writing through the parent transitions the
    /// child's permission from Active/Reserved to Disabled, making it invalid.
    ///
    /// This test demonstrates a case where BOTH models agree: writing through
    /// a parent while a shared child borrow is live is always UB.
    ///
    /// TODO(human): After reading, annotate whether this is UB under both
    /// models and explain why in a comment.
    #[test]
    #[ignore] // This is UB under BOTH models — don't run
    fn ex5_write_parent_with_shared_child_is_ub_everywhere() {
        let mut x: i32 = 10;
        let parent_ptr = &mut x as *mut i32;

        // Create a shared child borrow
        let child_ref = unsafe { &*parent_ptr };

        // Write through the parent — invalidates the child under BOTH models
        unsafe { *parent_ptr = 20; }

        // Reading through the invalidated child is UB under both models
        let _val = *child_ref;
    }

    /// Reborrow pattern: creating a child &mut from a parent *mut.
    ///
    /// This is valid under both models as long as the child is used BEFORE
    /// the parent is used again. Using the parent after the child's lifetime
    /// ends is fine (the child's permission is "returned").
    ///
    /// This is the standard pattern for FFI: receive a *mut, create a &mut
    /// for safe Rust code, then return.
    #[test]
    fn ex5_valid_reborrow_pattern() {
        let mut x: i32 = 10;
        let parent_ptr = &mut x as *mut i32;

        // Create a child &mut from the parent *mut
        let child_ref = unsafe { &mut *parent_ptr };
        *child_ref = 20;
        // child_ref is no longer used after this point

        // Using parent after child is done — valid under both models
        let val = unsafe { *parent_ptr };
        assert_eq!(val, 20);
    }

    /// Two raw pointers from the same &mut, used non-overlappingly.
    ///
    /// Under Stacked Borrows: Creating ptr2 may or may not invalidate ptr1
    /// depending on how they were derived. If both are created from the same
    /// &mut T via separate casts, the second cast creates a new pointer that
    /// pushes onto the stack, potentially popping the first.
    ///
    /// Under Tree Borrows: Both pointers are children of the same parent.
    /// As long as their accesses don't violate the tree structure, this can
    /// be valid.
    ///
    /// TODO(human): After running under both models, annotate the result.
    #[test]
    fn ex5_two_raw_ptrs_sequential_access() {
        let mut x: i32 = 0;

        // Both raw pointers derived from the same source
        let ptr1 = &mut x as *mut i32;
        // ptr2 is derived from ptr1 (same provenance chain)
        let ptr2 = ptr1;

        // Sequential access through different pointers with same provenance
        unsafe {
            *ptr1 = 10;
            *ptr2 = 20;
            let val = *ptr1;
            assert_eq!(val, 20);
        }
    }

    /// Interior mutability with UnsafeCell under both models.
    ///
    /// UnsafeCell is recognized by BOTH Stacked Borrows and Tree Borrows as
    /// a legal escape hatch for aliased mutation. This test should pass under
    /// both models.
    #[test]
    fn ex5_unsafecell_is_valid_under_both_models() {
        use std::cell::UnsafeCell;

        let cell = UnsafeCell::new(0_i32);

        // Get two raw pointers from the same UnsafeCell
        let ptr1 = cell.get();
        let ptr2 = cell.get();

        // Write through ptr1, read through ptr2 — valid because UnsafeCell
        // grants SharedReadWrite permission
        unsafe {
            *ptr1 = 42;
            let val = *ptr2;
            assert_eq!(val, 42);
        }
    }

    /// Interleaved reads and writes through raw pointers.
    ///
    /// This is a pattern that commonly occurs in linked list implementations.
    /// The behavior differs between Stacked Borrows and Tree Borrows.
    ///
    /// TODO(human): Run under both models, document which passes and why.
    #[test]
    fn ex5_interleaved_raw_ptr_access() {
        let mut x: i32 = 0;
        let ptr = &mut x as *mut i32;

        unsafe {
            // All accesses through the same raw pointer — always valid
            *ptr = 1;
            let a = *ptr;
            *ptr = 2;
            let b = *ptr;
            assert_eq!(a, 1);
            assert_eq!(b, 2);
        }
    }
}
