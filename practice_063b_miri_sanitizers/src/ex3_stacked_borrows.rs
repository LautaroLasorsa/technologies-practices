//! Exercise 3: Stacked Borrows Violations
//!
//! The Stacked Borrows model (Ralf Jung, POPL 2020) defines aliasing rules for
//! raw pointers in unsafe Rust. Each memory allocation has a "borrow stack" that
//! tracks which pointers are allowed to access it.
//!
//! Key rules:
//! - Creating `&mut T` pushes a Unique item — invalidates everything above it
//! - Creating `&T` pushes SharedReadOnly — only reads allowed through it
//! - Raw pointers derived from `&mut T` get SharedReadWrite permission
//! - Using a pointer that has been "popped off" the stack is UB
//!
//! The most common violation: creating two `&mut` to the same data via raw pointers.
//! Even if you "carefully" alternate accesses, Stacked Borrows says this is UB
//! because the compiler can optimize based on the assumption that `&mut` is unique.

use std::cell::UnsafeCell;

/// Demonstrate the concept behind `split_at_mut` — the canonical correct
/// way to have two mutable references to non-overlapping parts of a slice.
pub fn demonstrate_split_at_mut_concept() {
    let mut data = vec![1, 2, 3, 4, 5];

    // split_at_mut is safe because the two halves DON'T OVERLAP.
    // Internally it uses unsafe code, but the invariant (non-overlapping)
    // makes it sound.
    let (left, right) = data.split_at_mut(2);
    left[0] = 10;
    right[0] = 30;

    println!("  split_at_mut result: {data:?}");
}

// =============================================================================
// TODO(human): Implement safe_mutable_split
// =============================================================================

/// Split a mutable slice into two non-overlapping mutable halves using raw pointers.
///
/// This is how `slice::split_at_mut` works internally — it's the canonical example
/// of correct unsafe aliasing in Rust.
///
/// TODO(human): Implement this function using raw pointers.
///
/// The function receives a mutable slice and a midpoint index. It must return
/// two mutable slices: `&mut [T]` for `[0..mid]` and `&mut [T]` for `[mid..len]`.
///
/// Steps:
///
/// 1. Assert that `mid <= slice.len()` (panic if not).
///
/// 2. Get the raw pointer to the start of the slice:
///    ```rust
///    let ptr = slice.as_mut_ptr();
///    let len = slice.len();
///    ```
///
/// 3. Create two non-overlapping slices using `std::slice::from_raw_parts_mut`:
///    ```rust
///    unsafe {
///        (
///            std::slice::from_raw_parts_mut(ptr, mid),
///            std::slice::from_raw_parts_mut(ptr.add(mid), len - mid),
///        )
///    }
///    ```
///
/// Why this is sound (passes Miri):
/// - The two slices point to NON-OVERLAPPING memory regions
/// - `ptr` and `ptr.add(mid)` have the same provenance (derived from the original slice)
///   but point to disjoint byte ranges
/// - Stacked Borrows allows this because neither slice's borrow range overlaps the other's
///
/// Why the naive approach (two `&mut` to the same data) is UB:
/// - Creating `&mut slice[0]` and `&mut slice[1]` separately would invalidate
///   the first reference when the second is created, because `&mut` to ANY part
///   of the slice invalidates ALL existing `&mut` to that slice (the borrow is
///   on the whole slice, not individual elements)
/// - The raw pointer approach works because `from_raw_parts_mut` creates a NEW
///   slice with a new provenance that covers only its range
///
/// Miri verifies: no overlapping mutable borrows, correct pointer arithmetic,
/// no out-of-bounds access.
pub fn safe_mutable_split<T>(slice: &mut [T], mid: usize) -> (&mut [T], &mut [T]) {
    // TODO(human): Implement using raw pointers as described above.
    // This should behave identically to slice::split_at_mut(mid).
    todo!()
}

// =============================================================================
// TODO(human): Implement interior_mut_pattern
// =============================================================================

/// Use UnsafeCell to achieve interior mutability — the correct escape hatch
/// for aliased mutation.
///
/// TODO(human): Implement this function that demonstrates UnsafeCell.
///
/// `UnsafeCell<T>` is the ONLY legal way to mutate data behind a shared reference
/// in Rust. All interior mutability types (Cell, RefCell, Mutex, AtomicU64) are
/// built on top of UnsafeCell.
///
/// Why UnsafeCell works with Stacked Borrows:
/// - Normally, creating `&T` pushes SharedReadOnly on the borrow stack, which
///   forbids writes through any pointer derived from it.
/// - `UnsafeCell<T>` is special: creating `&UnsafeCell<T>` pushes SharedReadWrite
///   instead of SharedReadOnly. This allows mutation through the shared reference.
/// - The compiler knows about UnsafeCell and disables alias-based optimizations
///   for data behind it (it won't assume the value doesn't change between reads).
///
/// Steps:
/// 1. Create an `UnsafeCell<i32>` with initial value 0.
///    ```rust
///    let cell = UnsafeCell::new(0_i32);
///    ```
///
/// 2. Get a raw pointer via `cell.get()` — this returns `*mut i32`.
///    ```rust
///    let ptr = cell.get();
///    ```
///
/// 3. Write to the cell through the raw pointer:
///    ```rust
///    unsafe { *ptr = 42; }
///    ```
///
/// 4. Read back through the same pointer and return the value:
///    ```rust
///    unsafe { *ptr }
///    ```
///
/// This passes Miri because UnsafeCell legitimately allows aliased mutation.
/// Without UnsafeCell, writing through a pointer derived from `&T` would be
/// a Stacked Borrows violation.
pub fn interior_mut_pattern() -> i32 {
    // TODO(human): Implement using UnsafeCell as described above.
    todo!()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Intentionally buggy test (Miri WILL flag this) ----

    /// INTENTIONAL BUG: Two &mut references to the same data via raw pointers.
    ///
    /// Miri error will say something like:
    ///   "attempting a write access using <tag> at alloc123[0x0],
    ///    but that tag does not exist in the borrow stack for this location"
    ///
    /// What happens step by step in Stacked Borrows:
    /// 1. `let ptr = &mut x as *mut i32` — stack: [Unique(tag1)]
    /// 2. `let ref1 = &mut *ptr` — pushes Unique(tag2), stack: [Unique(tag1), Unique(tag2)]
    /// 3. `let ref2 = &mut *ptr` — pushes Unique(tag3), which POPS tag2!
    ///    stack: [Unique(tag1), Unique(tag3)]
    /// 4. `*ref1 = 1` — tries to use tag2, but tag2 was popped → UB!
    ///
    /// This is UB even though we "alternate" accesses. The compiler can see that
    /// ref1 is `&mut i32` and assume nobody else writes to it, potentially
    /// reordering or caching the read.
    #[test]
    #[ignore] // Remove #[ignore] to see Miri flag this as UB
    fn ex3_buggy_mutable_aliasing() {
        let mut x: i32 = 0;
        let ptr = &mut x as *mut i32;

        // Create two &mut references to the same data — UB!
        let ref1 = unsafe { &mut *ptr };
        let ref2 = unsafe { &mut *ptr }; // This invalidates ref1's borrow

        *ref1 = 1; // UB: ref1's Stacked Borrows tag was popped when ref2 was created
        *ref2 = 2;

        let _ = x;
    }

    /// INTENTIONAL BUG: Writing through a pointer derived from &T (shared ref).
    ///
    /// Miri error: attempting a write through a SharedReadOnly tag.
    ///
    /// Creating &T pushes SharedReadOnly on the borrow stack. Casting to *const T
    /// and then to *mut T doesn't change the permission — the pointer still has
    /// SharedReadOnly provenance. Writing through it is UB.
    ///
    /// This is the kind of bug that "works" natively because the CPU doesn't
    /// track permissions. But the compiler assumes &T data doesn't change and
    /// may cache the value in a register, never re-reading from memory.
    #[test]
    #[ignore] // Remove #[ignore] to see Miri flag this as UB
    #[allow(invalid_reference_casting)]
    fn ex3_buggy_write_through_shared_ref() {
        let x: i32 = 42;
        let shared_ref: &i32 = &x;

        // Cast &T → *const T → *mut T — the cast compiles but doesn't
        // change the Stacked Borrows permission!
        let ptr = shared_ref as *const i32 as *mut i32;

        // UB: writing through a pointer with SharedReadOnly permission!
        unsafe {
            *ptr = 100;
        }
    }

    // ---- Fix verification tests ----

    #[test]
    fn ex3_safe_mutable_split_basic() {
        let mut data = vec![1, 2, 3, 4, 5];
        let (left, right) = safe_mutable_split(&mut data, 2);

        left[0] = 10;
        left[1] = 20;
        right[0] = 30;
        right[1] = 40;
        right[2] = 50;

        // Verify modifications through both halves
        assert_eq!(data, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn ex3_safe_mutable_split_at_start() {
        let mut data = vec![1, 2, 3];
        let (left, right) = safe_mutable_split(&mut data, 0);

        assert!(left.is_empty());
        assert_eq!(right, &[1, 2, 3]);
    }

    #[test]
    fn ex3_safe_mutable_split_at_end() {
        let mut data = vec![1, 2, 3];
        let (left, right) = safe_mutable_split(&mut data, 3);

        assert_eq!(left, &[1, 2, 3]);
        assert!(right.is_empty());
    }

    #[test]
    fn ex3_interior_mut_pattern_works() {
        let result = interior_mut_pattern();
        assert_eq!(result, 42);
    }

    #[test]
    fn ex3_unsafecell_multiple_writes() {
        // Demonstrate that UnsafeCell allows multiple writes through shared refs.
        let cell = UnsafeCell::new(0_i32);
        let ptr = cell.get();

        unsafe {
            *ptr = 10;
            *ptr = 20;
            *ptr = 30;
        }

        assert_eq!(unsafe { *ptr }, 30);
    }
}
