//! Exercise 7: Capstone — Fix the Bugs
//!
//! This module contains 5 functions with INTENTIONAL undefined behavior bugs.
//! Your task:
//!
//! 1. Run `cargo miri test ex7` to see all the errors.
//! 2. Read each Miri error message carefully.
//! 3. Fix each bug so the function produces the same observable result but
//!    without any undefined behavior.
//! 4. Run `cargo miri test ex7` again until all tests pass.
//!
//! Each bug is documented with a comment explaining what kind of UB it triggers.
//! The fix patterns are all things you learned in exercises 1-6.

use std::cell::UnsafeCell;
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

// =============================================================================
// Bug 1: Use-after-free in a swap operation
// =============================================================================

/// Swap two values using raw pointers.
///
/// BUG: The temporary storage is freed before the swap completes.
///
/// This function should swap the values of `a` and `b` using a temporary
/// heap allocation. The buggy version frees the temporary too early.
///
/// Miri will report: "dereferencing pointer to deallocated memory"
///
/// TODO(human): Fix this function so it correctly swaps a and b without UB.
/// The fix is straightforward — change the order of operations so the
/// temporary is read BEFORE it is freed. Or better yet, use a stack
/// variable instead of a heap allocation (no need for Box here).
///
/// HINT: The simplest fix is to not use Box at all — just use a local variable
/// for the temporary. If you want to keep the Box (for learning purposes),
/// ensure you read from it BEFORE dropping it.
pub fn buggy_swap(a: &mut i32, b: &mut i32) {
    // Allocate temporary on the heap (unnecessarily, but this is for demonstration)
    let tmp = Box::new(*a);
    let tmp_ptr = &*tmp as *const i32;

    *a = *b;

    // BUG: tmp is dropped here (Box goes out of scope implicitly at the end),
    // but we read through tmp_ptr below. However, the actual bug is more subtle:
    // we need to use tmp_ptr AFTER potentially dropping tmp.
    // Let's make the bug explicit:
    drop(tmp); // Free the temporary

    // UB: tmp_ptr points to freed memory!
    *b = unsafe { *tmp_ptr };
}

// =============================================================================
// Bug 2: Misaligned pointer in a byte-to-struct cast
// =============================================================================

/// A simple 2D point with 4-byte alignment requirement.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f32, // 4-byte aligned
    pub y: f32,
}

/// Parse a Point from a byte buffer at a given offset.
///
/// BUG: Casts the byte pointer directly to *const Point without ensuring
/// alignment. If offset is not a multiple of 4, this is misaligned access.
///
/// Miri will report: "accessing memory with alignment 1, but alignment 4 is required"
///
/// TODO(human): Fix this function to handle arbitrary offsets correctly.
///
/// Options:
/// a) Use `ptr::read_unaligned` to read from potentially unaligned memory
/// b) Copy the bytes into a properly aligned local variable first
/// c) Use `f32::from_le_bytes` to parse each field from byte slices
///
/// Option (c) is the most idiomatic and portable — it doesn't even need unsafe.
pub fn buggy_parse_point(data: &[u8], offset: usize) -> Point {
    assert!(offset + 8 <= data.len(), "not enough data for a Point");

    let ptr = data[offset..].as_ptr() as *const Point;
    // BUG: ptr may not be aligned to align_of::<Point>() (4 bytes)!
    unsafe { *ptr }
}

// =============================================================================
// Bug 3: Stacked Borrows violation in a "clever" optimization
// =============================================================================

/// Increment a value, returning both the old and new values.
///
/// BUG: Creates two references (shared and mutable) to the same data simultaneously
/// through raw pointer tricks, violating Stacked Borrows.
///
/// Miri will report a Stacked Borrows violation: attempting to read through a
/// tag that has been invalidated.
///
/// TODO(human): Fix this function to return (old_value, new_value) without
/// aliasing violations.
///
/// The fix is simple: read the old value BEFORE creating the mutable reference
/// (or just copy the value before incrementing). There's no need for raw pointers.
pub fn buggy_increment_and_return(value: &mut i32) -> (i32, i32) {
    let ptr = value as *mut i32;

    // Create a shared reference to read the "old" value
    let old_ref = unsafe { &*ptr };

    // BUG: Creating a mutable reference invalidates the shared reference above!
    let new_ref = unsafe { &mut *ptr };
    *new_ref += 1;

    // UB: old_ref's Stacked Borrows tag was popped when new_ref was created
    let old = *old_ref;
    let new = *new_ref;

    (old, new)
}

// =============================================================================
// Bug 4: Data race in parallel accumulation
// =============================================================================

/// Sum an array in parallel using two threads.
///
/// BUG: Both threads write to a shared non-atomic accumulator without
/// synchronization — classic data race.
///
/// Miri will report: "Data race detected"
///
/// TODO(human): Fix this function to correctly sum in parallel without data races.
///
/// Options:
/// a) Use AtomicU32 with fetch_add
/// b) Have each thread compute a partial sum locally, then combine after join
/// c) Use a Mutex
///
/// Option (b) is the best for performance — each thread works on its own data,
/// then the results are combined. No synchronization needed during the hot loop.
/// This is the "map-reduce" pattern.
pub fn buggy_parallel_sum(data: &[u32]) -> u32 {
    if data.len() < 2 {
        return data.iter().sum();
    }

    let mid = data.len() / 2;

    // Shared mutable state — data race!
    struct SharedSum(UnsafeCell<u32>);
    unsafe impl Sync for SharedSum {}

    let sum = Arc::new(SharedSum(UnsafeCell::new(0)));

    let left_data: Vec<u32> = data[..mid].to_vec();
    let right_data: Vec<u32> = data[mid..].to_vec();

    let s1 = Arc::clone(&sum);
    let t1 = std::thread::spawn(move || {
        for &val in &left_data {
            // BUG: unsynchronized write!
            unsafe { *s1.0.get() += val; }
        }
    });

    let s2 = Arc::clone(&sum);
    let t2 = std::thread::spawn(move || {
        for &val in &right_data {
            // BUG: unsynchronized write — concurrent with t1!
            unsafe { *s2.0.get() += val; }
        }
    });

    t1.join().unwrap();
    t2.join().unwrap();

    unsafe { *sum.0.get() }
}

// =============================================================================
// Bug 5: Reading uninitialized memory
// =============================================================================

/// Create an array and return the sum of its elements.
///
/// BUG: Uses MaybeUninit but reads elements before writing all of them.
///
/// Miri will report: "using uninitialized data" or "type validation failed"
///
/// TODO(human): Fix this function to properly initialize ALL elements before
/// reading any of them.
///
/// The fix: initialize ALL elements in the array before calling assume_init().
/// The current code only initializes elements at even indices, leaving odd
/// indices uninitialized.
pub fn buggy_uninit_array() -> i32 {
    use std::mem::MaybeUninit;

    let mut arr: [MaybeUninit<i32>; 8] = unsafe { MaybeUninit::uninit().assume_init() };
    // ^ This is actually fine — an array of MaybeUninit is always valid,
    //   even when uninitialized. MaybeUninit<T> has no validity invariant.

    // BUG: Only initializing even indices!
    for i in (0..8).step_by(2) {
        arr[i] = MaybeUninit::new((i as i32) * 10);
    }
    // Indices 1, 3, 5, 7 are still uninitialized!

    // BUG: Assuming the entire array is initialized — indices 1,3,5,7 are not!
    let initialized: [i32; 8] = unsafe {
        // This transmutes MaybeUninit<i32> → i32, but odd indices contain
        // uninitialized bytes. Reading uninitialized i32 is UB.
        std::mem::transmute(arr)
    };

    initialized.iter().sum()
}

// =============================================================================
// Tests — fix all 5 bugs until these pass under `cargo miri test ex7`
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex7_bug1_swap() {
        let mut a = 10;
        let mut b = 20;
        buggy_swap(&mut a, &mut b);
        assert_eq!(a, 20);
        assert_eq!(b, 10);
    }

    #[test]
    fn ex7_bug2_parse_point_aligned() {
        // Offset 0 is aligned — should work even with the bug
        let data: Vec<u8> = {
            let mut v = Vec::new();
            v.extend_from_slice(&1.0_f32.to_le_bytes());
            v.extend_from_slice(&2.0_f32.to_le_bytes());
            v
        };
        let point = buggy_parse_point(&data, 0);
        assert_eq!(point, Point { x: 1.0, y: 2.0 });
    }

    #[test]
    fn ex7_bug2_parse_point_unaligned() {
        // Offset 1 is NOT aligned — triggers the bug
        let data: Vec<u8> = {
            let mut v = vec![0u8]; // 1 byte padding
            v.extend_from_slice(&3.0_f32.to_le_bytes());
            v.extend_from_slice(&4.0_f32.to_le_bytes());
            v
        };
        let point = buggy_parse_point(&data, 1);
        assert_eq!(point, Point { x: 3.0, y: 4.0 });
    }

    #[test]
    fn ex7_bug3_increment() {
        let mut val = 42;
        let (old, new) = buggy_increment_and_return(&mut val);
        assert_eq!(old, 42);
        assert_eq!(new, 43);
        assert_eq!(val, 43);
    }

    #[test]
    fn ex7_bug4_parallel_sum() {
        let data: Vec<u32> = (1..=100).collect();
        let expected: u32 = (1..=100).sum(); // 5050
        let iterations = if cfg!(miri) { 1 } else { 10 };

        for _ in 0..iterations {
            let result = buggy_parallel_sum(&data);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn ex7_bug5_uninit_array() {
        let sum = buggy_uninit_array();
        // After fix: all 8 elements should be initialized.
        // The expected sum depends on your fix — if you initialize
        // ALL indices with i*10: 0+10+20+30+40+50+60+70 = 280
        assert_eq!(sum, 280);
    }
}
