//! Exercise 2: Use-After-Free & Dangling Pointers
//!
//! This exercise demonstrates how Miri detects use-after-free and misaligned
//! memory access — two of the most common sources of CVEs in systems code.
//!
//! Miri catches these through **pointer provenance tracking**: every pointer
//! carries an invisible "tag" identifying which allocation it belongs to.
//! When an allocation is freed, all pointers with that tag become invalid.
//! Dereferencing an invalidated pointer is UB, even if the memory address
//! has been reused by a new allocation.

/// Demonstrate valid raw pointer usage — this passes Miri.
///
/// Raw pointers are not inherently UB. They become UB when:
/// - The allocation they point to has been freed (use-after-free)
/// - They point outside the bounds of their allocation
/// - They are misaligned for the type they're cast to
/// - They are dereferenced when the data is uninit
pub fn demonstrate_valid_raw_ptr() {
    let mut value: i32 = 42;
    let ptr: *mut i32 = &mut value;

    // SAFETY: `ptr` points to `value` which is still alive and we have
    // exclusive access (no other references exist).
    unsafe {
        *ptr = 100;
    }

    println!("  Valid raw pointer write: value = {value}");
}

// =============================================================================
// INTENTIONALLY BUGGY CODE — for Miri to detect
// =============================================================================
//
// The functions below contain UB that Miri will flag. They are placed in
// test modules so `cargo check` passes (the buggy code only runs in tests).
//
// Your task:
// 1. Run `cargo miri test ex2` and read the error messages carefully.
// 2. Implement the fix_* functions that do the same thing correctly.

// =============================================================================
// TODO(human): Fix use-after-free
// =============================================================================

/// Fix the use-after-free bug.
///
/// The buggy version (in the test below) does this:
/// ```ignore
/// let ptr = Box::into_raw(Box::new(42));
/// unsafe { Box::from_raw(ptr); } // frees the allocation
/// unsafe { *ptr }                 // UB: reading freed memory!
/// ```
///
/// TODO(human): Implement this function to safely:
/// 1. Allocate an i32 on the heap using `Box::new(42)`
/// 2. Convert it to a raw pointer with `Box::into_raw`
/// 3. Read the value through the raw pointer BEFORE freeing
/// 4. Free the allocation with `Box::from_raw` AFTER reading
/// 5. Return the read value
///
/// The key principle: a raw pointer is only valid while its allocation is alive.
/// `Box::into_raw` transfers ownership OUT of the Box (preventing automatic drop),
/// giving you a raw pointer. You must eventually call `Box::from_raw` to reclaim
/// ownership and free the memory — but only AFTER you're done using the pointer.
///
/// This pattern is used in FFI: you `Box::into_raw` to pass a heap object to C,
/// and `Box::from_raw` to reclaim it when C is done.
///
/// Miri enforces this with provenance: after `Box::from_raw` frees the allocation,
/// the pointer's provenance tag is invalidated. Any subsequent dereference through
/// that pointer is detected as UB, even if the memory address is coincidentally
/// reused by another allocation.
pub fn fix_use_after_free() -> i32 {
    // TODO(human): Implement the correct version as described above.
    // Allocate, read via raw pointer, THEN free. Return the read value.
    todo!()
}

// =============================================================================
// TODO(human): Fix misaligned access
// =============================================================================

/// Fix the misaligned memory access bug.
///
/// The buggy version (in the test below) does this:
/// ```ignore
/// let bytes: [u8; 8] = [1, 0, 0, 0, 2, 0, 0, 0];
/// let ptr = bytes.as_ptr();
/// let misaligned_ptr = unsafe { ptr.add(1) } as *const u32;
/// unsafe { *misaligned_ptr }  // UB: u32 requires 4-byte alignment!
/// ```
///
/// TODO(human): Implement this function to safely read a u32 from an
/// arbitrary byte offset in a byte array.
///
/// You have two correct approaches:
///
/// **Approach A — `ptr::read_unaligned`**:
///   Use `std::ptr::read_unaligned(ptr as *const u32)` which explicitly handles
///   unaligned reads. This compiles to the same code on x86 (which supports
///   unaligned access in hardware) but is correct on ARM/RISC-V (which may fault).
///
/// **Approach B — `u32::from_le_bytes`**:
///   Copy the bytes into a `[u8; 4]` array and use `u32::from_le_bytes`.
///   This is the most portable and idiomatic approach — no raw pointers needed.
///
/// The function should:
/// 1. Take a byte slice and an offset
/// 2. Read a little-endian u32 starting at that offset
/// 3. Return the u32 value
///
/// Why alignment matters: Most CPUs require that a T* pointer's address is a
/// multiple of `align_of::<T>()`. For u32, this means divisible by 4. Reading
/// a u32 from address 0x1001 is "misaligned." On x86, this works but is slower.
/// On ARM, it may fault. In Rust, it's always UB because the compiler may emit
/// aligned-only instructions based on the type's alignment guarantee.
///
/// Miri's `-Zmiri-symbolic-alignment-check` flag catches even "lucky" alignment
/// where the numeric address happens to be aligned but the pointer's symbolic
/// provenance doesn't guarantee it.
pub fn fix_misaligned_access(bytes: &[u8], offset: usize) -> u32 {
    // TODO(human): Implement a safe, alignment-correct read of a u32
    // from `bytes` at the given `offset`. Use either read_unaligned or
    // from_le_bytes. Panic if offset + 4 > bytes.len().
    todo!()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Intentionally buggy tests (Miri WILL flag these) ----
    // These tests demonstrate the UB that Miri catches. Run them to see
    // the error messages, then focus on fixing the fix_* functions.

    /// INTENTIONAL BUG: Use-after-free.
    ///
    /// Miri error will say something like:
    ///   "dereferencing pointer to deallocated memory"
    /// or "pointer to alloc123 was dereferenced after this allocation was freed"
    ///
    /// The pointer's provenance tag was invalidated when Box::from_raw freed
    /// the allocation. Even though the memory address might still be readable
    /// (the OS hasn't reclaimed the page), Rust's aliasing model says this is UB.
    #[test]
    #[ignore] // Remove #[ignore] to see Miri flag this as UB
    fn ex2_buggy_use_after_free() {
        let ptr = Box::into_raw(Box::new(42_i32));

        // This frees the heap allocation — ptr is now dangling
        unsafe {
            let _ = Box::from_raw(ptr);
        }

        // UB: reading through a dangling pointer!
        // In native execution this often "works" because the memory hasn't
        // been reused yet. Miri catches it immediately.
        let _value = unsafe { *ptr };
    }

    /// INTENTIONAL BUG: Misaligned pointer dereference.
    ///
    /// Miri error will say:
    ///   "accessing memory with alignment 1, but alignment 4 is required"
    ///
    /// The pointer was created from a [u8] (alignment 1), then cast to *const u32
    /// (requires alignment 4). The offset of 1 byte means the address is not
    /// 4-byte aligned.
    #[test]
    #[ignore] // Remove #[ignore] to see Miri flag this as UB
    fn ex2_buggy_misaligned_access() {
        let bytes: [u8; 8] = [1, 0, 0, 0, 2, 0, 0, 0];
        let ptr = bytes.as_ptr();

        // Offset by 1 byte — now misaligned for u32
        let misaligned_ptr = unsafe { ptr.add(1) } as *const u32;

        // UB: u32 requires 4-byte alignment, but this pointer is at alignment 1!
        let _value = unsafe { *misaligned_ptr };
    }

    // ---- Fix verification tests ----

    #[test]
    fn ex2_fix_use_after_free_works() {
        let value = fix_use_after_free();
        assert_eq!(value, 42);
    }

    #[test]
    fn ex2_fix_misaligned_access_at_offset_0() {
        let bytes: [u8; 8] = [1, 0, 0, 0, 2, 0, 0, 0];
        // Offset 0: reads bytes [1, 0, 0, 0] → u32 little-endian = 1
        assert_eq!(fix_misaligned_access(&bytes, 0), 1);
    }

    #[test]
    fn ex2_fix_misaligned_access_at_offset_1() {
        let bytes: [u8; 8] = [0xFF, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        // Offset 1: reads bytes [0x01, 0x00, 0x00, 0x00] → u32 little-endian = 1
        assert_eq!(fix_misaligned_access(&bytes, 1), 1);
    }

    #[test]
    fn ex2_fix_misaligned_access_at_offset_4() {
        let bytes: [u8; 8] = [1, 0, 0, 0, 2, 0, 0, 0];
        // Offset 4: reads bytes [2, 0, 0, 0] → u32 little-endian = 2
        assert_eq!(fix_misaligned_access(&bytes, 4), 2);
    }

    #[test]
    #[should_panic]
    fn ex2_fix_misaligned_access_out_of_bounds_panics() {
        let bytes: [u8; 4] = [1, 0, 0, 0];
        // Offset 2: would need bytes[2..6] but only 4 bytes exist → should panic
        let _ = fix_misaligned_access(&bytes, 2);
    }

    /// Verify that valid raw pointer usage passes Miri.
    #[test]
    fn ex2_valid_raw_ptr_is_fine() {
        let mut value: i32 = 42;
        let ptr: *mut i32 = &mut value;

        unsafe {
            *ptr = 100;
        }

        assert_eq!(value, 100);
    }
}
