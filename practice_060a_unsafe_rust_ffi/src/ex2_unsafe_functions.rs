//! Exercise 2: Unsafe Functions & Safe Wrappers
//!
//! This exercise teaches the core pattern of unsafe Rust: write an unsafe
//! implementation, then wrap it in a safe API that upholds all invariants.
//!
//! Key concepts:
//! - `unsafe fn` — functions with preconditions the compiler cannot verify
//! - Safe wrappers — public API that validates inputs before calling unsafe code
//! - `std::mem::transmute` — the nuclear option for type conversion
//! - `slice::from_raw_parts` / `slice::from_raw_parts_mut` — creating slices from pointers
//!
//! Why this matters:
//! Every good FFI binding follows this pattern: the raw `extern "C"` calls are
//! unsafe, but the public Rust API wraps them in safe functions that validate
//! inputs, check nulls, convert types, and manage lifetimes. Users of your
//! library never see `unsafe`.

pub fn run() {
    demo_unchecked_indexing();
    demo_split_at_mut_raw();
    demo_transmute();
}

// ─── Exercise 2a: Unchecked Indexing ────────────────────────────────

/// Index into a slice WITHOUT bounds checking, then wrap it safely.
///
/// # Part 1: The unsafe function
///
/// Implement `unsafe_get_unchecked` that returns `slice[index]` without any
/// bounds check. This is what `slice::get_unchecked()` does internally.
///
/// # Part 2: The safe wrapper
///
/// Implement `safe_get` that validates the index, then calls the unsafe function.
///
/// # Why this matters
///
/// This is the **canonical pattern** for unsafe Rust:
///
/// ```text
/// unsafe fn (preconditions documented but not checked)
///     ↓ wrapped by
/// pub fn (preconditions validated, then calls unsafe fn)
/// ```
///
/// In FFI, the pattern is identical:
/// - `unsafe extern "C" fn c_function(...)` — raw C call with pointer preconditions
/// - `pub fn safe_wrapper(...)` — validates arguments, converts types, calls C function
///
/// The standard library is full of this: `Vec::set_len()` is unsafe (you must
/// ensure elements are initialized), but `Vec::push()` is safe (it handles everything).
///
/// # What would break
///
/// If `unsafe_get_unchecked` is called with `index >= slice.len()`, it's
/// undefined behavior — not a panic, but UB. The compiler may assume it
/// never happens and optimize accordingly (potentially eliminating subsequent
/// bounds checks, reordering memory accesses, etc.).

/// UNSAFE: Index into a slice without bounds checking.
///
/// # Safety
///
/// Caller MUST ensure `index < slice.len()`. Violating this is UB.
///
/// # Implementation
///
/// Use `slice.as_ptr().add(index)` to get a pointer to the element,
/// then dereference it. This bypasses the bounds check that `slice[index]`
/// would normally perform.
unsafe fn unsafe_get_unchecked(slice: &[i32], index: usize) -> i32 {
    // TODO(human): Return the element at `index` using raw pointer arithmetic.
    //
    // Step 1: let ptr = slice.as_ptr();
    // Step 2: *ptr.add(index)    (already in unsafe context since this is an unsafe fn)
    //
    // Note: We're inside an `unsafe fn`, so the entire function body is an
    // implicit unsafe block. You don't need to write `unsafe { }` inside here.
    // However, Rust 2024 edition may change this — explicit unsafe blocks
    // inside unsafe fns are becoming the recommended style.
    todo!("Exercise 2a-unsafe: Index without bounds check using pointer arithmetic")
}

/// Safe wrapper around `unsafe_get_unchecked`.
///
/// Returns `Some(value)` if `index` is in bounds, `None` otherwise.
/// The caller never needs to use `unsafe` — this function handles it.
fn safe_get(slice: &[i32], index: usize) -> Option<i32> {
    // TODO(human): Validate the index, then call the unsafe function.
    //
    // Step 1: Check if index < slice.len()
    // Step 2: If yes, call unsafe { unsafe_get_unchecked(slice, index) }
    //         and wrap in Some(...)
    // Step 3: If no, return None
    //
    // This is the pattern you'll use for every FFI wrapper:
    // 1. Validate inputs (null checks, bounds checks, type checks)
    // 2. Convert types (Rust types → C types)
    // 3. Call the unsafe function
    // 4. Convert the result back (C types → Rust types)
    // 5. Return a safe Rust type (Option, Result, etc.)
    todo!("Exercise 2a-safe: Validate index then call unsafe_get_unchecked")
}

fn demo_unchecked_indexing() {
    let data = [100, 200, 300, 400, 500];

    assert_eq!(safe_get(&data, 0), Some(100));
    assert_eq!(safe_get(&data, 4), Some(500));
    assert_eq!(safe_get(&data, 5), None);
    assert_eq!(safe_get(&data, 100), None);

    println!("[2a] safe_get([100..500], 2) = {:?}", safe_get(&data, 2));
    println!("[2a] safe_get([100..500], 5) = {:?}", safe_get(&data, 5));
}

// ─── Exercise 2b: split_at_mut Using Raw Pointers ───────────────────

/// Reimplement `slice::split_at_mut` using raw pointers.
///
/// # What to implement
///
/// Given a mutable slice and a midpoint index, return two non-overlapping
/// mutable sub-slices: `&mut [T]` for `[0..mid]` and `&mut [T]` for `[mid..len]`.
///
/// # Why this matters
///
/// This is THE canonical example of why `unsafe` exists in Rust. The borrow
/// checker cannot allow two `&mut` references to parts of the same slice —
/// it doesn't understand that `[0..mid]` and `[mid..len]` don't overlap.
/// The standard library implements this using raw pointers and
/// `slice::from_raw_parts_mut`, which is exactly what you'll do here.
///
/// This same pattern appears in FFI when you need to split a buffer into
/// regions that C functions process independently (e.g., double buffering,
/// ring buffers, scatter-gather I/O).
///
/// # Hints
///
/// - `slice.as_mut_ptr()` gives `*mut i32` to the first element
/// - `std::slice::from_raw_parts_mut(ptr, len)` creates a `&mut [T]` from a raw pointer
///   Safety requirements: ptr must be valid for `len` elements, properly aligned,
///   and the resulting slice must not alias any other mutable reference
/// - The first sub-slice starts at `ptr` with length `mid`
/// - The second sub-slice starts at `ptr.add(mid)` with length `len - mid`
///
/// # What would break
///
/// If `mid > slice.len()`, the pointer arithmetic goes out of bounds → UB.
/// If you create overlapping slices (e.g., both include element at index `mid`),
/// you have two `&mut` references to the same memory → UB.
fn split_at_mut_raw(slice: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    assert!(mid <= slice.len(), "mid {} out of bounds for len {}", mid, slice.len());

    // TODO(human): Split the slice into two non-overlapping mutable sub-slices
    // using raw pointers and std::slice::from_raw_parts_mut.
    //
    // Step 1: let len = slice.len();
    // Step 2: let ptr = slice.as_mut_ptr();
    // Step 3: unsafe {
    //     let left = std::slice::from_raw_parts_mut(ptr, mid);
    //     let right = std::slice::from_raw_parts_mut(ptr.add(mid), len - mid);
    //     (left, right)
    // }
    //
    // Why can't the borrow checker do this? Because `slice.as_mut_ptr()` borrows
    // the entire slice mutably. Creating TWO &mut slices from it looks like
    // aliasing to the borrow checker, even though the regions don't overlap.
    // We use `unsafe` to assert: "I know these regions are disjoint."
    //
    // This is EXACTLY what std::slice::split_at_mut does internally. Check the
    // source: https://doc.rust-lang.org/src/core/slice/mod.rs.html
    todo!("Exercise 2b: Split a mutable slice using raw pointers")
}

fn demo_split_at_mut_raw() {
    let mut data = [1, 2, 3, 4, 5];
    let (left, right) = split_at_mut_raw(&mut data, 3);

    // Modify both halves independently — this would be impossible without raw pointers
    // (or the standard library's split_at_mut, which uses raw pointers internally)
    left[0] = 10;
    right[0] = 40;

    assert_eq!(left, &[10, 2, 3]);
    assert_eq!(right, &[40, 5]);
    println!("[2b] After split_at_mut_raw and modification: {:?}", data);
}

// ─── Exercise 2c: Transmute Demo ────────────────────────────────────

/// Demonstrate `std::mem::transmute` — the most dangerous unsafe operation.
///
/// # What to implement
///
/// Use `transmute` to reinterpret an `f32` as a `u32` (view its IEEE 754 bit pattern),
/// and to convert between pointer types.
///
/// # Why this matters
///
/// `transmute` reinterprets the raw bytes of one type as another type, with zero
/// runtime cost. It's used in FFI for:
/// - Converting between `*const c_void` and `*const ConcreteType`
/// - Viewing the bit pattern of floating-point numbers
/// - Converting between enum representations
///
/// # Hints
///
/// - `std::mem::transmute::<f32, u32>(value)` reinterprets an f32's bits as u32
/// - Both types MUST have the same size — transmute between different sizes is a compile error
/// - For f32 → u32, prefer `f32::to_bits()` (safe, same result, more readable)
/// - For u32 → f32, prefer `f32::from_bits()` (safe)
/// - `transmute` is shown here for education; in real code, use the safe alternatives
///
/// # What would break
///
/// - Transmuting a `u8` (value 2) to a `bool` is UB — bool only allows 0 or 1
/// - Transmuting an `i32` to a `&T` is UB — the integer probably isn't a valid pointer
/// - Transmuting between types of different sizes is a compile error (not UB)
/// - General rule: if a safe alternative exists (as_bytes, to_bits, etc.), use it
fn transmute_f32_to_bits(value: f32) -> u32 {
    // TODO(human): Use std::mem::transmute to view the bit pattern of an f32.
    //
    // unsafe { std::mem::transmute::<f32, u32>(value) }
    //
    // Then note: this is identical to `value.to_bits()`, which is safe.
    // transmute is educational here — in production, use to_bits().
    //
    // The IEEE 754 representation of 1.0f32 is:
    //   Sign: 0 (positive)
    //   Exponent: 01111111 (127 biased = 0 actual)
    //   Mantissa: 00000000000000000000000 (1.0 implicit)
    //   Full: 0_01111111_00000000000000000000000 = 0x3F800000
    todo!("Exercise 2c: Transmute f32 to u32 to see IEEE 754 bit pattern")
}

fn demo_transmute() {
    let bits = transmute_f32_to_bits(1.0);
    assert_eq!(bits, 0x3F80_0000);
    println!("[2c] transmute_f32_to_bits(1.0) = 0x{:08X}", bits);

    let bits_neg = transmute_f32_to_bits(-1.0);
    assert_eq!(bits_neg, 0xBF80_0000); // sign bit flipped
    println!("[2c] transmute_f32_to_bits(-1.0) = 0x{:08X}", bits_neg);

    let bits_zero = transmute_f32_to_bits(0.0);
    assert_eq!(bits_zero, 0x0000_0000);
    println!("[2c] transmute_f32_to_bits(0.0) = 0x{:08X}", bits_zero);

    // Note the safe alternative:
    println!("[2c] Safe alternative: (1.0f32).to_bits() = 0x{:08X}", 1.0f32.to_bits());
}
