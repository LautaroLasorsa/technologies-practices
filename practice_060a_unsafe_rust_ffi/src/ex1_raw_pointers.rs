//! Exercise 1: Raw Pointer Fundamentals
//!
//! This exercise teaches the most basic unsafe operation: working with raw pointers.
//!
//! Key concepts:
//! - `*const T` (immutable raw pointer) and `*mut T` (mutable raw pointer)
//! - Creating raw pointers is SAFE; only dereferencing them is UNSAFE
//! - Raw pointers have no lifetime, no borrowing rules, and can be null
//! - Pointer arithmetic with `.add()`, `.offset()`, `.sub()`
//! - Type casting (punning) between pointer types
//!
//! Why this matters for FFI:
//! Every C function that takes a pointer argument expects a raw pointer.
//! Understanding raw pointers is prerequisite to all FFI work.

pub fn run() {
    demo_pointer_from_reference();
    demo_pointer_arithmetic();
    demo_null_pointer_check();
    demo_cast_pointer_types();
    demo_swap_via_raw_pointers();
}

// ─── Exercise 1a: Pointer from Reference ────────────────────────────

/// Convert a reference to a raw pointer and read the value through it.
///
/// # What to implement
///
/// Given a reference `&i32`, convert it to a `*const i32` raw pointer,
/// then dereference the raw pointer inside an `unsafe` block to read the value.
///
/// # Why this matters
///
/// This is the most fundamental unsafe operation. In FFI, you constantly convert
/// between Rust references and raw pointers. The key insight: creating the raw
/// pointer (`&x as *const i32`) is safe — the danger is only at the point of
/// dereferencing (`*ptr`), because the compiler can no longer guarantee the
/// pointer is valid.
///
/// # Hints
///
/// - `let ptr: *const i32 = &x as *const i32;` or simply `let ptr: *const i32 = &x;`
///   (Rust coerces references to raw pointers automatically in many contexts)
/// - `unsafe { *ptr }` dereferences the raw pointer
/// - You can also use `ptr.read()` instead of `*ptr` — both require unsafe
///
/// # What would break
///
/// If you dereference a raw pointer OUTSIDE an `unsafe` block, the compiler
/// rejects it. This is Rust's safety boundary: you must explicitly opt into
/// the risk.
fn pointer_from_reference(value: &i32) -> i32 {
    // TODO(human): Convert the reference to a *const i32 raw pointer,
    // then dereference it in an unsafe block to return the value.
    //
    // Step 1: let ptr: *const i32 = ...;  (convert reference to raw pointer)
    // Step 2: unsafe { ... }              (dereference the pointer)
    //
    // Remember: creating the pointer is safe. Only `*ptr` or `ptr.read()` is unsafe.
    todo!("Exercise 1a: Convert reference to raw pointer and dereference it")
}

fn demo_pointer_from_reference() {
    let x = 42;
    let result = pointer_from_reference(&x);
    assert_eq!(result, 42);
    println!("[1a] pointer_from_reference(&42) = {}", result);
}

// ─── Exercise 1b: Pointer Arithmetic ────────────────────────────────

/// Walk through an array using raw pointer arithmetic and sum all elements.
///
/// # What to implement
///
/// Given a slice, get a raw pointer to its first element, then use `.add(i)`
/// to access each element and compute the sum.
///
/// # Why this matters
///
/// C array access is pointer arithmetic: `arr[i]` is syntactic sugar for
/// `*(arr + i)`. When you pass a Rust slice to C (via `.as_ptr()`), the C
/// code walks the array with pointer arithmetic. Understanding this is
/// essential for verifying that your FFI calls are correct.
///
/// # Hints
///
/// - `slice.as_ptr()` returns `*const T` pointing to the first element
/// - `ptr.add(i)` returns a pointer to the element at offset `i`
///   (it advances by `i * size_of::<T>()` bytes, not `i` bytes!)
/// - `ptr.add(i)` is unsafe because going out of bounds is UB (not a panic — UB)
/// - Dereference with `*ptr.add(i)` or `ptr.add(i).read()`
///
/// # What would break
///
/// If you call `ptr.add(n)` where `n > slice.len()`, you get undefined behavior.
/// Unlike `slice[i]` which panics on out-of-bounds, raw pointer arithmetic
/// silently reads garbage memory (or worse, crashes unpredictably).
fn sum_via_pointer_arithmetic(slice: &[i32]) -> i32 {
    // TODO(human): Sum all elements using raw pointer arithmetic.
    //
    // Step 1: let ptr = slice.as_ptr();            (get raw pointer to first element)
    // Step 2: let len = slice.len();               (get the length)
    // Step 3: Loop i from 0..len, accumulate sum using unsafe { *ptr.add(i) }
    //
    // Do NOT use slice indexing (slice[i]) — the whole point is to use pointer arithmetic.
    // This is exactly how C iterates over arrays, and it's what happens under the hood
    // when you pass a Rust slice to a C function via .as_ptr() + .len().
    todo!("Exercise 1b: Sum array elements using raw pointer arithmetic")
}

fn demo_pointer_arithmetic() {
    let data = [10, 20, 30, 40, 50];
    let sum = sum_via_pointer_arithmetic(&data);
    assert_eq!(sum, 150);
    println!("[1b] sum_via_pointer_arithmetic([10,20,30,40,50]) = {}", sum);

    let empty: [i32; 0] = [];
    let sum_empty = sum_via_pointer_arithmetic(&empty);
    assert_eq!(sum_empty, 0);
    println!("[1b] sum_via_pointer_arithmetic([]) = {}", sum_empty);
}

// ─── Exercise 1c: Null Pointer Check ────────────────────────────────

/// Safely dereference a raw pointer that might be null.
///
/// # What to implement
///
/// Given a `*const i32`, check if it's null. If null, return `None`.
/// If non-null, dereference it and return `Some(value)`.
///
/// # Why this matters
///
/// C functions frequently return NULL to indicate errors (e.g., `malloc` returns
/// NULL on failure, `fopen` returns NULL if the file doesn't exist). Every raw
/// pointer received from C code must be checked for null before dereferencing.
/// Dereferencing a null pointer is **undefined behavior** in both C and Rust —
/// it does not reliably cause a segfault; it might silently corrupt memory.
///
/// # Hints
///
/// - `ptr.is_null()` returns true if the pointer is null
/// - `std::ptr::null::<T>()` creates a null `*const T`
/// - Pattern: `if ptr.is_null() { None } else { Some(unsafe { *ptr }) }`
///
/// # What would break
///
/// Dereferencing a null pointer is UB. On most platforms it causes a segfault,
/// but the compiler is allowed to assume it never happens and may optimize away
/// your null checks if it can "prove" the pointer was already dereferenced.
///
/// # Safety contract
///
/// This function is safe to call because it handles the null case internally.
/// If the pointer is non-null, the caller must ensure it points to a valid,
/// aligned, initialized `i32`. In a real API, this function would be `unsafe`.
fn safe_deref(ptr: *const i32) -> Option<i32> {
    // TODO(human): Check for null, then dereference if non-null.
    //
    // Step 1: if ptr.is_null() { return None; }
    // Step 2: Some(unsafe { ... })   // dereference the pointer
    //
    // This is the #1 defensive FFI pattern. Every raw pointer from C is
    // potentially null, and you MUST check before dereferencing.
    todo!("Exercise 1c: Safely dereference a potentially-null raw pointer")
}

fn demo_null_pointer_check() {
    let x = 99;
    let ptr: *const i32 = &x;
    let null_ptr: *const i32 = std::ptr::null();

    assert_eq!(safe_deref(ptr), Some(99));
    assert_eq!(safe_deref(null_ptr), None);

    println!("[1c] safe_deref(valid_ptr) = {:?}", safe_deref(ptr));
    println!("[1c] safe_deref(null_ptr) = {:?}", safe_deref(null_ptr));
}

// ─── Exercise 1d: Pointer Type Casting ──────────────────────────────

/// Reinterpret 4 bytes as a u32 using pointer casting (type punning).
///
/// # What to implement
///
/// Given a `[u8; 4]` byte array, cast a pointer to its first byte
/// (`*const u8`) to a `*const u32`, then dereference to read the bytes
/// as a single `u32` value.
///
/// # Why this matters
///
/// FFI frequently requires reinterpreting raw bytes as structured data.
/// Network protocols, file formats, and hardware registers are defined
/// as byte sequences that must be cast to the appropriate types. This is
/// "type punning" — reading memory as a different type than it was written as.
///
/// # Hints
///
/// - `bytes.as_ptr()` gives `*const u8`
/// - `ptr as *const u32` casts the pointer type (does NOT convert the data!)
/// - Dereference with `unsafe { *casted_ptr }` or `unsafe { casted_ptr.read_unaligned() }`
/// - `read_unaligned()` is safer because `[u8; 4]` might not be 4-byte aligned
///   (though in practice, stack arrays usually are)
/// - The result depends on the platform's endianness:
///   On little-endian (x86/ARM): bytes [0x01, 0x02, 0x03, 0x04] → 0x04030201
///   On big-endian: bytes [0x01, 0x02, 0x03, 0x04] → 0x01020304
///
/// # What would break
///
/// Using `ptr.read()` (aligned read) on a pointer that isn't properly aligned
/// for `u32` (4-byte alignment) is UB. `read_unaligned()` handles this safely
/// but may be slower on some architectures. In practice, prefer `u32::from_ne_bytes()`
/// for this specific use case — pointer casting is shown here for educational purposes.
fn bytes_to_u32(bytes: &[u8; 4]) -> u32 {
    // TODO(human): Cast the byte pointer to a u32 pointer and read the value.
    //
    // Step 1: let byte_ptr: *const u8 = bytes.as_ptr();
    // Step 2: let u32_ptr: *const u32 = byte_ptr as *const u32;
    // Step 3: unsafe { u32_ptr.read_unaligned() }
    //
    // Note: In production code, you'd use u32::from_ne_bytes(*bytes) or
    // u32::from_le_bytes(*bytes) instead of pointer casting. We use pointer
    // casting here to teach the concept, because FFI often requires it for
    // structs more complex than a single integer.
    todo!("Exercise 1d: Cast *const u8 to *const u32 and read the value")
}

fn demo_cast_pointer_types() {
    let bytes: [u8; 4] = [0x01, 0x02, 0x03, 0x04];
    let value = bytes_to_u32(&bytes);

    // On little-endian (x86): 0x04030201 = 67305985
    // On big-endian: 0x01020304 = 16909060
    let expected = u32::from_ne_bytes(bytes);
    assert_eq!(value, expected);
    println!(
        "[1d] bytes_to_u32([0x01, 0x02, 0x03, 0x04]) = 0x{:08X} ({})",
        value, value
    );
}

// ─── Exercise 1e: Swap via Raw Pointers ─────────────────────────────

/// Swap two values using only raw pointers.
///
/// # What to implement
///
/// Given two mutable references, convert them to `*mut i32` raw pointers,
/// then swap the values they point to using only pointer reads and writes.
///
/// # Why this matters
///
/// The borrow checker prevents having two `&mut` references to the same data.
/// But sometimes you KNOW two pointers don't alias (point to different memory)
/// and need to do operations the borrow checker can't verify. `slice::split_at_mut`
/// in the standard library uses exactly this pattern: raw pointers to bypass
/// the borrow checker when the programmer can prove non-aliasing.
///
/// # Hints
///
/// - `a as *mut i32` converts `&mut i32` to `*mut i32`
/// - `ptr.read()` reads the value (unsafe)
/// - `ptr.write(value)` writes a value (unsafe)
/// - Alternatively: `std::ptr::swap(ptr_a, ptr_b)` does it in one call (still unsafe)
///
/// # What would break
///
/// If `a` and `b` point to the SAME memory location (aliasing), this function
/// would read the value, overwrite it, then read the overwritten value as the
/// "original" — resulting in both variables having the same value (the swap fails).
/// With references, the borrow checker prevents this. With raw pointers, it's
/// YOUR responsibility to ensure non-aliasing.
fn swap_raw(a: &mut i32, b: &mut i32) {
    // TODO(human): Swap the values using raw pointers.
    //
    // Option A (manual):
    //   let pa: *mut i32 = a;
    //   let pb: *mut i32 = b;
    //   unsafe {
    //       let tmp = pa.read();
    //       pa.write(pb.read());
    //       pb.write(tmp);
    //   }
    //
    // Option B (std::ptr::swap):
    //   unsafe { std::ptr::swap(a as *mut i32, b as *mut i32); }
    //
    // Try Option A first to understand the mechanics, then note that
    // std::ptr::swap exists as a convenience.
    todo!("Exercise 1e: Swap two values using raw pointer read/write")
}

fn demo_swap_via_raw_pointers() {
    let mut x = 10;
    let mut y = 20;
    swap_raw(&mut x, &mut y);
    assert_eq!(x, 20);
    assert_eq!(y, 10);
    println!("[1e] After swap_raw: x={}, y={}", x, y);
}
