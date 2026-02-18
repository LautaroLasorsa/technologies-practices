//! Exercise 4: Custom C Library via Manual FFI
//!
//! This exercise teaches calling functions from a custom C library (mathlib.c)
//! using hand-written `extern "C"` declarations.
//!
//! Key concepts:
//! - Writing `extern "C"` blocks with correct type signatures
//! - Passing Rust slices to C as (pointer, length) pairs
//! - Receiving heap-allocated C strings and freeing them with `libc::free`
//! - The danger of incorrect FFI declarations (silent UB, no compile-time check)
//!
//! Why this matters:
//! Before bindgen existed, ALL FFI bindings were hand-written. Understanding
//! manual FFI is essential because: (1) it teaches what bindgen automates,
//! (2) some projects still use manual declarations for fine-grained control,
//! (3) you need to verify bindgen's output matches your expectations.
//!
//! The C functions are defined in c_src/mathlib.c and compiled by build.rs.

use std::ffi::CStr;
use std::os::raw::c_char;

// Hand-written FFI declarations for c_src/mathlib.c.
//
// These MUST exactly match the C function signatures. A mismatch is
// **silent undefined behavior** — no compiler error, no runtime error,
// just corrupted data or crashes.
//
// Compare these with the auto-generated bindings in Exercise 6 to
// appreciate what bindgen does for you.
//
// SAFETY NOTE: In Rust 2024 edition, extern blocks must be explicitly marked
// `unsafe` because all functions in them are implicitly unsafe to call.
// The compiler cannot verify C code's behavior — it could write to invalid
// memory, dereference null pointers, or have any other C-level bugs.
//
// The `unsafe` on the extern block acknowledges that the declarations
// within are a trust boundary: we assert that the C functions exist,
// have these exact signatures, and follow C calling conventions.
unsafe extern "C" {
    // int32_t mathlib_add(int32_t a, int32_t b);
    fn mathlib_add(a: i32, b: i32) -> i32;

    // double mathlib_dot_product(const double *a, const double *b, size_t len);
    fn mathlib_dot_product(a: *const f64, b: *const f64, len: usize) -> f64;

    // char *mathlib_format_vector(const double *arr, size_t len);
    // Returns a malloc'd string — caller must free() it!
    fn mathlib_format_vector(arr: *const f64, len: usize) -> *mut c_char;

    // int32_t mathlib_multiply(int32_t a, int32_t b);
    fn mathlib_multiply(a: i32, b: i32) -> i32;
}

pub fn run() {
    demo_call_c_add();
    demo_call_c_dot_product();
    demo_call_c_format_vector();
}

// ─── Exercise 4a: Call C's mathlib_add ──────────────────────────────

/// Call the C `mathlib_add` function and return the result.
///
/// # What to implement
///
/// Wrap the unsafe FFI call to `mathlib_add(a, b)` in a safe function.
/// This is the simplest possible FFI call: two integers in, one integer out.
/// No pointers, no allocation, no ownership transfer.
///
/// # Why this matters
///
/// Even this trivial call teaches important concepts:
/// - All `extern "C"` functions are unsafe to call (Rust can't verify C code)
/// - You must wrap them in `unsafe { }` blocks
/// - The type mapping must be exact: `int32_t` → `i32`, NOT `i64` or `u32`
/// - If you declared `mathlib_add` as returning `i64` instead of `i32`, the
///   upper 32 bits would be garbage from whatever happened to be in the register
///
/// # Hints
///
/// - Simply: `unsafe { mathlib_add(a, b) }`
/// - Consider adding a `debug_assert!` or validation if the C function has
///   preconditions (mathlib_add doesn't, but many C functions do)
fn call_c_add(a: i32, b: i32) -> i32 {
    // TODO(human): Call the C function mathlib_add(a, b) in an unsafe block.
    //
    // This is a one-liner: unsafe { mathlib_add(a, b) }
    //
    // The unsafe block is required because ALL extern "C" functions are implicitly
    // unsafe. Rust cannot inspect or verify the C code's behavior — it might
    // dereference invalid pointers, write out of bounds, or trigger any other
    // C-level undefined behavior.
    //
    // For this specific function (integer addition), there are no safety concerns.
    // But the compiler doesn't know that — it treats ALL FFI calls uniformly as unsafe.
    todo!("Exercise 4a: Call C's mathlib_add in an unsafe block")
}

fn demo_call_c_add() {
    let result = call_c_add(17, 25);
    assert_eq!(result, 42);
    println!("[4a] mathlib_add(17, 25) = {}", result);

    let result_neg = call_c_add(-10, 3);
    assert_eq!(result_neg, -7);
    println!("[4a] mathlib_add(-10, 3) = {}", result_neg);
}

// ─── Exercise 4b: Call C's mathlib_dot_product ──────────────────────

/// Compute the dot product of two Rust slices by calling C's mathlib_dot_product.
///
/// # What to implement
///
/// Pass two Rust `&[f64]` slices to the C function as (pointer, length) pairs.
/// The C function expects `const double*` + `size_t` for each array.
///
/// # Why this matters
///
/// This teaches the **fundamental pattern for passing Rust collections to C**:
///
/// ```text
/// Rust slice &[T]  →  C array: (const T*, size_t)
///   .as_ptr()      →  const T*   (pointer to first element)
///   .len()         →  size_t     (number of elements)
/// ```
///
/// Rust slices are "fat pointers" (pointer + length). C has no concept of fat
/// pointers — you always pass the data pointer and length separately.
///
/// # Hints
///
/// - `a.as_ptr()` returns `*const f64` — the raw data pointer
/// - `a.len()` returns `usize` — same as C's `size_t`
/// - The C function takes the length once (both arrays must be same length)
/// - Add a bounds check: `assert_eq!(a.len(), b.len())` before the FFI call
///
/// # What would break
///
/// - If `a.len() != b.len()`, the C function reads beyond the shorter array → buffer over-read (UB)
/// - If you pass an empty slice, `as_ptr()` may return a dangling pointer (valid for zero-length though)
/// - Passing the wrong length is the #1 cause of buffer overflow bugs in C FFI
fn call_c_dot_product(a: &[f64], b: &[f64]) -> f64 {
    // TODO(human): Call the C function mathlib_dot_product.
    //
    // Step 1: assert_eq!(a.len(), b.len(), "arrays must have equal length");
    //
    // Step 2: unsafe {
    //     mathlib_dot_product(a.as_ptr(), b.as_ptr(), a.len())
    // }
    //
    // Note how we split the Rust slice into two parts for C:
    // - .as_ptr() → the raw data pointer (const double*)
    // - .len()    → the element count (size_t)
    //
    // C has no concept of "slices" or "fat pointers". Every array in C is
    // just a pointer + separate length. This is why buffer overflows are
    // so common in C — nothing enforces that the length matches the actual
    // allocation size.
    todo!("Exercise 4b: Pass Rust slices to C's mathlib_dot_product")
}

fn demo_call_c_dot_product() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    // dot product = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    let result = call_c_dot_product(&a, &b);
    assert!((result - 32.0).abs() < f64::EPSILON);
    println!("[4b] dot_product([1,2,3], [4,5,6]) = {}", result);

    // Edge case: single element
    let result_single = call_c_dot_product(&[3.0], &[7.0]);
    assert!((result_single - 21.0).abs() < f64::EPSILON);
    println!("[4b] dot_product([3], [7]) = {}", result_single);
}

// ─── Exercise 4c: Call C's mathlib_format_vector (heap-allocated return) ─

/// Call C's mathlib_format_vector and return the result as a Rust String.
///
/// # What to implement
///
/// Call the C function, which returns a `malloc`'d `char*`. Convert it to
/// a Rust `String`, then free the C memory using `libc::free`.
///
/// # Why this matters
///
/// This teaches **ownership transfer across the FFI boundary**. The C function
/// allocates memory with `malloc` and returns a pointer. The Rust side MUST:
/// 1. Check the pointer for null (malloc can fail)
/// 2. Convert the C string to a Rust string (CStr → &str → String)
/// 3. Free the C memory with `libc::free` (NOT Rust's deallocator!)
///
/// ```text
/// C side:  malloc → write data → return char*
/// Rust:    receive *mut c_char → read via CStr → clone to String → free() the C memory
/// ```
///
/// **CRITICAL:** The Rust `String` is a COPY of the data. After creating the
/// Rust string, we free the C memory. The Rust string lives independently.
/// If you forget to free, you leak C memory. If you free too early, the
/// Rust string might contain garbage (if you didn't copy first).
///
/// # Hints
///
/// - Check for null: `if ptr.is_null() { return "null".into(); }`
/// - `CStr::from_ptr(ptr)` wraps the C string (finds the null terminator)
///   This is unsafe because: (1) the pointer might not point to valid memory,
///   (2) there might not be a null terminator (reads until segfault)
/// - `.to_str().unwrap()` converts to `&str` (fails if not valid UTF-8)
/// - `.to_owned()` or `.to_string()` copies into a Rust `String`
/// - `libc::free(ptr as *mut libc::c_void)` deallocates the C memory
/// - The order matters: copy FIRST, then free
///
/// # What would break
///
/// - Forgetting `libc::free` → memory leak (C memory is not tracked by Rust's allocator)
/// - Calling `libc::free` BEFORE reading the string → use-after-free (UB)
/// - Using `CString::from_raw(ptr)` instead → WRONG! That tells Rust to free
///   using Rust's allocator, but the memory was allocated by C's malloc.
///   Mismatched allocators = UB (likely heap corruption).
fn call_c_format_vector(values: &[f64]) -> String {
    // TODO(human): Call mathlib_format_vector, convert result to String, free C memory.
    //
    // Step 1: let ptr = unsafe { mathlib_format_vector(values.as_ptr(), values.len()) };
    //
    // Step 2: Check for null (malloc failure)
    //   if ptr.is_null() { return String::from("(null)"); }
    //
    // Step 3: Convert to Rust String BEFORE freeing
    //   let rust_string = unsafe { CStr::from_ptr(ptr) }
    //       .to_str()
    //       .expect("C string is not valid UTF-8")
    //       .to_owned();
    //
    // Step 4: Free the C-allocated memory
    //   unsafe { libc::free(ptr as *mut libc::c_void); }
    //
    // Step 5: return rust_string;
    //
    // KEY INSIGHT: We COPY the string data into a Rust-owned String, THEN free
    // the C memory. The Rust String now owns its own copy. This is the safest
    // pattern for C-allocated strings: copy + free, not transfer ownership.
    todo!("Exercise 4c: Call C function returning malloc'd string, convert, free")
}

fn demo_call_c_format_vector() {
    let values = [1.5, 2.7, 3.14];
    let formatted = call_c_format_vector(&values);
    println!("[4c] mathlib_format_vector([1.5, 2.7, 3.14]) = \"{}\"", formatted);

    let single = call_c_format_vector(&[42.0]);
    println!("[4c] mathlib_format_vector([42.0]) = \"{}\"", single);

    let empty = call_c_format_vector(&[]);
    println!("[4c] mathlib_format_vector([]) = \"{}\"", empty);
}
