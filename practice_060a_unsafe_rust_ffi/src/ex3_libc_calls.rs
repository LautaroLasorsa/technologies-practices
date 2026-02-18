//! Exercise 3: Calling libc from Rust
//!
//! This exercise teaches calling C standard library functions from Rust
//! using the `libc` crate. This is the simplest form of FFI: calling
//! functions that are already linked into every program.
//!
//! Key concepts:
//! - The `libc` crate provides type definitions and function declarations
//!   for C standard library and POSIX functions
//! - `CString` / `CStr` bridge Rust strings and C strings
//! - Function pointers with `extern "C"` calling convention for callbacks
//! - `libc::strlen`, `libc::qsort` — real C functions called from Rust
//!
//! Why this matters:
//! The C standard library is the lowest common denominator of system programming.
//! OS APIs, database drivers, and crypto libraries all expose C interfaces.
//! Mastering the Rust→C string conversion pipeline is essential.

use std::ffi::CString;
use std::os::raw::c_void;

pub fn run() {
    demo_libc_strlen();
    demo_libc_qsort();
}

// ─── Exercise 3a: libc::strlen ──────────────────────────────────────

/// Get the length of a Rust string by calling C's `strlen`.
///
/// # What to implement
///
/// Convert a Rust `&str` to a C string (`CString`), get a raw pointer,
/// pass it to `libc::strlen`, and return the result as `usize`.
///
/// # Why this matters
///
/// This exercises the **complete Rust→C string pipeline**:
///
/// ```text
/// &str (UTF-8, length-prefixed, no null terminator)
///   ↓ CString::new(s)     — allocates, copies, adds \0, rejects interior \0
///   ↓ .as_ptr()            — returns *const c_char (the raw C string pointer)
///   ↓ libc::strlen(ptr)    — C walks bytes until it finds \0, returns count
///   ↓ as usize             — convert C's size_t to Rust's usize
/// ```
///
/// Every FFI call that takes a string follows this exact pattern. The conversion
/// is NOT zero-cost: CString::new allocates and copies the string. For
/// performance-critical paths, consider keeping CStrings alive and reusing them.
///
/// # Hints
///
/// - `CString::new("hello")` returns `Result<CString, NulError>`
///   It fails if the string contains interior null bytes (since C strings can't have them)
/// - `.unwrap()` or `.expect()` is fine here for simplicity
/// - `.as_ptr()` returns `*const c_char` — the pointer C expects
/// - `unsafe { libc::strlen(ptr) }` — all FFI calls are unsafe
/// - `strlen` returns `libc::size_t` which is `usize` on most platforms
///
/// # What would break
///
/// **CRITICAL LIFETIME BUG:** If you write `CString::new(s).unwrap().as_ptr()`,
/// the `CString` is a temporary that is dropped at the end of the expression.
/// The pointer then DANGLES — pointing to freed memory. You MUST bind the
/// `CString` to a variable first:
///
/// ```rust
/// let c_string = CString::new(s).unwrap();  // lives until end of scope
/// let ptr = c_string.as_ptr();              // pointer is valid while c_string lives
/// ```
///
/// This is one of the most common FFI bugs in Rust.
fn rust_strlen(s: &str) -> usize {
    // TODO(human): Call libc::strlen on the Rust string.
    //
    // Step 1: let c_string = CString::new(s).expect("string contains interior null byte");
    // Step 2: let ptr = c_string.as_ptr();
    //         ^^^^^^^^ IMPORTANT: c_string must be bound to a variable!
    //         If you inline it as CString::new(s).unwrap().as_ptr(), the CString
    //         is dropped immediately and the pointer dangles. This is a classic
    //         FFI lifetime bug.
    // Step 3: unsafe { libc::strlen(ptr) }
    //
    // Note: libc::strlen counts bytes until the null terminator.
    // For ASCII strings, this equals the number of characters.
    // For multi-byte UTF-8 (e.g., "café"), it returns byte count (5, not 4).
    todo!("Exercise 3a: Call libc::strlen on a Rust string via CString")
}

fn demo_libc_strlen() {
    assert_eq!(rust_strlen("hello"), 5);
    assert_eq!(rust_strlen(""), 0);
    assert_eq!(rust_strlen("hello world"), 11);

    println!("[3a] rust_strlen(\"hello\") = {}", rust_strlen("hello"));
    println!("[3a] rust_strlen(\"\") = {}", rust_strlen(""));
    println!("[3a] rust_strlen(\"hello world\") = {}", rust_strlen("hello world"));
}

// ─── Exercise 3b: libc::qsort ──────────────────────────────────────

/// Sort an array of i32 using C's `qsort`.
///
/// # What to implement
///
/// Call `libc::qsort` to sort a mutable slice of `i32` in ascending order.
/// You must provide a comparison callback function with `extern "C"` linkage.
///
/// # Why this matters
///
/// `qsort` demonstrates THREE FFI concepts at once:
///
/// 1. **Passing Rust data to C**: The slice's `.as_mut_ptr()` gives C a raw pointer
///    to Rust-owned memory. C reads/writes this memory directly.
///
/// 2. **Function pointers as callbacks**: C's `qsort` takes a comparison function
///    pointer. The callback MUST be `extern "C"` because C will call it using
///    the C calling convention. A Rust closure won't work (wrong ABI + captures).
///
/// 3. **`*const c_void` (void pointers)**: C's generic programming uses `void*`
///    to erase types. The callback receives `*const c_void` which you must cast
///    back to `*const i32`.
///
/// # Hints
///
/// The `qsort` signature is:
/// ```c
/// void qsort(void *base, size_t nmemb, size_t size,
///            int (*compar)(const void *, const void *));
/// ```
///
/// In Rust via libc:
/// ```rust
/// libc::qsort(
///     base,      // *mut c_void — pointer to array start
///     nmemb,     // usize — number of elements
///     size,      // usize — size of each element in bytes
///     compar,    // comparison function pointer
/// );
/// ```
///
/// The comparison callback:
/// - Receives two `*const c_void` pointers (type-erased pointers to elements)
/// - Must cast them to `*const i32` (since we know the array contains i32)
/// - Returns negative (a < b), zero (a == b), or positive (a > b)
/// - MUST be `extern "C"` — C calls this function, so it must use C's ABI
///
/// # What would break
///
/// - Forgetting `extern "C"` on the callback → wrong calling convention → stack corruption
/// - Casting to the wrong type (e.g., `*const u64` instead of `*const i32`) → reads wrong bytes
/// - Panicking inside the callback → unwinding across FFI boundary → UB
///   (wrap in `std::panic::catch_unwind` for production code)
fn sort_with_qsort(slice: &mut [i32]) {
    // TODO(human): Sort the slice using libc::qsort.
    //
    // Step 1: Define the comparison function:
    //
    //   extern "C" fn compare_i32(a: *const c_void, b: *const c_void) -> libc::c_int {
    //       unsafe {
    //           let a = *(a as *const i32);
    //           let b = *(b as *const i32);
    //           // Return negative if a < b, zero if a == b, positive if a > b
    //           // A common pattern: a.cmp(&b) as libc::c_int
    //           // Or simply: a - b (but beware overflow for large values!)
    //       }
    //   }
    //
    // Step 2: Call libc::qsort:
    //
    //   unsafe {
    //       libc::qsort(
    //           slice.as_mut_ptr() as *mut c_void,      // base pointer
    //           slice.len(),                              // number of elements
    //           std::mem::size_of::<i32>(),              // size of each element
    //           Some(compare_i32),                       // comparison callback
    //       );
    //   }
    //
    // Why `Some(compare_i32)`? libc declares the function pointer parameter as
    // `Option<unsafe extern "C" fn(...)>` because C allows NULL function pointers.
    // We wrap our function in Some() to indicate it's non-null.
    //
    // Why `as *mut c_void`? qsort is generic over element type (like a template).
    // It uses void* to accept any array. We cast our typed pointer to void*.
    todo!("Exercise 3b: Sort a slice using libc::qsort with an extern C callback")
}

fn demo_libc_qsort() {
    let mut data = [42, 17, 99, 3, 55, 8, 71];
    println!("[3b] Before qsort: {:?}", data);

    sort_with_qsort(&mut data);
    assert_eq!(data, [3, 8, 17, 42, 55, 71, 99]);
    println!("[3b] After qsort:  {:?}", data);

    // Edge case: empty slice
    let mut empty: [i32; 0] = [];
    sort_with_qsort(&mut empty);
    println!("[3b] Empty slice sorted: {:?}", empty);

    // Edge case: single element
    let mut single = [1];
    sort_with_qsort(&mut single);
    assert_eq!(single, [1]);
    println!("[3b] Single element sorted: {:?}", single);
}
