//! Exercise 7: Capstone — Bidirectional FFI
//!
//! This capstone exercise combines everything from exercises 1-6:
//! - Exposing Rust functions to C (extern "C" + #[unsafe(no_mangle)])
//! - Calling C from Rust (extern "C" blocks)
//! - Ownership transfer across the FFI boundary (Box::into_raw / from_raw)
//! - String conversion in both directions (CString, CStr)
//! - repr(C) structs shared between languages
//!
//! The scenario: Build a "text processor" where:
//! - Rust exposes a string processing function that C could call
//! - Rust calls C functions for numerical processing
//! - Ownership of heap-allocated data is carefully managed
//!
//! In a real project, this pattern appears when:
//! - You're wrapping a C library with a safe Rust API
//! - You're exposing a Rust library to C/Python/other languages
//! - You're gradually migrating a C codebase to Rust (the Firefox pattern)

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// A Rust struct that will be exposed to C via opaque pointer.
///
/// C code will hold a `*mut TextProcessor` but never access its fields
/// directly — all operations go through Rust functions. This is the
/// "opaque pointer" pattern, which is the safest way to expose Rust
/// objects to C.
///
/// Note: This struct does NOT need `#[repr(C)]` because C never reads
/// its fields. C only holds a pointer to it and passes that pointer
/// back to Rust functions. The internal layout is Rust's business.
pub struct TextProcessor {
    prefix: String,
    processed_count: u32,
}

unsafe extern "C" {
    fn mathlib_add(a: i32, b: i32) -> i32;
    fn mathlib_multiply(a: i32, b: i32) -> i32;
}

pub fn run() {
    demo_rust_exports();
    demo_opaque_pointer();
    demo_round_trip();
}

// ─── Exercise 7a: Expose Rust Functions to C ────────────────────────

/// Process a C string in Rust: receive *const c_char, return *mut c_char.
///
/// # What to implement
///
/// Write a function with `extern "C"` + `#[unsafe(no_mangle)]` that:
/// 1. Receives a `*const c_char` (C string pointer)
/// 2. Converts to Rust `&str` (via CStr)
/// 3. Transforms the string (convert to uppercase)
/// 4. Returns a new `*mut c_char` (heap-allocated C string)
///
/// # Why this matters
///
/// This is how you expose Rust functionality to C (or Python via ctypes,
/// or any other language that can call C functions). The key attributes:
///
/// - `extern "C"` — use C calling convention (so C/Python/etc. can call it)
/// - `#[unsafe(no_mangle)]` — preserve the function name (no Rust name mangling)
///
/// Together, these make the function visible as a regular C symbol in the
/// compiled library. You could compile this crate as a `cdylib` and load
/// it from Python: `ctypes.cdll.LoadLibrary("mylib.dll")`
///
/// # Hints
///
/// - Check for null: `if input.is_null() { return std::ptr::null_mut(); }`
/// - Convert to Rust string: `CStr::from_ptr(input).to_str().unwrap()`
/// - Process: `let upper = rust_str.to_uppercase();`
/// - Convert back to C string: `CString::new(upper).unwrap()`
/// - Return as raw pointer: `c_string.into_raw()` — this LEAKS the memory intentionally!
///   The caller (C code) is responsible for calling our `rust_free_string` to free it.
///
/// # What would break
///
/// - Missing `#[unsafe(no_mangle)]` → C can't find the function (linker error)
/// - Missing `extern "C"` → wrong calling convention → stack corruption
/// - Forgetting to handle the null case → null dereference in CStr::from_ptr → UB
/// - Returning a pointer to a local CString → CString is dropped, pointer dangles
///   (that's why we use `.into_raw()` which prevents the CString from being dropped)
///
/// # SAFETY
///
/// This function is `unsafe` from C's perspective (C has no safety guarantees).
/// The Rust side handles safety internally:
/// - Null check on input
/// - CStr::from_ptr is unsafe (we verify non-null first, assume valid C string)
/// - CString::into_raw transfers ownership to the caller
#[unsafe(no_mangle)]
pub extern "C" fn rust_process_string(input: *const c_char) -> *mut c_char {
    // TODO(human): Convert C string to Rust, process it, return as new C string.
    //
    // Step 1: Null check
    //   if input.is_null() {
    //       return std::ptr::null_mut();
    //   }
    //
    // Step 2: Convert to Rust string
    //   let c_str = unsafe { CStr::from_ptr(input) };
    //   let rust_str = c_str.to_str().unwrap_or("(invalid UTF-8)");
    //
    // Step 3: Process (uppercase)
    //   let processed = rust_str.to_uppercase();
    //
    // Step 4: Convert back to C string and return raw pointer
    //   match CString::new(processed) {
    //       Ok(c_string) => c_string.into_raw(),
    //       Err(_) => std::ptr::null_mut(),  // processed string contained a null byte
    //   }
    //
    // CRITICAL: `into_raw()` transfers ownership of the CString's heap allocation
    // to the caller. Rust will NOT free this memory. The caller MUST call
    // `rust_free_string()` to reclaim it. This is the standard pattern for
    // Rust functions that return heap-allocated data to C.
    todo!("Exercise 7a: Expose a Rust string processing function with extern C")
}

/// Free a string that was allocated by `rust_process_string`.
///
/// # What to implement
///
/// Reclaim ownership of a `*mut c_char` that was created by `CString::into_raw()`.
///
/// # Why this matters
///
/// Every `into_raw()` MUST have a matching deallocation function. This is the
/// **ownership contract** of FFI: whoever allocates must provide a way to free.
///
/// Without this function, every call to `rust_process_string` leaks memory.
/// The C caller cannot use `free()` because the memory was allocated by Rust's
/// allocator (not C's malloc). Mixing allocators is undefined behavior.
///
/// # Hints
///
/// - `CString::from_raw(ptr)` reclaims ownership (unsafe)
/// - The CString is immediately dropped, which frees the memory
/// - Check for null first — `from_raw(null)` is UB
/// - Alternatively: `unsafe { drop(CString::from_raw(ptr)); }`
///   The explicit `drop` makes the intent clear
///
/// # What would break
///
/// - Calling this with a pointer NOT from `into_raw()` → double-free or heap corruption
/// - Calling this with a NULL pointer → UB (from_raw requires non-null)
/// - Calling `libc::free()` instead → wrong allocator → heap corruption
/// - Calling this twice on the same pointer → double-free → heap corruption
#[unsafe(no_mangle)]
pub extern "C" fn rust_free_string(ptr: *mut c_char) {
    // TODO(human): Reclaim ownership and free the CString.
    //
    // Step 1: Check for null
    //   if ptr.is_null() { return; }
    //
    // Step 2: Reclaim ownership (this frees the memory when dropped)
    //   unsafe { drop(CString::from_raw(ptr)); }
    //
    // After this call, the pointer is invalid. Any further use by C is
    // use-after-free (UB). In a real API, you'd document this clearly.
    //
    // The `drop()` call is technically redundant (Rust drops at end of scope),
    // but it makes the intent explicit: "we're reclaiming this to free it."
    todo!("Exercise 7b: Free a CString that was returned to C via into_raw()")
}

fn demo_rust_exports() {
    // Simulate C calling our Rust function:
    // In a real scenario, C code would call rust_process_string and rust_free_string.
    // Here we test the round trip from Rust for verification.

    let input = CString::new("hello from c").unwrap();
    let result_ptr = rust_process_string(input.as_ptr());

    assert!(!result_ptr.is_null());
    let result = unsafe { CStr::from_ptr(result_ptr) }
        .to_str()
        .unwrap()
        .to_owned();
    println!("[7a] rust_process_string(\"hello from c\") = \"{}\"", result);
    assert_eq!(result, "HELLO FROM C");

    // Free the result (this is what C would call)
    rust_free_string(result_ptr);
    println!("[7a] rust_free_string called — memory freed");

    // Test null input
    let null_result = rust_process_string(std::ptr::null());
    assert!(null_result.is_null());
    println!("[7a] rust_process_string(NULL) = NULL (correct)");
}

// ─── Exercise 7b: Opaque Pointer Pattern ────────────────────────────

/// Create a TextProcessor and expose it as an opaque pointer.
///
/// # What to implement
///
/// Three functions that form a complete lifecycle API:
/// 1. `processor_new` — allocate and return an opaque pointer
/// 2. `processor_process` — use the processor through its pointer
/// 3. `processor_free` — deallocate the processor
///
/// # Why this matters
///
/// The "opaque pointer" pattern is the safest way to expose Rust objects to C:
///
/// ```text
/// C side:   struct TextProcessor;                // opaque — no field access
///           TextProcessor* p = processor_new();  // create
///           processor_process(p, "hello");       // use
///           processor_free(p);                   // destroy
/// ```
///
/// C never sees inside the struct — it's a black box. All operations go through
/// Rust functions that handle the raw pointer safely. This is how many Rust
/// libraries expose themselves to C: hyper, rusqlite, etc.
///
/// The pattern uses `Box::into_raw` / `Box::from_raw` for ownership transfer:
/// - `Box::new(value)` → allocates on the heap
/// - `Box::into_raw(boxed)` → gives C a raw pointer (Rust "forgets" the allocation)
/// - `Box::from_raw(ptr)` → Rust reclaims ownership (destructor runs when dropped)
///
/// # Hints
///
/// For `processor_new`:
/// - `Box::into_raw(Box::new(TextProcessor { ... }))`
///
/// For `processor_process`:
/// - Check for null: `if processor.is_null() { return std::ptr::null_mut(); }`
/// - Convert to reference: `let proc = unsafe { &mut *processor };`
/// - Use the processor (process the string, increment counter)
/// - Return result as `CString::into_raw()`
///
/// For `processor_free`:
/// - Check for null
/// - `unsafe { drop(Box::from_raw(processor)); }`
///
/// # What would break
///
/// - Calling `processor_process` after `processor_free` → use-after-free (UB)
/// - Calling `processor_free` twice → double-free (UB)
/// - Forgetting `processor_free` entirely → memory leak
/// - Using `&*processor` (shared ref) when you need `&mut *processor` → can't mutate
#[unsafe(no_mangle)]
pub extern "C" fn processor_new(prefix: *const c_char) -> *mut TextProcessor {
    // TODO(human): Create a TextProcessor and return it as an opaque raw pointer.
    //
    // Step 1: Convert the C string prefix to a Rust String
    //   let prefix_str = if prefix.is_null() {
    //       String::from("DEFAULT")
    //   } else {
    //       unsafe { CStr::from_ptr(prefix) }
    //           .to_str()
    //           .unwrap_or("DEFAULT")
    //           .to_owned()
    //   };
    //
    // Step 2: Create the processor on the heap and return raw pointer
    //   let processor = TextProcessor {
    //       prefix: prefix_str,
    //       processed_count: 0,
    //   };
    //   Box::into_raw(Box::new(processor))
    //
    // Box::into_raw does two things:
    // 1. Takes ownership of the Box (heap allocation)
    // 2. Returns the raw pointer WITHOUT deallocating
    // Rust "forgets" this allocation — it won't be dropped automatically.
    // The caller MUST eventually pass this pointer to processor_free.
    todo!("Exercise 7b-new: Allocate TextProcessor and return opaque pointer")
}

/// Process a string using the TextProcessor.
///
/// Returns a new heap-allocated C string. Caller must free with `rust_free_string`.
#[unsafe(no_mangle)]
pub extern "C" fn processor_process(
    processor: *mut TextProcessor,
    input: *const c_char,
) -> *mut c_char {
    // TODO(human): Use the processor to transform the input string.
    //
    // Step 1: Null checks
    //   if processor.is_null() || input.is_null() {
    //       return std::ptr::null_mut();
    //   }
    //
    // Step 2: Convert raw pointer to mutable reference (unsafe!)
    //   let proc = unsafe { &mut *processor };
    //   This is one of the most dangerous operations: we're asserting that
    //   the pointer is valid, properly aligned, and not aliased. If C passed
    //   a freed pointer, this is use-after-free UB.
    //
    // Step 3: Convert input C string to Rust &str
    //   let input_str = unsafe { CStr::from_ptr(input) }
    //       .to_str()
    //       .unwrap_or("(invalid)");
    //
    // Step 4: Process the string (prefix + uppercase input)
    //   let result = format!("[{}] {}", proc.prefix, input_str.to_uppercase());
    //   proc.processed_count += 1;
    //
    // Step 5: Return as C string
    //   CString::new(result).unwrap().into_raw()
    //
    // The returned pointer must be freed by the caller using rust_free_string.
    // This is the ownership contract: Rust allocates, Rust must free (through
    // the provided free function).
    todo!("Exercise 7b-process: Process a string through the opaque TextProcessor")
}

/// Free a TextProcessor that was created by `processor_new`.
#[unsafe(no_mangle)]
pub extern "C" fn processor_free(processor: *mut TextProcessor) {
    // TODO(human): Reclaim ownership of the TextProcessor and free it.
    //
    // Step 1: if processor.is_null() { return; }
    //
    // Step 2: unsafe { drop(Box::from_raw(processor)); }
    //
    // Box::from_raw reclaims ownership of the heap allocation.
    // When the Box is dropped, it:
    //   1. Runs the destructor for TextProcessor (drops the String field)
    //   2. Deallocates the heap memory
    //
    // After this call, the processor pointer is INVALID. Any further use
    // (processor_process with this pointer) is use-after-free UB.
    //
    // In a production API, consider using a generation counter or a flag
    // to detect double-free attempts (though this adds overhead).
    todo!("Exercise 7b-free: Deallocate the opaque TextProcessor")
}

fn demo_opaque_pointer() {
    // Create processor with prefix
    let prefix = CString::new("RUST").unwrap();
    let processor = processor_new(prefix.as_ptr());
    assert!(!processor.is_null());
    println!("[7b] Created TextProcessor with prefix \"RUST\"");

    // Process two strings
    let input1 = CString::new("hello world").unwrap();
    let result1_ptr = processor_process(processor, input1.as_ptr());
    assert!(!result1_ptr.is_null());
    let result1 = unsafe { CStr::from_ptr(result1_ptr) }.to_str().unwrap().to_owned();
    println!("[7b] processor_process(\"hello world\") = \"{}\"", result1);
    rust_free_string(result1_ptr);

    let input2 = CString::new("goodbye").unwrap();
    let result2_ptr = processor_process(processor, input2.as_ptr());
    assert!(!result2_ptr.is_null());
    let result2 = unsafe { CStr::from_ptr(result2_ptr) }.to_str().unwrap().to_owned();
    println!("[7b] processor_process(\"goodbye\") = \"{}\"", result2);
    rust_free_string(result2_ptr);

    // Check processed count
    let count = unsafe { (*processor).processed_count };
    assert_eq!(count, 2);
    println!("[7b] processed_count = {}", count);

    // Free the processor
    processor_free(processor);
    println!("[7b] processor_free called — TextProcessor deallocated");
}

// ─── Exercise 7c: Full Round Trip ───────────────────────────────────

/// Demonstrate a complete round trip: Rust data → C processing → Rust result.
///
/// # What to implement
///
/// Combine C function calls with Rust processing:
/// 1. Use C's `mathlib_add` and `mathlib_multiply` for arithmetic
/// 2. Use `rust_process_string` for string processing
/// 3. Manage all allocations correctly (no leaks, no dangling pointers)
///
/// This simulates a real-world scenario where you mix Rust and C functionality
/// in a single workflow, carefully managing the ownership boundary.
///
/// # Why this matters
///
/// In production FFI code, you rarely use just one direction. A typical workflow:
/// 1. Prepare data in Rust
/// 2. Call C library for specialized processing (math, crypto, compression)
/// 3. Receive results back in Rust
/// 4. Process further in Rust
/// 5. Optionally expose results back to C
///
/// Getting the ownership right at each step is the primary challenge.
fn round_trip_demo() {
    // TODO(human): Build a round trip that combines Rust and C operations.
    //
    // Step 1: Use C for arithmetic
    //   let sum = unsafe { mathlib_add(10, 20) };
    //   let product = unsafe { mathlib_multiply(sum, 3) };
    //   println!("C arithmetic: (10 + 20) * 3 = {}", product);
    //
    // Step 2: Format the result as a string in Rust
    //   let message = format!("Result is {}", product);
    //
    // Step 3: Process the string through our Rust-exported function
    //   (simulating C calling our Rust function)
    //   let c_message = CString::new(message).unwrap();
    //   let processed_ptr = rust_process_string(c_message.as_ptr());
    //
    // Step 4: Read back the processed result
    //   if !processed_ptr.is_null() {
    //       let processed = unsafe { CStr::from_ptr(processed_ptr) }
    //           .to_str().unwrap().to_owned();
    //       println!("Processed: {}", processed);
    //       rust_free_string(processed_ptr);  // Don't forget to free!
    //   }
    //
    // Step 5: Verify everything was cleaned up
    //   println!("Round trip complete — all memory freed, no leaks");
    //
    // In a real project, you'd wrap steps 3-4 in a safe Rust function that
    // handles the CString conversion, null checks, and freeing internally.
    // Callers would just see: fn process(input: &str) -> String
    todo!("Exercise 7c: Full round trip combining Rust and C operations")
}

fn demo_round_trip() {
    round_trip_demo();
}
