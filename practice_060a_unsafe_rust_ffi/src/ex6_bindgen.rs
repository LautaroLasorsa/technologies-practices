//! Exercise 6: bindgen Auto-Generated Bindings
//!
//! This exercise uses bindgen-generated Rust declarations instead of
//! hand-written `extern "C"` blocks. Compare with Exercise 4 to appreciate
//! what bindgen automates.
//!
//! Key concepts:
//! - `include!()` pulls in the generated `bindings.rs` at compile time
//! - bindgen auto-generates: extern fn declarations, repr(C) structs, #define constants
//! - The generated code may have different naming conventions (e.g., no Rust-style snake_case)
//! - bindgen-generated types are equivalent to hand-written ones but less ergonomic
//!
//! Why this matters:
//! For any non-trivial C library (hundreds of functions, complex structs),
//! hand-writing FFI declarations is impractical and error-prone. bindgen
//! eliminates this entire class of bugs — the declarations are guaranteed
//! to match the C header because they're generated FROM the header.
//!
//! PREREQUISITE: This exercise requires LLVM/libclang to be installed for
//! bindgen to work. If `cargo build` fails with a bindgen error, you can
//! still complete exercises 1-5 and 7 using the hand-written declarations.
//!
//! On Windows: Install LLVM from https://releases.llvm.org/ and set
//! LIBCLANG_PATH=C:\Program Files\LLVM\bin (adjust path as needed).

/// Include the auto-generated bindings from build.rs.
///
/// This file is generated at compile time by bindgen in build.rs.
/// It contains:
/// - `extern "C"` function declarations for all mathlib_* functions
/// - `#[repr(C)]` struct definitions for Point3D and StatResult
/// - `pub const MATHLIB_MAX_DIM: ...` and `pub const MATHLIB_VERSION: ...`
///
/// The `#[allow(...)` attributes suppress warnings about bindgen's naming
/// conventions (C uses snake_case for structs, which Rust warns about).
#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(dead_code)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub fn run() {
    demo_bindgen_call_add();
    demo_bindgen_use_struct();
    demo_bindgen_use_constant();
}

// ─── Exercise 6a: Call C Function via bindgen Bindings ──────────────

/// Call mathlib_add using bindgen-generated declarations.
///
/// # What to implement
///
/// Same as Exercise 4a, but using `bindings::mathlib_add` instead of
/// hand-written declarations. The function signature is identical —
/// bindgen generated it from the same C header.
///
/// # Why this matters
///
/// Compare this with Exercise 4a. The code is nearly identical, but the
/// key difference is WHERE the `extern "C"` declaration comes from:
/// - Exercise 4: hand-written in Rust (error-prone, must match C manually)
/// - Exercise 6: auto-generated from C header (guaranteed to match)
///
/// For a library with 5 functions, hand-writing is fine. For a library with
/// 500 functions (OpenSSL, SQLite, etc.), bindgen is essential.
///
/// # Hints
///
/// - `unsafe { bindings::mathlib_add(a, b) }`
/// - The function lives in the `bindings` module because of how we included it
fn bindgen_call_add(a: i32, b: i32) -> i32 {
    // TODO(human): Call mathlib_add using the bindgen-generated binding.
    //
    // unsafe { bindings::mathlib_add(a, b) }
    //
    // That's it. The call is identical to Exercise 4a, just namespaced
    // under `bindings::`. bindgen generated the exact same extern "C"
    // declaration that you wrote by hand in Exercise 4.
    //
    // Open the generated file to compare:
    //   target/debug/build/unsafe-ffi-practice-*/out/bindings.rs
    // You'll see something like:
    //   extern "C" { pub fn mathlib_add(a: i32, b: i32) -> i32; }
    // Which is exactly what you wrote in ex4_manual_ffi.rs.
    todo!("Exercise 6a: Call C function using bindgen-generated binding")
}

fn demo_bindgen_call_add() {
    let result = bindgen_call_add(100, 200);
    assert_eq!(result, 300);
    println!("[6a] bindgen::mathlib_add(100, 200) = {}", result);
}

// ─── Exercise 6b: Use bindgen-Generated Struct ──────────────────────

/// Use bindgen's auto-generated Point3D struct with C functions.
///
/// # What to implement
///
/// 1. Create a `bindings::Point3D` struct
/// 2. Pass it to `bindings::mathlib_point_magnitude`
/// 3. Call `bindings::mathlib_point_add` with two points
///
/// # Why this matters
///
/// bindgen generates `#[repr(C)]` structs automatically from C typedef/struct
/// declarations. The generated struct is guaranteed to have:
/// - Correct field order (matching C declaration order)
/// - Correct field types (C `double` → Rust `f64`)
/// - Correct alignment and padding
/// - `#[repr(C)]` attribute
///
/// With hand-written structs (Exercise 5), a field reorder or type mismatch
/// is a silent bug. With bindgen, the struct is derived from the C header,
/// so mismatches are impossible (as long as the header is correct).
///
/// # Hints
///
/// - `bindings::Point3D { x: 3.0, y: 4.0, z: 0.0 }`
/// - `unsafe { bindings::mathlib_point_magnitude(&point) }`
/// - `unsafe { bindings::mathlib_point_add(a, b) }`
/// - The bindgen-generated Point3D might have slightly different derives
///   than your hand-written one (check the generated file)
fn bindgen_use_struct() -> (f64, bindings::Point3D) {
    // TODO(human): Create bindgen::Point3D structs and pass to C functions.
    //
    // Step 1: Create two points using bindgen's generated struct type:
    //   let a = bindings::Point3D { x: 3.0, y: 4.0, z: 0.0 };
    //   let b = bindings::Point3D { x: 1.0, y: 1.0, z: 1.0 };
    //
    // Step 2: Compute magnitude of point `a`:
    //   let magnitude = unsafe { bindings::mathlib_point_magnitude(&a) };
    //
    // Step 3: Add the two points:
    //   let sum = unsafe { bindings::mathlib_point_add(a, b) };
    //
    // Step 4: Return (magnitude, sum)
    //
    // The bindgen-generated Point3D is a different Rust type than the
    // hand-written one in ex5_repr_c_structs.rs, even though they have
    // identical memory layout. Rust's type system treats them as distinct.
    // To convert between them, you'd use transmute (same layout) or
    // field-by-field copy. In practice, pick one source of truth.
    todo!("Exercise 6b: Use bindgen-generated struct with C functions")
}

fn demo_bindgen_use_struct() {
    let (magnitude, sum) = bindgen_use_struct();
    assert!((magnitude - 5.0).abs() < 1e-10);
    println!("[6b] magnitude via bindgen struct = {:.6}", magnitude);
    println!(
        "[6b] point_add via bindgen = ({:.1}, {:.1}, {:.1})",
        sum.x, sum.y, sum.z
    );
}

// ─── Exercise 6c: Use bindgen-Generated Constants ───────────────────

/// Access C `#define` constants that bindgen converted to Rust `const`.
///
/// # What to implement
///
/// Read the values of `MATHLIB_MAX_DIM` and `MATHLIB_VERSION` from the
/// bindgen-generated bindings and return them.
///
/// # Why this matters
///
/// C libraries define configuration through `#define` macros:
/// ```c
/// #define MATHLIB_MAX_DIM 1024
/// #define MATHLIB_VERSION 1
/// ```
///
/// bindgen converts simple integer `#define` macros to Rust `pub const` values:
/// ```rust
/// pub const MATHLIB_MAX_DIM: u32 = 1024;
/// pub const MATHLIB_VERSION: u32 = 1;
/// ```
///
/// This is safe (no `unsafe` needed) — they're just compile-time constants.
/// But note: bindgen can only convert SIMPLE `#define` macros (integer literals,
/// simple expressions). Complex macros (involving casts, function calls, or
/// token pasting) are silently skipped — you'll need to define those manually.
///
/// # Hints
///
/// - `bindings::MATHLIB_MAX_DIM` — the constant value
/// - `bindings::MATHLIB_VERSION` — the version number
/// - These are `u32` (bindgen's default for integer #define without suffix)
/// - No `unsafe` needed — these are plain Rust constants, not FFI calls
fn bindgen_use_constant() -> (u32, u32) {
    // TODO(human): Return the values of MATHLIB_MAX_DIM and MATHLIB_VERSION.
    //
    // (bindings::MATHLIB_MAX_DIM, bindings::MATHLIB_VERSION)
    //
    // No unsafe needed! bindgen converted the C #define macros to plain
    // Rust constants. This is purely a compile-time operation.
    //
    // Open the generated bindings.rs to verify the types and values.
    // Note that bindgen defaults to u32 for #define integer constants
    // (since C's preprocessor integer type is `int` = 32-bit on most platforms).
    todo!("Exercise 6c: Access bindgen-generated constants from C #define macros")
}

fn demo_bindgen_use_constant() {
    let (max_dim, version) = bindgen_use_constant();
    assert_eq!(max_dim, 1024);
    assert_eq!(version, 1);
    println!("[6c] MATHLIB_MAX_DIM = {}", max_dim);
    println!("[6c] MATHLIB_VERSION = {}", version);
}
