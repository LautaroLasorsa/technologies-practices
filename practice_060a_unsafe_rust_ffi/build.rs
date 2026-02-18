//! Build script for the unsafe FFI practice.
//!
//! This file runs BEFORE the rest of the crate compiles. It does two things:
//!
//! 1. **Compiles C source files** into a static library using the `cc` crate.
//!    The `cc` crate auto-detects the system C compiler (MSVC on Windows,
//!    gcc/clang on Linux/macOS) and produces a `.lib`/`.a` that Cargo links
//!    into the final Rust binary.
//!
//! 2. **Generates Rust FFI bindings** from the C header using `bindgen` (if enabled).
//!    When the `bindgen` feature is active, it parses `c_src/mathlib.h` and produces
//!    Rust `extern "C"` declarations in `$OUT_DIR/bindings.rs`.
//!    When bindgen is NOT available, it writes pre-written fallback bindings instead.
//!
//! The generated bindings are included in the Rust source via:
//!     include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

fn main() {
    compile_c_library();
    generate_bindings();
}

/// Compile c_src/mathlib.c into a static library named "mathlib".
///
/// The `cc` crate handles:
/// - Finding the C compiler (cl.exe on MSVC, gcc on Linux, clang on macOS)
/// - Setting appropriate flags (-O2, warning flags, etc.)
/// - Producing a static library (.lib on Windows, .a on Unix)
/// - Telling Cargo to link the library into the final binary
///
/// After this runs, any `extern "C"` function declared in Rust that matches
/// a symbol in mathlib.c will be resolved by the linker.
fn compile_c_library() {
    cc::Build::new()
        .file("c_src/mathlib.c")
        // Tell the C compiler where to find mathlib.h when mathlib.c does #include "mathlib.h"
        .include("c_src")
        // Enable warnings so we catch C issues early
        .warnings(true)
        // Name the output library "mathlib" (produces mathlib.lib or libmathlib.a)
        .compile("mathlib");

    // Tell Cargo to re-run this build script if any C source file changes.
    // Without this, modifying mathlib.c would NOT trigger recompilation.
    println!("cargo:rerun-if-changed=c_src/mathlib.c");
    println!("cargo:rerun-if-changed=c_src/mathlib.h");
}

/// Generate Rust FFI bindings from c_src/mathlib.h.
///
/// When the `bindgen` feature is enabled, this uses bindgen to parse the C header
/// using libclang and produce Rust declarations. When bindgen is not available,
/// it writes pre-written fallback bindings that are equivalent to what bindgen
/// would generate.
///
/// The output goes to `$OUT_DIR/bindings.rs`, which Exercise 6 includes via
/// `include!(concat!(env!("OUT_DIR"), "/bindings.rs"))`.
fn generate_bindings() {
    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let bindings_file = out_path.join("bindings.rs");

    #[cfg(feature = "bindgen")]
    {
        // ── bindgen path: auto-generate from C header ──────────────────
        //
        // bindgen parses the C header using libclang and produces a Rust file with:
        // - `extern "C"` function declarations for each C function
        // - `#[repr(C)]` struct definitions for each C struct/typedef
        // - `pub const` values for each `#define` macro (if it's a simple constant)
        //
        // REQUIREMENT: libclang (part of LLVM) must be installed.
        // On Windows: install LLVM and set LIBCLANG_PATH to the LLVM bin/ directory.
        let bindings = bindgen::Builder::default()
            .header("c_src/mathlib.h")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate_comments(true)
            .allowlist_function("mathlib_.*")
            .allowlist_type("Point3D")
            .allowlist_type("StatResult")
            .allowlist_var("MATHLIB_.*")
            .derive_debug(true)
            .derive_copy(true)
            .generate()
            .expect("bindgen failed to generate bindings — is libclang/LLVM installed?");

        bindings
            .write_to_file(&bindings_file)
            .expect("Failed to write bindings.rs");

        println!("cargo:warning=bindgen: generated bindings from c_src/mathlib.h");
    }

    #[cfg(not(feature = "bindgen"))]
    {
        // ── Fallback path: write pre-written bindings ──────────────────
        //
        // These are equivalent to what bindgen would generate from mathlib.h.
        // Used when LLVM/libclang is not installed.
        //
        // In a real project, you would:
        // 1. Generate bindings once on a machine with LLVM
        // 2. Check the generated file into version control
        // 3. Use the checked-in file as the fallback
        //
        // This approach is used by many crates (e.g., libgit2-sys has
        // pre-generated bindings for platforms where bindgen doesn't work).
        let fallback = r#"
/* Fallback bindings — equivalent to what bindgen would generate from mathlib.h.
 * Generated manually to match the C header declarations.
 * If you have LLVM installed, use `cargo build --features bindgen` to
 * auto-generate these from the C header instead.
 */

pub const MATHLIB_MAX_DIM: u32 = 1024;
pub const MATHLIB_VERSION: u32 = 1;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct StatResult {
    pub value: f64,
    pub error_code: i32,
}

unsafe extern "C" {
    pub fn mathlib_add(a: i32, b: i32) -> i32;
    pub fn mathlib_multiply(a: i32, b: i32) -> i32;
    pub fn mathlib_dot_product(a: *const f64, b: *const f64, len: usize) -> f64;
    pub fn mathlib_sum_array(arr: *const i32, len: usize) -> i64;
    pub fn mathlib_format_vector(arr: *const f64, len: usize) -> *mut std::os::raw::c_char;
    pub fn mathlib_point_magnitude(p: *const Point3D) -> f64;
    pub fn mathlib_point_add(a: Point3D, b: Point3D) -> Point3D;
    pub fn mathlib_mean(arr: *const f64, len: usize) -> StatResult;
    pub fn mathlib_transform_array(
        arr: *mut f64,
        len: usize,
        transform: ::std::option::Option<unsafe extern "C" fn(f64) -> f64>,
    );
}
"#;
        std::fs::write(&bindings_file, fallback).expect("Failed to write fallback bindings.rs");
        println!("cargo:warning=bindgen: using fallback bindings (LLVM not required)");
        println!("cargo:warning=bindgen: for auto-generated bindings, use: cargo build --features bindgen");
    }
}
