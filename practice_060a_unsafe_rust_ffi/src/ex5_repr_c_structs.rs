//! Exercise 5: repr(C) Structs & Complex Data Across FFI
//!
//! This exercise teaches sharing structured data between Rust and C
//! using `#[repr(C)]` to guarantee memory layout compatibility.
//!
//! Key concepts:
//! - `#[repr(C)]` forces C-compatible struct layout (field order, padding, alignment)
//! - Without repr(C), Rust may reorder fields → C reads garbage
//! - Passing structs by pointer vs by value across FFI
//! - Passing arrays of structs (Vec<T> → *const T + len)
//! - Receiving structs from C (return by value)
//!
//! Why this matters:
//! Real-world FFI rarely passes only integers. C APIs use structs extensively
//! (OpenGL vertices, network packet headers, database rows, OS structures).
//! Getting the layout right is critical — a layout mismatch is silent UB.

/// A 3D point — must match C's `Point3D` typedef in mathlib.h.
///
/// `#[repr(C)]` guarantees the layout matches C's struct:
/// - Fields are in declaration order (x, y, z)
/// - Alignment of each field follows C rules (f64 = 8-byte aligned)
/// - Padding between fields matches C (no padding here since all f64)
/// - Total size = 24 bytes, alignment = 8 bytes
///
/// WITHOUT `#[repr(C)]`, Rust could reorder fields (e.g., z, x, y) for
/// optimization. The C function would then read `z` when it expects `x`,
/// producing wrong results with no error — silent data corruption.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Result of a statistical computation — must match C's `StatResult`.
///
/// Note the mixed types: f64 (8 bytes) + i32 (4 bytes). On most platforms,
/// the C compiler inserts 4 bytes of padding after `error_code` to maintain
/// 8-byte alignment of the struct. `#[repr(C)]` ensures Rust does the same.
///
/// Layout (on 64-bit platforms):
///   offset 0:  value (f64, 8 bytes)
///   offset 8:  error_code (i32, 4 bytes)
///   offset 12: [4 bytes padding to align struct size to 8]
///   total: 16 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StatResult {
    pub value: f64,
    pub error_code: i32,
}

// FFI declarations for struct-based C functions.
// In Rust 2024 edition, extern blocks must be explicitly marked `unsafe`.
unsafe extern "C" {
    // double mathlib_point_magnitude(const Point3D *p);
    fn mathlib_point_magnitude(p: *const Point3D) -> f64;

    // Point3D mathlib_point_add(Point3D a, Point3D b);
    fn mathlib_point_add(a: Point3D, b: Point3D) -> Point3D;

    // StatResult mathlib_mean(const double *arr, size_t len);
    fn mathlib_mean(arr: *const f64, len: usize) -> StatResult;

    // int64_t mathlib_sum_array(const int32_t *arr, size_t len);
    fn mathlib_sum_array(arr: *const i32, len: usize) -> i64;

    // void mathlib_transform_array(double *arr, size_t len,
    //                              double (*transform)(double));
    fn mathlib_transform_array(
        arr: *mut f64,
        len: usize,
        transform: Option<unsafe extern "C" fn(f64) -> f64>,
    );
}

pub fn run() {
    demo_create_and_pass_point();
    demo_receive_struct_from_c();
    demo_array_of_structs_to_c();
}

// ─── Exercise 5a: Pass a repr(C) Struct to C ────────────────────────

/// Create a Point3D in Rust and pass it to C's mathlib_point_magnitude.
///
/// # What to implement
///
/// 1. Create a `Point3D` struct with the given x, y, z values
/// 2. Pass a pointer to it to `mathlib_point_magnitude`
/// 3. Return the magnitude (Euclidean length)
///
/// # Why this matters
///
/// Passing structs by pointer is the most common FFI pattern. The C function
/// receives a `const Point3D*` — a read-only pointer to Rust-owned memory.
/// For this to work, the Rust struct's memory layout MUST exactly match C's.
///
/// `#[repr(C)]` is the critical ingredient. Without it:
/// - Rust might reorder fields → C reads `y` where it expects `x`
/// - Rust might use different padding → C reads misaligned data
/// - The struct size might differ → subsequent fields are at wrong offsets
///
/// The result would be wrong values (not a crash), which is the worst kind
/// of bug — silent data corruption that only manifests downstream.
///
/// # Hints
///
/// - Create the struct: `Point3D { x, y, z }`
/// - Get a pointer: `&point as *const Point3D` (or just `&point` — Rust coerces)
/// - Call: `unsafe { mathlib_point_magnitude(&point) }`
///   (Rust auto-coerces `&Point3D` to `*const Point3D` in extern fn calls)
///
/// # What would break
///
/// Remove `#[repr(C)]` from Point3D and the magnitude calculation would
/// produce wrong results. The C function reads bytes at fixed offsets
/// (0, 8, 16 for x, y, z). If Rust reorders fields, those offsets
/// contain different values.
fn compute_magnitude(x: f64, y: f64, z: f64) -> f64 {
    // TODO(human): Create a Point3D and pass it to C's mathlib_point_magnitude.
    //
    // Step 1: let point = Point3D { x, y, z };
    // Step 2: unsafe { mathlib_point_magnitude(&point) }
    //
    // The & reference is automatically coerced to *const Point3D when passed
    // to an extern "C" function expecting a raw pointer. Rust does this
    // coercion implicitly for FFI calls.
    //
    // The C function computes sqrt(x*x + y*y + z*z) using the struct fields
    // at their expected memory offsets. repr(C) guarantees those offsets match.
    todo!("Exercise 5a: Create a repr(C) struct and pass to C function")
}

fn demo_create_and_pass_point() {
    // magnitude of (3, 4, 0) = sqrt(9+16+0) = 5.0
    let mag = compute_magnitude(3.0, 4.0, 0.0);
    assert!((mag - 5.0).abs() < 1e-10);
    println!("[5a] magnitude(3, 4, 0) = {:.6}", mag);

    // magnitude of (1, 1, 1) = sqrt(3) ≈ 1.732
    let mag2 = compute_magnitude(1.0, 1.0, 1.0);
    assert!((mag2 - 3.0f64.sqrt()).abs() < 1e-10);
    println!("[5a] magnitude(1, 1, 1) = {:.6}", mag2);

    // Verify struct size matches C's expectation
    println!(
        "[5a] size_of::<Point3D>() = {} bytes (expected 24)",
        std::mem::size_of::<Point3D>()
    );
    assert_eq!(std::mem::size_of::<Point3D>(), 24);
}

// ─── Exercise 5b: Receive a Struct from C ───────────────────────────

/// Call C functions that RETURN structs by value.
///
/// # What to implement
///
/// 1. Call `mathlib_point_add` which takes two `Point3D` BY VALUE and returns a `Point3D` by value
/// 2. Call `mathlib_mean` which returns a `StatResult` (struct with error code)
///
/// # Why this matters
///
/// C functions can return structs by value (not just pointers). The C ABI
/// handles this either by returning in registers (small structs) or via a
/// hidden output pointer (large structs). Rust handles this transparently
/// when using `extern "C"` — you just declare the return type.
///
/// The `StatResult` demonstrates a common C pattern: returning a struct with
/// both the result AND an error code. This is C's equivalent of Rust's
/// `Result<T, E>` — since C has no sum types, it bundles success and error
/// info into one struct.
///
/// # Hints
///
/// For point_add:
/// - `unsafe { mathlib_point_add(a, b) }` — pass structs by value
/// - Point3D is `Copy`, so passing by value works (it's copied to the C stack)
///
/// For mean:
/// - `unsafe { mathlib_mean(data.as_ptr(), data.len()) }` — returns StatResult
/// - Check `result.error_code`: 0 = success, non-zero = error
/// - Convert to Rust Result: `if result.error_code == 0 { Ok(result.value) } else { Err(...) }`
///
/// # What would break
///
/// If `StatResult`'s field order in Rust doesn't match C (e.g., `error_code`
/// before `value`), the f64 value field would read 4 bytes of error_code +
/// 4 bytes of padding as a floating point number → complete garbage.
fn add_points(a: Point3D, b: Point3D) -> Point3D {
    // TODO(human): Call mathlib_point_add which returns a Point3D by value.
    //
    // unsafe { mathlib_point_add(a, b) }
    //
    // Note: Point3D is Copy, so passing by value copies the 24 bytes onto
    // the C call stack. The C function returns another 24-byte struct
    // (either in registers or via hidden pointer, depending on ABI).
    todo!("Exercise 5b-1: Call C function that returns a struct by value")
}

fn compute_mean(data: &[f64]) -> Result<f64, String> {
    // TODO(human): Call mathlib_mean, check error_code, return Result.
    //
    // Step 1: let result = unsafe { mathlib_mean(data.as_ptr(), data.len()) };
    //
    // Step 2: Convert the C-style error to Rust Result:
    //   if result.error_code == 0 {
    //       Ok(result.value)
    //   } else {
    //       Err(format!("mathlib_mean error code: {}", result.error_code))
    //   }
    //
    // This pattern — calling a C function that returns a struct with an error
    // code, then converting to Rust's Result — is extremely common in FFI.
    // C has no exceptions and no Result type, so error codes in return structs
    // are the standard approach.
    todo!("Exercise 5b-2: Call C function returning StatResult, convert to Result")
}

fn demo_receive_struct_from_c() {
    let a = Point3D { x: 1.0, y: 2.0, z: 3.0 };
    let b = Point3D { x: 4.0, y: 5.0, z: 6.0 };
    let sum = add_points(a, b);
    assert!((sum.x - 5.0).abs() < f64::EPSILON);
    assert!((sum.y - 7.0).abs() < f64::EPSILON);
    assert!((sum.z - 9.0).abs() < f64::EPSILON);
    println!("[5b] point_add({:?}, {:?}) = {:?}", a, b, sum);

    // StatResult — success case
    let mean = compute_mean(&[10.0, 20.0, 30.0]);
    assert_eq!(mean, Ok(20.0));
    println!("[5b] mean([10, 20, 30]) = {:?}", mean);

    // StatResult — error case (empty array)
    let mean_err = compute_mean(&[]);
    assert!(mean_err.is_err());
    println!("[5b] mean([]) = {:?}", mean_err);

    // Verify StatResult size (f64 + i32 + padding = 16)
    println!(
        "[5b] size_of::<StatResult>() = {} bytes (expected 16)",
        std::mem::size_of::<StatResult>()
    );
}

// ─── Exercise 5c: Array of Structs & Callbacks ──────────────────────

/// Pass a Vec of data to C and use a C function with a Rust callback.
///
/// # What to implement
///
/// 1. Call `mathlib_sum_array` with a Rust `&[i32]` slice
/// 2. Call `mathlib_transform_array` with a mutable `&mut [f64]` and a Rust callback
///
/// # Why this matters
///
/// These demonstrate two important FFI patterns:
///
/// **Pattern 1: Rust data → C processing**
/// `Vec<T>` / `&[T]` with `repr(C)` elements are layout-compatible with C arrays.
/// `slice.as_ptr()` gives C a pointer to the contiguous data. C processes it
/// and returns the result. Rust still owns the memory.
///
/// **Pattern 2: Rust callback called by C**
/// `mathlib_transform_array` takes a function pointer. Rust provides an
/// `extern "C" fn` that C calls for each element. This is the reverse
/// of calling C from Rust — now C is calling Rust.
///
/// # Hints
///
/// For sum_array:
/// - `unsafe { mathlib_sum_array(data.as_ptr(), data.len()) }`
///
/// For transform_array:
/// - Define a callback: `extern "C" fn double_it(x: f64) -> f64 { x * 2.0 }`
/// - Call: `unsafe { mathlib_transform_array(data.as_mut_ptr(), data.len(), Some(double_it)) }`
/// - The callback MUST be `extern "C"` — C calls it, so it must use C's calling convention
/// - Closures cannot be used (they have captures + different ABI)
///
/// # What would break
///
/// - Forgetting `extern "C"` on the callback → wrong calling convention → stack corruption
/// - Using a closure instead of a function pointer → closures are fat pointers (pointer + environment),
///   C expects a thin function pointer → crash or UB
/// - Panicking in the callback → unwinding across FFI boundary → UB (wrap with catch_unwind in production)
fn sum_i32_array(data: &[i32]) -> i64 {
    // TODO(human): Call mathlib_sum_array with the slice data.
    //
    // unsafe { mathlib_sum_array(data.as_ptr(), data.len()) }
    //
    // Note: i32 is already repr(C)-compatible (same as C's int32_t).
    // Primitive types don't need #[repr(C)] — they have the same layout in Rust and C.
    todo!("Exercise 5c-1: Pass a Rust i32 slice to C's mathlib_sum_array")
}

fn transform_array_double(data: &mut [f64]) {
    // TODO(human): Call mathlib_transform_array with a callback that doubles each element.
    //
    // Step 1: Define an extern "C" callback function:
    //   extern "C" fn double_it(x: f64) -> f64 {
    //       x * 2.0
    //   }
    //
    // Step 2: Call the C function:
    //   unsafe {
    //       mathlib_transform_array(data.as_mut_ptr(), data.len(), Some(double_it));
    //   }
    //
    // Why `Some(double_it)`? The C function declares the callback as a function
    // pointer which could be NULL. Rust models this as Option<extern "C" fn(...)>.
    // We use Some() to pass a non-null function pointer.
    //
    // Why can't we use a closure? Closures in Rust are anonymous structs that
    // capture their environment. They're represented as (data_ptr, vtable_ptr) —
    // a "fat pointer". C function pointers are simple code addresses — "thin pointers".
    // You can only pass `fn` items (no captures) as C function pointers.
    todo!("Exercise 5c-2: Call C transform function with a Rust extern C callback")
}

fn demo_array_of_structs_to_c() {
    // Sum array
    let data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let sum = sum_i32_array(&data);
    assert_eq!(sum, 55);
    println!("[5c] sum_array([1..10]) = {}", sum);

    // Transform array with callback
    let mut values = [1.0, 2.0, 3.0, 4.0, 5.0];
    println!("[5c] Before transform: {:?}", values);
    transform_array_double(&mut values);
    assert_eq!(values, [2.0, 4.0, 6.0, 8.0, 10.0]);
    println!("[5c] After transform (doubled): {:?}", values);
}
