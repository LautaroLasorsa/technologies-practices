//! Exercise 5: Assembly Inspection with cargo-show-asm
//!
//! Sometimes profiling identifies a hot function, and you need to understand
//! WHAT the compiler actually generated. Did it auto-vectorize? Are there
//! unnecessary bounds checks? Is it inlining the right things?
//!
//! cargo-show-asm lets you answer these questions by showing the actual machine
//! code (x86-64 assembly), LLVM-IR, or MIR for any public function.
//!
//! To view assembly for functions in this module:
//!   cargo asm --lib profiling_flamegraphs::ex5_assembly::sum_slice
//!   cargo asm --lib --rust profiling_flamegraphs::ex5_assembly::sum_slice
//!
//! The `--rust` flag interleaves the Rust source code with the assembly,
//! making it much easier to see which Rust code maps to which instructions.
//!
//! IMPORTANT: Functions must be `pub` and `#[inline(never)]` to appear in
//! the assembly output. The compiler inlines small functions by default,
//! which removes them from the symbol table.

/// Sum all elements of a `f64` slice using iterators.
///
/// This function demonstrates auto-vectorization: when the compiler can prove
/// that a loop has no side effects and a predictable access pattern, it may
/// emit SIMD instructions (SSE/AVX) that process 2-4 f64 values per instruction
/// instead of one at a time (scalar).
///
/// To check for auto-vectorization, look for these instructions in the assembly:
/// - `addpd` / `vaddpd` — SIMD double-precision add (processes 2 or 4 f64s at once)
/// - `addsd` — scalar double-precision add (processes 1 f64 — NOT vectorized)
///
/// If you see `vaddpd ymm0, ymm0, [rdi+rcx*8]`, that's AVX processing 4 f64s
/// per instruction — a 4x throughput improvement over scalar code.
///
/// The `#[inline(never)]` attribute prevents the compiler from inlining this
/// function into its callers, which would make it invisible in assembly output
/// and in profiler flamegraphs (the caller would get the samples instead).
#[inline(never)]
pub fn sum_slice(data: &[f64]) -> f64 {
    // TODO(human): Implement a simple sum using iterators.
    //
    // This is intentionally simple — the point is not the implementation
    // (which is one line) but the ASSEMBLY it generates.
    //
    // Steps:
    //
    // 1. Use the iterator `.iter().sum()` pattern:
    //    ```
    //    data.iter().copied().sum()
    //    ```
    //    Or equivalently: `data.iter().sum::<f64>()`
    //
    // 2. After implementing, inspect the assembly:
    //    ```
    //    cargo asm --lib profiling_flamegraphs::ex5_assembly::sum_slice
    //    ```
    //
    // 3. Look for SIMD instructions:
    //    - `vaddpd` (AVX, 4 f64s per op) or `addpd` (SSE2, 2 f64s per op)
    //      → Auto-vectorized! The compiler recognized the reduction pattern.
    //    - `addsd` (scalar, 1 f64 per op)
    //      → NOT vectorized. This can happen if the compiler cannot prove
    //        that reordering floating-point additions is safe (FP addition is
    //        not associative due to rounding). Try adding `-C target-cpu=native`
    //        to RUSTFLAGS or using `.fold(0.0, |acc, x| acc + x)`.
    //
    // 4. Also inspect with interleaved source:
    //    ```
    //    cargo asm --lib --rust profiling_flamegraphs::ex5_assembly::sum_slice
    //    ```
    //    This shows which Rust line maps to which assembly instructions.
    //
    // Why this matters: Auto-vectorization is "free" performance — the compiler
    // does the work if you write the code in a vectorization-friendly way.
    // But it is fragile: small changes (like adding a branch in the loop body)
    // can prevent vectorization. Inspecting the assembly tells you whether you
    // got the "free" speedup or not.

    todo!("Exercise 5a: Implement sum_slice and inspect assembly")
}

/// Sum all elements using indexing (with bounds checks).
///
/// This function demonstrates the performance cost of bounds checking.
/// When you access `slice[i]`, Rust inserts a conditional branch:
///   if i >= slice.len() { panic!("index out of bounds") }
///
/// In the assembly, this appears as a `cmp` + `jae` (jump if above or equal)
/// before every array access. The branch is almost never taken (the index is
/// almost always valid), but the branch predictor and instruction cache still
/// pay a cost — and more importantly, the bounds check prevents the compiler
/// from auto-vectorizing the loop.
///
/// Compare the assembly output of this function with `sum_slice()`. The
/// iterator version avoids bounds checks entirely because `iter()` is
/// guaranteed to produce valid references. This is a key example of how
/// idiomatic Rust (iterators) generates BETTER code than C-style loops.
#[inline(never)]
pub fn sum_with_bounds_checks(data: &[f64]) -> f64 {
    // TODO(human): Implement sum using C-style indexing.
    //
    // Steps:
    //
    // 1. Use a manual index loop:
    //    ```
    //    let mut sum = 0.0_f64;
    //    for i in 0..data.len() {
    //        sum += data[i];  // bounds check inserted here
    //    }
    //    sum
    //    ```
    //
    // 2. After implementing, inspect the assembly:
    //    ```
    //    cargo asm --lib profiling_flamegraphs::ex5_assembly::sum_with_bounds_checks
    //    ```
    //
    // 3. Compare with sum_slice's assembly. Look for:
    //    - `cmp` + `jae` or `jb` instructions near the array access → bounds check
    //    - Whether `vaddpd`/`addpd` (SIMD) appears or only `addsd` (scalar)
    //    - Total instruction count: bounds-checked version typically has more
    //
    // 4. Optional experiment: try using `unsafe { *data.get_unchecked(i) }`
    //    instead of `data[i]` and inspect the assembly again. It should match
    //    the iterator version — confirming that bounds checks were the difference.
    //    (DO NOT use get_unchecked in production without proof that indices are valid.)
    //
    // Why this matters: This is a concrete example of Rust's zero-cost abstractions.
    // Iterators are not just "nice syntax" — they generate genuinely better machine
    // code than manual indexing because they statically guarantee valid access,
    // eliminating the need for runtime bounds checks. This is one of the key
    // performance advantages of Rust over C/C++, where array access is unchecked
    // (and buffer overflows are a major source of security vulnerabilities).

    todo!("Exercise 5b: Implement sum with indexing and compare assembly")
}

/// Dot product of two f64 slices.
///
/// The dot product (sum of element-wise products) is one of the most important
/// operations in scientific computing, machine learning, and graphics. It maps
/// perfectly to SIMD: multiply pairs of elements and accumulate the results.
///
/// In the assembly, look for:
/// - `vmulpd` + `vaddpd` — separate multiply and add (not ideal)
/// - `vfmadd213pd` / `vfmadd231pd` — Fused Multiply-Add (FMA), which computes
///   `a * b + c` in a single instruction with better precision and throughput.
///   FMA is available on Haswell+ (2013+) CPUs.
///
/// FMA requires either `-C target-cpu=native` (uses the host CPU's features)
/// or `-C target-feature=+fma`. Without these, the compiler targets a generic
/// x86-64 baseline that does not include FMA.
#[inline(never)]
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    // TODO(human): Implement dot product of two slices.
    //
    // Steps:
    //
    // 1. Use the idiomatic Rust approach with iterators:
    //    ```
    //    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    //    ```
    //    This is concise and generates excellent code because:
    //    - `zip` produces pairs without bounds checking
    //    - `map` applies the multiply without materializing an intermediate Vec
    //    - `sum` accumulates — the compiler can often auto-vectorize this pattern
    //
    // 2. After implementing, inspect the assembly:
    //    ```
    //    cargo asm --lib profiling_flamegraphs::ex5_assembly::dot_product
    //    ```
    //
    // 3. Look for FMA instructions: `vfmadd213pd`, `vfmadd231pd`
    //    - If you see separate `vmulpd` + `vaddpd`: FMA is not being used.
    //      To enable FMA, set the environment variable:
    //      `RUSTFLAGS="-C target-cpu=native" cargo asm --lib ...::dot_product`
    //      This tells the compiler to use all features available on your CPU.
    //
    // 4. Also compare the instruction count with a manual loop version to see
    //    if the iterator version is equivalent (it should be — zero-cost abstractions).
    //
    // Why this matters: FMA is a 2x throughput improvement for multiply-accumulate
    // patterns. It is critical for numerical code, ML inference, and graphics.
    // Knowing how to check if the compiler uses FMA (and how to enable it) is a
    // practical performance engineering skill.

    todo!("Exercise 5c: Implement dot product and check for FMA instructions")
}

/// Demo runner for Exercise 5 — calls the assembly-inspectable functions.
pub fn demo_sum_slice() {
    let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let result = sum_slice(&data);
    println!("  sum_slice({} elements) = {}", data.len(), result);
    println!("  Inspect with: cargo asm --lib profiling_flamegraphs::ex5_assembly::sum_slice");
}

pub fn demo_sum_with_bounds_checks() {
    let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let result = sum_with_bounds_checks(&data);
    println!("  sum_with_bounds_checks({} elements) = {}", data.len(), result);
    println!("  Inspect with: cargo asm --lib profiling_flamegraphs::ex5_assembly::sum_with_bounds_checks");
}

pub fn demo_dot_product() {
    let a: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.5).collect();
    let result = dot_product(&a, &b);
    println!("  dot_product({} elements) = {}", a.len(), result);
    println!("  Inspect with: cargo asm --lib profiling_flamegraphs::ex5_assembly::dot_product");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = sum_slice(&data);
        assert!((result - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sum_with_bounds_checks() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = sum_with_bounds_checks(&data);
        assert!((result - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((result - 32.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sum_functions_agree() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let iter_sum = sum_slice(&data);
        let index_sum = sum_with_bounds_checks(&data);
        assert!(
            (iter_sum - index_sum).abs() < 1e-10,
            "Iterator and indexing sums should agree"
        );
    }

    #[test]
    fn test_dot_product_empty() {
        let result = dot_product(&[], &[]);
        assert!((result - 0.0).abs() < f64::EPSILON);
    }
}
