//! Exercise 1: Auto-Vectorization — Observe and Break It
//!
//! The Rust compiler (via LLVM) can automatically convert scalar loops into SIMD
//! instructions — this is called auto-vectorization. When it works, you get SIMD
//! performance for free. When it silently fails, your code runs 4-8x slower than
//! it could.
//!
//! This exercise builds intuition for what patterns auto-vectorize and what breaks it.
//!
//! **How to verify:** Use `cargo asm --lib "function_name"` to inspect the generated
//! assembly. Look for packed instructions (addps, mulps, vaddps, vmulps) vs scalar
//! ones (addss, mulss). IMPORTANT: only `--release` builds auto-vectorize — debug
//! builds disable optimizations.

/// Sum all elements of a float slice — the "hello world" of auto-vectorization.
///
/// This is the simplest reduction loop: `sum += a[i]`. LLVM can vectorize this
/// by splitting the accumulator into N parallel partial sums (one per SIMD lane),
/// adding N elements per iteration, and combining the partial sums at the end.
///
/// The key requirement for auto-vectorization here is that floating-point addition
/// is treated as associative for optimization purposes (which it technically isn't
/// due to rounding — this is why `-ffast-math` exists in C/C++, and Rust's LLVM
/// does it by default for simple reductions).
///
/// After implementing, inspect the assembly with:
///   `cargo asm --lib "sum_array_autovec" --release`
///
/// You should see packed SIMD instructions like `addps` (SSE2, 4 floats) or
/// `vaddps` (AVX2, 8 floats) instead of scalar `addss`.
#[inline(never)] // Prevent inlining so cargo-show-asm can find this function
pub fn sum_array_autovec(data: &[f32]) -> f32 {
    // TODO(human): Implement a simple summation loop.
    //
    // Write the most straightforward loop you can — the compiler does the rest:
    //
    //   let mut sum = 0.0f32;
    //   for &x in data {
    //       sum += x;
    //   }
    //   sum
    //
    // Or equivalently: `data.iter().sum()` — the iterator version auto-vectorizes
    // equally well because LLVM sees through the iterator abstraction.
    //
    // WHY THIS WORKS: Each loop iteration's `sum += x` looks like a loop-carried
    // dependency (each iteration needs the previous sum). But LLVM transforms it:
    //
    //   Original:  sum = ((sum + a[0]) + a[1]) + a[2] + ...
    //   Vectorized: sum = (a[0]+a[4]+a[8]+...) + (a[1]+a[5]+a[9]+...) + ...
    //                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                     4 independent partial sums computed in parallel
    //
    // This reordering changes the floating-point result slightly (different rounding
    // order), but LLVM considers this acceptable for auto-vectorization of simple
    // reductions.
    //
    // TIP: Compile with `--release` — debug builds have no auto-vectorization.

    todo!("Exercise 1a: Implement sum_array_autovec")
}

/// Element-wise addition of two slices into an output slice.
///
/// This is the ideal case for auto-vectorization: a pure vertical operation with
/// no loop-carried dependencies. Each `output[i] = a[i] + b[i]` is completely
/// independent, so LLVM can process 4 (SSE2) or 8 (AVX2) elements per iteration.
///
/// After implementing, inspect assembly with:
///   `cargo asm --lib "add_arrays_autovec" --release`
///
/// Look for `movups` + `addps` + `movups` (SSE2 load-add-store pattern) or
/// the `v`-prefixed AVX equivalents.
#[inline(never)]
pub fn add_arrays_autovec(a: &[f32], b: &[f32], output: &mut [f32]) {
    // TODO(human): Implement element-wise addition.
    //
    // The simplest approach uses zip iterators:
    //
    //   for ((out, &a_val), &b_val) in output.iter_mut().zip(a).zip(b) {
    //       *out = a_val + b_val;
    //   }
    //
    // Or an index-based loop:
    //
    //   let n = a.len().min(b.len()).min(output.len());
    //   for i in 0..n {
    //       output[i] = a[i] + b[i];
    //   }
    //
    // IMPORTANT: Both versions auto-vectorize, but for different reasons:
    //
    // - The iterator version auto-vectorizes because `zip()` guarantees no
    //   out-of-bounds access — LLVM trusts iterators not to alias or overflow.
    //
    // - The index version auto-vectorizes because the explicit `n = min(...)`
    //   bound lets LLVM prove all accesses are in-bounds without runtime checks.
    //   If you instead wrote `for i in 0..a.len()` without checking `output.len()`,
    //   LLVM would insert bounds checks that may prevent vectorization.
    //
    // TIP: The iterator approach is generally more robust for auto-vectorization
    // because Rust's iterator design was intentionally built to enable it.

    todo!("Exercise 1b: Implement add_arrays_autovec")
}

/// Conditional sum: only add elements greater than a threshold.
///
/// This is the adversarial case for auto-vectorization. The branch inside the
/// loop (`if x > threshold`) creates a data-dependent control flow that is
/// harder for LLVM to vectorize.
///
/// LLVM MAY still vectorize this by converting the branch into a masked operation
/// (compare → mask → blend with zero → add), but it depends on the LLVM version,
/// optimization level, and target features. Check the assembly to see which path
/// your compiler takes.
///
/// Compare the assembly of this function with `sum_array_autovec` to see the
/// difference between unconditional and conditional vectorization:
///   `cargo asm --lib "sum_with_branch_autovec" --release`
///   `cargo asm --lib "sum_array_autovec" --release`
#[inline(never)]
pub fn sum_with_branch_autovec(data: &[f32], threshold: f32) -> f32 {
    // TODO(human): Implement a conditional sum — sum only elements > threshold.
    //
    // Write it as a straightforward loop with an if:
    //
    //   let mut sum = 0.0f32;
    //   for &x in data {
    //       if x > threshold {
    //           sum += x;
    //       }
    //   }
    //   sum
    //
    // After implementing, inspect the assembly. There are three possible outcomes:
    //
    // 1. FULLY VECTORIZED: LLVM converts the branch to a mask operation:
    //    - Compare: `cmpltps` produces a per-lane mask
    //    - Blend: `andps` or `blendvps` zeros out lanes that fail the condition
    //    - Add: `addps` adds the (possibly zeroed) result to the accumulator
    //    This is the best case — no branch, constant throughput regardless of data.
    //
    // 2. PARTIALLY VECTORIZED: LLVM vectorizes the comparison but falls back to
    //    scalar for the conditional accumulation. Look for a mix of packed and
    //    scalar instructions.
    //
    // 3. SCALAR ONLY: LLVM gives up and processes one element at a time.
    //    Look for `ucomiss` (scalar compare) + branch instructions.
    //
    // This is exactly why explicit SIMD (Exercises 4-5) exists: you can GUARANTEE
    // the masked approach regardless of what the compiler decides.
    //
    // TIP: Try an alternative formulation that helps auto-vectorization:
    //   sum += if x > threshold { x } else { 0.0 };
    // The branchless ternary is easier for LLVM to convert to a blend operation.

    todo!("Exercise 1c: Implement sum_with_branch_autovec")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_array() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let result = sum_array_autovec(&data);
        let expected: f32 = (0..100).map(|i| i as f32).sum();
        assert!((result - expected).abs() < 1e-2, "sum: got {result}, expected {expected}");
    }

    #[test]
    fn test_add_arrays() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut out = vec![0.0f32; 5];
        add_arrays_autovec(&a, &b, &mut out);
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0, 55.0]);
    }

    #[test]
    fn test_sum_with_branch() {
        let data = vec![1.0f32, 5.0, 3.0, 7.0, 2.0, 8.0];
        let result = sum_with_branch_autovec(&data, 4.0);
        let expected: f32 = 5.0 + 7.0 + 8.0;
        assert!((result - expected).abs() < 1e-6, "got {result}, expected {expected}");
    }
}
