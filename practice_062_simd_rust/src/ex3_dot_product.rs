//! Exercise 3: SIMD Dot Product
//!
//! The dot product `sum(a[i] * b[i])` is the canonical SIMD exercise because it
//! demonstrates the **accumulator pattern** — the most important SIMD idiom:
//!
//!   1. Initialize a SIMD accumulator to zero
//!   2. Main loop: load chunks, vertical multiply, vertical add to accumulator
//!   3. Handle tail elements
//!   4. ONE horizontal reduction at the very end
//!
//! This pattern is the backbone of matrix multiplication (GEMM), convolution,
//! correlation, neural network forward passes, and essentially all compute-intensive
//! numerical algorithms. Master this pattern and you can SIMD-ify almost anything.
//!
//! The dot product also demonstrates a key performance principle: **multiple
//! accumulators** to exploit instruction-level parallelism (ILP). Modern CPUs can
//! execute multiple SIMD additions in parallel if they are independent — using two
//! or four accumulator vectors can double or quadruple throughput.

use std::simd::prelude::*;

/// Scalar dot product — the baseline for comparison.
///
/// This is the simplest possible implementation. The compiler MAY auto-vectorize
/// this (see Exercise 1), but we mark it `#[inline(never)]` to prevent that for
/// benchmarking purposes.
#[inline(never)]
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    // TODO(human): Implement the scalar dot product.
    //
    // This is intentionally trivial — it serves as the performance baseline:
    //
    //   assert_eq!(a.len(), b.len());
    //   let mut sum = 0.0f32;
    //   for i in 0..a.len() {
    //       sum += a[i] * b[i];
    //   }
    //   sum
    //
    // Or using iterators:
    //   a.iter().zip(b).map(|(x, y)| x * y).sum()
    //
    // IMPORTANT: We use `#[inline(never)]` to prevent the compiler from inlining
    // this into the caller and potentially auto-vectorizing the whole thing.
    // In a benchmark, we want to measure the actual scalar vs SIMD difference.
    //
    // NUMERICAL NOTE: Sequential floating-point addition accumulates rounding error.
    // For N elements, the error bound is O(N * epsilon). SIMD reductions have different
    // error characteristics because they sum in a tree pattern rather than sequentially.
    // For most applications the difference is negligible, but for scientific computing
    // you may need compensated summation (Kahan) or higher precision accumulators.

    todo!("Exercise 3a: Implement dot_product_scalar")
}

/// Dot product using portable_simd with a single accumulator.
///
/// The accumulator pattern:
///   acc = [0, 0, 0, 0, 0, 0, 0, 0]   ← f32x8 of zeros
///   for each chunk of 8 elements:
///       a_chunk = load a[i..i+8]       ← f32x8
///       b_chunk = load b[i..i+8]       ← f32x8
///       acc += a_chunk * b_chunk        ← vertical mul + vertical add
///   result = acc.reduce_sum()           ← ONE horizontal reduction
///
/// Why accumulate vertically first? Because vertical add (addps) has 1 cycle
/// throughput, while horizontal reduce_sum requires ~4 instructions with shuffles.
/// By accumulating into a SIMD vector and reducing once at the end, we do
/// (N/8) vertical adds + 1 reduction, instead of N individual scalar adds.
#[inline(never)]
pub fn dot_product_portable_simd(a: &[f32], b: &[f32]) -> f32 {
    // TODO(human): Implement the SIMD dot product with an f32x8 accumulator.
    //
    // Steps:
    //
    // 1. Assert equal lengths:
    //      assert_eq!(a.len(), b.len());
    //
    // 2. Choose lane count and initialize accumulator:
    //      const LANES: usize = 8;
    //      let mut acc = f32x8::splat(0.0);
    //
    //    The accumulator is a SIMD vector, NOT a scalar. Each lane independently
    //    accumulates partial products from elements at offsets 0, 1, 2, ..., 7
    //    within each chunk.
    //
    // 3. Main loop — process LANES elements per iteration:
    //
    //      let chunks = a.len() / LANES;
    //      for i in 0..chunks {
    //          let offset = i * LANES;
    //          let va = f32x8::from_slice(&a[offset..]);
    //          let vb = f32x8::from_slice(&b[offset..]);
    //          acc += va * vb;
    //          // This compiles to:
    //          //   vmovups ymm1, [rdi + offset]     ; load 8 floats from a
    //          //   vmovups ymm2, [rsi + offset]     ; load 8 floats from b
    //          //   vmulps  ymm1, ymm1, ymm2         ; 8 multiplications in parallel
    //          //   vaddps  ymm0, ymm0, ymm1         ; 8 additions to accumulator
    //          // Total: 8 multiply-adds in ~2-3 cycles (vs 8 cycles scalar)
    //      }
    //
    // 4. Handle tail elements (when a.len() % LANES != 0):
    //
    //      let tail_start = chunks * LANES;
    //      let mut tail_sum = 0.0f32;
    //      for i in tail_start..a.len() {
    //          tail_sum += a[i] * b[i];
    //      }
    //
    //    The tail is at most LANES-1 = 7 elements — negligible overhead.
    //
    // 5. Final horizontal reduction — combine all 8 accumulator lanes + tail:
    //
    //      acc.reduce_sum() + tail_sum
    //
    //    reduce_sum() combines: lane0 + lane1 + lane2 + ... + lane7 → scalar f32.
    //    This is the ONLY horizontal operation in the entire function.
    //
    // PERFORMANCE ANALYSIS:
    //   For N elements:
    //     - Main loop: N/8 iterations, each doing 1 mul + 1 add = 2 SIMD ops
    //     - Reduction: 1 call (~4 shuffle+add instructions)
    //     - Total SIMD ops: N/4 + 4
    //     - Scalar equivalent: 2N ops (N muls + N adds)
    //     - Theoretical speedup: ~8x (for large N where reduction cost is amortized)

    todo!("Exercise 3b: Implement dot_product_portable_simd")
}

/// Dot product with multiple accumulators for instruction-level parallelism.
///
/// Modern out-of-order CPUs can execute multiple independent SIMD operations in
/// the same cycle if they don't depend on each other. A single accumulator creates
/// a dependency chain:
///
///   acc = acc + (a0 * b0)     ← must wait for previous acc
///   acc = acc + (a1 * b1)     ← must wait for previous acc
///   acc = acc + (a2 * b2)     ← must wait for previous acc
///
/// With two accumulators, operations alternate and can overlap:
///
///   acc0 = acc0 + (a0 * b0)   ← independent chain 0
///   acc1 = acc1 + (a1 * b1)   ← independent chain 1 (can start immediately!)
///   acc0 = acc0 + (a2 * b2)   ← chain 0 continues
///   acc1 = acc1 + (a3 * b3)   ← chain 1 continues
///
/// The CPU's out-of-order execution engine sees two independent dependency chains
/// and can issue operations from both chains simultaneously. With FMA latency of
/// 4 cycles and throughput of 0.5 cycles, using 4 accumulators fully saturates
/// the execution units.
#[inline(never)]
pub fn dot_product_multi_accumulator(a: &[f32], b: &[f32]) -> f32 {
    // TODO(human): Implement dot product with TWO f32x8 accumulators.
    //
    // Steps:
    //
    // 1. Initialize two accumulators:
    //      const LANES: usize = 8;
    //      let mut acc0 = f32x8::splat(0.0);
    //      let mut acc1 = f32x8::splat(0.0);
    //
    // 2. Main loop — process 2 * LANES = 16 elements per iteration:
    //
    //      let double_lanes = LANES * 2;
    //      let chunks = a.len() / double_lanes;
    //      for i in 0..chunks {
    //          let offset = i * double_lanes;
    //
    //          // First chunk of 8 → accumulator 0
    //          let va0 = f32x8::from_slice(&a[offset..]);
    //          let vb0 = f32x8::from_slice(&b[offset..]);
    //          acc0 += va0 * vb0;
    //
    //          // Second chunk of 8 → accumulator 1
    //          let va1 = f32x8::from_slice(&a[offset + LANES..]);
    //          let vb1 = f32x8::from_slice(&b[offset + LANES..]);
    //          acc1 += va1 * vb1;
    //
    //          // CPU can execute acc0 and acc1 updates in parallel!
    //      }
    //
    // 3. Handle tail elements (up to 2*LANES - 1 = 15 elements):
    //
    //      let tail_start = chunks * double_lanes;
    //      let mut tail_sum = 0.0f32;
    //      for i in tail_start..a.len() {
    //          tail_sum += a[i] * b[i];
    //      }
    //
    // 4. Combine accumulators and reduce:
    //
    //      (acc0 + acc1).reduce_sum() + tail_sum
    //
    //    First combine the two accumulator vectors (vertical add — fast), then
    //    do one horizontal reduction.
    //
    // WHY THIS IS FASTER:
    //   On a Haswell/Skylake CPU, vaddps (or vfmadd) has latency=4, throughput=0.5.
    //   - 1 accumulator: limited to 1 add per 4 cycles (waiting for result)
    //   - 2 accumulators: 1 add per 2 cycles (alternating chains)
    //   - 4 accumulators: 1 add per 1 cycle (fully saturated)
    //   - 8 accumulators: 1 add per 0.5 cycles (theoretical max throughput)
    //
    //   In practice, 2-4 accumulators capture most of the benefit. Beyond that,
    //   register pressure and loop overhead start to dominate.
    //
    // TIP: For even more parallelism, try 4 accumulators in the benchmark exercise.

    todo!("Exercise 3c: Implement dot_product_multi_accumulator")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn test_dot_product_scalar() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0];
        let result = dot_product_scalar(&a, &b);
        let expected = reference_dot(&a, &b); // 2+6+12+20+30 = 70
        assert!((result - expected).abs() < 1e-4, "got {result}, expected {expected}");
    }

    #[test]
    fn test_dot_product_portable_simd() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 0.001).collect();
        let result = dot_product_portable_simd(&a, &b);
        let expected = reference_dot(&a, &b);
        assert!(
            (result - expected).abs() < 1.0,
            "got {result}, expected {expected}, diff {}",
            (result - expected).abs()
        );
    }

    #[test]
    fn test_dot_product_multi_accumulator() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 0.001).collect();
        let result = dot_product_multi_accumulator(&a, &b);
        let expected = reference_dot(&a, &b);
        assert!(
            (result - expected).abs() < 1.0,
            "got {result}, expected {expected}, diff {}",
            (result - expected).abs()
        );
    }

    #[test]
    fn test_dot_product_small() {
        // Test with less than LANES elements
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let expected = reference_dot(&a, &b); // 4 + 10 + 18 = 32
        assert!((dot_product_portable_simd(&a, &b) - expected).abs() < 1e-4);
        assert!((dot_product_multi_accumulator(&a, &b) - expected).abs() < 1e-4);
    }
}
