//! Exercise 2: portable_simd Fundamentals
//!
//! The `std::simd` module (feature `portable_simd`) provides Rust's hardware-agnostic
//! SIMD abstraction. The core type `Simd<T, N>` represents a fixed-size vector of N
//! elements of type T, where operations execute in parallel across all lanes.
//!
//! On x86-64, `Simd<f32, 4>` compiles to SSE2 instructions, `Simd<f32, 8>` to AVX2
//! (if available). On AArch64, the same code compiles to NEON instructions. On targets
//! without SIMD, it falls back to scalar operations — your code still works, just slower.
//!
//! Key insight: operator overloads on `Simd` (`+`, `-`, `*`, `/`) are all VERTICAL
//! (lane-wise) operations. Each maps to a single SIMD instruction. Horizontal operations
//! (reduce_sum, reduce_max) combine lanes and are more expensive.

use std::simd::prelude::*;

/// Basic SIMD arithmetic: addition, subtraction, multiplication, division.
///
/// Each arithmetic operator on `Simd<f32, 4>` compiles to a single SSE2 instruction:
///   + → addps (packed single-precision add)
///   - → subps
///   * → mulps
///   / → divps
///
/// These are all VERTICAL operations: lane 0 of the result comes from lane 0 of
/// each operand, lane 1 from lane 1, etc. No cross-lane interaction occurs.
pub fn simd_arithmetic() {
    // TODO(human): Perform basic SIMD arithmetic and print the results.
    //
    // The fundamental type is `Simd<f32, N>` with alias `f32x4` for N=4.
    // Construction methods:
    //
    //   f32x4::from_array([1.0, 2.0, 3.0, 4.0])  — from a literal array
    //   f32x4::splat(5.0)                          — all lanes = 5.0: [5.0, 5.0, 5.0, 5.0]
    //
    // Steps:
    //
    // 1. Create two f32x4 vectors:
    //      let a = f32x4::from_array([1.0, 2.0, 3.0, 4.0]);
    //      let b = f32x4::from_array([10.0, 20.0, 30.0, 40.0]);
    //
    // 2. Perform each arithmetic operation and print the result:
    //      let sum = a + b;         // [11.0, 22.0, 33.0, 44.0]
    //      let diff = a - b;        // [-9.0, -18.0, -27.0, -36.0]
    //      let prod = a * b;        // [10.0, 40.0, 90.0, 160.0]
    //      let quot = b / a;        // [10.0, 10.0, 10.0, 10.0]
    //
    //    Use `println!("  sum:  {:?}", sum)` — Simd implements Debug.
    //
    // 3. Also try splat + scalar broadcast:
    //      let scaled = a * f32x4::splat(100.0);  // [100.0, 200.0, 300.0, 400.0]
    //
    //    `splat` creates a vector where every lane has the same value. This is how
    //    you multiply a vector by a scalar in SIMD — broadcast the scalar to all lanes
    //    first, then do a vertical multiply. The hardware has a dedicated broadcast
    //    instruction for this (vbroadcastss on AVX).
    //
    // 4. Convert back to a regular array if needed:
    //      let arr: [f32; 4] = sum.to_array();
    //
    // CONCEPT: Every operation here is "vertical" — lane i of the output depends only
    // on lane i of the inputs. This is why SIMD addition costs the same as scalar
    // addition but processes 4 (or 8, or 16) elements. The hardware executes all
    // lanes in lockstep within a single clock cycle.

    todo!("Exercise 2a: Implement simd_arithmetic")
}

/// Horizontal reductions: combining all lanes into a single scalar.
///
/// Reductions are the EXPENSIVE part of SIMD. While vertical operations (add, mul)
/// each take 1 cycle regardless of lane count, reductions require log2(N) shuffle +
/// add steps to combine all lanes.
///
/// For f32x4, reduce_sum compiles to roughly:
///   1. movhlps (shuffle high lanes to low)
///   2. addps (add pairs)
///   3. movhlps (shuffle again)
///   4. addss (final scalar add)
///   = ~4 instructions for 4 lanes
///
/// For f32x8 (AVX2), it's even more steps because you first need to extract the
/// upper 128-bit half with vextractf128 before combining.
///
/// RULE: Minimize reductions. The ideal SIMD pattern is:
///   1. Load → 2. Vertical ops (repeat many times) → 3. ONE reduction at the end.
pub fn simd_reductions() {
    // TODO(human): Create a vector and compute various reductions.
    //
    // Steps:
    //
    // 1. Create a f32x8 vector:
    //      let v = f32x8::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    //
    //    f32x8 = Simd<f32, 8> — this maps to a single 256-bit AVX register (ymm0-ymm15)
    //    on CPUs with AVX support. Without AVX, LLVM splits it into two SSE2 operations.
    //
    // 2. Compute and print each reduction:
    //
    //    let sum = v.reduce_sum();      // 1+2+3+4+5+6+7+8 = 36.0
    //    let prod = v.reduce_product(); // 1*2*3*4*5*6*7*8 = 40320.0
    //    let min = v.reduce_min();      // 1.0
    //    let max = v.reduce_max();      // 8.0
    //
    //    println!("  sum={sum}, product={prod}, min={min}, max={max}");
    //
    // 3. Note: reduce_sum and reduce_product return a scalar f32.
    //    These are "horizontal" operations — they combine values ACROSS lanes,
    //    which requires shuffling data between SIMD lanes (slow).
    //
    //    reduce_min and reduce_max also exist on SimdOrd (for integers) and
    //    SimdFloat (for floats). They handle NaN propagation according to IEEE 754.
    //
    // IMPORTANT: reduce_sum() for floats may give slightly different results than
    // the sequential sum due to different rounding order. For most applications this
    // is acceptable, but for numerically sensitive algorithms, you may need to use
    // compensated summation (Kahan summation) even in SIMD code.

    todo!("Exercise 2b: Implement simd_reductions")
}

/// Load SIMD vectors from slices, process, and write back.
///
/// Real SIMD code processes arrays in chunks: load N elements into a SIMD register,
/// operate on them, write N results back. The tricky part is handling the "tail" —
/// the remaining elements when the array length isn't a multiple of N.
///
/// This function multiplies every element of `input` by `scalar` and writes to `output`.
pub fn simd_from_slice(input: &[f32], output: &mut [f32], scalar: f32) {
    // TODO(human): Process a slice using SIMD with proper tail handling.
    //
    // This is the fundamental SIMD processing loop that you'll use in every
    // SIMD algorithm. Master this pattern and you can SIMD-ify anything.
    //
    // Choose a lane width — let's use 8 (f32x8, maps to AVX2 register):
    //
    //   const LANES: usize = 8;
    //   let scale = f32x8::splat(scalar);
    //
    // Main loop — process LANES elements at a time:
    //
    //   let n = input.len().min(output.len());
    //   let chunks = n / LANES;  // number of full SIMD iterations
    //
    //   for i in 0..chunks {
    //       let offset = i * LANES;
    //       // Load LANES elements from the input slice into a SIMD register:
    //       let v = f32x8::from_slice(&input[offset..]);
    //       // from_slice reads LANES elements starting at the given offset.
    //       // It panics if the slice is shorter than LANES — our loop bound
    //       // guarantees this won't happen.
    //
    //       // Vertical multiply: all LANES multiplications happen in parallel
    //       let result = v * scale;
    //
    //       // Write LANES results back to the output slice:
    //       result.copy_to_slice(&mut output[offset..]);
    //   }
    //
    // Handle the tail — remaining elements that don't fill a full SIMD vector:
    //
    //   let tail_start = chunks * LANES;
    //   for i in tail_start..n {
    //       output[i] = input[i] * scalar;
    //   }
    //
    // ALTERNATIVE tail handling with portable_simd:
    //   Instead of scalar tail, you can use `Simd::load_or_default()` which loads
    //   from a slice and fills missing lanes with the default value (0.0 for f32).
    //   But then you must be careful not to write the zero-filled lanes to output.
    //
    // PERFORMANCE NOTE: The main loop processes 8 elements per iteration.
    // With AVX2, each iteration is roughly:
    //   vmovups  → load 8 floats (1 cycle)
    //   vmulps   → multiply 8 floats (4-5 cycle latency, 0.5 cycle throughput)
    //   vmovups  → store 8 floats (1 cycle)
    // That's 8 multiplications in ~2-3 cycles, vs 8 cycles for scalar.

    todo!("Exercise 2c: Implement simd_from_slice")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_arithmetic() {
        // Should not panic
        simd_arithmetic();
    }

    #[test]
    fn test_simd_reductions() {
        // Should not panic
        simd_reductions();
    }

    #[test]
    fn test_simd_from_slice() {
        let input: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; 20];
        simd_from_slice(&input, &mut output, 2.0);
        for i in 0..20 {
            assert!(
                (output[i] - input[i] * 2.0).abs() < 1e-6,
                "output[{i}] = {}, expected {}",
                output[i],
                input[i] * 2.0
            );
        }
    }

    #[test]
    fn test_simd_from_slice_non_multiple() {
        // Test with a length that's NOT a multiple of 8
        let input: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; 13];
        simd_from_slice(&input, &mut output, 3.0);
        for i in 0..13 {
            assert!(
                (output[i] - input[i] * 3.0).abs() < 1e-6,
                "output[{i}] = {}, expected {}",
                output[i],
                input[i] * 3.0
            );
        }
    }
}
