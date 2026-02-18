//! Exercise 4: Masks and Conditional SIMD
//!
//! In scalar code, `if/else` is handled by the branch predictor — when it guesses
//! wrong, the CPU stalls for 10-20 cycles (branch misprediction penalty). In SIMD,
//! there is no per-lane branch predictor. Instead, you compute BOTH outcomes for
//! all lanes simultaneously and use a **mask** to select which result each lane keeps.
//!
//! This is called **branchless** or **predicated** execution. It has constant,
//! predictable performance regardless of the data pattern — no branch mispredictions.
//!
//! A `Mask<i32, N>` is a per-lane boolean vector. It's produced by comparison
//! operations (`simd_gt`, `simd_lt`, `simd_eq`) and consumed by `select`:
//!
//!   mask.select(true_value, false_value) → per-lane choice
//!
//! Under the hood, a mask is a bitmask in a SIMD register. The `select` operation
//! compiles to a single `blendvps` (SSE4.1) or `vblendvps` (AVX) instruction.

use std::simd::prelude::*;

/// ReLU activation function: max(0, x) for each element.
///
/// ReLU (Rectified Linear Unit) is the most common activation function in neural
/// networks. It clips all negative values to zero while keeping positive values
/// unchanged. In inference engines (PyTorch, TensorFlow, ONNX Runtime), ReLU
/// on f32 tensors is implemented exactly like this — SIMD comparison + select.
///
/// Two approaches:
///   1. `simd_max(zero)` — if the portable_simd SimdFloat trait provides it
///   2. `compare + select` — explicit mask-based approach (more general)
pub fn relu_simd(input: &[f32], output: &mut [f32]) {
    // TODO(human): Implement ReLU using SIMD masks.
    //
    // This teaches the fundamental mask → select pattern that replaces branches
    // in all SIMD conditional logic.
    //
    // Steps:
    //
    // 1. Setup:
    //      const LANES: usize = 8;
    //      let zero = f32x8::splat(0.0);
    //      let n = input.len().min(output.len());
    //      let chunks = n / LANES;
    //
    // 2. Main SIMD loop:
    //
    //      for i in 0..chunks {
    //          let offset = i * LANES;
    //          let v = f32x8::from_slice(&input[offset..]);
    //
    //          // APPROACH A — Using simd_max (simplest):
    //          //   let result = v.simd_max(zero);
    //          //   This compiles to vmaxps — a single instruction that computes
    //          //   max(v[lane], 0.0) for each lane. Perfect for ReLU.
    //
    //          // APPROACH B — Using compare + select (more general):
    //          //   let mask = v.simd_ge(zero);     // per-lane: v[i] >= 0.0?
    //          //   let result = mask.select(v, zero); // true → v[i], false → 0.0
    //          //   This compiles to vcmpps + vblendvps — two instructions.
    //          //   Use this approach when the condition is more complex than min/max.
    //
    //          // UNDERSTANDING THE MASK:
    //          //   v    = [-3.0, -1.5,  0.0,  1.0,  2.5, -0.5,  4.0, -2.0]
    //          //   mask = [false, false, true, true, true, false, true, false]
    //          //   select(v, zero):
    //          //        = [ 0.0,  0.0,  0.0,  1.0,  2.5,  0.0,  4.0,  0.0]
    //          //
    //          //   The mask is a SIMD register of all-ones (0xFFFFFFFF) or all-zeros
    //          //   (0x00000000) per lane. The blend instruction uses each lane's
    //          //   mask bit to select from one source or the other — no branching.
    //
    //          result.copy_to_slice(&mut output[offset..]);
    //      }
    //
    // 3. Scalar tail:
    //      let tail_start = chunks * LANES;
    //      for i in tail_start..n {
    //          output[i] = input[i].max(0.0);
    //      }
    //
    // TRY BOTH APPROACHES and verify they produce identical results.
    // simd_max is simpler for ReLU specifically, but the mask+select pattern
    // generalizes to ANY conditional operation.

    todo!("Exercise 4a: Implement relu_simd")
}

/// Clamp each element to a [min, max] range.
///
/// Clamping is a compound conditional:
///   if x < min: min
///   elif x > max: max
///   else: x
///
/// In scalar code this is two branches. In SIMD, it's two comparisons + two selects,
/// or a single `simd_clamp()` call which does the same thing.
pub fn clamp_simd(input: &[f32], output: &mut [f32], min_val: f32, max_val: f32) {
    // TODO(human): Clamp each element to [min_val, max_val] using SIMD.
    //
    // Steps:
    //
    // 1. Setup:
    //      const LANES: usize = 8;
    //      let vmin = f32x8::splat(min_val);
    //      let vmax = f32x8::splat(max_val);
    //      let n = input.len().min(output.len());
    //      let chunks = n / LANES;
    //
    // 2. Main SIMD loop:
    //
    //      for i in 0..chunks {
    //          let offset = i * LANES;
    //          let v = f32x8::from_slice(&input[offset..]);
    //
    //          // APPROACH A — Using simd_clamp (simplest):
    //          //   let result = v.simd_clamp(vmin, vmax);
    //          //   This calls simd_max(vmin) then simd_min(vmax) internally.
    //
    //          // APPROACH B — Manual with two masks (educational):
    //          //   let below = v.simd_lt(vmin);       // lanes where v < min
    //          //   let above = v.simd_gt(vmax);       // lanes where v > max
    //          //   let result = below.select(vmin, v); // replace too-low with min
    //          //   let result = above.select(vmax, result); // replace too-high with max
    //          //
    //          //   Understanding:
    //          //     v      = [-3.0, -1.5,  0.0,  1.0,  2.5, -0.5,  4.0, -2.0]
    //          //     min=-1.0, max=3.0
    //          //     below  = [true, true, false, false, false, false, false, true]
    //          //     above  = [false, false, false, false, false, false, true, false]
    //          //     step1  = [-1.0, -1.0,  0.0,  1.0,  2.5, -0.5,  4.0, -1.0]
    //          //     step2  = [-1.0, -1.0,  0.0,  1.0,  2.5, -0.5,  3.0, -1.0]
    //
    //          // APPROACH C — Using simd_max + simd_min (most common in practice):
    //          //   let result = v.simd_max(vmin).simd_min(vmax);
    //          //   This compiles to vmaxps + vminps — two fast instructions.
    //          //   Equivalent to clamp but may handle NaN differently.
    //
    //          result.copy_to_slice(&mut output[offset..]);
    //      }
    //
    // 3. Scalar tail:
    //      let tail_start = chunks * LANES;
    //      for i in tail_start..n {
    //          output[i] = input[i].clamp(min_val, max_val);
    //      }
    //
    // Implement approach A or C (both compile to the same instructions), then
    // try approach B to understand how compound conditions decompose into masks.

    todo!("Exercise 4b: Implement clamp_simd")
}

/// Count how many elements in a slice exceed a threshold.
///
/// This demonstrates mask-to-scalar conversion: comparing SIMD vectors produces
/// masks, and you need to extract a scalar count from those masks.
///
/// Two approaches:
///   1. Convert mask to bitmask integer, count set bits with `count_ones()`
///   2. Select 1s and 0s, accumulate with SIMD, reduce at the end
pub fn count_threshold_simd(data: &[f32], threshold: f32) -> usize {
    // TODO(human): Count elements exceeding threshold using SIMD masks.
    //
    // This pattern is useful for filtering, histogram building, and data
    // quality checks — anywhere you need to count elements matching a condition.
    //
    // APPROACH A — Bitmask extraction (most efficient):
    //
    //   const LANES: usize = 8;
    //   let thresh = f32x8::splat(threshold);
    //   let n = data.len();
    //   let chunks = n / LANES;
    //   let mut count: usize = 0;
    //
    //   for i in 0..chunks {
    //       let offset = i * LANES;
    //       let v = f32x8::from_slice(&data[offset..]);
    //       let mask = v.simd_gt(thresh);
    //       // Convert mask to a bitmask: each lane → one bit
    //       //   mask = [true, false, true, true, false, false, true, false]
    //       //   bitmask = 0b00101101 = bits representing which lanes are true
    //       count += mask.to_bitmask().count_ones() as usize;
    //       // to_bitmask() returns a u8 (for 8 lanes) where bit i = mask lane i.
    //       // count_ones() counts set bits (popcount) — a single CPU instruction.
    //   }
    //
    //   // Handle tail
    //   let tail_start = chunks * LANES;
    //   for i in tail_start..n {
    //       if data[i] > threshold {
    //           count += 1;
    //       }
    //   }
    //
    //   count
    //
    // APPROACH B — Accumulate-and-reduce (alternative):
    //
    //   Use mask.select(i32x8::splat(1), i32x8::splat(0)) to get a vector of 1s and 0s,
    //   then accumulate into an i32x8 accumulator and reduce_sum at the end.
    //   This avoids the bitmask extraction but uses more register space.
    //
    // Approach A is typically faster because count_ones compiles to a single `popcnt`
    // instruction, while approach B requires an extra SIMD add per chunk.

    todo!("Exercise 4c: Implement count_threshold_simd")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let input = vec![-3.0f32, -1.5, 0.0, 1.0, 2.5, -0.5, 4.0, -2.0, 3.0, -4.0];
        let expected: Vec<f32> = input.iter().map(|&x| x.max(0.0)).collect();
        let mut output = vec![0.0f32; input.len()];
        relu_simd(&input, &mut output);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_clamp() {
        let input = vec![-3.0f32, -1.5, 0.0, 1.0, 2.5, -0.5, 4.0, -2.0, 3.0, -4.0];
        let expected: Vec<f32> = input.iter().map(|&x| x.clamp(-1.0, 3.0)).collect();
        let mut output = vec![0.0f32; input.len()];
        clamp_simd(&input, &mut output, -1.0, 3.0);
        for i in 0..input.len() {
            assert!(
                (output[i] - expected[i]).abs() < 1e-6,
                "clamp[{i}]: got {}, expected {}",
                output[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_count_threshold() {
        let data = vec![-3.0f32, -1.5, 0.0, 1.0, 2.5, -0.5, 4.0, -2.0,
                        3.0, -4.0, 1.5, 0.5, -1.0, 2.0, -0.1, 5.0];
        let expected = data.iter().filter(|&&x| x > 1.0).count();
        let result = count_threshold_simd(&data, 1.0);
        assert_eq!(result, expected, "got {result}, expected {expected}");
    }

    #[test]
    fn test_count_threshold_small() {
        let data = vec![0.5f32, 1.5, 2.5];
        let result = count_threshold_simd(&data, 1.0);
        assert_eq!(result, 2); // 1.5 and 2.5 exceed 1.0
    }
}
