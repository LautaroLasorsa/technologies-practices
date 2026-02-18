//! Exercise 6: Swizzle and Data Rearrangement
//!
//! Swizzle (also called shuffle or permute) rearranges which data sits in which lane
//! of a SIMD vector. While vertical operations (add, mul) keep data in the same lane,
//! swizzle operations move data BETWEEN lanes.
//!
//! Common uses:
//! - Matrix transpose (swap rows ↔ columns for cache-friendly access patterns)
//! - AoS → SoA conversion (interleaved RGB → separate R/G/B channels)
//! - Prefix sum / scan operations
//! - Complex number multiplication (real/imaginary interleave)
//!
//! portable_simd provides:
//! - `simd_swizzle!(vec, [indices])` — compile-time lane selection
//! - `interleave(a, b)` → weave elements alternately
//! - `deinterleave(a, b)` → split alternating elements apart
//! - `reverse()` — reverse lane order
//! - `rotate_elements_left::<N>()` / `rotate_elements_right::<N>()` — cyclic shift
//!
//! The key difference between `simd_swizzle!` and runtime indexing is that swizzle
//! indices are COMPILE-TIME constants. This lets the compiler emit a single shuffle
//! instruction (like `vpshufd` or `vpermps`) rather than a gather from memory.

use std::simd::prelude::*;
use std::simd::simd_swizzle;

/// Transpose a 2x2 matrix stored in two f32x4 vectors.
///
/// Matrix layout (row-major in two vectors):
///   row0 = [a, b, _, _]    →   col0 = [a, c, _, _]
///   row1 = [c, d, _, _]    →   col1 = [b, d, _, _]
///
/// This is the building block for larger matrix transposes. A 4x4 transpose
/// is four 2x2 transposes + recombination. An NxN transpose decomposes into
/// smaller block transposes — the same approach used in cache-oblivious algorithms.
///
/// Why transpose matters for SIMD: matrix multiplication in row-major layout
/// requires accessing columns (stride-N access), which kills cache performance.
/// Transposing the second matrix converts column access to row access (stride-1),
/// enabling efficient SIMD loads.
pub fn transpose_2x2_simd() {
    // TODO(human): Transpose a 2x2 matrix using SIMD swizzle operations.
    //
    // Steps:
    //
    // 1. Define two f32x4 vectors representing rows:
    //      let row0 = f32x4::from_array([1.0, 2.0, 0.0, 0.0]);  // [a, b, _, _]
    //      let row1 = f32x4::from_array([3.0, 4.0, 0.0, 0.0]);  // [c, d, _, _]
    //
    // 2. Use simd_swizzle! to transpose.
    //
    //    simd_swizzle! with TWO source vectors uses indices 0..N for the first
    //    vector and N..2N for the second:
    //
    //      // For two f32x4 vectors, indices 0-3 refer to row0's lanes,
    //      // and indices 4-7 refer to row1's lanes.
    //      let col0 = simd_swizzle!(row0, row1, [0, 4, 2, 6]);  // [a, c, _, _]
    //      let col1 = simd_swizzle!(row0, row1, [1, 5, 3, 7]);  // [b, d, _, _]
    //
    //    UNDERSTANDING THE INDICES:
    //      row0 = [1.0, 2.0, 0.0, 0.0]  (indices 0, 1, 2, 3)
    //      row1 = [3.0, 4.0, 0.0, 0.0]  (indices 4, 5, 6, 7)
    //
    //      col0[0] = index 0 → row0[0] = 1.0 (a)
    //      col0[1] = index 4 → row1[0] = 3.0 (c)
    //      col0[2] = index 2 → row0[2] = 0.0 (_)
    //      col0[3] = index 6 → row1[2] = 0.0 (_)
    //      → col0 = [1.0, 3.0, 0.0, 0.0] = [a, c, _, _] ✓
    //
    //    This compiles to a single `vshufps` or `vunpcklps` instruction on x86.
    //
    // 3. ALTERNATIVE using interleave:
    //      let (interleaved_lo, interleaved_hi) = row0.interleave(row1);
    //      // interleaved_lo = [row0[0], row1[0], row0[1], row1[1]] = [a, c, b, d]
    //      // interleaved_hi = [row0[2], row1[2], row0[3], row1[3]] = [_, _, _, _]
    //      // Then extract col0 and col1 from interleaved_lo
    //
    // 4. Print results:
    //      println!("  row0: {:?}", row0);
    //      println!("  row1: {:?}", row1);
    //      println!("  col0: {:?}", col0);
    //      println!("  col1: {:?}", col1);
    //
    // 5. Verify the transpose:
    //      assert_eq!(col0.to_array()[0], row0.to_array()[0]); // a
    //      assert_eq!(col0.to_array()[1], row1.to_array()[0]); // c
    //      assert_eq!(col1.to_array()[0], row0.to_array()[1]); // b
    //      assert_eq!(col1.to_array()[1], row1.to_array()[1]); // d
    //
    // ALSO TRY these other swizzle operations and print their results:
    //
    //   let reversed = row0.reverse();
    //   // reverse() flips all lanes: [a, b, _, _] → [_, _, b, a]
    //
    //   let rotated = row0.rotate_elements_left::<1>();
    //   // rotate_elements_left::<1>: [a, b, c, d] → [b, c, d, a]
    //   // This is a cyclic shift — element 0 wraps around to the end.

    todo!("Exercise 6a: Implement transpose_2x2_simd")
}

/// Deinterleave RGB pixel data from AoS (Array of Structures) to SoA (Structure of Arrays).
///
/// Image data is often stored interleaved: [R0,G0,B0, R1,G1,B1, R2,G2,B2, ...]
/// But SIMD processes same-typed data best: [R0,R1,R2,...], [G0,G1,G2,...], [B0,B1,B2,...]
///
/// Converting AoS → SoA is called "deinterleaving" and is a fundamental SIMD
/// pattern for image processing, audio processing (interleaved L/R channels),
/// and any domain with packed multi-field records.
///
/// On ARM NEON, there's a dedicated instruction for this (vld3q_f32 — load and
/// deinterleave 3 channels). On x86, you have to do it manually with shuffles.
pub fn deinterleave_rgb() {
    // TODO(human): Deinterleave RGB data from AoS to SoA layout.
    //
    // This is a real-world SIMD pattern used in every image processing library
    // (OpenCV, Pillow, stb_image). Understanding it reveals why SoA layout is
    // preferred for SIMD-heavy code.
    //
    // Input: 12 floats representing 4 RGB pixels in interleaved order:
    //
    //   let rgb_data: [f32; 12] = [
    //       1.0, 0.2, 0.3,   // pixel 0: R=1.0, G=0.2, B=0.3
    //       0.4, 1.0, 0.6,   // pixel 1: R=0.4, G=1.0, B=0.6
    //       0.7, 0.8, 1.0,   // pixel 2: R=0.7, G=0.8, B=1.0
    //       0.1, 0.2, 0.3,   // pixel 3: R=0.1, G=0.2, B=0.3
    //   ];
    //
    // Goal: separate into R, G, B channels:
    //   r = [1.0, 0.4, 0.7, 0.1]  — red values of all 4 pixels
    //   g = [0.2, 1.0, 0.8, 0.2]  — green values of all 4 pixels
    //   b = [0.3, 0.6, 1.0, 0.3]  — blue values of all 4 pixels
    //
    // APPROACH A — Direct gather with known indices:
    //
    //   Since we know the RGB structure, we can load directly with compile-time
    //   index arithmetic. Load all 12 values into three f32x4 vectors and
    //   use simd_swizzle! to extract channels:
    //
    //   let v0 = f32x4::from_slice(&rgb_data[0..4]);   // [R0, G0, B0, R1]
    //   let v1 = f32x4::from_slice(&rgb_data[4..8]);   // [G1, B1, R2, G2]
    //   let v2 = f32x4::from_slice(&rgb_data[8..12]);  // [B2, R3, G3, B3]
    //
    //   // Extract red channel: indices 0 (R0 from v0), 3 (R1 from v0),
    //   //                              6 (R2 from v1→idx 2), 9 (R3 from v2→idx 1)
    //   // With two-vector swizzle:
    //   let r01 = simd_swizzle!(v0, v1, [0, 3, 6, 6]);  // need to be creative
    //
    // APPROACH B — Simple scalar gather (clearest):
    //
    //   For small data, just gather elements by index:
    //     let r = f32x4::from_array([rgb_data[0], rgb_data[3], rgb_data[6], rgb_data[9]]);
    //     let g = f32x4::from_array([rgb_data[1], rgb_data[4], rgb_data[7], rgb_data[10]]);
    //     let b = f32x4::from_array([rgb_data[2], rgb_data[5], rgb_data[8], rgb_data[11]]);
    //
    //   This is fine for learning. In production, you'd use SIMD shuffles for larger
    //   data sets where the gather overhead amortizes.
    //
    // APPROACH C — Using gather_or with index vectors (portable_simd):
    //
    //   let r_indices = Simd::<usize, 4>::from_array([0, 3, 6, 9]);
    //   let r = f32x4::gather_or(&rgb_data, r_indices, f32x4::splat(0.0));
    //
    //   gather_or reads elements at non-contiguous indices from a slice.
    //   On CPUs with AVX2, this compiles to vgatherdps — a hardware scatter/gather
    //   instruction. On other CPUs, it falls back to scalar loads.
    //
    // Steps:
    //
    // 1. Create the interleaved RGB data array (given above)
    // 2. Extract R, G, B channels using your preferred approach
    // 3. Print the results:
    //      println!("  Interleaved: {:?}", rgb_data);
    //      println!("  R channel:   {:?}", r);
    //      println!("  G channel:   {:?}", g);
    //      println!("  B channel:   {:?}", b);
    //
    // 4. Demonstrate a SIMD operation on separated channels.
    //    For example, convert to grayscale using the luminance formula:
    //      gray = 0.299 * R + 0.587 * G + 0.114 * B
    //
    //    With SoA layout, this is trivially vectorized:
    //      let gray = r * f32x4::splat(0.299)
    //               + g * f32x4::splat(0.587)
    //               + b * f32x4::splat(0.114);
    //      println!("  Grayscale:   {:?}", gray);
    //
    //    This computes grayscale for ALL 4 pixels in parallel. With AoS layout,
    //    you'd need to extract each pixel's RGB, compute grayscale, and store —
    //    no SIMD benefit. SoA enables SIMD; AoS prevents it.
    //
    // 5. Verify:
    //      let expected_gray_0 = 1.0 * 0.299 + 0.2 * 0.587 + 0.3 * 0.114;
    //      assert!((gray.to_array()[0] - expected_gray_0).abs() < 1e-5);

    todo!("Exercise 6b: Implement deinterleave_rgb")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        // Should not panic
        transpose_2x2_simd();
    }

    #[test]
    fn test_deinterleave() {
        // Should not panic
        deinterleave_rgb();
    }
}
