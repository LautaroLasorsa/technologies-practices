//! Exercise 5: std::arch — SSE2 and AVX2 Intrinsics
//!
//! `std::arch` exposes the raw CPU intrinsics, giving you 1:1 control over which
//! machine instructions are generated. This is the same set of intrinsics available
//! in C/C++ (`#include <immintrin.h>`), with the same names and semantics.
//!
//! The x86 SIMD intrinsic naming convention:
//!   _mm_<op>_<suffix>      → 128-bit SSE (4 floats, 2 doubles)
//!   _mm256_<op>_<suffix>   → 256-bit AVX/AVX2 (8 floats, 4 doubles)
//!   _mm512_<op>_<suffix>   → 512-bit AVX-512 (16 floats, 8 doubles)
//!
//! Suffix meanings:
//!   ps = packed single (f32)    pd = packed double (f64)
//!   ss = scalar single          sd = scalar double
//!   epi32 = packed 32-bit int   epi64 = packed 64-bit int
//!   si128 = 128-bit integer     si256 = 256-bit integer
//!
//! All std::arch intrinsics are `unsafe` because calling them on a CPU that doesn't
//! support the required ISA extension is undefined behavior. The safety contract:
//!
//! 1. Mark the function with `#[target_feature(enable = "sse2")]` (or "avx2", etc.)
//! 2. The function must be declared `unsafe fn`
//! 3. The caller must verify the CPU supports the feature (via `is_x86_feature_detected!`)
//!
//! This is the standard "runtime dispatch" pattern: compile multiple versions of a
//! hot function targeting different ISA levels, then select the best one at startup.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Dot product using SSE2 intrinsics (128-bit, 4 floats per vector).
///
/// SSE2 is the baseline SIMD for x86-64 — every 64-bit x86 CPU supports it.
/// Rust enables SSE2 by default, so this function works on ALL x86-64 systems.
///
/// SSE2 intrinsics used:
///   _mm_setzero_ps()   → create [0.0, 0.0, 0.0, 0.0]
///   _mm_loadu_ps(ptr)  → load 4 floats from unaligned memory
///   _mm_mul_ps(a, b)   → [a0*b0, a1*b1, a2*b2, a3*b3]
///   _mm_add_ps(a, b)   → [a0+b0, a1+b1, a2+b2, a3+b3]
///   _mm_store_ss(ptr, v) → store lane 0 to a scalar pointer
///
/// For the horizontal sum at the end (SSE3+):
///   _mm_hadd_ps(a, b)  → horizontal add adjacent pairs
/// Or manually with SSE2 shuffles:
///   _mm_shuffle_ps + _mm_add_ps to fold lanes together
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn dot_product_sse2(a: &[f32], b: &[f32]) -> f32 {
    // TODO(human): Implement dot product using SSE2 intrinsics.
    //
    // This is the low-level equivalent of dot_product_portable_simd from Exercise 3,
    // but using raw intrinsics instead of the portable abstraction.
    //
    // Steps:
    //
    // 1. Assert equal lengths:
    //      assert_eq!(a.len(), b.len());
    //
    // 2. Initialize a 128-bit accumulator:
    //      let mut acc = _mm_setzero_ps();
    //      // acc = [0.0, 0.0, 0.0, 0.0] — a __m128 value (128-bit SSE register)
    //
    // 3. Main loop — process 4 floats per iteration:
    //
    //      const LANES: usize = 4;
    //      let chunks = a.len() / LANES;
    //
    //      for i in 0..chunks {
    //          let offset = i * LANES;
    //          // _mm_loadu_ps loads 4 floats from a POINTER. The "u" means unaligned —
    //          // the data doesn't need to be 16-byte aligned. Without "u" (_mm_load_ps),
    //          // the pointer MUST be 16-byte aligned or you get a segfault (SIGBUS).
    //          let va = _mm_loadu_ps(a.as_ptr().add(offset));
    //          let vb = _mm_loadu_ps(b.as_ptr().add(offset));
    //
    //          // _mm_mul_ps: 4 parallel multiplications
    //          let prod = _mm_mul_ps(va, vb);
    //
    //          // _mm_add_ps: 4 parallel additions to accumulator
    //          acc = _mm_add_ps(acc, prod);
    //      }
    //
    // 4. Horizontal sum of the 4 accumulator lanes:
    //
    //    There is no single SSE2 instruction for horizontal sum. You must shuffle
    //    and add in a tree pattern:
    //
    //      // acc = [s0, s1, s2, s3]
    //      //
    //      // Step 1: shuffle high pair to low position
    //      let hi = _mm_movehl_ps(acc, acc);
    //      // hi = [s2, s3, s2, s3]
    //      //
    //      // Step 2: add high and low pairs
    //      let sum01 = _mm_add_ps(acc, hi);
    //      // sum01 = [s0+s2, s1+s3, ?, ?]
    //      //
    //      // Step 3: shuffle lane 1 to lane 0 position
    //      let shuf = _mm_shuffle_ps::<0x01>(sum01, sum01);
    //      // shuf = [s1+s3, ?, ?, ?]
    //      //
    //      // Step 4: final add
    //      let total = _mm_add_ss(sum01, shuf);
    //      // total lane 0 = (s0+s2) + (s1+s3) = s0+s1+s2+s3
    //      //
    //      // Note: _mm_add_ss only adds lane 0 (scalar single), ignoring other lanes.
    //      // More efficient than _mm_add_ps when we only need one result.
    //
    // 5. Extract the scalar result:
    //      let mut result = 0.0f32;
    //      _mm_store_ss(&mut result, total);
    //
    // 6. Handle tail elements:
    //      let tail_start = chunks * LANES;
    //      for i in tail_start..a.len() {
    //          result += a[i] * b[i];
    //      }
    //
    //      result
    //
    // IMPORTANT SAFETY:
    //   - This function is `unsafe` because of raw pointer arithmetic (as_ptr().add())
    //     and because it requires SSE2 support (guaranteed on x86-64, but the type
    //     system doesn't know that).
    //   - `_mm_loadu_ps` takes a `*const f32`, not a slice reference. We get the
    //     pointer from `slice.as_ptr()` and offset with `.add(n)`.
    //   - The `u` in `loadu` means unaligned — safe for any pointer alignment.
    //     Without `u`, the pointer MUST be 16-byte aligned.

    todo!("Exercise 5a: Implement dot_product_sse2")
}

/// Dot product using AVX2 + FMA intrinsics (256-bit, 8 floats per vector).
///
/// AVX2 doubles the width to 256 bits (8 floats). FMA (Fused Multiply-Add) provides
/// `_mm256_fmadd_ps(a, b, c)` which computes `a * b + c` in a single instruction
/// with a single rounding step — both faster and more precise than separate mul + add.
///
/// AVX2 intrinsics used:
///   _mm256_setzero_ps()       → create 8 zeros
///   _mm256_loadu_ps(ptr)      → load 8 floats (unaligned)
///   _mm256_fmadd_ps(a, b, c)  → a*b + c (fused, single rounding)
///   _mm256_add_ps(a, b)       → 8-wide addition
///   _mm256_castps256_ps128(v) → extract lower 128 bits
///   _mm256_extractf128_ps(v, 1) → extract upper 128 bits
///
/// FMA is critical for performance:
///   Without FMA: vmulps + vaddps = 2 instructions, 2 roundings
///   With FMA:    vfmadd231ps    = 1 instruction, 1 rounding
///   FMA has the same latency as multiplication alone — the addition is "free."
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    // TODO(human): Implement dot product using AVX2 + FMA intrinsics.
    //
    // This is twice as wide as SSE2 (8 lanes vs 4) and uses fused multiply-add.
    //
    // Steps:
    //
    // 1. Assert equal lengths:
    //      assert_eq!(a.len(), b.len());
    //
    // 2. Initialize 256-bit accumulator:
    //      let mut acc = _mm256_setzero_ps();
    //      // acc = [0.0; 8] — a __m256 value (256-bit AVX register, ymm0-ymm15)
    //
    // 3. Main loop — process 8 floats per iteration:
    //
    //      const LANES: usize = 8;
    //      let chunks = a.len() / LANES;
    //
    //      for i in 0..chunks {
    //          let offset = i * LANES;
    //          let va = _mm256_loadu_ps(a.as_ptr().add(offset));
    //          let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
    //
    //          // Fused multiply-add: acc = va * vb + acc
    //          // _mm256_fmadd_ps(a, b, c) = a * b + c
    //          acc = _mm256_fmadd_ps(va, vb, acc);
    //
    //          // IMPORTANT: _mm256_fmadd_ps computes a*b+c with ONE rounding step,
    //          // while separate _mm256_mul_ps + _mm256_add_ps has TWO roundings.
    //          // This matters for numerical precision in scientific computing.
    //          // FMA is also faster: same latency as mul alone, so the add is "free."
    //      }
    //
    // 4. Horizontal sum of 8 accumulator lanes:
    //
    //    AVX horizontal sum is more complex because 256-bit operations are split
    //    into two 128-bit "lanes" internally. You must:
    //
    //      // Step 1: Extract upper 128 bits and add to lower 128 bits
    //      let hi128 = _mm256_extractf128_ps::<1>(acc);  // upper [4,5,6,7]
    //      let lo128 = _mm256_castps256_ps128(acc);       // lower [0,1,2,3]
    //      let sum128 = _mm_add_ps(lo128, hi128);         // [0+4, 1+5, 2+6, 3+7]
    //
    //      // Step 2: SSE2 horizontal sum of the 128-bit result (same as Exercise 5a)
    //      let hi = _mm_movehl_ps(sum128, sum128);
    //      let sum01 = _mm_add_ps(sum128, hi);
    //      let shuf = _mm_shuffle_ps::<0x01>(sum01, sum01);
    //      let total = _mm_add_ss(sum01, shuf);
    //
    //      let mut result = 0.0f32;
    //      _mm_store_ss(&mut result, total);
    //
    // 5. Handle tail and return:
    //      let tail_start = chunks * LANES;
    //      for i in tail_start..a.len() {
    //          result += a[i] * b[i];
    //      }
    //      result
    //
    // NOTE: After using 256-bit AVX operations, some CPUs require a `_mm256_zeroupper()`
    // call (vzeroupper instruction) before returning to non-AVX code. This avoids a
    // performance penalty caused by mixing AVX and SSE code. The compiler usually
    // inserts this automatically, but be aware of it in hand-written assembly.

    todo!("Exercise 5b: Implement dot_product_avx2_fma")
}

/// Runtime dispatch: select the best dot product implementation for this CPU.
///
/// This is the standard pattern for shipping portable high-performance binaries:
/// compile multiple ISA-specific implementations, then select at runtime based on
/// what the CPU actually supports. The `is_x86_feature_detected!` macro caches its
/// result after the first call, so the overhead is negligible (one branch per call).
///
/// Dispatch order (best → worst):
///   1. AVX2 + FMA (8 lanes, fused multiply-add)
///   2. SSE2 (4 lanes, baseline x86-64)
///   3. Scalar fallback (1 lane)
pub fn dispatch_dot_product(a: &[f32], b: &[f32]) -> f32 {
    // TODO(human): Implement runtime feature detection and dispatch.
    //
    // The dispatch function itself is safe (not `unsafe`). It checks CPU features
    // at runtime and only calls the unsafe intrinsic functions when the CPU supports
    // them — making the `unsafe` blocks justified.
    //
    // Pattern:
    //
    //   #[cfg(target_arch = "x86_64")]
    //   {
    //       if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
    //           println!("  [dispatch] Using AVX2 + FMA (8 lanes, fused multiply-add)");
    //           return unsafe { dot_product_avx2_fma(a, b) };
    //       }
    //
    //       // SSE2 is baseline on x86-64, but check explicitly for clarity
    //       if is_x86_feature_detected!("sse2") {
    //           println!("  [dispatch] Using SSE2 (4 lanes)");
    //           return unsafe { dot_product_sse2(a, b) };
    //       }
    //   }
    //
    //   // Fallback for non-x86 architectures or as a safety net
    //   println!("  [dispatch] Using scalar fallback");
    //   crate::ex3_dot_product::dot_product_scalar(a, b)
    //
    // HOW is_x86_feature_detected! WORKS:
    //   On first call, it executes the CPUID instruction to query CPU capabilities.
    //   The result is cached in a static variable (initialized once via std::sync::Once).
    //   Subsequent calls are just a load + branch — essentially free.
    //
    // WHY this pattern matters:
    //   Your binary runs on ANY x86-64 CPU (even old ones with only SSE2), but
    //   automatically uses AVX2+FMA on modern CPUs for 2x the throughput.
    //   This is how libraries like BLAS (OpenBLAS, MKL), codecs (x264, dav1d),
    //   and runtimes (V8, WASM) ship a single binary that's fast everywhere.
    //
    // COMPILE-TIME ALTERNATIVE:
    //   Instead of runtime dispatch, you can compile the entire binary for a specific
    //   CPU with RUSTFLAGS="-C target-cpu=native". This enables AVX2/FMA globally
    //   but produces a binary that crashes on older CPUs.

    todo!("Exercise 5c: Implement dispatch_dot_product")
}

/// Print which SIMD features the current CPU supports.
///
/// This is a utility function for understanding your hardware. It prints the
/// available x86 SIMD extensions detected at runtime.
pub fn print_cpu_features() {
    println!("  CPU SIMD features detected:");

    #[cfg(target_arch = "x86_64")]
    {
        let features = [
            ("SSE2", is_x86_feature_detected!("sse2")),
            ("SSE3", is_x86_feature_detected!("sse3")),
            ("SSSE3", is_x86_feature_detected!("ssse3")),
            ("SSE4.1", is_x86_feature_detected!("sse4.1")),
            ("SSE4.2", is_x86_feature_detected!("sse4.2")),
            ("AVX", is_x86_feature_detected!("avx")),
            ("AVX2", is_x86_feature_detected!("avx2")),
            ("FMA", is_x86_feature_detected!("fma")),
            ("AVX-512F", is_x86_feature_detected!("avx512f")),
        ];

        for (name, supported) in features {
            let status = if supported { "YES" } else { "no" };
            println!("    {name:>10}: {status}");
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("    (not x86_64 — std::arch x86 features not available)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dot_product_sse2() {
        let a: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..100).map(|i| 1.0 - (i as f32) * 0.005).collect();
        let result = unsafe { dot_product_sse2(&a, &b) };
        let expected = reference_dot(&a, &b);
        assert!(
            (result - expected).abs() < 0.1,
            "SSE2: got {result}, expected {expected}"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dot_product_avx2_fma() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("Skipping AVX2+FMA test — CPU does not support it");
            return;
        }
        let a: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..100).map(|i| 1.0 - (i as f32) * 0.005).collect();
        let result = unsafe { dot_product_avx2_fma(&a, &b) };
        let expected = reference_dot(&a, &b);
        assert!(
            (result - expected).abs() < 0.1,
            "AVX2+FMA: got {result}, expected {expected}"
        );
    }

    #[test]
    fn test_dispatch_dot_product() {
        let a: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..100).map(|i| 1.0 - (i as f32) * 0.005).collect();
        let result = dispatch_dot_product(&a, &b);
        let expected = reference_dot(&a, &b);
        assert!(
            (result - expected).abs() < 0.1,
            "dispatch: got {result}, expected {expected}"
        );
    }
}
