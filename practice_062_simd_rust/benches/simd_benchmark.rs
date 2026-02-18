//! Exercise 7: Capstone Benchmark — Scalar vs Auto-Vectorized vs Explicit SIMD
//!
//! This criterion benchmark measures the real-world speedup of each SIMD approach
//! for the dot product operation across different array sizes.
//!
//! Run with: `cargo bench`
//!
//! The benchmark compares:
//!   1. Scalar (no SIMD) — the baseline
//!   2. portable_simd (single accumulator) — explicit SIMD, portable
//!   3. portable_simd (multi-accumulator) — exploit ILP for more throughput
//!   4. std::arch SSE2 — raw intrinsics, 4 lanes
//!   5. std::arch AVX2+FMA — raw intrinsics, 8 lanes + fused multiply-add
//!
//! Expected results (on a modern x86-64 CPU with AVX2+FMA):
//!   - For small arrays (64 elements): overhead dominates, scalar may win
//!   - For medium arrays (1024): SIMD starts showing 3-6x speedup
//!   - For large arrays (16K+): SIMD approaches theoretical 4-8x speedup
//!   - AVX2+FMA should be ~2x faster than SSE2 (8 lanes vs 4, plus FMA)
//!   - Multi-accumulator should beat single-accumulator by 20-50%

#![feature(portable_simd)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// Import the implementations from the library crate
use simd_rust::ex3_dot_product;
use simd_rust::ex5_stdarch;

/// Generate test data for benchmarking.
///
/// Uses deterministic values so results are reproducible.
fn generate_data(n: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..n).map(|i| ((i * 7 + 13) % 1000) as f32 * 0.001).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i * 11 + 3) % 1000) as f32 * 0.001).collect();
    (a, b)
}

fn bench_dot_products(c: &mut Criterion) {
    // TODO(human): Create criterion benchmark groups comparing all implementations.
    //
    // This exercise teaches:
    //   - How to set up criterion benchmarks with parameterized sizes
    //   - How to measure throughput (elements/second) not just time
    //   - How to interpret SIMD benchmark results and identify crossover points
    //
    // Steps:
    //
    // 1. Define the array sizes to benchmark:
    //      let sizes = [64, 256, 1024, 4096, 16384, 65536];
    //
    //    Small sizes reveal overhead; large sizes reveal peak throughput.
    //    The crossover where SIMD overhead pays off is typically around 64-256 elements.
    //
    // 2. Create a benchmark group:
    //      let mut group = c.benchmark_group("dot_product");
    //
    // 3. For each size, benchmark each implementation:
    //
    //      for &size in &sizes {
    //          let (a, b) = generate_data(size);
    //
    //          // Report throughput in elements/second (not just time)
    //          group.throughput(Throughput::Elements(size as u64));
    //
    //          // Scalar baseline
    //          group.bench_with_input(
    //              BenchmarkId::new("scalar", size),
    //              &size,
    //              |bencher, &_size| {
    //                  bencher.iter(|| {
    //                      // black_box prevents the compiler from optimizing away the result
    //                      black_box(dot_product::dot_product_scalar(
    //                          black_box(&a),
    //                          black_box(&b),
    //                      ))
    //                  });
    //              },
    //          );
    //
    //          // portable_simd (single accumulator)
    //          group.bench_with_input(
    //              BenchmarkId::new("portable_simd", size),
    //              &size,
    //              |bencher, &_size| {
    //                  bencher.iter(|| {
    //                      black_box(dot_product::dot_product_portable_simd(
    //                          black_box(&a),
    //                          black_box(&b),
    //                      ))
    //                  });
    //              },
    //          );
    //
    //          // portable_simd (multi-accumulator)
    //          group.bench_with_input(
    //              BenchmarkId::new("multi_acc", size),
    //              &size,
    //              |bencher, &_size| {
    //                  bencher.iter(|| {
    //                      black_box(dot_product::dot_product_multi_accumulator(
    //                          black_box(&a),
    //                          black_box(&b),
    //                      ))
    //                  });
    //              },
    //          );
    //
    //          // std::arch SSE2 (4 lanes)
    //          #[cfg(target_arch = "x86_64")]
    //          group.bench_with_input(
    //              BenchmarkId::new("sse2", size),
    //              &size,
    //              |bencher, &_size| {
    //                  bencher.iter(|| {
    //                      black_box(unsafe {
    //                          ex5_stdarch::dot_product_sse2(
    //                              black_box(&a),
    //                              black_box(&b),
    //                          )
    //                      })
    //                  });
    //              },
    //          );
    //
    //          // std::arch AVX2+FMA (8 lanes + fused multiply-add)
    //          #[cfg(target_arch = "x86_64")]
    //          if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
    //              group.bench_with_input(
    //                  BenchmarkId::new("avx2_fma", size),
    //                  &size,
    //                  |bencher, &_size| {
    //                      bencher.iter(|| {
    //                          black_box(unsafe {
    //                              ex5_stdarch::dot_product_avx2_fma(
    //                                  black_box(&a),
    //                                  black_box(&b),
    //                              )
    //                          })
    //                      });
    //                  },
    //              );
    //          }
    //      }
    //
    //      group.finish();
    //
    // INTERPRETING RESULTS:
    //
    //   Criterion outputs time per iteration and throughput. Look for:
    //
    //   1. SPEEDUP: Compare "scalar" time vs others at each size.
    //      Expected: portable_simd ~4-8x faster, AVX2+FMA ~8-16x faster.
    //
    //   2. CROSSOVER: At what size does SIMD overhead pay off?
    //      For very small arrays (< 32 elements), function call overhead and
    //      the horizontal reduction dominate — SIMD may be SLOWER than scalar.
    //
    //   3. THROUGHPUT CEILING: At large sizes, throughput plateaus when the
    //      computation becomes memory-bandwidth bound. At this point, faster SIMD
    //      doesn't help because the CPU is waiting for data from RAM.
    //      For f32 dot product: each element needs 2 loads (8 bytes) and 1 FMA.
    //      If memory bandwidth is 40 GB/s, max throughput is 40/8 = 5 billion
    //      elements/sec — regardless of SIMD width.
    //
    //   4. ILP BENEFIT: Multi-accumulator should beat single-accumulator by
    //      20-50%, demonstrating that instruction-level parallelism matters
    //      even when using SIMD.
    //
    // IMPORTANT: Run with `cargo bench`, NOT `cargo test --bench`. The benchmark
    // harness needs to control timing. Also ensure no other heavy processes are
    // running — CPU frequency scaling and thermal throttling affect results.

    todo!("Exercise 7: Implement criterion benchmarks for dot product comparison")
}

criterion_group!(benches, bench_dot_products);
criterion_main!(benches);
