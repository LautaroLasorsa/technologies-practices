//! Practice 062: SIMD in Rust — portable_simd & std::arch
//!
//! This program runs 6 exercises exploring SIMD programming in Rust:
//!
//! 1. Auto-vectorization — observe when the compiler vectorizes and when it fails
//! 2. portable_simd basics — Simd<T,N>, arithmetic, reductions, slice processing
//! 3. SIMD dot product — the canonical accumulator pattern
//! 4. Masks and conditional SIMD — branchless selection and counting
//! 5. std::arch SSE2/AVX2 intrinsics — raw intrinsics with runtime dispatch
//! 6. Swizzle and data rearrangement — lane permutation patterns
//!
//! Run all exercises: `cargo run`
//! Run one exercise: `cargo run -- <number>`
//!
//! Exercise 7 (capstone benchmark) is in `benches/simd_benchmark.rs` — run with `cargo bench`.

#![feature(portable_simd)]

mod ex1_autovec;
mod ex2_portable_basics;
mod ex3_dot_product;
mod ex4_masks;
mod ex5_stdarch;
mod ex6_swizzle;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let exercise: Option<u32> = args.get(1).and_then(|s| s.parse().ok());

    println!("=== Practice 062: SIMD in Rust — portable_simd & std::arch ===");
    println!();

    match exercise {
        Some(1) => run_exercise_1(),
        Some(2) => run_exercise_2(),
        Some(3) => run_exercise_3(),
        Some(4) => run_exercise_4(),
        Some(5) => run_exercise_5(),
        Some(6) => run_exercise_6(),
        Some(n) => eprintln!("Unknown exercise: {}. Valid range: 1-6", n),
        None => run_all(),
    }
}

fn run_all() {
    run_exercise_1();
    run_exercise_2();
    run_exercise_3();
    run_exercise_4();
    run_exercise_5();
    run_exercise_6();

    println!();
    println!("{}", "=".repeat(60));
    println!("All exercises completed!");
    println!("Run `cargo bench` for Exercise 7 (capstone benchmark).");
}

fn run_exercise_1() {
    println!("--- Exercise 1: Auto-Vectorization ---\n");

    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let data2: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.5).collect();
    let mut output = vec![0.0f32; 1024];

    println!("[1a] Sum array (auto-vectorizable):");
    let sum = ex1_autovec::sum_array_autovec(&data);
    println!("  sum of 0..1023 = {sum}");
    println!();

    println!("[1b] Add arrays element-wise (auto-vectorizable):");
    ex1_autovec::add_arrays_autovec(&data, &data2, &mut output);
    println!("  output[0]={}, output[100]={}, output[1023]={}", output[0], output[100], output[1023]);
    println!();

    println!("[1c] Conditional sum (harder to auto-vectorize):");
    let cond_sum = ex1_autovec::sum_with_branch_autovec(&data, 500.0);
    println!("  sum of elements > 500.0 = {cond_sum}");
    println!();
}

fn run_exercise_2() {
    println!("--- Exercise 2: portable_simd Fundamentals ---\n");

    println!("[2a] Basic SIMD arithmetic:");
    ex2_portable_basics::simd_arithmetic();
    println!();

    println!("[2b] Horizontal reductions:");
    ex2_portable_basics::simd_reductions();
    println!();

    println!("[2c] Processing slices with SIMD:");
    let input: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let mut output = vec![0.0f32; 20];
    ex2_portable_basics::simd_from_slice(&input, &mut output, 3.0);
    println!("  input:  {:?}", &input[..8]);
    println!("  output: {:?}", &output[..8]);
    println!();
}

fn run_exercise_3() {
    println!("--- Exercise 3: SIMD Dot Product ---\n");

    let n = 1024;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 0.001).collect();

    let scalar = ex3_dot_product::dot_product_scalar(&a, &b);
    println!("[3a] Scalar dot product:           {scalar:.4}");

    let simd = ex3_dot_product::dot_product_portable_simd(&a, &b);
    println!("[3b] portable_simd dot product:    {simd:.4}");

    let multi = ex3_dot_product::dot_product_multi_accumulator(&a, &b);
    println!("[3c] Multi-accumulator dot product: {multi:.4}");

    let max_diff = (scalar - simd).abs().max((scalar - multi).abs());
    println!("  Max difference from scalar: {max_diff:.2e}");
    println!();
}

fn run_exercise_4() {
    println!("--- Exercise 4: Masks and Conditional SIMD ---\n");

    let data: Vec<f32> = vec![-3.0, -1.5, 0.0, 1.0, 2.5, -0.5, 4.0, -2.0,
                               3.0, -4.0, 1.5, 0.5, -1.0, 2.0, -0.1, 5.0];

    println!("[4a] ReLU (max(0, x)):");
    let mut relu_out = vec![0.0f32; data.len()];
    ex4_masks::relu_simd(&data, &mut relu_out);
    println!("  input:  {:?}", &data[..8]);
    println!("  output: {:?}", &relu_out[..8]);
    println!();

    println!("[4b] Clamp to [-1.0, 3.0]:");
    let mut clamp_out = vec![0.0f32; data.len()];
    ex4_masks::clamp_simd(&data, &mut clamp_out, -1.0, 3.0);
    println!("  input:  {:?}", &data[..8]);
    println!("  output: {:?}", &clamp_out[..8]);
    println!();

    println!("[4c] Count elements > 1.0:");
    let count = ex4_masks::count_threshold_simd(&data, 1.0);
    let expected = data.iter().filter(|&&x| x > 1.0).count();
    println!("  SIMD count: {count}, expected: {expected}");
    println!();
}

fn run_exercise_5() {
    println!("--- Exercise 5: std::arch Intrinsics ---\n");

    let n = 1024;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 0.001).collect();

    let scalar = ex3_dot_product::dot_product_scalar(&a, &b);
    println!("  Scalar baseline: {scalar:.4}");

    let dispatched = ex5_stdarch::dispatch_dot_product(&a, &b);
    println!("  Dispatched result: {dispatched:.4}");
    println!("  Difference: {:.2e}", (scalar - dispatched).abs());

    ex5_stdarch::print_cpu_features();
    println!();
}

fn run_exercise_6() {
    println!("--- Exercise 6: Swizzle and Data Rearrangement ---\n");

    println!("[6a] 2x2 matrix transpose:");
    ex6_swizzle::transpose_2x2_simd();
    println!();

    println!("[6b] Deinterleave RGB:");
    ex6_swizzle::deinterleave_rgb();
    println!();
}
