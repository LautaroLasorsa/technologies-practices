use rand::Rng;
use std::f64::consts::PI;

// =============================================================================
// Rastrigin function (same as Phase 1)
// =============================================================================

fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n + x.iter().map(|&xi| xi * xi - 10.0 * (2.0 * PI * xi).cos()).sum::<f64>()
}

// =============================================================================
// Neighbor generation (same as Phase 1)
// =============================================================================

fn random_neighbor_continuous(x: &[f64], step_size: f64, rng: &mut impl Rng) -> Vec<f64> {
    x.iter()
        .map(|&xi| {
            let u1: f64 = rng.gen_range(0.0001..1.0);
            let u2: f64 = rng.gen_range(0.0..2.0 * PI);
            let noise = step_size * (-2.0 * u1.ln()).sqrt() * u2.cos();
            xi + noise
        })
        .collect()
}

// =============================================================================
// Hill climbing (copy your implementation from Phase 1)
// =============================================================================

/// Greedy local search — same as Phase 1.
/// Copy your working implementation here.
fn hill_climbing(
    f: fn(&[f64]) -> f64,
    x0: &[f64],
    step_size: f64,
    max_iter: usize,
) -> (Vec<f64>, f64) {
    todo!("TODO(human): copy your hill_climbing implementation from Phase 1")
}

// =============================================================================
// Output helpers
// =============================================================================

fn print_result(name: &str, solution: &[f64], value: f64) {
    let coords: Vec<String> = solution.iter().map(|x| format!("{:.4}", x)).collect();
    println!("  {}: f({}) = {:.6}", name, coords.join(", "), value);
}

// =============================================================================
// TODO(human): Multi-start hill climbing
// =============================================================================

/// Multi-start hill climbing: run hill climbing from many random starting points.
///
/// Strategy:
///   1. For each of `n_starts` restarts:
///      a. Generate a random starting point x0 where each component is
///         sampled uniformly from [bounds.0, bounds.1]
///      b. Run hill_climbing(f, x0, step_size, max_iter_per_start)
///      c. Record the (solution, cost) pair
///   2. Return the (solution, cost) with the lowest cost across all runs
///
/// Why this helps:
///   Each hill climbing run converges to the local minimum of whatever basin
///   it starts in. By sampling many starting points, we explore many different
///   basins. If we happen to start in the global minimum's basin, we'll find it.
///
/// Limitations:
///   - No guarantee: the global basin might be small and we might miss it
///   - No learning: each restart is independent, doesn't use info from previous runs
///   - Diminishing returns: going from 100→200 starts helps less than 10→100
///   - For Rastrigin in d dimensions, there are ~10^d local minima, so even
///     many restarts may not suffice in high dimensions
fn multi_start_hill_climbing(
    f: fn(&[f64]) -> f64,
    dim: usize,
    bounds: (f64, f64),
    n_starts: usize,
    step_size: f64,
    max_iter_per_start: usize,
) -> (Vec<f64>, f64) {
    todo!("TODO(human): not implemented")
}

// =============================================================================
// Main — compare single-start vs multi-start
// =============================================================================

fn main() {
    println!("=== Phase 2: Multi-start Hill Climbing on Rastrigin ===\n");
    println!("Global minimum: f(0, 0, ..., 0) = 0.0\n");

    let dim = 2;
    let bounds = (-5.12, 5.12);
    let step_size = 0.1;
    let max_iter = 5_000;

    // --- Single start (baseline) ---
    println!("--- Single start (1 run, {} iterations) ---", max_iter);
    let (sol, val) = multi_start_hill_climbing(rastrigin, dim, bounds, 1, step_size, max_iter);
    print_result("Single", &sol, val);
    println!();

    // --- Multi-start: increasing number of restarts ---
    for &n_starts in &[10, 50, 200] {
        println!(
            "--- Multi-start ({} runs, {} iterations each) ---",
            n_starts, max_iter
        );
        let (sol, val) =
            multi_start_hill_climbing(rastrigin, dim, bounds, n_starts, step_size, max_iter);
        print_result(&format!("{} starts", n_starts), &sol, val);
    }

    println!();

    // --- Higher dimension: 5D ---
    println!("--- 5D Rastrigin: Multi-start comparison ---");
    let dim5 = 5;
    for &n_starts in &[10, 50, 200] {
        let (sol, val) =
            multi_start_hill_climbing(rastrigin, dim5, bounds, n_starts, step_size, max_iter);
        print_result(&format!("{} starts (5D)", n_starts), &sol, val);
    }

    println!("\nObservation: More starts generally find better solutions, but with diminishing returns.");
    println!("In higher dimensions, even 200 starts may not find the global optimum.");
    println!("Multi-start is better than single-start, but still fundamentally limited.");
}
