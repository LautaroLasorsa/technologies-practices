use rand::Rng;
use std::f64::consts::PI;

// =============================================================================
// Rastrigin function — standard non-convex test function
// =============================================================================

/// Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
///
/// Properties:
///   - Global minimum: f(0, 0, ..., 0) = 0
///   - Many local minima: approximately 10^n in the hypercube [-5.12, 5.12]^n
///   - Local minima are arranged in a regular grid, spaced ~1 unit apart
///   - The cosine term creates the "bumps" that trap local search
fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n + x.iter().map(|&xi| xi * xi - 10.0 * (2.0 * PI * xi).cos()).sum::<f64>()
}

// =============================================================================
// Neighbor generation
// =============================================================================

/// Generate a neighbor by adding Gaussian noise to each component.
///
/// Each component x_i is perturbed by N(0, step_size^2).
/// The step_size controls the "radius" of the neighborhood:
///   - Small step_size → fine-grained search near current solution
///   - Large step_size → coarse exploration over larger region
fn random_neighbor_continuous(x: &[f64], step_size: f64, rng: &mut impl Rng) -> Vec<f64> {
    x.iter()
        .map(|&xi| {
            // Box-Muller transform for Gaussian noise
            let u1: f64 = rng.gen_range(0.0001..1.0);
            let u2: f64 = rng.gen_range(0.0..2.0 * PI);
            let noise = step_size * (-2.0 * u1.ln()).sqrt() * u2.cos();
            xi + noise
        })
        .collect()
}

// =============================================================================
// Output helpers
// =============================================================================

fn print_result(name: &str, solution: &[f64], value: f64, iterations: usize) {
    let coords: Vec<String> = solution.iter().map(|x| format!("{:.4}", x)).collect();
    println!(
        "  {}: f({}) = {:.6}  ({} iterations)",
        name,
        coords.join(", "),
        value,
        iterations
    );
}

// =============================================================================
// TODO(human): Hill climbing — greedy local search
// =============================================================================

/// Greedy local search (hill climbing for minimization).
///
/// Starting from `x0`, iteratively generate random neighbors and accept
/// them ONLY if they improve (decrease) the objective value.
///
/// Algorithm:
///   1. Set current = x0, current_cost = f(x0)
///   2. Track the best solution found (initially = current)
///   3. For each iteration up to max_iter:
///      a. Generate a neighbor using random_neighbor_continuous(current, step_size)
///      b. Compute neighbor_cost = f(neighbor)
///      c. If neighbor_cost < current_cost:
///           → Accept: set current = neighbor, current_cost = neighbor_cost
///           → Update best if this is the best seen so far
///         Else:
///           → Reject: do nothing (stay at current)
///   4. Return (best_solution, best_cost)
///
/// This is the simplest possible local search. Its fatal flaw is that it
/// can NEVER escape a local minimum — once all neighbors are worse, it stops
/// improving. The quality of the result depends entirely on the starting point.
fn hill_climbing(
    f: fn(&[f64]) -> f64,
    x0: &[f64],
    step_size: f64,
    max_iter: usize,
) -> (Vec<f64>, f64) {
    todo!("TODO(human): not implemented")
}

// =============================================================================
// Main — demonstrate hill climbing getting stuck in local optima
// =============================================================================

fn main() {
    let mut rng = rand::thread_rng();

    println!("=== Phase 1: Hill Climbing on Rastrigin Function ===\n");
    println!("Global minimum: f(0, 0, ..., 0) = 0.0");
    println!("The Rastrigin function has ~10^n local minima in [-5.12, 5.12]^n.\n");

    // --- 2D Rastrigin ---
    println!("--- 2D Rastrigin (step_size=0.1, 10000 iterations) ---");
    for trial in 1..=5 {
        let x0: Vec<f64> = (0..2).map(|_| rng.gen_range(-5.12..5.12)).collect();
        let start_val = rastrigin(&x0);
        let (sol, val) = hill_climbing(rastrigin, &x0, 0.1, 10_000);
        print_result(&format!("Trial {}", trial), &sol, val, 10_000);
        println!("    Started at f = {:.4}, improved to {:.4}", start_val, val);
    }

    println!();

    // --- 5D Rastrigin ---
    println!("--- 5D Rastrigin (step_size=0.1, 50000 iterations) ---");
    for trial in 1..=5 {
        let x0: Vec<f64> = (0..5).map(|_| rng.gen_range(-5.12..5.12)).collect();
        let start_val = rastrigin(&x0);
        let (sol, val) = hill_climbing(rastrigin, &x0, 0.1, 50_000);
        print_result(&format!("Trial {}", trial), &sol, val, 50_000);
        println!("    Started at f = {:.4}, improved to {:.4}", start_val, val);
    }

    println!("\nObservation: Hill climbing almost never finds f ≈ 0.");
    println!("Each run gets stuck in a different local minimum, determined by the starting point.");
    println!("Higher dimensions make it even harder — more local minima to get trapped in.");
}
