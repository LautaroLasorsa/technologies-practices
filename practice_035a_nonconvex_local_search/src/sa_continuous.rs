use rand::Rng;
use std::f64::consts::PI;

// =============================================================================
// Rastrigin function
// =============================================================================

fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n + x.iter().map(|&xi| xi * xi - 10.0 * (2.0 * PI * xi).cos()).sum::<f64>()
}

// =============================================================================
// Neighbor generation
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
// Cooling schedule
// =============================================================================

/// Geometric cooling: T_{k+1} = alpha * T_k
///
/// This is the most common cooling schedule in practice.
///   alpha close to 1 (e.g., 0.999) → slow cooling → better solutions, more iterations
///   alpha far from 1 (e.g., 0.9)   → fast cooling → quick convergence, worse solutions
fn geometric_cooling(t: f64, alpha: f64) -> f64 {
    t * alpha
}

// =============================================================================
// SA result struct
// =============================================================================

struct SAResult {
    best_solution: Vec<f64>,
    best_cost: f64,
    final_temperature: f64,
    acceptance_count: usize,
    total_iterations: usize,
}

impl SAResult {
    fn acceptance_rate(&self) -> f64 {
        self.acceptance_count as f64 / self.total_iterations as f64
    }
}

// =============================================================================
// TODO(human): Simulated Annealing
// =============================================================================

/// Simulated Annealing with geometric cooling for continuous minimization.
///
/// Algorithm:
///   1. Initialize: temperature = t_init, current = x0, current_cost = f(x0)
///      Track best_solution = current, best_cost = current_cost
///
///   2. For each iteration (0..max_iter):
///      a. Generate neighbor = random_neighbor_continuous(current, step_size)
///      b. Compute neighbor_cost = f(neighbor)
///      c. Compute delta = neighbor_cost - current_cost
///         (positive delta means neighbor is WORSE for minimization)
///
///      d. Acceptance decision:
///         - If delta < 0 (improving): ALWAYS accept
///         - If delta >= 0 (worsening): accept with probability exp(-delta / temperature)
///           → Generate u ~ Uniform(0, 1)
///           → If u < exp(-delta / temperature): accept (this is the Metropolis criterion)
///           → Otherwise: reject
///
///      e. If accepted: current = neighbor, current_cost = neighbor_cost
///         Count this as an acceptance (for statistics)
///      f. If current_cost < best_cost: update best_solution and best_cost
///
///      g. Cool: temperature = geometric_cooling(temperature, alpha)
///
///   3. Return SAResult { best_solution, best_cost, final_temperature,
///                        acceptance_count, total_iterations }
///
/// The key insight: at high temperature, exp(-delta/T) is close to 1 for moderate
/// delta, so most moves are accepted (exploration). As T decreases, only small
/// worsening moves are accepted, and eventually only improvements (exploitation).
fn simulated_annealing(
    f: fn(&[f64]) -> f64,
    x0: &[f64],
    t_init: f64,
    alpha: f64,
    max_iter: usize,
    step_size: f64,
) -> SAResult {
    todo!("TODO(human): not implemented")
}

// =============================================================================
// Hill climbing for comparison (copy from Phase 1)
// =============================================================================

fn hill_climbing(
    f: fn(&[f64]) -> f64,
    x0: &[f64],
    step_size: f64,
    max_iter: usize,
) -> (Vec<f64>, f64) {
    todo!("TODO(human): copy your hill_climbing implementation from Phase 1")
}

fn multi_start_hill_climbing(
    f: fn(&[f64]) -> f64,
    dim: usize,
    bounds: (f64, f64),
    n_starts: usize,
    step_size: f64,
    max_iter_per_start: usize,
) -> (Vec<f64>, f64) {
    todo!("TODO(human): copy your multi_start implementation from Phase 2")
}

// =============================================================================
// Output helpers
// =============================================================================

fn print_sa_result(name: &str, result: &SAResult) {
    let coords: Vec<String> = result
        .best_solution
        .iter()
        .map(|x| format!("{:.4}", x))
        .collect();
    println!("  {}", name);
    println!("    Best: f({}) = {:.6}", coords.join(", "), result.best_cost);
    println!(
        "    Acceptance rate: {:.1}%, final T: {:.6}",
        result.acceptance_rate() * 100.0,
        result.final_temperature
    );
}

fn print_result(name: &str, solution: &[f64], value: f64) {
    let coords: Vec<String> = solution.iter().map(|x| format!("{:.4}", x)).collect();
    println!("  {}: f({}) = {:.6}", name, coords.join(", "), value);
}

// =============================================================================
// Main — SA with different cooling rates, compared with multi-start
// =============================================================================

fn main() {
    let mut rng = rand::thread_rng();

    println!("=== Phase 3: Simulated Annealing on Rastrigin ===\n");
    println!("Global minimum: f(0, 0, ..., 0) = 0.0\n");

    let dim = 2;
    let max_iter = 100_000;
    let step_size = 0.5;

    // --- SA with different cooling rates ---
    println!("--- 2D Rastrigin: Cooling rate comparison ---");
    for &alpha in &[0.9, 0.99, 0.999, 0.9999] {
        let x0: Vec<f64> = (0..dim).map(|_| rng.gen_range(-5.12..5.12)).collect();
        let result = simulated_annealing(rastrigin, &x0, 100.0, alpha, max_iter, step_size);
        print_sa_result(&format!("alpha={}", alpha), &result);
    }

    println!();

    // --- SA vs multi-start hill climbing ---
    println!("--- 2D Rastrigin: SA vs Multi-start (same total evaluations) ---");

    // SA: 100k iterations
    let x0: Vec<f64> = (0..dim).map(|_| rng.gen_range(-5.12..5.12)).collect();
    let sa_result = simulated_annealing(rastrigin, &x0, 100.0, 0.9999, max_iter, step_size);
    print_sa_result("SA (100k iter, alpha=0.9999)", &sa_result);

    // Multi-start: 20 starts x 5k iterations = 100k total evaluations
    let (ms_sol, ms_val) =
        multi_start_hill_climbing(rastrigin, dim, (-5.12, 5.12), 20, 0.1, 5_000);
    print_result("Multi-start (20 x 5k)", &ms_sol, ms_val);

    println!();

    // --- 5D comparison ---
    println!("--- 5D Rastrigin: SA vs Multi-start ---");
    let dim5 = 5;
    let x0_5d: Vec<f64> = (0..dim5).map(|_| rng.gen_range(-5.12..5.12)).collect();
    let sa_5d = simulated_annealing(rastrigin, &x0_5d, 100.0, 0.9999, 200_000, step_size);
    print_sa_result("SA (200k, alpha=0.9999)", &sa_5d);

    let (ms5_sol, ms5_val) =
        multi_start_hill_climbing(rastrigin, dim5, (-5.12, 5.12), 40, 0.1, 5_000);
    print_result("Multi-start (40 x 5k)", &ms5_sol, ms5_val);

    println!("\nObservation: SA with slow cooling typically outperforms multi-start,");
    println!("especially in higher dimensions where random restarts can't cover the space.");
    println!("The Metropolis criterion allows SA to systematically explore and escape local optima.");
}
