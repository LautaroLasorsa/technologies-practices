// common.rs — Shared types and test functions for NSGA-II.
//
// This module is included via `#[path = "common.rs"] mod common;` in each binary,
// since each binary has its own `fn main()` and Cargo treats them as separate crates.

use rand::Rng;

/// An individual in the population.
///
/// Contains decision variables, evaluated objective values, and NSGA-II metadata
/// (Pareto rank and crowding distance) assigned during non-dominated sorting.
#[derive(Clone, Debug)]
pub struct Individual {
    /// Decision variables (the solution representation).
    pub x: Vec<f64>,
    /// Objective values (all minimization). Computed by evaluating the test function.
    pub objectives: Vec<f64>,
    /// Pareto front rank: 0 = first front (non-dominated), 1 = second front, etc.
    pub rank: usize,
    /// Crowding distance: how isolated this solution is within its front.
    /// Higher = more isolated = preferred for diversity.
    pub crowding_distance: f64,
}

impl Individual {
    /// Create a new individual with decision variables, no objectives computed yet.
    pub fn new(x: Vec<f64>) -> Self {
        Self {
            x,
            objectives: vec![],
            rank: usize::MAX,
            crowding_distance: 0.0,
        }
    }

    /// Create an individual with known objectives (for testing).
    pub fn with_objectives(x: Vec<f64>, objectives: Vec<f64>) -> Self {
        Self {
            x,
            objectives,
            rank: usize::MAX,
            crowding_distance: 0.0,
        }
    }
}

/// Result of running NSGA-II.
pub struct NSGA2Result {
    /// The final population partitioned into Pareto fronts.
    pub fronts: Vec<Vec<Individual>>,
    /// Number of generations completed.
    pub generations: usize,
}

// ===========================================================================
// ZDT Test Functions
//
// All ZDT problems have the form:
//   f1(x) = x_1
//   f2(x) = g(x_2, ..., x_n) * h(f1, g)
//
// Where:
//   g captures distance from the Pareto front (optimal when g = 1)
//   h determines the front shape
//   x_i in [0, 1] for all i
//   n = 30 (standard, but works for any n >= 2)
//
// The true Pareto front is achieved when g = 1, i.e., x_2 = ... = x_n = 0.
// ===========================================================================

/// ZDT1: Convex Pareto front.
///
/// f1(x) = x_1
/// g(x) = 1 + 9 * sum(x_2..x_n) / (n-1)
/// f2(x) = g * (1 - sqrt(f1/g))
///
/// True Pareto front: f2 = 1 - sqrt(f1), f1 in [0, 1]
/// Shape: convex curve from (0, 1) to (1, 0)
pub fn zdt1(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let f1 = x[0];
    let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (n as f64 - 1.0);
    let f2 = g * (1.0 - (f1 / g).sqrt());
    vec![f1, f2]
}

/// ZDT2: Non-convex (concave) Pareto front.
///
/// f1(x) = x_1
/// g(x) = 1 + 9 * sum(x_2..x_n) / (n-1)
/// f2(x) = g * (1 - (f1/g)^2)
///
/// True Pareto front: f2 = 1 - f1^2, f1 in [0, 1]
/// Shape: concave curve — weighted sum method CANNOT find middle points
pub fn zdt2(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let f1 = x[0];
    let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (n as f64 - 1.0);
    let f2 = g * (1.0 - (f1 / g).powi(2));
    vec![f1, f2]
}

/// ZDT3: Disconnected Pareto front.
///
/// f1(x) = x_1
/// g(x) = 1 + 9 * sum(x_2..x_n) / (n-1)
/// f2(x) = g * (1 - sqrt(f1/g) - (f1/g)*sin(10*pi*f1))
///
/// True Pareto front: f2 = 1 - sqrt(f1) - f1*sin(10*pi*f1), in several disconnected segments
/// Shape: multiple disconnected convex segments
pub fn zdt3(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let f1 = x[0];
    let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (n as f64 - 1.0);
    let h = 1.0 - (f1 / g).sqrt() - (f1 / g) * (10.0 * std::f64::consts::PI * f1).sin();
    let f2 = g * h;
    vec![f1, f2]
}

// ===========================================================================
// Utility functions
// ===========================================================================

/// Check if solution A dominates solution B (all objectives minimized).
///
/// A dominates B if:
///   - A[i] <= B[i] for ALL objectives i  (A is no worse on every objective)
///   - A[j] <  B[j] for SOME objective j  (A is strictly better on at least one)
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    debug_assert_eq!(a.len(), b.len(), "Objective vectors must have same length");
    let mut strictly_better = false;
    for (ai, bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false; // A is worse on this objective → cannot dominate
        }
        if ai < bi {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Create a random population with decision variables uniformly in [bounds.0, bounds.1].
pub fn random_population(
    pop_size: usize,
    n_vars: usize,
    bounds: &[(f64, f64)],
    rng: &mut impl Rng,
) -> Vec<Individual> {
    (0..pop_size)
        .map(|_| {
            let x: Vec<f64> = (0..n_vars)
                .map(|j| {
                    let (lo, hi) = bounds[j];
                    rng.gen_range(lo..=hi)
                })
                .collect();
            Individual::new(x)
        })
        .collect()
}

/// Evaluate an individual's objectives using the given function.
pub fn evaluate(individual: &mut Individual, objective_fn: &dyn Fn(&[f64]) -> Vec<f64>) {
    individual.objectives = objective_fn(&individual.x);
}

/// Evaluate all individuals in a population.
pub fn evaluate_population(
    population: &mut [Individual],
    objective_fn: &dyn Fn(&[f64]) -> Vec<f64>,
) {
    for ind in population.iter_mut() {
        ind.objectives = objective_fn(&ind.x);
    }
}

/// Print a summary of a generation: front sizes, best objectives, etc.
pub fn print_generation_summary(gen: usize, population: &[Individual]) {
    let n = population.len();
    let front0_count = population.iter().filter(|ind| ind.rank == 0).count();

    // Find extreme objective values in the first front
    let front0: Vec<&Individual> = population.iter().filter(|ind| ind.rank == 0).collect();
    if front0.is_empty() {
        println!("Gen {:>4}: pop={}, front0=0", gen, n);
        return;
    }

    let f1_min = front0.iter().map(|ind| ind.objectives[0]).fold(f64::INFINITY, f64::min);
    let f1_max = front0.iter().map(|ind| ind.objectives[0]).fold(f64::NEG_INFINITY, f64::max);
    let f2_min = front0.iter().map(|ind| ind.objectives[1]).fold(f64::INFINITY, f64::min);
    let f2_max = front0.iter().map(|ind| ind.objectives[1]).fold(f64::NEG_INFINITY, f64::max);

    println!(
        "Gen {:>4}: pop={}, front0={}, f1=[{:.4}, {:.4}], f2=[{:.4}, {:.4}]",
        gen, n, front0_count, f1_min, f1_max, f2_min, f2_max
    );
}

/// Print the fronts with individual details.
pub fn print_fronts(fronts: &[Vec<usize>], population: &[Individual]) {
    for (rank, front) in fronts.iter().enumerate() {
        println!("  Front {} (rank {}): {} individuals", rank + 1, rank, front.len());
        for &idx in front {
            let ind = &population[idx];
            let objs: Vec<String> = ind.objectives.iter().map(|o| format!("{:.4}", o)).collect();
            println!("    [{}] objectives=[{}]", idx, objs.join(", "));
        }
    }
}

/// Text-based scatter plot of a 2D Pareto front.
///
/// Renders objectives[0] on x-axis and objectives[1] on y-axis using terminal characters.
pub fn text_scatter_plot(
    individuals: &[Individual],
    title: &str,
    width: usize,
    height: usize,
) {
    if individuals.is_empty() {
        println!("  (no individuals to plot)");
        return;
    }

    let f1: Vec<f64> = individuals.iter().map(|ind| ind.objectives[0]).collect();
    let f2: Vec<f64> = individuals.iter().map(|ind| ind.objectives[1]).collect();

    let f1_min = f1.iter().cloned().fold(f64::INFINITY, f64::min);
    let f1_max = f1.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let f2_min = f2.iter().cloned().fold(f64::INFINITY, f64::min);
    let f2_max = f2.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let f1_range = if (f1_max - f1_min).abs() < 1e-12 { 1.0 } else { f1_max - f1_min };
    let f2_range = if (f2_max - f2_min).abs() < 1e-12 { 1.0 } else { f2_max - f2_min };

    // Create grid
    let mut grid = vec![vec![' '; width]; height];

    for (x, y) in f1.iter().zip(f2.iter()) {
        let col = (((x - f1_min) / f1_range) * (width as f64 - 1.0)).round() as usize;
        let row = ((1.0 - (y - f2_min) / f2_range) * (height as f64 - 1.0)).round() as usize;
        let col = col.min(width - 1);
        let row = row.min(height - 1);
        grid[row][col] = '*';
    }

    // Print
    println!("\n  {} ({} points)", title, individuals.len());
    println!("  f2");
    println!("  {:.2} |{}", f2_max, grid[0].iter().collect::<String>());
    for row in 1..height - 1 {
        println!("       |{}", grid[row].iter().collect::<String>());
    }
    println!("  {:.2} |{}", f2_min, grid[height - 1].iter().collect::<String>());
    println!("       +{}+", "-".repeat(width));
    println!("     {:.2}{}  {:.2}", f1_min, " ".repeat(width - 8), f1_max);
    println!("                f1");
}
