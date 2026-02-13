// Phase 2: Differential Evolution for Continuous Optimization
//
// This binary implements Differential Evolution (DE/rand/1/bin) — one of the
// most effective algorithms for continuous black-box optimization. You implement
// the three core DE operators: differential mutation, binomial crossover, and
// the main DE loop with greedy selection.

use rand::prelude::*;
use std::f64::consts::{E, PI};

// ============================================================================
// Provided: Benchmark functions
// ============================================================================

/// Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
///
/// Highly multimodal with a regular grid of local minima.
/// Global minimum: f(0, 0, ..., 0) = 0.
/// Typical search bounds: [-5.12, 5.12] per dimension.
fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n
        + x.iter()
            .map(|&xi| xi * xi - 10.0 * (2.0 * PI * xi).cos())
            .sum::<f64>()
}

/// Ackley function.
///
/// Has a large basin near the global optimum with many small local minima.
/// Global minimum: f(0, 0, ..., 0) = 0.
/// Typical search bounds: [-5, 5] per dimension.
fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>();
    let sum_cos: f64 = x.iter().map(|&xi| (2.0 * PI * xi).cos()).sum::<f64>();
    -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + 20.0 + E
}

/// Result of a DE run.
#[derive(Debug, Clone)]
struct DEResult {
    best_position: Vec<f64>,
    best_fitness: f64,
    generations_run: usize,
    fitness_history: Vec<f64>,
}

// ============================================================================
// Provided: Initialization helpers
// ============================================================================

/// Initialize a random population within the given bounds.
///
/// bounds: Vec of (lower, upper) per dimension.
/// Returns pop_size vectors, each of length bounds.len().
fn random_population(
    pop_size: usize,
    bounds: &[(f64, f64)],
    rng: &mut impl Rng,
) -> Vec<Vec<f64>> {
    let dim = bounds.len();
    (0..pop_size)
        .map(|_| {
            (0..dim)
                .map(|d| rng.gen_range(bounds[d].0..=bounds[d].1))
                .collect()
        })
        .collect()
}

/// Clip a vector to respect bounds.
fn clip_to_bounds(x: &mut [f64], bounds: &[(f64, f64)]) {
    for (xi, &(lo, hi)) in x.iter_mut().zip(bounds.iter()) {
        *xi = xi.clamp(lo, hi);
    }
}

/// Find the index of the minimum value in a slice.
fn argmin(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
}

/// Print DE generation summary.
fn print_de_generation(gen: usize, best_fitness: f64, avg_fitness: f64) {
    println!(
        "  Gen {:4} | Best: {:12.6} | Avg: {:12.6}",
        gen, best_fitness, avg_fitness
    );
}

// ============================================================================
// TODO(human): DE Mutation (DE/rand/1)
// ============================================================================

/// Perform DE/rand/1 mutation: v = x_a + F * (x_b - x_c).
///
/// # TODO(human): Implement DE mutation
///
/// Pick three random distinct individuals from the population, none of which
/// is the target individual (index `target_idx`). Compute the mutant vector:
///   v[d] = population[a][d] + f_scale * (population[b][d] - population[c][d])
/// for each dimension d.
///
/// Algorithm:
///   1. Collect candidate indices: all indices except target_idx
///   2. Shuffle or sample 3 distinct indices from candidates
///      (use candidates.choose(rng) or similar, removing each chosen one)
///   3. Let a, b, c be the three chosen indices
///   4. For each dimension d:
///        v[d] = population[a][d] + f_scale * (population[b][d] - population[c][d])
///   5. Return v
///
/// Note: Do NOT clip to bounds here — that happens after crossover.
/// f_scale (F) is typically 0.5 to 0.9. Higher F = larger mutations = more exploration.
fn de_mutation(
    population: &[Vec<f64>],
    target_idx: usize,
    f_scale: f64,
    rng: &mut impl Rng,
) -> Vec<f64> {
    // TODO(human): Implement DE/rand/1 mutation as described above.
    //
    // Pseudocode:
    //   candidates = (0..pop_size).filter(|&i| i != target_idx).collect::<Vec<_>>()
    //   shuffle candidates (or use choose_multiple)
    //   a, b, c = candidates[0], candidates[1], candidates[2]
    //   dim = population[0].len()
    //   v = vec![0.0; dim]
    //   for d in 0..dim:
    //       v[d] = population[a][d] + f_scale * (population[b][d] - population[c][d])
    //   return v
    todo!()
}

// ============================================================================
// TODO(human): DE Binomial Crossover
// ============================================================================

/// Perform binomial (uniform) crossover between target and mutant vectors.
///
/// # TODO(human): Implement binomial crossover
///
/// For each dimension d, choose the mutant component with probability `cr`,
/// otherwise keep the target component. Additionally, pick one random
/// dimension `j_rand` that ALWAYS takes the mutant component — this guarantees
/// the trial vector differs from the target in at least one dimension.
///
/// Algorithm:
///   1. Let dim = target.len()
///   2. Pick j_rand = rng.gen_range(0..dim)  (forced mutant dimension)
///   3. For each dimension d in 0..dim:
///        if d == j_rand OR rng.gen::<f64>() < cr:
///            trial[d] = mutant[d]
///        else:
///            trial[d] = target[d]
///   4. Return trial
///
/// CR controls how many dimensions come from the mutant:
///   - CR=0: only j_rand from mutant (very conservative, mostly target)
///   - CR=1: all dimensions from mutant (aggressive, pure mutation)
///   - CR=0.9: typical for DE/rand/1/bin on multimodal functions
fn de_crossover(
    target: &[f64],
    mutant: &[f64],
    cr: f64,
    rng: &mut impl Rng,
) -> Vec<f64> {
    // TODO(human): Implement binomial crossover as described above.
    //
    // Pseudocode:
    //   dim = target.len()
    //   j_rand = rng.gen_range(0..dim)
    //   trial = vec![0.0; dim]
    //   for d in 0..dim:
    //       if d == j_rand || rng.gen::<f64>() < cr:
    //           trial[d] = mutant[d]
    //       else:
    //           trial[d] = target[d]
    //   return trial
    todo!()
}

// ============================================================================
// TODO(human): DE Main Loop
// ============================================================================

/// Run Differential Evolution (DE/rand/1/bin).
///
/// # TODO(human): Implement the DE optimization loop
///
/// DE evolves a population by, for each individual in each generation:
///   1. Generate a mutant vector via de_mutation
///   2. Create a trial vector via de_crossover(target, mutant, cr)
///   3. Clip trial to bounds
///   4. Greedy selection: if f(trial) <= f(target), replace target with trial
///
/// This greedy per-individual selection is what makes DE stable and efficient:
/// no individual ever gets worse, and the population monotonically improves.
///
/// Track the best solution found across all generations.
///
/// Parameters:
///   - f: objective function (minimization)
///   - bounds: search bounds per dimension
///   - pop_size: population size (typically 5*dim to 10*dim)
///   - max_gen: maximum number of generations
///   - f_scale: DE mutation scale factor F (0.5 to 0.9)
///   - cr: crossover rate CR (0.5 to 1.0)
fn differential_evolution(
    f: fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    pop_size: usize,
    max_gen: usize,
    f_scale: f64,
    cr: f64,
    rng: &mut impl Rng,
) -> DEResult {
    // TODO(human): Implement the DE loop as described above.
    //
    // Pseudocode:
    //   population = random_population(pop_size, bounds, rng)
    //   fitnesses = population.iter().map(|x| f(x)).collect::<Vec<_>>()
    //   best_idx = argmin(&fitnesses)
    //   global_best = population[best_idx].clone()
    //   global_best_fit = fitnesses[best_idx]
    //   fitness_history = vec![]
    //
    //   for gen in 0..max_gen:
    //       for i in 0..pop_size:
    //           mutant = de_mutation(&population, i, f_scale, rng)
    //           trial = de_crossover(&population[i], &mutant, cr, rng)
    //           clip_to_bounds(&mut trial, bounds)
    //           trial_fit = f(&trial)
    //           if trial_fit <= fitnesses[i]:
    //               population[i] = trial
    //               fitnesses[i] = trial_fit
    //               if trial_fit < global_best_fit:
    //                   global_best = population[i].clone()
    //                   global_best_fit = trial_fit
    //       fitness_history.push(global_best_fit)
    //       // print every 50 gens
    //
    //   return DEResult { best_position: global_best, best_fitness: global_best_fit, ... }
    todo!()
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== Phase 2: Differential Evolution ===\n");

    let dim = 10;
    let pop_size = 50; // 5 * dim
    let max_gen = 500;
    let f_scale = 0.8;
    let cr = 0.9;
    let mut rng = StdRng::seed_from_u64(42);

    // --- Rastrigin 10D ---
    println!("--- DE on Rastrigin {}D ---", dim);
    println!(
        "Parameters: pop={}, gens={}, F={}, CR={}",
        pop_size, max_gen, f_scale, cr
    );
    println!("Global optimum: f(0, ..., 0) = 0.0\n");

    let bounds_rast: Vec<(f64, f64)> = vec![(-5.12, 5.12); dim];
    let result_rast = differential_evolution(
        rastrigin,
        &bounds_rast,
        pop_size,
        max_gen,
        f_scale,
        cr,
        &mut rng,
    );

    println!("\nRastrigin result:");
    println!("  Best fitness: {:.6}", result_rast.best_fitness);
    println!(
        "  Best position: [{}]",
        result_rast
            .best_position
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  Generations: {}", result_rast.generations_run);

    // --- Ackley 10D ---
    println!("\n--- DE on Ackley {}D ---", dim);
    println!("Global optimum: f(0, ..., 0) = 0.0\n");

    let bounds_ack: Vec<(f64, f64)> = vec![(-5.0, 5.0); dim];
    let result_ack = differential_evolution(
        ackley,
        &bounds_ack,
        pop_size,
        max_gen,
        f_scale,
        cr,
        &mut rng,
    );

    println!("\nAckley result:");
    println!("  Best fitness: {:.6}", result_ack.best_fitness);
    println!(
        "  Best position: [{}]",
        result_ack
            .best_position
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  Generations: {}", result_ack.generations_run);

    // Summary
    println!("\n=== DE Summary ===");
    println!(
        "  Rastrigin {}D: {:.6} (optimum: 0.0)",
        dim, result_rast.best_fitness
    );
    println!(
        "  Ackley    {}D: {:.6} (optimum: 0.0)",
        dim, result_ack.best_fitness
    );
}
