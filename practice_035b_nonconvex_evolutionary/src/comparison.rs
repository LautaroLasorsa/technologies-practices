// Phase 4: Comparison — GA (continuous), DE, and PSO on the same problems
//
// This binary runs all three evolutionary algorithms on the same continuous
// benchmark functions and compares solution quality, convergence speed, and
// wall-clock time. You implement the comparison runner that orchestrates
// fair experiments.

use rand::prelude::*;
use std::f64::consts::{E, PI};
use std::time::Instant;

// ============================================================================
// Provided: Benchmark functions
// ============================================================================

/// Rastrigin function. Global minimum: f(0,...,0) = 0. Bounds: [-5.12, 5.12].
fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n
        + x.iter()
            .map(|&xi| xi * xi - 10.0 * (2.0 * PI * xi).cos())
            .sum::<f64>()
}

/// Ackley function. Global minimum: f(0,...,0) = 0. Bounds: [-5, 5].
fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>();
    let sum_cos: f64 = x.iter().map(|&xi| (2.0 * PI * xi).cos()).sum::<f64>();
    -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + 20.0 + E
}

/// Schwefel function. Global minimum: f(420.9687,...) ~ 0. Bounds: [-500, 500].
///
/// Deceptive: the global optimum is far from the next-best local optimum,
/// making it hard for algorithms that converge to the nearest good region.
fn schwefel(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    418.9829 * n - x.iter().map(|&xi| xi * (xi.abs().sqrt()).sin()).sum::<f64>()
}

// ============================================================================
// Provided: Algorithm result type
// ============================================================================

/// Unified result from any algorithm.
#[derive(Debug, Clone)]
struct AlgorithmResult {
    name: String,
    best_fitness: f64,
    best_position: Vec<f64>,
    evaluations: usize,
    elapsed_ms: f64,
}

// ============================================================================
// Provided: GA adapted for continuous optimization
// ============================================================================

/// Real-coded GA for continuous optimization.
///
/// Uses real-valued chromosomes, tournament selection, BLX-alpha crossover
/// (blend crossover), and Gaussian mutation. This is the standard way to
/// adapt GA for continuous problems.
fn ga_continuous(
    f: fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    pop_size: usize,
    max_gen: usize,
    rng: &mut impl Rng,
) -> AlgorithmResult {
    let dim = bounds.len();
    let crossover_rate = 0.8;
    let mutation_rate = 0.1;
    let mutation_sigma = 0.1; // fraction of range
    let tournament_size = 3;
    let mut evals = 0usize;

    // Initialize population
    let mut population: Vec<Vec<f64>> = (0..pop_size)
        .map(|_| {
            (0..dim)
                .map(|d| rng.gen_range(bounds[d].0..=bounds[d].1))
                .collect()
        })
        .collect();

    let mut fitnesses: Vec<f64> = population.iter().map(|x| { evals += 1; f(x) }).collect();
    let mut best_idx = argmin_f64(&fitnesses);
    let mut global_best = population[best_idx].clone();
    let mut global_best_fit = fitnesses[best_idx];

    for _gen in 0..max_gen {
        // Tournament selection + crossover + mutation
        let elite = population[best_idx].clone();
        let mut new_pop = vec![elite];

        while new_pop.len() < pop_size {
            // Tournament select two parents
            let p1 = tournament_select(&population, &fitnesses, tournament_size, rng);
            let p2 = tournament_select(&population, &fitnesses, tournament_size, rng);

            let (mut c1, mut c2) = if rng.gen::<f64>() < crossover_rate {
                blx_alpha_crossover(&p1, &p2, 0.5, bounds, rng)
            } else {
                (p1, p2)
            };

            gaussian_mutate(&mut c1, mutation_rate, mutation_sigma, bounds, rng);
            gaussian_mutate(&mut c2, mutation_rate, mutation_sigma, bounds, rng);

            new_pop.push(c1);
            new_pop.push(c2);
        }
        new_pop.truncate(pop_size);

        population = new_pop;
        fitnesses = population.iter().map(|x| { evals += 1; f(x) }).collect();
        best_idx = argmin_f64(&fitnesses);

        if fitnesses[best_idx] < global_best_fit {
            global_best = population[best_idx].clone();
            global_best_fit = fitnesses[best_idx];
        }
    }

    AlgorithmResult {
        name: "GA (continuous)".to_string(),
        best_fitness: global_best_fit,
        best_position: global_best,
        evaluations: evals,
        elapsed_ms: 0.0, // set by caller
    }
}

/// Tournament selection (minimization).
fn tournament_select(
    pop: &[Vec<f64>],
    fitnesses: &[f64],
    k: usize,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let mut best_idx = rng.gen_range(0..pop.len());
    for _ in 1..k {
        let idx = rng.gen_range(0..pop.len());
        if fitnesses[idx] < fitnesses[best_idx] {
            best_idx = idx;
        }
    }
    pop[best_idx].clone()
}

/// BLX-alpha crossover for real-coded GA.
fn blx_alpha_crossover(
    p1: &[f64],
    p2: &[f64],
    alpha: f64,
    bounds: &[(f64, f64)],
    rng: &mut impl Rng,
) -> (Vec<f64>, Vec<f64>) {
    let dim = p1.len();
    let mut c1 = vec![0.0; dim];
    let mut c2 = vec![0.0; dim];
    for d in 0..dim {
        let lo = p1[d].min(p2[d]);
        let hi = p1[d].max(p2[d]);
        let range = hi - lo;
        let ext_lo = (lo - alpha * range).max(bounds[d].0);
        let ext_hi = (hi + alpha * range).min(bounds[d].1);
        c1[d] = rng.gen_range(ext_lo..=ext_hi);
        c2[d] = rng.gen_range(ext_lo..=ext_hi);
    }
    (c1, c2)
}

/// Gaussian mutation for real-coded GA.
fn gaussian_mutate(
    x: &mut [f64],
    rate: f64,
    sigma_frac: f64,
    bounds: &[(f64, f64)],
    rng: &mut impl Rng,
) {
    for d in 0..x.len() {
        if rng.gen::<f64>() < rate {
            let range = bounds[d].1 - bounds[d].0;
            let sigma = sigma_frac * range;
            // Box-Muller for normal random
            let u1: f64 = rng.gen_range(1e-10..1.0);
            let u2: f64 = rng.gen::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            x[d] = (x[d] + sigma * z).clamp(bounds[d].0, bounds[d].1);
        }
    }
}

// ============================================================================
// Provided: DE implementation (copy from Phase 2, fully implemented)
// ============================================================================

fn de_run(
    f: fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    pop_size: usize,
    max_gen: usize,
    f_scale: f64,
    cr: f64,
    rng: &mut impl Rng,
) -> AlgorithmResult {
    let dim = bounds.len();
    let mut evals = 0usize;

    let mut population: Vec<Vec<f64>> = (0..pop_size)
        .map(|_| {
            (0..dim)
                .map(|d| rng.gen_range(bounds[d].0..=bounds[d].1))
                .collect()
        })
        .collect();
    let mut fitnesses: Vec<f64> = population.iter().map(|x| { evals += 1; f(x) }).collect();
    let mut best_idx = argmin_f64(&fitnesses);
    let mut global_best = population[best_idx].clone();
    let mut global_best_fit = fitnesses[best_idx];

    for _gen in 0..max_gen {
        for i in 0..pop_size {
            // DE/rand/1 mutation
            let mut candidates: Vec<usize> = (0..pop_size).filter(|&j| j != i).collect();
            let a = candidates.remove(rng.gen_range(0..candidates.len()));
            let b = candidates.remove(rng.gen_range(0..candidates.len()));
            let c = candidates.remove(rng.gen_range(0..candidates.len()));

            let mutant: Vec<f64> = (0..dim)
                .map(|d| population[a][d] + f_scale * (population[b][d] - population[c][d]))
                .collect();

            // Binomial crossover
            let j_rand = rng.gen_range(0..dim);
            let mut trial: Vec<f64> = (0..dim)
                .map(|d| {
                    if d == j_rand || rng.gen::<f64>() < cr {
                        mutant[d]
                    } else {
                        population[i][d]
                    }
                })
                .collect();

            // Clip
            for d in 0..dim {
                trial[d] = trial[d].clamp(bounds[d].0, bounds[d].1);
            }

            let trial_fit = f(&trial);
            evals += 1;
            if trial_fit <= fitnesses[i] {
                population[i] = trial;
                fitnesses[i] = trial_fit;
                if trial_fit < global_best_fit {
                    global_best = population[i].clone();
                    global_best_fit = trial_fit;
                }
            }
        }
    }

    AlgorithmResult {
        name: "DE".to_string(),
        best_fitness: global_best_fit,
        best_position: global_best,
        evaluations: evals,
        elapsed_ms: 0.0,
    }
}

// ============================================================================
// Provided: PSO implementation (copy from Phase 3, fully implemented)
// ============================================================================

fn pso_run(
    f: fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    n_particles: usize,
    max_iter: usize,
    w: f64,
    c1: f64,
    c2: f64,
    rng: &mut impl Rng,
) -> AlgorithmResult {
    let dim = bounds.len();
    let mut evals = 0usize;

    // Initialize swarm
    let mut positions: Vec<Vec<f64>> = (0..n_particles)
        .map(|_| {
            (0..dim)
                .map(|d| rng.gen_range(bounds[d].0..=bounds[d].1))
                .collect()
        })
        .collect();
    let mut velocities: Vec<Vec<f64>> = (0..n_particles)
        .map(|_| {
            (0..dim)
                .map(|d| {
                    let range = bounds[d].1 - bounds[d].0;
                    rng.gen_range(-0.1 * range..=0.1 * range)
                })
                .collect()
        })
        .collect();
    let mut pbest: Vec<Vec<f64>> = positions.clone();
    let mut pbest_cost: Vec<f64> = positions.iter().map(|x| { evals += 1; f(x) }).collect();
    let mut gbest_idx = argmin_f64(&pbest_cost);
    let mut gbest = positions[gbest_idx].clone();
    let mut gbest_cost = pbest_cost[gbest_idx];

    for _iter in 0..max_iter {
        for i in 0..n_particles {
            // Update velocity
            for d in 0..dim {
                let r1: f64 = rng.gen();
                let r2: f64 = rng.gen();
                velocities[i][d] = w * velocities[i][d]
                    + c1 * r1 * (pbest[i][d] - positions[i][d])
                    + c2 * r2 * (gbest[d] - positions[i][d]);
            }
            // Update position
            for d in 0..dim {
                positions[i][d] += velocities[i][d];
                positions[i][d] = positions[i][d].clamp(bounds[d].0, bounds[d].1);
            }
            let cost = f(&positions[i]);
            evals += 1;
            if cost < pbest_cost[i] {
                pbest[i] = positions[i].clone();
                pbest_cost[i] = cost;
                if cost < gbest_cost {
                    gbest = positions[i].clone();
                    gbest_cost = cost;
                }
            }
        }
    }

    AlgorithmResult {
        name: "PSO".to_string(),
        best_fitness: gbest_cost,
        best_position: gbest,
        evaluations: evals,
        elapsed_ms: 0.0,
    }
}

// ============================================================================
// Provided: Helper functions
// ============================================================================

fn argmin_f64(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
}

/// Print a formatted comparison table.
fn print_comparison_table(results: &[AlgorithmResult], label: &str) {
    println!("\n  {:<20} {:>14} {:>12} {:>10}", "Algorithm", "Best Fitness", "Evaluations", "Time (ms)");
    println!("  {}", "-".repeat(60));
    for r in results {
        println!(
            "  {:<20} {:>14.6} {:>12} {:>10.1}",
            r.name, r.best_fitness, r.evaluations, r.elapsed_ms
        );
    }
    println!();

    // Rank by best fitness
    let mut ranked: Vec<(usize, f64)> = results.iter().enumerate().map(|(i, r)| (i, r.best_fitness)).collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    println!("  Ranking on {}:", label);
    for (rank, (idx, _fit)) in ranked.iter().enumerate() {
        println!("    {}. {} (fitness: {:.6})", rank + 1, results[*idx].name, results[*idx].best_fitness);
    }
}

// ============================================================================
// TODO(human): Comparison Runner
// ============================================================================

/// Run all three algorithms on the same function and compare results.
///
/// # TODO(human): Implement the comparison runner
///
/// For a fair comparison, all algorithms should use:
///   - The same function and bounds
///   - Similar total function evaluations (pop_size * generations)
///   - Independent RNG seeds (so results are reproducible but independent)
///
/// Algorithm:
///   1. Set up parameters so all three algorithms do approximately the same
///      number of function evaluations:
///        - GA:  pop_size=50,  generations=500   → ~25,000 evals
///        - DE:  pop_size=50,  generations=500   → ~25,000 evals
///        - PSO: particles=50, iterations=500    → ~25,000 evals
///   2. For each algorithm:
///        a. Create an RNG with a distinct seed (e.g., base_seed + algo_index)
///        b. Record start time: Instant::now()
///        c. Run the algorithm
///        d. Record elapsed time: start.elapsed().as_secs_f64() * 1000.0
///        e. Store the result
///   3. Call print_comparison_table with all results
///
/// Use the provided ga_continuous(), de_run(), pso_run() functions.
/// Each returns an AlgorithmResult — you just need to set elapsed_ms.
fn run_comparison(
    f: fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    _dim: usize,
    label: &str,
    base_seed: u64,
) {
    // TODO(human): Implement comparison runner as described above.
    //
    // Pseudocode:
    //   let pop_size = 50;
    //   let generations = 500;
    //   let mut results = vec![];
    //
    //   // GA
    //   let mut rng_ga = StdRng::seed_from_u64(base_seed);
    //   let start = Instant::now();
    //   let mut ga_result = ga_continuous(f, bounds, pop_size, generations, &mut rng_ga);
    //   ga_result.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    //   results.push(ga_result);
    //
    //   // DE
    //   let mut rng_de = StdRng::seed_from_u64(base_seed + 1);
    //   let start = Instant::now();
    //   let mut de_result = de_run(f, bounds, pop_size, generations, 0.8, 0.9, &mut rng_de);
    //   de_result.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    //   results.push(de_result);
    //
    //   // PSO
    //   let mut rng_pso = StdRng::seed_from_u64(base_seed + 2);
    //   let start = Instant::now();
    //   let mut pso_result = pso_run(f, bounds, pop_size, generations, 0.7, 1.5, 1.5, &mut rng_pso);
    //   pso_result.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    //   results.push(pso_result);
    //
    //   print_comparison_table(&results, label);
    todo!()
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== Phase 4: Algorithm Comparison ===\n");
    println!("Comparing GA (continuous), DE, and PSO on the same benchmark functions.");
    println!("Each algorithm gets ~25,000 function evaluations (pop=50, gens=500).\n");

    let dim = 10;

    // --- Rastrigin 10D ---
    println!("========== Rastrigin {}D ==========", dim);
    println!("Global optimum: f(0,...,0) = 0.0, bounds [-5.12, 5.12]");
    let bounds_rast: Vec<(f64, f64)> = vec![(-5.12, 5.12); dim];
    run_comparison(rastrigin, &bounds_rast, dim, "Rastrigin 10D", 100);

    // --- Ackley 10D ---
    println!("\n========== Ackley {}D ==========", dim);
    println!("Global optimum: f(0,...,0) = 0.0, bounds [-5, 5]");
    let bounds_ack: Vec<(f64, f64)> = vec![(-5.0, 5.0); dim];
    run_comparison(ackley, &bounds_ack, dim, "Ackley 10D", 200);

    // --- Schwefel 10D ---
    println!("\n========== Schwefel {}D ==========", dim);
    println!("Global optimum: f(420.9687,...) ~ 0.0, bounds [-500, 500]");
    let bounds_sch: Vec<(f64, f64)> = vec![(-500.0, 500.0); dim];
    run_comparison(schwefel, &bounds_sch, dim, "Schwefel 10D", 300);

    println!("\n=== Comparison Complete ===");
    println!("\nExpected general trends:");
    println!("  - DE typically performs best on Rastrigin and Ackley (continuous, multimodal)");
    println!("  - PSO converges fast on Ackley (large basin near optimum)");
    println!("  - Schwefel is deceptive — tests ability to escape distant local optima");
    println!("  - GA (continuous) is generally weakest due to less sophisticated operators");
    println!("\nNote: Results vary with random seed. Run multiple trials for robust conclusions.");
}
