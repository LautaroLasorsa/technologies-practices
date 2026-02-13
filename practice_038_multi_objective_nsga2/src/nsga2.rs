// Phase 3: Full NSGA-II
//
// This binary assembles the complete NSGA-II evolutionary algorithm.
// Non-dominated sorting, crowding distance, and comparison are provided
// (implemented). You implement the genetic operators and main loop.

#[path = "common.rs"]
mod common;

use common::*;
use rand::Rng;

// ============================================================================
// Provided: Non-dominated sorting (implemented).
// Copy your working version from Phase 1.
// ============================================================================

fn fast_nondominated_sort(population: &mut [Individual]) -> Vec<Vec<usize>> {
    todo!("TODO(human): Copy your fast_nondominated_sort from Phase 1")
}

// ============================================================================
// Provided: Crowding distance and comparison (implemented).
// Copy your working versions from Phase 2.
// ============================================================================

fn compute_crowding_distance(population: &mut [Individual], front_indices: &[usize]) {
    todo!("TODO(human): Copy your compute_crowding_distance from Phase 2")
}

fn crowded_comparison(a: &Individual, b: &Individual) -> std::cmp::Ordering {
    todo!("TODO(human): Copy your crowded_comparison from Phase 2")
}

// ============================================================================
// TODO(human): Tournament Selection
// ============================================================================

/// Binary tournament selection: pick two random individuals, return the better one.
///
/// # TODO(human): Implement this function
///
/// Binary tournament selection using crowded comparison:
///
///   1. Pick two random indices i and j from the population (i != j)
///      Use rng.gen_range(0..population.len()) for each
///      If i == j, re-pick j until i != j
///
///   2. Compare population[i] and population[j] using crowded_comparison
///      - If i is better (Less): return i
///      - If j is better (Greater): return j
///      - If Equal: return either (e.g., i)
///
/// This is called twice per offspring (once for each parent).
/// The crowded comparison ensures selection pressure toward:
///   (a) individuals on better Pareto fronts (lower rank)
///   (b) individuals in less crowded regions (higher crowding distance)
///
/// Binary tournament is simple but effective — it doesn't require sorting
/// the whole population, just comparing two random samples.
pub fn tournament_selection(population: &[Individual], rng: &mut impl Rng) -> usize {
    todo!("TODO(human): Implement binary tournament selection using crowded comparison")
}

// ============================================================================
// TODO(human): SBX Crossover
// ============================================================================

/// Simulated Binary Crossover (SBX) for real-valued variables.
///
/// # TODO(human): Implement this function
///
/// SBX (Simulated Binary Crossover, Deb & Agrawal 1995):
///
/// Creates two children from two parents. The distribution index eta_c
/// controls how far children can be from parents:
///   - Large eta_c (e.g., 20): children close to parents (exploitation)
///   - Small eta_c (e.g., 2): children far from parents (exploration)
///
/// For each variable i:
///   u = rng.gen::<f64>()   // uniform random in [0, 1)
///
///   if u <= 0.5:
///     beta = (2.0 * u).powf(1.0 / (eta_c + 1.0))
///   else:
///     beta = (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta_c + 1.0))
///
///   child1[i] = 0.5 * ((1.0 + beta) * parent1[i] + (1.0 - beta) * parent2[i])
///   child2[i] = 0.5 * ((1.0 - beta) * parent1[i] + (1.0 + beta) * parent2[i])
///
///   // Clip to bounds
///   child1[i] = child1[i].clamp(bounds[i].0, bounds[i].1)
///   child2[i] = child2[i].clamp(bounds[i].0, bounds[i].1)
///
/// Return (child1, child2) as Vec<f64>.
///
/// The key insight: SBX mimics single-point crossover from binary GAs but
/// works on real-valued variables. The probability distribution of children
/// around parents is controlled by eta_c. Common value: eta_c = 20.
pub fn sbx_crossover(
    parent1: &[f64],
    parent2: &[f64],
    eta_c: f64,
    bounds: &[(f64, f64)],
    rng: &mut impl Rng,
) -> (Vec<f64>, Vec<f64>) {
    todo!("TODO(human): Implement SBX crossover for real-valued variables")
}

// ============================================================================
// TODO(human): Polynomial Mutation
// ============================================================================

/// Polynomial mutation for real-valued variables.
///
/// # TODO(human): Implement this function
///
/// Polynomial Mutation (Deb & Goyal 1996):
///
/// For each variable i:
///   if rng.gen::<f64>() < mutation_prob:
///     u = rng.gen::<f64>()  // uniform random in [0, 1)
///
///     if u < 0.5:
///       delta = (2.0 * u).powf(1.0 / (eta_m + 1.0)) - 1.0
///     else:
///       delta = 1.0 - (2.0 * (1.0 - u)).powf(1.0 / (eta_m + 1.0))
///
///     x[i] = x[i] + delta * (bounds[i].1 - bounds[i].0)
///     x[i] = x[i].clamp(bounds[i].0, bounds[i].1)
///
/// Parameters:
///   eta_m: distribution index (large = small perturbations, common: 20)
///   mutation_prob: probability per variable (common: 1/n_vars)
///
/// The polynomial distribution creates perturbations centered at zero,
/// with most perturbations being small (near the parent) but occasional
/// large jumps. This balance between exploitation and exploration is key.
pub fn polynomial_mutation(
    x: &mut [f64],
    eta_m: f64,
    bounds: &[(f64, f64)],
    mutation_prob: f64,
    rng: &mut impl Rng,
) {
    todo!("TODO(human): Implement polynomial mutation for real-valued variables")
}

// ============================================================================
// TODO(human): Full NSGA-II Main Loop
// ============================================================================

/// Run NSGA-II on the given objective function.
///
/// # TODO(human): Implement this function
///
/// NSGA-II Main Loop:
///
///   1. Initialize random population P of size pop_size using random_population()
///      Evaluate all individuals: evaluate_population(&mut pop, objective_fn)
///
///   2. For each generation (0..generations):
///
///      a. Non-dominated sort the current population:
///         let fronts = fast_nondominated_sort(&mut pop);
///         Compute crowding distance for each front:
///         for front in &fronts { compute_crowding_distance(&mut pop, front); }
///
///      b. Create offspring population Q of size pop_size:
///         Repeat pop_size/2 times (each iteration produces 2 children):
///           - Select parent1 index via tournament_selection(&pop, rng)
///           - Select parent2 index via tournament_selection(&pop, rng)
///           - Create (child1_x, child2_x) via sbx_crossover(
///               &pop[p1].x, &pop[p2].x, eta_c=20.0, bounds, rng)
///           - Apply polynomial_mutation to each child's x
///             (mutation_prob = 1.0 / n_vars as f64, eta_m = 20.0)
///           - Create Individual::new(child1_x) and Individual::new(child2_x)
///           - Push both to offspring
///         Evaluate all offspring: evaluate_population(&mut offspring, objective_fn)
///
///      c. Combine R = P union Q (size 2 * pop_size):
///         let mut combined = pop;
///         combined.extend(offspring);
///
///      d. Non-dominated sort the combined population:
///         let fronts = fast_nondominated_sort(&mut combined);
///
///      e. Fill next generation P' from fronts:
///         let mut next_pop = Vec::with_capacity(pop_size);
///         for front in &fronts:
///           if next_pop.len() + front.len() <= pop_size:
///             // Entire front fits — add all individuals
///             compute_crowding_distance(&mut combined, front);
///             for &idx in front:
///               next_pop.push(combined[idx].clone());
///           else:
///             // Partial front — need to select by crowding distance
///             compute_crowding_distance(&mut combined, front);
///             // Sort front indices by crowded comparison (best first)
///             let mut sorted_front = front.clone();
///             sorted_front.sort_by(|&a, &b| crowded_comparison(&combined[a], &combined[b]));
///             // Take as many as needed to fill pop_size
///             let remaining = pop_size - next_pop.len();
///             for &idx in sorted_front.iter().take(remaining):
///               next_pop.push(combined[idx].clone());
///             break;  // Population is full
///
///      f. pop = next_pop
///         Print progress every 10 generations using print_generation_summary()
///
///   3. Final non-dominated sort to build result:
///      let final_fronts = fast_nondominated_sort(&mut pop);
///      Compute crowding distance for each front.
///      Build NSGA2Result { fronts: vec of vec of Individuals per front, generations }
pub fn nsga2(
    objective_fn: &dyn Fn(&[f64]) -> Vec<f64>,
    n_vars: usize,
    bounds: &[(f64, f64)],
    pop_size: usize,
    generations: usize,
) -> NSGA2Result {
    todo!("TODO(human): Implement the complete NSGA-II evolutionary loop")
}

fn main() {
    println!("{}", "=".repeat(60));
    println!("  Phase 3: Full NSGA-II");
    println!("{}\n", "=".repeat(60));

    // -----------------------------------------------------------------------
    // Run NSGA-II on ZDT1
    //
    // ZDT1: 30 variables, all in [0, 1], convex Pareto front
    // Expected: converge to f2 = 1 - sqrt(f1) in ~100 generations
    // -----------------------------------------------------------------------
    let n_vars = 30;
    let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); n_vars];
    let pop_size = 100;
    let generations = 100;

    println!("=== NSGA-II on ZDT1 ===\n");
    println!("  Variables:    {}", n_vars);
    println!("  Bounds:       [0, 1] for all");
    println!("  Population:   {}", pop_size);
    println!("  Generations:  {}", generations);
    println!("  True front:   f2 = 1 - sqrt(f1)\n");

    let result = nsga2(&zdt1, n_vars, &bounds, pop_size, generations);

    // Print final Pareto front
    println!("\n=== Final Pareto Front (Front 0) ===\n");
    if let Some(front0) = result.fronts.first() {
        println!("  {} solutions on the first front\n", front0.len());

        // Sort by f1 for display
        let mut sorted: Vec<&Individual> = front0.iter().collect();
        sorted.sort_by(|a, b| a.objectives[0].partial_cmp(&b.objectives[0]).unwrap());

        println!("  {:>8}  {:>8}  {:>8}  {:>10}", "f1", "f2", "f2_true", "gap");
        println!("  {:>8}  {:>8}  {:>8}  {:>10}", "----", "----", "-------", "---");
        for ind in sorted.iter().take(20) {
            let f1 = ind.objectives[0];
            let f2 = ind.objectives[1];
            let f2_true = 1.0 - f1.sqrt(); // true ZDT1 Pareto front
            let gap = (f2 - f2_true).abs();
            println!(
                "  {:>8.4}  {:>8.4}  {:>8.4}  {:>10.6}",
                f1, f2, f2_true, gap
            );
        }
        if sorted.len() > 20 {
            println!("  ... ({} more)", sorted.len() - 20);
        }

        // Visualize
        text_scatter_plot(front0, "ZDT1 Pareto Front", 60, 20);
    }

    println!("\n  Total fronts: {}", result.fronts.len());
    println!("  Generations:  {}", result.generations);

    println!("\nPhase 3 complete.");
}
