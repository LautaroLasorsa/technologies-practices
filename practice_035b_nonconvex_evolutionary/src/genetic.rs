// Phase 1: Genetic Algorithm for Binary Knapsack
//
// This binary implements a Genetic Algorithm (GA) to solve the 0-1 knapsack
// problem. The chromosome is a boolean vector where each gene indicates
// whether an item is included. You implement the core GA operators:
// tournament selection, single-point crossover, bit-flip mutation, and
// the main generational loop.

use rand::prelude::*;

// ============================================================================
// Knapsack problem definition
// ============================================================================

#[derive(Debug, Clone)]
struct KnapsackItem {
    name: &'static str,
    weight: f64,
    value: f64,
}

/// Result of a GA run.
#[derive(Debug, Clone)]
struct GAResult {
    best_chromosome: Vec<bool>,
    best_fitness: f64,
    best_value: f64,
    generations_run: usize,
    fitness_history: Vec<f64>,
}

/// Build the knapsack problem instance (15 items).
fn knapsack_items() -> Vec<KnapsackItem> {
    vec![
        KnapsackItem { name: "Laptop",       weight: 3.0,  value: 26.0 },
        KnapsackItem { name: "Camera",       weight: 2.0,  value: 22.0 },
        KnapsackItem { name: "Headphones",   weight: 1.0,  value: 12.0 },
        KnapsackItem { name: "Book",         weight: 1.5,  value: 8.0  },
        KnapsackItem { name: "Jacket",       weight: 2.5,  value: 14.0 },
        KnapsackItem { name: "Tent",         weight: 5.0,  value: 30.0 },
        KnapsackItem { name: "Sleeping bag", weight: 3.5,  value: 18.0 },
        KnapsackItem { name: "Stove",        weight: 2.0,  value: 10.0 },
        KnapsackItem { name: "Water filter", weight: 0.5,  value: 15.0 },
        KnapsackItem { name: "First aid",    weight: 1.0,  value: 20.0 },
        KnapsackItem { name: "Rope",         weight: 1.5,  value: 6.0  },
        KnapsackItem { name: "Flashlight",   weight: 0.5,  value: 9.0  },
        KnapsackItem { name: "Map",          weight: 0.2,  value: 5.0  },
        KnapsackItem { name: "Compass",      weight: 0.1,  value: 7.0  },
        KnapsackItem { name: "Binoculars",   weight: 1.5,  value: 11.0 },
    ]
}

const CAPACITY: f64 = 15.0;

// ============================================================================
// Provided: Fitness function
// ============================================================================

/// Evaluate the fitness of a chromosome for the knapsack problem.
///
/// If total weight <= capacity: fitness = total value.
/// If total weight > capacity: fitness = 0.0 (death penalty for infeasibility).
///
/// The death penalty is simple but effective for knapsack. More sophisticated
/// approaches use repair operators or penalty proportional to excess weight,
/// but the death penalty works well with tournament selection because infeasible
/// solutions are always worse than any feasible solution.
fn fitness(chromosome: &[bool], items: &[KnapsackItem], capacity: f64) -> f64 {
    let (total_weight, total_value) = chromosome
        .iter()
        .zip(items.iter())
        .filter(|(&gene, _)| gene)
        .fold((0.0, 0.0), |(w, v), (_, item)| {
            (w + item.weight, v + item.value)
        });

    if total_weight <= capacity {
        total_value
    } else {
        0.0
    }
}

// ============================================================================
// Provided: Random chromosome generation
// ============================================================================

/// Generate a random chromosome (bit vector of length n).
fn random_chromosome(n: usize, rng: &mut impl Rng) -> Vec<bool> {
    (0..n).map(|_| rng.gen_bool(0.5)).collect()
}

/// Generate a random population of chromosomes.
fn random_population(pop_size: usize, n_genes: usize, rng: &mut impl Rng) -> Vec<Vec<bool>> {
    (0..pop_size)
        .map(|_| random_chromosome(n_genes, rng))
        .collect()
}

// ============================================================================
// Provided: Display helpers
// ============================================================================

/// Print a chromosome as selected items with total weight and value.
fn print_solution(chromosome: &[bool], items: &[KnapsackItem]) {
    let mut total_weight = 0.0;
    let mut total_value = 0.0;
    let selected: Vec<&str> = chromosome
        .iter()
        .zip(items.iter())
        .filter(|(&gene, _)| gene)
        .map(|(_, item)| {
            total_weight += item.weight;
            total_value += item.value;
            item.name
        })
        .collect();
    println!("  Items: {:?}", selected);
    println!("  Weight: {:.1} / {:.1}", total_weight, CAPACITY);
    println!("  Value:  {:.1}", total_value);
}

/// Print generation summary.
fn print_generation(gen: usize, best_fitness: f64, avg_fitness: f64, pop_size: usize) {
    println!(
        "  Gen {:4} | Best: {:7.1} | Avg: {:7.1} | Pop: {}",
        gen, best_fitness, avg_fitness, pop_size
    );
}

// ============================================================================
// TODO(human): Tournament Selection
// ============================================================================

/// Select one individual via tournament selection.
///
/// # TODO(human): Implement tournament selection
///
/// Tournament selection picks `tournament_size` random individuals from the
/// population, then returns (a clone of) the one with the highest fitness.
///
/// Algorithm:
///   1. Pick `tournament_size` random indices from 0..population.len()
///      (with replacement — same individual can appear twice in a tournament)
///   2. Among those indices, find the one with the highest fitness
///   3. Return a clone of that individual's chromosome
///
/// Tournament size controls selection pressure:
///   - k=1: no selection pressure (random selection)
///   - k=2: moderate pressure (common default)
///   - k=pop_size: always selects the best (too greedy, kills diversity)
///
/// Use rng.gen_range(0..population.len()) to pick random indices.
/// Use fitnesses[idx] to look up fitness of each candidate.
/// Return population[best_idx].clone().
fn tournament_selection(
    population: &[Vec<bool>],
    fitnesses: &[f64],
    tournament_size: usize,
    rng: &mut impl Rng,
) -> Vec<bool> {
    // TODO(human): Implement tournament selection as described above.
    //
    // Pseudocode:
    //   best_idx = random index
    //   for _ in 1..tournament_size:
    //       candidate_idx = random index
    //       if fitnesses[candidate_idx] > fitnesses[best_idx]:
    //           best_idx = candidate_idx
    //   return population[best_idx].clone()
    todo!()
}

// ============================================================================
// TODO(human): Single-Point Crossover
// ============================================================================

/// Perform single-point crossover on two parent chromosomes.
///
/// # TODO(human): Implement single-point crossover
///
/// Single-point crossover chooses a random crossover point `p` in [1, len-1],
/// then creates two children by swapping the tails:
///   child1 = parent1[0..p] + parent2[p..]
///   child2 = parent2[0..p] + parent1[p..]
///
/// Algorithm:
///   1. Let n = parent1.len() (both parents have same length)
///   2. Pick random crossover point p in range 1..n (exclusive of 0 and n
///      so both segments are non-empty)
///   3. child1 = parent1[..p] concatenated with parent2[p..]
///   4. child2 = parent2[..p] concatenated with parent1[p..]
///   5. Return (child1, child2)
///
/// Use rng.gen_range(1..n) for the crossover point.
/// Use [slice1, slice2].concat() or iterators to build child vectors.
fn single_point_crossover(
    parent1: &[bool],
    parent2: &[bool],
    rng: &mut impl Rng,
) -> (Vec<bool>, Vec<bool>) {
    // TODO(human): Implement single-point crossover as described above.
    //
    // Pseudocode:
    //   n = parent1.len()
    //   p = random in 1..n
    //   child1 = parent1[..p].to_vec() extended with parent2[p..].iter()
    //   child2 = parent2[..p].to_vec() extended with parent1[p..].iter()
    //   return (child1, child2)
    todo!()
}

// ============================================================================
// TODO(human): Bit-Flip Mutation
// ============================================================================

/// Mutate a chromosome by flipping each gene with probability `mutation_rate`.
///
/// # TODO(human): Implement bit-flip mutation
///
/// For each gene in the chromosome, generate a random number in [0, 1).
/// If it is less than mutation_rate, flip the gene (true <-> false).
///
/// Typical mutation_rate for binary GAs: 1/n to 0.05 where n is chromosome length.
/// Too low: population loses diversity, gets stuck.
/// Too high: search becomes random walk, destroys good solutions.
///
/// Algorithm:
///   for each gene in chromosome (mutable):
///       if rng.gen::<f64>() < mutation_rate:
///           *gene = !*gene
///
/// This mutates in-place (chromosome is &mut [bool]).
/// Use rng.gen::<f64>() to generate uniform random in [0, 1).
fn mutate(chromosome: &mut [bool], mutation_rate: f64, rng: &mut impl Rng) {
    // TODO(human): Implement bit-flip mutation as described above.
    //
    // Pseudocode:
    //   for gene in chromosome.iter_mut():
    //       if rng.gen::<f64>() < mutation_rate:
    //           *gene = !*gene
    todo!()
}

// ============================================================================
// TODO(human): Main GA Loop
// ============================================================================

/// Run the Genetic Algorithm on the knapsack problem.
///
/// # TODO(human): Implement the GA generational loop
///
/// The GA evolves a population over `generations` iterations. Each generation:
///   1. Evaluate fitness of all individuals
///   2. Record the best individual (elitism — keep it for next generation)
///   3. Build a new population:
///      a. Start with the elite individual (elitism, 1 copy)
///      b. While new_population.len() < pop_size:
///         - Select two parents via tournament_selection
///         - With probability crossover_rate, apply single_point_crossover;
///           otherwise, clone parents as children
///         - Apply mutate to each child
///         - Add children to new_population
///      c. Truncate new_population to exactly pop_size (in case of odd overflow)
///   4. Replace old population with new_population
///
/// Track the best solution found across ALL generations (not just the last one),
/// because elitism only preserves one copy and mutation could theoretically
/// degrade it (though unlikely with low mutation rates).
///
/// Return a GAResult with the best chromosome, its fitness, the total value,
/// number of generations, and a history of best fitness per generation.
fn genetic_algorithm(
    items: &[KnapsackItem],
    capacity: f64,
    pop_size: usize,
    generations: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    tournament_size: usize,
    rng: &mut impl Rng,
) -> GAResult {
    // TODO(human): Implement the GA loop as described above.
    //
    // Pseudocode:
    //   population = random_population(pop_size, items.len(), rng)
    //   global_best = None
    //   fitness_history = vec![]
    //
    //   for gen in 0..generations:
    //       fitnesses = population.iter().map(|c| fitness(c, items, capacity)).collect()
    //       best_idx = argmax(fitnesses)
    //       if fitnesses[best_idx] > global_best_fitness:
    //           global_best = (population[best_idx].clone(), fitnesses[best_idx])
    //       fitness_history.push(fitnesses[best_idx])
    //       print_generation(gen, ...) // every 10 or 25 gens
    //
    //       // Build next generation
    //       elite = population[best_idx].clone()
    //       new_pop = vec![elite]
    //       while new_pop.len() < pop_size:
    //           p1 = tournament_selection(...)
    //           p2 = tournament_selection(...)
    //           if rng.gen::<f64>() < crossover_rate:
    //               (c1, c2) = single_point_crossover(&p1, &p2, rng)
    //           else:
    //               (c1, c2) = (p1, p2)
    //           mutate(&mut c1, mutation_rate, rng)
    //           mutate(&mut c2, mutation_rate, rng)
    //           new_pop.push(c1); new_pop.push(c2)
    //       new_pop.truncate(pop_size)
    //       population = new_pop
    //
    //   return GAResult { ... }
    todo!()
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== Phase 1: Genetic Algorithm for Binary Knapsack ===\n");

    let items = knapsack_items();
    let n_items = items.len();

    println!("Problem: {} items, capacity {:.1} kg", n_items, CAPACITY);
    println!("Items:");
    for (i, item) in items.iter().enumerate() {
        println!(
            "  [{:2}] {:<15} weight={:.1}  value={:.1}  ratio={:.2}",
            i,
            item.name,
            item.weight,
            item.value,
            item.value / item.weight
        );
    }
    println!();

    // GA parameters
    let pop_size = 50;
    let generations = 200;
    let crossover_rate = 0.8;
    let mutation_rate = 0.02;
    let tournament_size = 3;
    let mut rng = StdRng::seed_from_u64(42);

    println!(
        "GA Parameters: pop={}, gens={}, cx_rate={}, mut_rate={}, tournament_k={}",
        pop_size, generations, crossover_rate, mutation_rate, tournament_size
    );
    println!("\nRunning GA...\n");

    let result = genetic_algorithm(
        &items,
        CAPACITY,
        pop_size,
        generations,
        crossover_rate,
        mutation_rate,
        tournament_size,
        &mut rng,
    );

    println!("\n=== GA Result ===");
    println!("Best fitness: {:.1}", result.best_fitness);
    println!("Generations run: {}", result.generations_run);
    print_solution(&result.best_chromosome, &items);

    // Show convergence: first 5 and last 5 fitness values
    let h = &result.fitness_history;
    if h.len() > 10 {
        println!("\nConvergence (first 5 / last 5 generations):");
        for i in 0..5 {
            println!("  Gen {:4}: {:.1}", i, h[i]);
        }
        println!("  ...");
        for i in (h.len() - 5)..h.len() {
            println!("  Gen {:4}: {:.1}", i, h[i]);
        }
    }

    // Greedy baseline for comparison
    println!("\n--- Greedy baseline (by value/weight ratio) ---");
    let mut indices: Vec<usize> = (0..n_items).collect();
    indices.sort_by(|&a, &b| {
        let ra = items[a].value / items[a].weight;
        let rb = items[b].value / items[b].weight;
        rb.partial_cmp(&ra).unwrap()
    });
    let mut greedy_chromosome = vec![false; n_items];
    let mut remaining = CAPACITY;
    for &i in &indices {
        if items[i].weight <= remaining {
            greedy_chromosome[i] = true;
            remaining -= items[i].weight;
        }
    }
    let greedy_val = fitness(&greedy_chromosome, &items, CAPACITY);
    println!("Greedy fitness: {:.1}", greedy_val);
    print_solution(&greedy_chromosome, &items);

    if result.best_fitness >= greedy_val {
        println!("\n[OK] GA matched or beat the greedy solution.");
    } else {
        println!(
            "\n[INFO] GA found {:.1}, greedy found {:.1}. Try more generations or larger population.",
            result.best_fitness, greedy_val
        );
    }
}
