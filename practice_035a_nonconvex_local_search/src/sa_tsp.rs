use rand::Rng;

// =============================================================================
// City and distance
// =============================================================================

#[derive(Clone, Debug)]
struct City {
    x: f64,
    y: f64,
}

/// Euclidean distance between two cities.
fn distance(a: &City, b: &City) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

/// Total tour length: sum of distances between consecutive cities, plus return to start.
/// `tour` is a permutation of city indices [0..n).
fn tour_length(cities: &[City], tour: &[usize]) -> f64 {
    let n = tour.len();
    let mut total = 0.0;
    for i in 0..n {
        let from = &cities[tour[i]];
        let to = &cities[tour[(i + 1) % n]];
        total += distance(from, to);
    }
    total
}

// =============================================================================
// City generation
// =============================================================================

/// Generate `n` random cities with coordinates in [0, 100] x [0, 100].
fn generate_random_cities(n: usize, rng: &mut impl Rng) -> Vec<City> {
    (0..n)
        .map(|_| City {
            x: rng.gen_range(0.0..100.0),
            y: rng.gen_range(0.0..100.0),
        })
        .collect()
}

// =============================================================================
// Nearest-neighbor heuristic (greedy initial tour)
// =============================================================================

/// Construct a tour using the nearest-neighbor heuristic:
///   1. Start at city 0
///   2. Repeatedly visit the nearest unvisited city
///   3. Return to start
///
/// This produces a reasonable initial tour (typically within 20-25% of optimal)
/// that SA can then improve.
fn nearest_neighbor_tour(cities: &[City]) -> Vec<usize> {
    let n = cities.len();
    let mut visited = vec![false; n];
    let mut tour = Vec::with_capacity(n);

    tour.push(0);
    visited[0] = true;

    for _ in 1..n {
        let current = *tour.last().unwrap();
        let mut best_next = 0;
        let mut best_dist = f64::MAX;

        for j in 0..n {
            if !visited[j] {
                let d = distance(&cities[current], &cities[j]);
                if d < best_dist {
                    best_dist = d;
                    best_next = j;
                }
            }
        }

        tour.push(best_next);
        visited[best_next] = true;
    }

    tour
}

// =============================================================================
// TODO(human): 2-opt swap
// =============================================================================

/// Perform a 2-opt swap: reverse the segment of the tour between positions i and j (inclusive).
///
/// 2-opt is the most fundamental TSP neighborhood operator:
///   Given a tour and two positions i and j (where i < j):
///   1. Keep tour[0..i] unchanged
///   2. Reverse tour[i..=j]  (the segment from position i to j, inclusive)
///   3. Keep tour[j+1..] unchanged
///
/// Example:
///   tour = [0, 1, 2, 3, 4, 5], i = 1, j = 4
///   Result: [0, 4, 3, 2, 1, 5]
///   The segment [1, 2, 3, 4] was reversed to [4, 3, 2, 1].
///
/// Geometrically, 2-opt removes two edges from the tour and reconnects
/// the two resulting paths in the other possible way. This can "uncross"
/// intersecting edges, which always improves the tour.
///
/// Implementation hint: build a new Vec by concatenating:
///   tour[0..i]  ++  tour[i..=j].reversed()  ++  tour[j+1..]
/// Or equivalently, clone the tour and call .reverse() on the slice [i..=j].
fn two_opt_swap(tour: &[usize], i: usize, j: usize) -> Vec<usize> {
    todo!("TODO(human): not implemented")
}

// =============================================================================
// TODO(human): Simulated Annealing for TSP
// =============================================================================

/// Simulated Annealing for TSP using 2-opt neighborhood.
///
/// This is the same SA framework as Phase 3, but applied to a combinatorial
/// problem. The only difference is the neighborhood operator (2-opt instead
/// of Gaussian perturbation).
///
/// Algorithm:
///   1. Initialize: current_tour = initial_tour (clone it)
///      current_cost = tour_length(cities, current_tour)
///      best_tour = current_tour, best_cost = current_cost
///      temperature = t_init
///
///   2. For each iteration (0..max_iter):
///      a. Pick two random positions i, j where 1 <= i < j <= n-1
///         (we keep position 0 fixed to avoid equivalent tours)
///         Use rng.gen_range(1..n) for both, then ensure i < j by swapping if needed.
///
///      b. Compute new_tour = two_opt_swap(&current_tour, i, j)
///      c. Compute new_cost = tour_length(cities, &new_tour)
///      d. Compute delta = new_cost - current_cost
///
///      e. Acceptance decision (Metropolis criterion, same as Phase 3):
///         - If delta < 0.0: always accept
///         - If delta >= 0.0: accept with probability exp(-delta / temperature)
///           Generate u ~ Uniform(0,1), accept if u < exp(-delta / temperature)
///
///      f. If accepted: current_tour = new_tour, current_cost = new_cost
///      g. If current_cost < best_cost: update best_tour, best_cost
///      h. Cool: temperature *= alpha
///
///   3. Return (best_tour, best_cost)
///
/// Note: computing full tour_length each iteration is O(n) per step.
/// An optimization is to compute only the delta from the 4 affected edges,
/// but for learning purposes, full recomputation is fine for n <= 100.
fn sa_tsp(
    cities: &[City],
    initial_tour: &[usize],
    t_init: f64,
    alpha: f64,
    max_iter: usize,
) -> (Vec<usize>, f64) {
    todo!("TODO(human): not implemented")
}

// =============================================================================
// Output helpers
// =============================================================================

fn print_tour(label: &str, cities: &[City], tour: &[usize], cost: f64) {
    println!("  {}: tour length = {:.2}", label, cost);
    if cities.len() <= 20 {
        let order: Vec<String> = tour.iter().map(|&i| format!("{}", i)).collect();
        println!("    Order: [{}]", order.join(" -> "));
    }
}

// =============================================================================
// Main — SA for TSP
// =============================================================================

fn main() {
    let mut rng = rand::thread_rng();

    println!("=== Phase 4: Simulated Annealing for TSP ===\n");

    // --- Generate random cities ---
    let n_cities = 20;
    let cities = generate_random_cities(n_cities, &mut rng);
    println!("Generated {} random cities in [0, 100] x [0, 100]\n", n_cities);

    // --- Nearest-neighbor initial tour ---
    let nn_tour = nearest_neighbor_tour(&cities);
    let nn_cost = tour_length(&cities, &nn_tour);
    print_tour("Nearest-neighbor", &cities, &nn_tour, nn_cost);
    println!();

    // --- SA with different parameter settings ---
    println!("--- SA with different cooling rates ---");

    for &(alpha, max_iter, label) in &[
        (0.99, 50_000, "Fast cooling (alpha=0.99, 50k)"),
        (0.999, 100_000, "Medium cooling (alpha=0.999, 100k)"),
        (0.9999, 500_000, "Slow cooling (alpha=0.9999, 500k)"),
    ] {
        let (best_tour, best_cost) =
            sa_tsp(&cities, &nn_tour, 1000.0, alpha, max_iter);
        print_tour(label, &cities, &best_tour, best_cost);
        let improvement = (nn_cost - best_cost) / nn_cost * 100.0;
        println!("    Improvement over NN: {:.1}%", improvement);
    }

    println!();

    // --- Larger instance ---
    println!("--- Larger instance: 50 cities ---");
    let n_large = 50;
    let large_cities = generate_random_cities(n_large, &mut rng);
    let large_nn = nearest_neighbor_tour(&large_cities);
    let large_nn_cost = tour_length(&large_cities, &large_nn);
    println!("  Nearest-neighbor tour: {:.2}", large_nn_cost);

    let (large_best, large_best_cost) =
        sa_tsp(&large_cities, &large_nn, 1000.0, 0.9999, 1_000_000);
    println!("  SA (alpha=0.9999, 1M iter): {:.2}", large_best_cost);
    let improvement = (large_nn_cost - large_best_cost) / large_nn_cost * 100.0;
    println!("  Improvement: {:.1}%", improvement);

    println!("\nObservation: SA consistently improves the nearest-neighbor tour.");
    println!("Slow cooling and more iterations yield better results.");
    println!("The 2-opt neighborhood is simple but effective — it can 'uncross' edges.");
}
