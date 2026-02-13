// Phase 4: Benchmarks & Visualization
//
// This binary runs NSGA-II on the ZDT test suite (ZDT1, ZDT2, ZDT3),
// computes the hypervolume indicator for each, and visualizes the
// Pareto front approximations using text-based scatter plots.

#[path = "common.rs"]
mod common;

use common::*;
use rand::Rng;

// ============================================================================
// Provided: Full NSGA-II (implemented).
// Copy all your working implementations from Phases 1-3.
// ============================================================================

fn fast_nondominated_sort(population: &mut [Individual]) -> Vec<Vec<usize>> {
    todo!("TODO(human): Copy your fast_nondominated_sort from Phase 1")
}

fn compute_crowding_distance(population: &mut [Individual], front_indices: &[usize]) {
    todo!("TODO(human): Copy your compute_crowding_distance from Phase 2")
}

fn crowded_comparison(a: &Individual, b: &Individual) -> std::cmp::Ordering {
    todo!("TODO(human): Copy your crowded_comparison from Phase 2")
}

fn tournament_selection(population: &[Individual], rng: &mut impl Rng) -> usize {
    todo!("TODO(human): Copy your tournament_selection from Phase 3")
}

fn sbx_crossover(
    parent1: &[f64],
    parent2: &[f64],
    eta_c: f64,
    bounds: &[(f64, f64)],
    rng: &mut impl Rng,
) -> (Vec<f64>, Vec<f64>) {
    todo!("TODO(human): Copy your sbx_crossover from Phase 3")
}

fn polynomial_mutation(
    x: &mut [f64],
    eta_m: f64,
    bounds: &[(f64, f64)],
    mutation_prob: f64,
    rng: &mut impl Rng,
) {
    todo!("TODO(human): Copy your polynomial_mutation from Phase 3")
}

fn nsga2(
    objective_fn: &dyn Fn(&[f64]) -> Vec<f64>,
    n_vars: usize,
    bounds: &[(f64, f64)],
    pop_size: usize,
    generations: usize,
) -> NSGA2Result {
    todo!("TODO(human): Copy your nsga2 from Phase 3")
}

// ============================================================================
// TODO(human): Hypervolume Indicator (2D)
// ============================================================================

/// Compute the 2D hypervolume indicator for a Pareto front approximation.
///
/// The hypervolume is the area of objective space that is dominated by
/// the front and bounded by the reference point. Higher = better front.
///
/// # TODO(human): Implement this function
///
/// 2D Hypervolume Algorithm:
///
///   1. Filter: keep only points that dominate the reference point
///      (i.e., point.objectives[0] < ref_point[0] AND point.objectives[1] < ref_point[1])
///      If no points dominate the reference, hypervolume = 0.
///
///   2. Sort the filtered points by first objective (f1) ascending.
///
///   3. Compute area as sum of non-overlapping rectangles:
///      prev_f2 = ref_point[1]  (start from the reference point's f2 boundary)
///
///      For each point p (in sorted order by f1):
///        width  = ref_point[0] - p.objectives[0]  (horizontal distance to ref)
///        height = prev_f2 - p.objectives[1]        (vertical strip height)
///        hv += width * height  ... BUT this double-counts!
///
///      Correct approach — incremental rectangles:
///        Sort by f1 ascending.
///        prev_f2 = ref_point[1]
///        hv = 0.0
///        for i in 0..sorted.len():
///          // Rectangle from this point to the next (or to ref_point[0])
///          let f1_i = sorted[i].objectives[0]
///          let f2_i = sorted[i].objectives[1]
///          let f1_next = if i + 1 < sorted.len() { sorted[i+1].objectives[0] } else { ref_point[0] }
///          hv += (f1_next - f1_i) * (ref_point[1] - f2_i)
///
///      This computes the area correctly by sweeping left-to-right.
///      Each point contributes a rectangle extending to the right until the
///      next point (or the reference point), and upward to the reference f2.
///
/// The hypervolume is the ONLY unary quality indicator that is Pareto-compliant:
/// if front A has higher HV than front B (with the same reference), then A is
/// not worse than B in Pareto terms. This makes it the gold standard metric.
///
/// Reference point should be chosen to dominate all points on the front
/// (e.g., slightly beyond the worst objective values: [1.1, 1.1] for ZDT).
pub fn hypervolume_2d(front: &[Individual], ref_point: &[f64]) -> f64 {
    todo!("TODO(human): Implement 2D hypervolume indicator computation")
}

// ============================================================================
// Provided: Benchmark runner and visualization
// ============================================================================

/// Run NSGA-II on a test problem and report results.
fn run_benchmark(
    name: &str,
    objective_fn: &dyn Fn(&[f64]) -> Vec<f64>,
    true_front_fn: &dyn Fn(f64) -> f64,
    n_vars: usize,
    pop_size: usize,
    generations: usize,
    ref_point: &[f64],
) {
    println!("\n{}", "=".repeat(60));
    println!("  Benchmark: {}", name);
    println!("{}\n", "=".repeat(60));

    let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); n_vars];

    let start = std::time::Instant::now();
    let result = nsga2(objective_fn, n_vars, &bounds, pop_size, generations);
    let elapsed = start.elapsed();

    println!("  Time: {:.2?}", elapsed);
    println!("  Generations: {}", result.generations);

    if let Some(front0) = result.fronts.first() {
        println!("  Front 0 size: {}", front0.len());

        // Compute hypervolume
        let hv = hypervolume_2d(front0, ref_point);
        println!("  Hypervolume (ref={:?}): {:.6}", ref_point, hv);

        // Compute average distance to true Pareto front
        let mut total_gap = 0.0;
        let mut count = 0;
        for ind in front0 {
            let f1 = ind.objectives[0];
            if f1 >= 0.0 && f1 <= 1.0 {
                let f2_true = true_front_fn(f1);
                let f2_actual = ind.objectives[1];
                total_gap += (f2_actual - f2_true).abs();
                count += 1;
            }
        }
        if count > 0 {
            println!(
                "  Average gap to true front: {:.6} (over {} points)",
                total_gap / count as f64,
                count
            );
        }

        // Print sample points
        let mut sorted: Vec<&Individual> = front0.iter().collect();
        sorted.sort_by(|a, b| a.objectives[0].partial_cmp(&b.objectives[0]).unwrap());

        println!("\n  Sample points (sorted by f1):");
        println!("  {:>8}  {:>8}  {:>8}", "f1", "f2", "f2_true");
        println!("  {:>8}  {:>8}  {:>8}", "----", "----", "-------");
        let step = if sorted.len() > 10 {
            sorted.len() / 10
        } else {
            1
        };
        for (i, ind) in sorted.iter().enumerate() {
            if i % step == 0 || i == sorted.len() - 1 {
                let f1 = ind.objectives[0];
                let f2_true = true_front_fn(f1);
                println!(
                    "  {:>8.4}  {:>8.4}  {:>8.4}",
                    f1, ind.objectives[1], f2_true
                );
            }
        }

        // Visualize
        text_scatter_plot(front0, &format!("{} Pareto Front", name), 60, 20);
    } else {
        println!("  No fronts found!");
    }
}

fn main() {
    println!("{}", "=".repeat(60));
    println!("  Phase 4: Benchmarks & Visualization");
    println!("{}\n", "=".repeat(60));

    let n_vars = 30;
    let pop_size = 100;
    let generations = 200;
    let ref_point = vec![1.1, 1.1]; // slightly beyond [1, 1] — dominates all ZDT front points

    // -----------------------------------------------------------------------
    // ZDT1: Convex Pareto front
    // True front: f2 = 1 - sqrt(f1)
    // -----------------------------------------------------------------------
    run_benchmark(
        "ZDT1 (convex)",
        &zdt1,
        &|f1: f64| 1.0 - f1.sqrt(),
        n_vars,
        pop_size,
        generations,
        &ref_point,
    );

    // -----------------------------------------------------------------------
    // ZDT2: Non-convex (concave) Pareto front
    // True front: f2 = 1 - f1^2
    // This is where weighted sum methods fail!
    // -----------------------------------------------------------------------
    run_benchmark(
        "ZDT2 (non-convex)",
        &zdt2,
        &|f1: f64| 1.0 - f1 * f1,
        n_vars,
        pop_size,
        generations,
        &ref_point,
    );

    // -----------------------------------------------------------------------
    // ZDT3: Disconnected Pareto front
    // True front: f2 = 1 - sqrt(f1) - f1 * sin(10*pi*f1) (in disconnected segments)
    // Hardest for algorithms that assume connected fronts.
    // -----------------------------------------------------------------------
    run_benchmark(
        "ZDT3 (disconnected)",
        &zdt3,
        &|f1: f64| 1.0 - f1.sqrt() - f1 * (10.0 * std::f64::consts::PI * f1).sin(),
        n_vars,
        pop_size,
        generations,
        &ref_point,
    );

    // -----------------------------------------------------------------------
    // Summary comparison
    // -----------------------------------------------------------------------
    println!("\n{}", "=".repeat(60));
    println!("  Summary");
    println!("{}\n", "=".repeat(60));
    println!("  Problem         Front Shape     Challenge");
    println!("  --------------- --------------- -------------------------");
    println!("  ZDT1            Convex          Baseline (easiest)");
    println!("  ZDT2            Non-convex      Weighted sum fails here");
    println!("  ZDT3            Disconnected    Must find multiple segments");
    println!();
    println!("  NSGA-II handles ALL three shapes because it uses Pareto");
    println!("  dominance directly — no convexity assumptions needed.");
    println!("  This is the key advantage over scalarization methods.");

    println!("\nPhase 4 complete.");
}
