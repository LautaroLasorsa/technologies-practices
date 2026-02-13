// Phase 2: Crowding Distance
//
// This binary teaches the diversity preservation mechanism of NSGA-II.
// Crowding distance measures how isolated a solution is within its Pareto
// front. Solutions in sparse regions get higher crowding distance, which
// makes them preferred during selection — preserving diversity.

#[path = "common.rs"]
mod common;

use common::*;
use std::cmp::Ordering;

// ============================================================================
// Provided: fast_nondominated_sort (implemented).
// You implemented this in Phase 1 — copy your working version here.
// ============================================================================

/// Partition the population into Pareto fronts.
/// (Copy your implementation from Phase 1.)
fn fast_nondominated_sort(population: &mut [Individual]) -> Vec<Vec<usize>> {
    todo!("TODO(human): Copy your fast_nondominated_sort from Phase 1")
}

// ============================================================================
// TODO(human): Crowding Distance
// ============================================================================

/// Compute crowding distance for all individuals within a single front.
///
/// The `front_indices` are indices into the `population` array. This function
/// modifies `population[idx].crowding_distance` for each idx in `front_indices`.
///
/// # TODO(human): Implement this function
///
/// Crowding Distance (NSGA-II):
///
/// For each front, compute how "crowded" each solution is.
/// Solutions in sparse regions get higher crowding distance (preferred for diversity).
///
/// Algorithm for a single front:
///   1. Initialize crowding_distance = 0 for all individuals in the front
///   2. For each objective m (0..num_objectives):
///      a. Sort the front indices by objective m value (ascending)
///      b. Set boundary solutions to infinity:
///         population[sorted[0]].crowding_distance = INFINITY
///         population[sorted[last]].crowding_distance = INFINITY
///      c. Compute the normalization range:
///         obj_max = population[sorted[last]].objectives[m]
///         obj_min = population[sorted[0]].objectives[m]
///      d. If obj_max == obj_min (all same value on this objective), skip this objective
///      e. For middle solutions (i = 1 to len-2):
///         population[sorted[i]].crowding_distance +=
///           (population[sorted[i+1]].objectives[m] - population[sorted[i-1]].objectives[m])
///           / (obj_max - obj_min)
///
/// The normalization (obj_max - obj_min) ensures all objectives contribute equally
/// regardless of their scale. Without it, an objective with range [0, 1000] would
/// dominate the distance calculation over one with range [0, 1].
///
/// Boundary solutions get INFINITY so they're always preserved — they define
/// the extent of the front and losing them would shrink the front's coverage.
///
/// The result is: high CD = isolated (valuable for diversity), low CD = crowded
/// (expendable when truncating the population).
///
/// Important: work with indices into the population array, not copies.
/// Sort a local Vec<usize> of front_indices by objective m, then use the
/// sorted order to index into population.
pub fn compute_crowding_distance(population: &mut [Individual], front_indices: &[usize]) {
    todo!("TODO(human): Implement crowding distance computation for a single front")
}

// ============================================================================
// TODO(human): Crowded Comparison Operator
// ============================================================================

/// Compare two individuals using the crowded comparison operator.
///
/// # TODO(human): Implement this function
///
/// Crowded comparison (used in tournament selection and truncation):
///
///   1. If a.rank < b.rank → a is better (Less)
///      (Lower rank = closer to Pareto front = preferred)
///
///   2. If a.rank > b.rank → b is better (Greater)
///
///   3. If a.rank == b.rank → prefer higher crowding distance:
///      - If a.crowding_distance > b.crowding_distance → a is better (Less)
///      - If a.crowding_distance < b.crowding_distance → b is better (Greater)
///      - If equal → Equal
///
/// This operator creates dual selection pressure:
///   - Primary: toward the Pareto front (convergence)
///   - Secondary: toward less crowded regions (diversity)
///
/// Returns Ordering::Less if a is preferred, Ordering::Greater if b is preferred.
///
/// Note: use partial_cmp for f64 comparisons and unwrap_or(Ordering::Equal)
/// to handle NaN gracefully.
pub fn crowded_comparison(a: &Individual, b: &Individual) -> Ordering {
    todo!("TODO(human): Implement crowded comparison operator (rank, then crowding distance)")
}

fn main() {
    println!("{}", "=".repeat(60));
    println!("  Phase 2: Crowding Distance");
    println!("{}\n", "=".repeat(60));

    // -----------------------------------------------------------------------
    // Test 1: Crowding distance on a known front
    //
    // Front with 5 solutions (minimize both f1 and f2):
    //   A = (0.1, 0.9)  — boundary (should get INFINITY)
    //   B = (0.3, 0.7)  — middle
    //   C = (0.5, 0.5)  — middle
    //   D = (0.7, 0.3)  — middle
    //   E = (0.9, 0.1)  — boundary (should get INFINITY)
    //
    // All are on the same front (non-dominated with respect to each other).
    // Boundary solutions (min/max on each objective) should get INFINITY.
    // Middle solutions get finite crowding distance based on neighbor distances.
    // -----------------------------------------------------------------------
    println!("=== Test 1: Crowding Distance on Known Front ===\n");

    let mut population = vec![
        Individual::with_objectives(vec![], vec![0.1, 0.9]), // A
        Individual::with_objectives(vec![], vec![0.3, 0.7]), // B
        Individual::with_objectives(vec![], vec![0.5, 0.5]), // C
        Individual::with_objectives(vec![], vec![0.7, 0.3]), // D
        Individual::with_objectives(vec![], vec![0.9, 0.1]), // E
    ];

    // All in the same front
    for ind in &mut population {
        ind.rank = 0;
    }

    let front_indices: Vec<usize> = (0..population.len()).collect();
    compute_crowding_distance(&mut population, &front_indices);

    let names = ["A", "B", "C", "D", "E"];
    println!("  Solution   f1    f2    Crowding Distance");
    println!("  --------  ----  ----  ------------------");
    for (i, ind) in population.iter().enumerate() {
        println!(
            "  {}         {:.1}   {:.1}   {}",
            names[i],
            ind.objectives[0],
            ind.objectives[1],
            if ind.crowding_distance.is_infinite() {
                "INFINITY".to_string()
            } else {
                format!("{:.4}", ind.crowding_distance)
            }
        );
    }

    // Verify boundaries get infinity
    assert!(
        population[0].crowding_distance.is_infinite(),
        "A (boundary) should have infinite crowding distance"
    );
    assert!(
        population[4].crowding_distance.is_infinite(),
        "E (boundary) should have infinite crowding distance"
    );
    println!("\n  [PASS] Boundary solutions A and E have INFINITY crowding distance");

    // Middle solutions should have finite positive crowding distance
    for i in 1..4 {
        assert!(
            population[i].crowding_distance > 0.0 && population[i].crowding_distance.is_finite(),
            "{} should have finite positive crowding distance",
            names[i]
        );
    }
    println!("  [PASS] Middle solutions B, C, D have finite positive crowding distance");

    // For uniformly spaced points, middle solutions should have equal CD
    let cd_b = population[1].crowding_distance;
    let cd_c = population[2].crowding_distance;
    let cd_d = population[3].crowding_distance;
    println!("  B CD={:.4}, C CD={:.4}, D CD={:.4}", cd_b, cd_c, cd_d);
    if (cd_b - cd_c).abs() < 1e-6 && (cd_c - cd_d).abs() < 1e-6 {
        println!("  [PASS] Uniformly spaced → equal crowding distance for B, C, D\n");
    } else {
        println!("  [NOTE] Middle CDs differ — check normalization\n");
    }

    // -----------------------------------------------------------------------
    // Test 2: Non-uniform spacing (different crowding distances)
    //
    //   P = (0.0, 1.0)  — boundary
    //   Q = (0.1, 0.8)  — close to P → low CD (crowded region)
    //   R = (0.5, 0.5)  — far from neighbors → high CD (sparse region)
    //   S = (0.9, 0.2)  — close to T → low CD
    //   T = (1.0, 0.0)  — boundary
    //
    // R should have the highest finite CD (most isolated).
    // -----------------------------------------------------------------------
    println!("=== Test 2: Non-Uniform Spacing ===\n");

    let mut population2 = vec![
        Individual::with_objectives(vec![], vec![0.0, 1.0]), // P
        Individual::with_objectives(vec![], vec![0.1, 0.8]), // Q (close to P)
        Individual::with_objectives(vec![], vec![0.5, 0.5]), // R (isolated)
        Individual::with_objectives(vec![], vec![0.9, 0.2]), // S (close to T)
        Individual::with_objectives(vec![], vec![1.0, 0.0]), // T
    ];

    for ind in &mut population2 {
        ind.rank = 0;
    }

    let indices2: Vec<usize> = (0..population2.len()).collect();
    compute_crowding_distance(&mut population2, &indices2);

    let names2 = ["P", "Q", "R", "S", "T"];
    println!("  Solution   f1    f2    Crowding Distance");
    println!("  --------  ----  ----  ------------------");
    for (i, ind) in population2.iter().enumerate() {
        println!(
            "  {}         {:.1}   {:.1}   {}",
            names2[i],
            ind.objectives[0],
            ind.objectives[1],
            if ind.crowding_distance.is_infinite() {
                "INFINITY".to_string()
            } else {
                format!("{:.4}", ind.crowding_distance)
            }
        );
    }

    // R should have higher CD than Q and S (more isolated)
    let cd_q = population2[1].crowding_distance;
    let cd_r = population2[2].crowding_distance;
    let cd_s = population2[3].crowding_distance;
    if cd_r > cd_q && cd_r > cd_s {
        println!("\n  [PASS] R (most isolated) has highest finite CD: {:.4}", cd_r);
    } else {
        println!("\n  [CHECK] R CD={:.4}, Q CD={:.4}, S CD={:.4}", cd_r, cd_q, cd_s);
    }

    // -----------------------------------------------------------------------
    // Test 3: Crowded comparison operator
    // -----------------------------------------------------------------------
    println!("\n=== Test 3: Crowded Comparison Operator ===\n");

    let a = Individual {
        x: vec![],
        objectives: vec![],
        rank: 0,
        crowding_distance: 2.0,
    };
    let b = Individual {
        x: vec![],
        objectives: vec![],
        rank: 1,
        crowding_distance: 5.0,
    };

    let cmp = crowded_comparison(&a, &b);
    println!("  a(rank=0, cd=2.0) vs b(rank=1, cd=5.0): {:?}", cmp);
    assert_eq!(cmp, Ordering::Less, "Lower rank should be preferred");
    println!("  [PASS] Lower rank preferred over higher crowding distance");

    let c = Individual {
        x: vec![],
        objectives: vec![],
        rank: 0,
        crowding_distance: 1.0,
    };
    let d = Individual {
        x: vec![],
        objectives: vec![],
        rank: 0,
        crowding_distance: 3.0,
    };

    let cmp2 = crowded_comparison(&c, &d);
    println!("  c(rank=0, cd=1.0) vs d(rank=0, cd=3.0): {:?}", cmp2);
    assert_eq!(cmp2, Ordering::Greater, "Same rank → higher CD preferred");
    println!("  [PASS] Same rank → higher crowding distance preferred");

    // -----------------------------------------------------------------------
    // Test 4: ZDT1 population — sort by front, then crowding within front
    // -----------------------------------------------------------------------
    println!("\n=== Test 4: ZDT1 Population with Sorting ===\n");

    // Create a small ZDT1 population
    let mut zdt_pop: Vec<Individual> = (0..10)
        .map(|i| {
            let mut x = vec![0.0; 30]; // optimal g=1
            x[0] = i as f64 / 9.0; // spread f1 from 0 to 1
            // Add some non-optimal solutions
            if i >= 7 {
                for j in 1..30 {
                    x[j] = 0.3; // non-optimal g > 1
                }
            }
            Individual::new(x)
        })
        .collect();

    evaluate_population(&mut zdt_pop, &zdt1);

    let fronts = fast_nondominated_sort(&mut zdt_pop);
    println!("  {} fronts found:", fronts.len());
    for (rank, front) in fronts.iter().enumerate() {
        compute_crowding_distance(&mut zdt_pop, front);
        println!("  Front {} ({} individuals):", rank, front.len());
        // Sort within front by crowding distance (descending) for display
        let mut sorted_front = front.clone();
        sorted_front.sort_by(|&a, &b| {
            crowded_comparison(&zdt_pop[a], &zdt_pop[b])
        });
        for &idx in &sorted_front {
            let ind = &zdt_pop[idx];
            println!(
                "    [{:>2}] f1={:.4}, f2={:.4}, cd={}",
                idx,
                ind.objectives[0],
                ind.objectives[1],
                if ind.crowding_distance.is_infinite() {
                    "INF".to_string()
                } else {
                    format!("{:.4}", ind.crowding_distance)
                }
            );
        }
    }

    println!("\nPhase 2 complete.");
}
