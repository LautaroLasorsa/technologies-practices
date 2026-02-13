// Phase 1: Non-Dominated Sorting
//
// This binary teaches Pareto dominance and the fast non-dominated sorting
// algorithm from NSGA-II (Deb et al. 2002). You rank a population into
// Pareto fronts: F1 (non-dominated), F2 (dominated only by F1), etc.

#[path = "common.rs"]
mod common;

use common::*;

// ============================================================================
// Provided: dominates() is fully implemented in common.rs.
// It checks if solution A Pareto-dominates solution B.
// ============================================================================

// ============================================================================
// TODO(human): Fast Non-Dominated Sorting
// ============================================================================

/// Partition the population into Pareto fronts using Deb's fast algorithm.
///
/// Returns a vector of fronts, where each front is a vector of indices into
/// the population. Also sets each individual's `rank` field.
///
/// # TODO(human): Implement this function
///
/// Fast Non-Dominated Sorting (NSGA-II, Deb et al. 2002):
///
/// For each individual p in population:
///   S_p = set of individuals that p dominates
///   n_p = number of individuals that dominate p
///
/// Front F1 = all individuals with n_p = 0 (non-dominated)
///
/// For each individual p in current front F_k:
///   For each q in S_p:
///     n_q -= 1
///     If n_q == 0: q belongs to next front F_{k+1}
///
/// Repeat until all individuals are assigned to a front.
///
/// Complexity: O(M * N^2) where M = number of objectives, N = population size.
///
/// Implementation:
///   let n = population.len();
///   let mut domination_count: Vec<usize> = vec![0; n]; // n_p for each p
///   let mut dominated_set: Vec<Vec<usize>> = vec![vec![]; n]; // S_p for each p
///   let mut fronts: Vec<Vec<usize>> = vec![];
///
///   // Phase 1: compute domination relationships
///   // Compare every pair of individuals (i, j) where j > i:
///   //   if dominates(pop[i].objectives, pop[j].objectives):
///   //     dominated_set[i].push(j)  — i dominates j, so j is in S_i
///   //     domination_count[j] += 1  — j is dominated by one more individual
///   //   else if dominates(pop[j].objectives, pop[i].objectives):
///   //     dominated_set[j].push(i)  — j dominates i
///   //     domination_count[i] += 1
///   //   (if neither dominates, do nothing — they are non-dominated w.r.t. each other)
///
///   // Phase 2: extract fronts layer by layer
///   // current_front = all indices where domination_count[i] == 0
///   // Set rank = 0 for these individuals
///   //
///   // while current_front is not empty:
///   //   fronts.push(current_front.clone())
///   //   next_front = vec![]
///   //   for &p in &current_front:
///   //     for &q in &dominated_set[p]:
///   //       domination_count[q] -= 1
///   //       if domination_count[q] == 0:
///   //         population[q].rank = fronts.len()  (next rank)
///   //         next_front.push(q)
///   //   current_front = next_front
///
///   // Return fronts
pub fn fast_nondominated_sort(population: &mut [Individual]) -> Vec<Vec<usize>> {
    todo!("TODO(human): Implement fast non-dominated sorting (Deb's O(MN^2) algorithm)")
}

fn main() {
    println!("{}", "=".repeat(60));
    println!("  Phase 1: Non-Dominated Sorting");
    println!("{}\n", "=".repeat(60));

    // -----------------------------------------------------------------------
    // Test 1: Hand-crafted population with known Pareto structure (2 objectives)
    //
    // Objectives (minimize both f1 and f2):
    //   A = (1.0, 5.0) — non-dominated (best f1)
    //   B = (2.0, 3.0) — non-dominated
    //   C = (4.0, 1.0) — non-dominated (best f2)
    //   D = (3.0, 4.0) — dominated by B (B <= D on both, B < D on f2)
    //   E = (5.0, 2.0) — dominated by C (C < E on f1, C < E on f2)
    //   F = (3.0, 5.0) — dominated by A and B
    //   G = (5.0, 5.0) — dominated by A, B, C, D, F
    // -----------------------------------------------------------------------
    println!("=== Test 1: Hand-Crafted Population ===\n");
    println!("  Objectives (minimize both):");
    println!("  A=(1,5)  B=(2,3)  C=(4,1)  D=(3,4)  E=(5,2)  F=(3,5)  G=(5,5)\n");
    println!("  Expected fronts:");
    println!("    Front 1 (rank 0): A, B, C  (non-dominated)");
    println!("    Front 2 (rank 1): D, E     (dominated only by front 1)");
    println!("    Front 3 (rank 2): F        (dominated by front 1 and 2)");
    println!("    Front 4 (rank 3): G        (dominated by almost everything)\n");

    let mut population = vec![
        Individual::with_objectives(vec![], vec![1.0, 5.0]), // A
        Individual::with_objectives(vec![], vec![2.0, 3.0]), // B
        Individual::with_objectives(vec![], vec![4.0, 1.0]), // C
        Individual::with_objectives(vec![], vec![3.0, 4.0]), // D
        Individual::with_objectives(vec![], vec![5.0, 2.0]), // E
        Individual::with_objectives(vec![], vec![3.0, 5.0]), // F
        Individual::with_objectives(vec![], vec![5.0, 5.0]), // G
    ];

    let names = ["A", "B", "C", "D", "E", "F", "G"];

    let fronts = fast_nondominated_sort(&mut population);

    println!("  Result:");
    for (rank, front) in fronts.iter().enumerate() {
        let members: Vec<&str> = front.iter().map(|&idx| names[idx]).collect();
        println!("    Front {} (rank {}): {:?}", rank + 1, rank, members);
        for &idx in front {
            assert_eq!(
                population[idx].rank, rank,
                "Individual {} should have rank {}, got {}",
                names[idx], rank, population[idx].rank
            );
        }
    }

    // Verify front 1 = {A, B, C}
    let front1_names: Vec<&str> = fronts[0].iter().map(|&i| names[i]).collect();
    assert!(front1_names.contains(&"A"), "A should be in front 1");
    assert!(front1_names.contains(&"B"), "B should be in front 1");
    assert!(front1_names.contains(&"C"), "C should be in front 1");
    println!("\n  [PASS] Front 1 contains A, B, C (non-dominated)");

    // Verify front 2 = {D, E}
    let front2_names: Vec<&str> = fronts[1].iter().map(|&i| names[i]).collect();
    assert!(front2_names.contains(&"D"), "D should be in front 2");
    assert!(front2_names.contains(&"E"), "E should be in front 2");
    println!("  [PASS] Front 2 contains D, E");

    // Verify total fronts
    assert!(fronts.len() >= 3, "Should have at least 3 fronts");
    println!("  [PASS] {} fronts total\n", fronts.len());

    // -----------------------------------------------------------------------
    // Test 2: Population evaluated on ZDT1
    //
    // Create a small population with known decision variables, evaluate on
    // ZDT1, and verify that non-dominated sorting produces reasonable fronts.
    // -----------------------------------------------------------------------
    println!("=== Test 2: ZDT1 Evaluated Population ===\n");

    let mut zdt_pop = vec![
        Individual::new(vec![0.0; 30]),  // f1=0.0, f2=1.0 (on true Pareto front)
        Individual::new(vec![1.0; 30]),  // f1=1.0, f2=0.0 (NOT on front — g > 1 for x2..xn=1)
        Individual::new(vec![0.5; 30]),  // somewhere in the middle
    ];

    // Set x1 specifically for different trade-offs
    zdt_pop[0].x[0] = 0.0;
    zdt_pop[1].x[0] = 1.0;
    zdt_pop[2].x[0] = 0.5;

    // Optimal: all x2..xn = 0
    let mut optimal_a = Individual::new(vec![0.0; 30]);
    optimal_a.x[0] = 0.1;
    let mut optimal_b = Individual::new(vec![0.0; 30]);
    optimal_b.x[0] = 0.5;
    let mut optimal_c = Individual::new(vec![0.0; 30]);
    optimal_c.x[0] = 0.9;
    // Non-optimal: x2..xn != 0
    let mut non_optimal = Individual::new(vec![0.5; 30]);
    non_optimal.x[0] = 0.3;

    let mut population2 = vec![optimal_a, optimal_b, optimal_c, non_optimal];
    evaluate_population(&mut population2, &zdt1);

    println!("  Population (evaluated on ZDT1):");
    for (i, ind) in population2.iter().enumerate() {
        println!(
            "    [{}] x1={:.1}, f1={:.4}, f2={:.4}{}",
            i,
            ind.x[0],
            ind.objectives[0],
            ind.objectives[1],
            if ind.x[1..].iter().all(|&v| v == 0.0) {
                " (optimal g=1)"
            } else {
                " (non-optimal g>1)"
            }
        );
    }

    let fronts2 = fast_nondominated_sort(&mut population2);
    println!("\n  Fronts:");
    print_fronts(&fronts2, &population2);

    // Optimal solutions (g=1) should be in front 1
    // Non-optimal (g>1) should be in a later front (dominated by optimal with same f1)
    println!("\n  The three optimal solutions (g=1) should dominate the non-optimal one (g>1).");
    println!("  Front 1 should contain the optimal solutions.\n");

    println!("Phase 1 complete.");
}
