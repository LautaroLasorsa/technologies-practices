// Phase 4: Applications — Sudoku, N-Queens, Graph Coloring
//
// This binary applies the MAC solver from Phase 3 to three classic CSP
// benchmarks. Sudoku and N-Queens CSP builders are provided (they are
// modeling exercises, not algorithmic). You implement graph coloring as a CSP.

#[path = "common.rs"]
mod common;

use common::*;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Solver (provided — same as Phase 3)
// ---------------------------------------------------------------------------

fn revise(
    domains: &mut [Domain],
    var_i: usize,
    var_j: usize,
    check: &dyn Fn(i32, i32) -> bool,
) -> bool {
    let values_i: Vec<i32> = domains[var_i].iter().copied().collect();
    let mut revised = false;
    for v in values_i {
        let has_support = domains[var_j].iter().any(|&w| check(v, w));
        if !has_support {
            domains[var_i].remove(&v);
            revised = true;
        }
    }
    revised
}

fn ac3(csp: &mut CSP, stats: &mut CSPResult) -> bool {
    let neighbor_map = csp.build_neighbor_map();
    let mut domains: Vec<Domain> = csp.variables.iter().map(|v| v.domain.clone()).collect();

    let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();
    for (ci, c) in csp.constraints.iter().enumerate() {
        queue.push_back((c.var_i, c.var_j, ci));
        queue.push_back((c.var_j, c.var_i, ci));
    }

    while let Some((i, j, ci)) = queue.pop_front() {
        stats.propagations += 1;
        let check = &csp.constraints[ci].check;
        let is_forward = csp.constraints[ci].var_i == i;

        let revised = if is_forward {
            revise(&mut domains, i, j, &|a, b| check(a, b))
        } else {
            revise(&mut domains, i, j, &|a, b| check(b, a))
        };

        if revised {
            if domains[i].is_empty() {
                for (vi, var) in csp.variables.iter_mut().enumerate() {
                    var.domain = domains[vi].clone();
                }
                return false;
            }
            if let Some(neighbors) = neighbor_map.get(&i) {
                for &(nci, k) in neighbors {
                    if k != j {
                        queue.push_back((k, i, nci));
                    }
                }
            }
        }
    }

    for (vi, var) in csp.variables.iter_mut().enumerate() {
        var.domain = domains[vi].clone();
    }
    true
}

fn select_unassigned_variable(csp: &CSP) -> Option<usize> {
    csp.variables
        .iter()
        .enumerate()
        .filter(|(_, v)| v.domain.len() > 1)
        .min_by_key(|(_, v)| v.domain.len())
        .map(|(i, _)| i)
}

fn backtrack(csp: &mut CSP, stats: &mut CSPResult) -> bool {
    if csp.is_solved() {
        return true;
    }

    let var = match select_unassigned_variable(csp) {
        Some(v) => v,
        None => return csp.is_solved(),
    };

    let values: Vec<i32> = csp.variables[var].domain.iter().copied().collect();

    for v in values {
        let saved = csp.save_domains();
        csp.variables[var].domain.clear();
        csp.variables[var].domain.insert(v);
        stats.nodes_explored += 1;

        let consistent = ac3(csp, stats);
        if consistent && backtrack(csp, stats) {
            return true;
        }

        csp.restore_domains(&saved);
    }

    false
}

fn solve(csp: &mut CSP) -> CSPResult {
    let mut stats = CSPResult::new();
    let consistent = ac3(csp, &mut stats);
    if !consistent {
        return stats;
    }
    if csp.is_solved() {
        stats.solution = csp.extract_solution();
        return stats;
    }
    if backtrack(csp, &mut stats) {
        stats.solution = csp.extract_solution();
    }
    stats
}

// ---------------------------------------------------------------------------
// Sudoku CSP builder (provided)
// ---------------------------------------------------------------------------

/// Build a Sudoku CSP from a 9x9 grid. Zeros represent empty cells.
///
/// Variables: 81 (one per cell), indexed as row * 9 + col.
/// Domains: {given_value} for pre-filled cells, {1..9} for empty cells.
/// Constraints: AllDifferent decomposed into pairwise != for:
///   - Each row (9 variables, C(9,2) = 36 constraints per row)
///   - Each column (36 constraints per column)
///   - Each 3x3 box (36 constraints per box)
/// Total: 3 * 9 * 36 = 972 binary constraints (with some overlap between
/// row/col/box — actual unique pairs ≈ 810).
fn build_sudoku_csp(grid: &[[i32; 9]; 9]) -> CSP {
    let mut csp = CSP::new();

    // Create 81 variables
    for row in 0..9 {
        for col in 0..9 {
            let val = grid[row][col];
            if val != 0 {
                // Pre-filled: domain is just the given value
                csp.add_variable(Variable::new_set(
                    &format!("R{}C{}", row, col),
                    &[val],
                ));
            } else {
                // Empty: domain is 1..9
                csp.add_variable(Variable::new_range(
                    &format!("R{}C{}", row, col),
                    1,
                    9,
                ));
            }
        }
    }

    // Helper: variable index for (row, col)
    let idx = |r: usize, c: usize| r * 9 + c;

    // Track added pairs to avoid duplicates
    let mut added = std::collections::HashSet::new();

    // Row constraints: all cells in same row must differ
    for row in 0..9 {
        for i in 0..9 {
            for j in (i + 1)..9 {
                let vi = idx(row, i);
                let vj = idx(row, j);
                if added.insert((vi, vj)) {
                    csp.add_not_equal(vi, vj, &format!("row{}: C{} != C{}", row, i, j));
                }
            }
        }
    }

    // Column constraints: all cells in same column must differ
    for col in 0..9 {
        for i in 0..9 {
            for j in (i + 1)..9 {
                let vi = idx(i, col);
                let vj = idx(j, col);
                if added.insert((vi, vj)) {
                    csp.add_not_equal(vi, vj, &format!("col{}: R{} != R{}", col, i, j));
                }
            }
        }
    }

    // Box constraints: all cells in same 3x3 box must differ
    for box_row in 0..3 {
        for box_col in 0..3 {
            let mut cells = Vec::new();
            for r in 0..3 {
                for c in 0..3 {
                    cells.push(idx(box_row * 3 + r, box_col * 3 + c));
                }
            }
            for i in 0..cells.len() {
                for j in (i + 1)..cells.len() {
                    let vi = cells[i];
                    let vj = cells[j];
                    if added.insert((vi, vj)) {
                        csp.add_not_equal(
                            vi,
                            vj,
                            &format!("box({},{}): {} != {}", box_row, box_col, vi, vj),
                        );
                    }
                }
            }
        }
    }

    csp
}

// ---------------------------------------------------------------------------
// N-Queens CSP builder (provided)
// ---------------------------------------------------------------------------

/// Build an N-Queens CSP.
///
/// Variables: n (one per row), domain {0..n-1} (column placement).
/// Constraints: for each pair (i, j) where i < j:
///   - queens[i] != queens[j] (not same column)
///   - |queens[i] - queens[j]| != |i - j| (not same diagonal)
fn build_nqueens_csp(n: usize) -> CSP {
    let mut csp = CSP::new();
    for i in 0..n {
        csp.add_variable(Variable::new_range(&format!("Q{}", i), 0, (n - 1) as i32));
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = (j - i) as i32;
            csp.add_constraint(BinaryConstraint {
                var_i: i,
                var_j: j,
                name: format!("Q{} vs Q{}", i, j),
                check: Box::new(move |qi, qj| qi != qj && (qi - qj).abs() != diff),
            });
        }
    }
    csp
}

// ---------------------------------------------------------------------------
// Graph Coloring CSP builder (TODO)
// ---------------------------------------------------------------------------

/// Build a graph coloring CSP.
///
/// Given a graph with `n_vertices` and a list of edges, and `n_colors` colors,
/// find an assignment of colors to vertices such that no two adjacent vertices
/// share the same color.
///
/// # TODO(human): Implement graph coloring as a CSP
///
/// Graph Coloring as CSP:
///   Variables: one per vertex, named "V0", "V1", ..., "V{n-1}"
///   Domains: {0, 1, ..., n_colors - 1} (each color is an integer)
///   Constraints: for each edge (u, v), add a not-equal constraint:
///     color[u] != color[v]
///
/// Use csp.add_variable(Variable::new_range(...)) for each vertex.
/// Use csp.add_not_equal(u, v, name) for each edge.
///
/// The chromatic number of a graph is the minimum n_colors for which
/// a proper coloring exists. K_n (complete graph on n vertices) has
/// chromatic number n. The Petersen graph has chromatic number 3.
/// Planar graphs have chromatic number <= 4 (four color theorem).
///
/// This is a pure modeling exercise: translate the graph coloring problem
/// into the CSP framework. The solver (AC-3 + backtracking) does the rest.
pub fn build_graph_coloring_csp(
    n_vertices: usize,
    edges: &[(usize, usize)],
    n_colors: usize,
) -> CSP {
    todo!("TODO(human): Create CSP for graph coloring — variables, domains, not-equal constraints")
}

// ---------------------------------------------------------------------------
// Sample problems
// ---------------------------------------------------------------------------

/// Easy Sudoku puzzle.
fn easy_sudoku() -> [[i32; 9]; 9] {
    [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
}

/// Medium Sudoku puzzle.
fn medium_sudoku() -> [[i32; 9]; 9] {
    [
        [0, 0, 0, 6, 0, 0, 4, 0, 0],
        [7, 0, 0, 0, 0, 3, 6, 0, 0],
        [0, 0, 0, 0, 9, 1, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 0, 1, 8, 0, 0, 0, 3],
        [0, 0, 0, 3, 0, 6, 0, 4, 5],
        [0, 4, 0, 2, 0, 0, 0, 6, 0],
        [9, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 1, 0, 0],
    ]
}

/// Hard Sudoku puzzle (requires more backtracking).
fn hard_sudoku() -> [[i32; 9]; 9] {
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 8, 5],
        [0, 0, 1, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 5, 0, 7, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 1, 0, 0],
        [0, 9, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 7, 3],
        [0, 0, 2, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 9],
    ]
}

/// Petersen graph: 10 vertices, 15 edges. Chromatic number = 3.
fn petersen_graph() -> (usize, Vec<(usize, usize)>) {
    let edges = vec![
        // Outer cycle
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        // Inner pentagram
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        // Spokes
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    ];
    (10, edges)
}

/// Australia map coloring: 7 states/territories, adjacency edges.
/// Chromatic number = 3 (planar graph).
/// States: 0=WA, 1=NT, 2=SA, 3=QLD, 4=NSW, 5=VIC, 6=TAS
fn australia_map() -> (usize, Vec<(usize, usize)>, Vec<&'static str>) {
    let names = vec!["WA", "NT", "SA", "QLD", "NSW", "VIC", "TAS"];
    let edges = vec![
        (0, 1), // WA-NT
        (0, 2), // WA-SA
        (1, 2), // NT-SA
        (1, 3), // NT-QLD
        (2, 3), // SA-QLD
        (2, 4), // SA-NSW
        (2, 5), // SA-VIC
        (3, 4), // QLD-NSW
        (4, 5), // NSW-VIC
    ];
    (7, edges, names)
}

/// Find the chromatic number of a graph by trying increasing numbers of colors.
fn find_chromatic_number(
    n_vertices: usize,
    edges: &[(usize, usize)],
    max_colors: usize,
) -> Option<(usize, Vec<i32>)> {
    for k in 1..=max_colors {
        let mut csp = build_graph_coloring_csp(n_vertices, edges, k);
        let result = solve(&mut csp);
        if let Some(sol) = result.solution {
            return Some((k, sol));
        }
    }
    None
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Phase 4: Applications — Sudoku, N-Queens, Coloring    ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // --- Sudoku: Easy ---
    println!("━━━ Sudoku: Easy ━━━\n");
    let grid = easy_sudoku();
    println!("  Input:");
    print_sudoku_grid(&grid.iter().flat_map(|r| r.iter()).copied().collect::<Vec<_>>()
        .try_into().unwrap());

    let mut csp = build_sudoku_csp(&grid);
    let empties = csp.variables.iter().filter(|v| v.domain.len() > 1).count();
    println!("\n  {} empty cells, {} constraints", empties, csp.constraints.len());

    let start = Instant::now();
    let result = solve(&mut csp);
    let elapsed = start.elapsed();

    match &result.solution {
        Some(sol) => {
            println!("\n  Solution:");
            let arr: [i32; 81] = sol.as_slice().try_into().unwrap();
            print_sudoku_grid(&arr);
        }
        None => println!("  No solution!"),
    }
    println!("  Nodes: {}, Propagations: {}, Time: {:.1?}", result.nodes_explored, result.propagations, elapsed);

    // --- Sudoku: Medium ---
    println!("\n━━━ Sudoku: Medium ━━━\n");
    let grid = medium_sudoku();
    println!("  Input:");
    print_sudoku_grid(&grid.iter().flat_map(|r| r.iter()).copied().collect::<Vec<_>>()
        .try_into().unwrap());

    let mut csp = build_sudoku_csp(&grid);
    let empties = csp.variables.iter().filter(|v| v.domain.len() > 1).count();
    println!("\n  {} empty cells, {} constraints", empties, csp.constraints.len());

    let start = Instant::now();
    let result = solve(&mut csp);
    let elapsed = start.elapsed();

    match &result.solution {
        Some(sol) => {
            println!("\n  Solution:");
            let arr: [i32; 81] = sol.as_slice().try_into().unwrap();
            print_sudoku_grid(&arr);
        }
        None => println!("  No solution!"),
    }
    println!("  Nodes: {}, Propagations: {}, Time: {:.1?}", result.nodes_explored, result.propagations, elapsed);

    // --- Sudoku: Hard ---
    println!("\n━━━ Sudoku: Hard ━━━\n");
    let grid = hard_sudoku();
    println!("  Input:");
    print_sudoku_grid(&grid.iter().flat_map(|r| r.iter()).copied().collect::<Vec<_>>()
        .try_into().unwrap());

    let mut csp = build_sudoku_csp(&grid);
    let empties = csp.variables.iter().filter(|v| v.domain.len() > 1).count();
    println!("\n  {} empty cells, {} constraints", empties, csp.constraints.len());

    let start = Instant::now();
    let result = solve(&mut csp);
    let elapsed = start.elapsed();

    match &result.solution {
        Some(sol) => {
            println!("\n  Solution:");
            let arr: [i32; 81] = sol.as_slice().try_into().unwrap();
            print_sudoku_grid(&arr);
        }
        None => println!("  No solution!"),
    }
    println!("  Nodes: {}, Propagations: {}, Time: {:.1?}", result.nodes_explored, result.propagations, elapsed);

    // --- 8-Queens ---
    println!("\n━━━ 8-Queens ━━━\n");
    let mut csp = build_nqueens_csp(8);
    let start = Instant::now();
    let result = solve(&mut csp);
    let elapsed = start.elapsed();

    match &result.solution {
        Some(sol) => {
            println!("  Solution: {:?}", sol);
            println!("\n  Board:");
            print_queens_board(8, sol);
        }
        None => println!("  No solution!"),
    }
    println!("  Nodes: {}, Propagations: {}, Time: {:.1?}", result.nodes_explored, result.propagations, elapsed);

    // --- 12-Queens (larger) ---
    println!("\n━━━ 12-Queens ━━━\n");
    let mut csp = build_nqueens_csp(12);
    let start = Instant::now();
    let result = solve(&mut csp);
    let elapsed = start.elapsed();

    match &result.solution {
        Some(sol) => {
            println!("  Solution: {:?}", sol);
            println!("\n  Board:");
            print_queens_board(12, sol);
        }
        None => println!("  No solution!"),
    }
    println!("  Nodes: {}, Propagations: {}, Time: {:.1?}", result.nodes_explored, result.propagations, elapsed);

    // --- Graph Coloring: Australia Map ---
    println!("\n━━━ Graph Coloring: Australia Map ━━━\n");
    let (n, edges, names) = australia_map();
    println!("  {} states, {} borders", n, edges.len());
    for (u, v) in &edges {
        println!("    {} — {}", names[*u], names[*v]);
    }

    let color_names = ["Red", "Green", "Blue", "Yellow"];

    match find_chromatic_number(n, &edges, 4) {
        Some((k, sol)) => {
            println!("\n  Chromatic number: {}", k);
            println!("  Coloring:");
            for (i, &c) in sol.iter().enumerate() {
                println!("    {} = {} ({})", names[i], c, color_names[c as usize]);
            }
        }
        None => println!("  No coloring found with <= 4 colors!"),
    }

    // --- Graph Coloring: Petersen Graph ---
    println!("\n━━━ Graph Coloring: Petersen Graph ━━━\n");
    let (n, edges) = petersen_graph();
    println!("  {} vertices, {} edges", n, edges.len());

    // Try 2 colors first (should fail — Petersen is not bipartite)
    println!("\n  Trying 2 colors...");
    let mut csp = build_graph_coloring_csp(n, &edges, 2);
    let result = solve(&mut csp);
    match &result.solution {
        Some(_) => println!("  2 colors: feasible (unexpected!)"),
        None => println!("  2 colors: infeasible (correct — Petersen has odd cycles)"),
    }

    // Try 3 colors (should succeed — chromatic number = 3)
    println!("\n  Trying 3 colors...");
    let mut csp = build_graph_coloring_csp(n, &edges, 3);
    let start = Instant::now();
    let result = solve(&mut csp);
    let elapsed = start.elapsed();

    match &result.solution {
        Some(sol) => {
            println!("  3 colors: feasible (correct — chromatic number = 3)");
            println!("  Coloring: {:?}", sol);
            // Verify
            let valid = edges.iter().all(|&(u, v)| sol[u] != sol[v]);
            println!("  Valid coloring: {}", valid);
        }
        None => println!("  3 colors: infeasible (unexpected!)"),
    }
    println!("  Nodes: {}, Propagations: {}, Time: {:.1?}", result.nodes_explored, result.propagations, elapsed);

    println!("\n━━━ Phase 4 Complete ━━━");
    println!("  You modeled graph coloring as a CSP and solved Sudoku, N-Queens,");
    println!("  and graph coloring using your AC-3 + backtracking solver.");
    println!("  Key insight: the same solver handles all three — the modeling is");
    println!("  what differs. This is the power of constraint programming.");
}
