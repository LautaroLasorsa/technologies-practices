// common.rs — Shared types for the Constraint Programming CSP solver.
//
// This module is included via `#[path = "common.rs"] mod common;` in each binary,
// since each binary has its own `fn main()` and Cargo treats them as separate crates.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A variable's domain: set of possible integer values.
/// BTreeSet keeps values sorted, which makes output deterministic and debugging easier.
pub type Domain = BTreeSet<i32>;

/// A CSP variable with name and domain.
#[derive(Clone, Debug)]
pub struct Variable {
    pub name: String,
    pub domain: Domain,
}

impl Variable {
    /// Create a variable with a range domain [low, high].
    pub fn new_range(name: &str, low: i32, high: i32) -> Self {
        Variable {
            name: name.to_string(),
            domain: (low..=high).collect(),
        }
    }

    /// Create a variable with an explicit set of values.
    pub fn new_set(name: &str, values: &[i32]) -> Self {
        Variable {
            name: name.to_string(),
            domain: values.iter().copied().collect(),
        }
    }
}

/// A binary constraint between two variables.
/// The constraint function returns true if (val_i, val_j) is an allowed pair.
pub struct BinaryConstraint {
    /// Index of the first variable.
    pub var_i: usize,
    /// Index of the second variable.
    pub var_j: usize,
    /// Human-readable name for display/debugging.
    pub name: String,
    /// The constraint predicate: returns true if (val_i, val_j) is allowed.
    pub check: Box<dyn Fn(i32, i32) -> bool>,
}

impl fmt::Debug for BinaryConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Constraint({} <-> {}: {})", self.var_i, self.var_j, self.name)
    }
}

/// A Constraint Satisfaction Problem: variables + binary constraints.
pub struct CSP {
    pub variables: Vec<Variable>,
    pub constraints: Vec<BinaryConstraint>,
}

impl CSP {
    /// Create a new empty CSP.
    pub fn new() -> Self {
        CSP {
            variables: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Add a variable and return its index.
    pub fn add_variable(&mut self, var: Variable) -> usize {
        let idx = self.variables.len();
        self.variables.push(var);
        idx
    }

    /// Add a binary constraint.
    pub fn add_constraint(&mut self, constraint: BinaryConstraint) {
        self.constraints.push(constraint);
    }

    /// Add a not-equal constraint between two variables.
    pub fn add_not_equal(&mut self, var_i: usize, var_j: usize, name: &str) {
        self.constraints.push(BinaryConstraint {
            var_i,
            var_j,
            name: name.to_string(),
            check: Box::new(|a, b| a != b),
        });
    }

    /// Build an adjacency map: for each variable index, list (constraint_index, other_variable).
    /// This is used by AC-3 to quickly find which arcs to re-enqueue when a domain changes.
    pub fn build_neighbor_map(&self) -> HashMap<usize, Vec<(usize, usize)>> {
        let mut map: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (ci, c) in self.constraints.iter().enumerate() {
            map.entry(c.var_i).or_default().push((ci, c.var_j));
            map.entry(c.var_j).or_default().push((ci, c.var_i));
        }
        map
    }

    /// Get a snapshot of all domains (for save/restore during backtracking).
    pub fn save_domains(&self) -> Vec<Domain> {
        self.variables.iter().map(|v| v.domain.clone()).collect()
    }

    /// Restore all domains from a snapshot.
    pub fn restore_domains(&mut self, domains: &[Domain]) {
        for (var, dom) in self.variables.iter_mut().zip(domains.iter()) {
            var.domain = dom.clone();
        }
    }

    /// Check if the CSP is solved: every variable has exactly one value in its domain.
    pub fn is_solved(&self) -> bool {
        self.variables.iter().all(|v| v.domain.len() == 1)
    }

    /// Extract the solution (one value per variable) if all domains are singletons.
    pub fn extract_solution(&self) -> Option<Vec<i32>> {
        if self.is_solved() {
            Some(
                self.variables
                    .iter()
                    .map(|v| *v.domain.iter().next().unwrap())
                    .collect(),
            )
        } else {
            None
        }
    }
}

/// Result of solving a CSP.
#[derive(Debug)]
pub struct CSPResult {
    /// Assignment for each variable (None if unsolved/infeasible).
    pub solution: Option<Vec<i32>>,
    /// Number of search tree nodes explored.
    pub nodes_explored: usize,
    /// Number of REVISE operations performed.
    pub propagations: usize,
}

impl CSPResult {
    pub fn new() -> Self {
        CSPResult {
            solution: None,
            nodes_explored: 0,
            propagations: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

/// Print the current state of all variables and their domains.
pub fn print_domains(csp: &CSP) {
    println!("  Variables and domains:");
    for (i, var) in csp.variables.iter().enumerate() {
        let vals: Vec<String> = var.domain.iter().map(|v| v.to_string()).collect();
        println!(
            "    [{}] {} : {{{}}} (size {})",
            i,
            var.name,
            vals.join(", "),
            var.domain.len()
        );
    }
}

/// Print the constraints in a CSP.
pub fn print_constraints(csp: &CSP) {
    println!("  Constraints:");
    for (i, c) in csp.constraints.iter().enumerate() {
        println!(
            "    [{}] {} (vars {} <-> {})",
            i, c.name, c.var_i, c.var_j
        );
    }
}

/// Print a solution as variable=value pairs.
pub fn print_solution(csp: &CSP, solution: &[i32]) {
    println!("  Solution:");
    for (i, (&val, var)) in solution.iter().zip(csp.variables.iter()).enumerate() {
        println!("    [{}] {} = {}", i, var.name, val);
    }
}

/// Print an N-Queens board.
pub fn print_queens_board(n: usize, queens: &[i32]) {
    for row in 0..n {
        let col = queens[row] as usize;
        let mut line = String::with_capacity(2 * n);
        for c in 0..n {
            if c == col {
                line.push_str(" Q");
            } else {
                line.push_str(" .");
            }
        }
        println!("   {}", line);
    }
}

/// Print a 9x9 Sudoku grid.
pub fn print_sudoku_grid(grid: &[i32; 81]) {
    for row in 0..9 {
        if row % 3 == 0 && row > 0 {
            println!("    ------+-------+------");
        }
        let mut line = String::new();
        for col in 0..9 {
            if col % 3 == 0 && col > 0 {
                line.push_str(" |");
            }
            let val = grid[row * 9 + col];
            if val == 0 {
                line.push_str(" .");
            } else {
                line.push_str(&format!(" {}", val));
            }
        }
        println!("    {}", line);
    }
}

// ---------------------------------------------------------------------------
// Convenience: allow unused imports in binaries that only use a subset.
// ---------------------------------------------------------------------------
#[allow(unused_imports)]
pub use std::collections::{BTreeSet as _BTreeSet, HashMap as _HashMap, VecDeque as _VecDeque};
