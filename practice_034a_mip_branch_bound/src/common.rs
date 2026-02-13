// common.rs — Shared types for the Branch & Bound MIP solver.
//
// This module is included via `#[path = "common.rs"] mod common;` in each binary,
// since each binary has its own `fn main()` and Cargo treats them as separate crates.

/// Variable type: Continuous (no integrality constraint), Integer, or Binary (0-1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VarType {
    Continuous,
    Integer,
    Binary,
}

/// Optimization direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sense {
    Maximize,
    Minimize,
}

/// Constraint type (all constraints are of the form: row * x {<=, >=, =} rhs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstraintType {
    Leq, // <=
    Geq, // >=
    Eq,  // =
}

/// A MIP problem definition.
///
/// minimize/maximize  c^T x
/// subject to         A[i] * x  {<=, >=, =}  b[i]   for each constraint i
///                    lb[j] <= x[j] <= ub[j]          for each variable j
///                    x[j] in Z  if var_types[j] is Integer or Binary
#[derive(Debug, Clone)]
pub struct MIPProblem {
    /// Objective coefficients (one per variable).
    pub objective: Vec<f64>,
    /// Optimization direction.
    pub sense: Sense,
    /// Constraint matrix: constraints[i][j] is coefficient of variable j in constraint i.
    pub constraints: Vec<Vec<f64>>,
    /// Constraint right-hand sides.
    pub rhs: Vec<f64>,
    /// Constraint types.
    pub constraint_types: Vec<ConstraintType>,
    /// Variable types.
    pub var_types: Vec<VarType>,
    /// Lower bounds per variable.
    pub lower_bounds: Vec<f64>,
    /// Upper bounds per variable.
    pub upper_bounds: Vec<f64>,
    /// Variable names (for display).
    pub var_names: Vec<String>,
}

impl MIPProblem {
    /// Number of variables.
    pub fn num_vars(&self) -> usize {
        self.objective.len()
    }

    /// Number of constraints.
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }
}

/// A node in the Branch & Bound tree.
///
/// Each node represents a subproblem: the original MIP with additional
/// variable bound restrictions imposed by branching decisions.
#[derive(Debug, Clone)]
pub struct BBNode {
    /// Tightened lower bounds (overrides problem defaults for this node).
    pub lower_bounds: Vec<f64>,
    /// Tightened upper bounds (overrides problem defaults for this node).
    pub upper_bounds: Vec<f64>,
    /// LP relaxation bound at this node (set after solving).
    pub lp_bound: f64,
    /// Depth in the B&B tree (root = 0).
    pub depth: usize,
    /// Node ID (for tracking/display).
    pub id: usize,
}

/// Result of the Branch & Bound solver.
#[derive(Debug, Clone)]
pub struct BBResult {
    /// Optimal solution (variable values), or None if infeasible.
    pub solution: Option<Vec<f64>>,
    /// Optimal objective value, or None if infeasible.
    pub objective: Option<f64>,
    /// Total nodes explored (LP relaxations solved).
    pub nodes_explored: usize,
    /// Nodes pruned by bound.
    pub pruned_by_bound: usize,
    /// Nodes pruned by infeasibility.
    pub pruned_by_infeasibility: usize,
    /// Nodes pruned by integrality (LP solution was integer-feasible).
    pub pruned_by_integrality: usize,
}

/// Default tolerance for checking if a value is integer.
pub const INT_TOLERANCE: f64 = 1e-6;

/// Check if a single value is close enough to an integer.
pub fn is_integer(value: f64, tolerance: f64) -> bool {
    let rounded = value.round();
    (value - rounded).abs() < tolerance
}

/// Check if a solution is integer-feasible: all integer/binary variables
/// have values within `tolerance` of an integer.
pub fn is_integer_feasible(solution: &[f64], var_types: &[VarType], tolerance: f64) -> bool {
    solution.iter().zip(var_types.iter()).all(|(&val, &vt)| match vt {
        VarType::Continuous => true,
        VarType::Integer | VarType::Binary => is_integer(val, tolerance),
    })
}

/// Find all fractional integer/binary variables in a solution.
/// Returns vec of (variable_index, fractional_value).
pub fn find_fractional_variables(
    solution: &[f64],
    var_types: &[VarType],
    tolerance: f64,
) -> Vec<(usize, f64)> {
    solution
        .iter()
        .enumerate()
        .zip(var_types.iter())
        .filter_map(|((idx, &val), &vt)| match vt {
            VarType::Continuous => None,
            VarType::Integer | VarType::Binary => {
                if is_integer(val, tolerance) {
                    None
                } else {
                    Some((idx, val))
                }
            }
        })
        .collect()
}

/// Pretty-print a solution with variable names.
pub fn print_solution(solution: &[f64], var_names: &[String]) {
    println!("  Solution:");
    for (i, (val, name)) in solution.iter().zip(var_names.iter()).enumerate() {
        println!("    x[{}] ({}) = {:.6}", i, name, val);
    }
}

/// Print a BBResult summary.
pub fn print_bb_result(result: &BBResult, problem: &MIPProblem) {
    println!("\n=== Branch & Bound Result ===");
    match (&result.solution, &result.objective) {
        (Some(sol), Some(obj)) => {
            println!("  Status: OPTIMAL");
            println!("  Objective: {:.6}", obj);
            print_solution(sol, &problem.var_names);
        }
        _ => {
            println!("  Status: INFEASIBLE (no integer-feasible solution found)");
        }
    }
    println!("  Nodes explored:          {}", result.nodes_explored);
    println!("  Pruned by bound:         {}", result.pruned_by_bound);
    println!("  Pruned by infeasibility:  {}", result.pruned_by_infeasibility);
    println!("  Pruned by integrality:    {}", result.pruned_by_integrality);
    let total_pruned =
        result.pruned_by_bound + result.pruned_by_infeasibility + result.pruned_by_integrality;
    println!("  Total pruned:            {}", total_pruned);
    println!("==============================\n");
}

/// Create a root BBNode from a MIPProblem.
pub fn make_root_node(problem: &MIPProblem) -> BBNode {
    BBNode {
        lower_bounds: problem.lower_bounds.clone(),
        upper_bounds: problem.upper_bounds.clone(),
        lp_bound: f64::INFINITY, // will be set after LP solve
        depth: 0,
        id: 0,
    }
}

// ---------------------------------------------------------------------------
// Test problems
// ---------------------------------------------------------------------------

/// Binary knapsack problem:
///   maximize  8*x1 + 5*x2 + 4*x3
///   s.t.      6*x1 + 4*x2 + 3*x3 <= 12
///             x1, x2, x3 in {0, 1}
///
/// LP relaxation optimum: x1=1.0, x2=1.0, x3=0.667 → obj=13.667
/// MIP optimum: x1=1, x2=1, x3=0 → obj=13  (or x1=1, x2=0, x3=1 → obj=12)
/// Actually MIP optimum: x1=1, x2=1, x3=0 → obj=13
pub fn knapsack_problem() -> MIPProblem {
    MIPProblem {
        objective: vec![8.0, 5.0, 4.0],
        sense: Sense::Maximize,
        constraints: vec![vec![6.0, 4.0, 3.0]],
        rhs: vec![12.0],
        constraint_types: vec![ConstraintType::Leq],
        var_types: vec![VarType::Binary, VarType::Binary, VarType::Binary],
        lower_bounds: vec![0.0, 0.0, 0.0],
        upper_bounds: vec![1.0, 1.0, 1.0],
        var_names: vec!["x1".into(), "x2".into(), "x3".into()],
    }
}

/// Larger knapsack for benchmarking strategies:
///   maximize  10*x1 + 9*x2 + 8*x3 + 7*x4 + 6*x5 + 5*x6
///   s.t.      6*x1 + 5*x2 + 5*x3 + 4*x4 + 3*x5 + 3*x6 <= 15
///             x_i in {0, 1}
pub fn large_knapsack_problem() -> MIPProblem {
    MIPProblem {
        objective: vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
        sense: Sense::Maximize,
        constraints: vec![vec![6.0, 5.0, 5.0, 4.0, 3.0, 3.0]],
        rhs: vec![15.0],
        constraint_types: vec![ConstraintType::Leq],
        var_types: vec![VarType::Binary; 6],
        lower_bounds: vec![0.0; 6],
        upper_bounds: vec![1.0; 6],
        var_names: (1..=6).map(|i| format!("x{}", i)).collect(),
    }
}

/// Set cover problem:
///   minimize  3*x1 + 2*x2 + 4*x3 + 3*x4
///   s.t.      x1 + x3           >= 1   (element 1 must be covered)
///             x1 + x2 + x4      >= 1   (element 2 must be covered)
///                  x2 + x3 + x4 >= 1   (element 3 must be covered)
///             x_i in {0, 1}
///
/// Each x_i represents "select set i". Each constraint says "at least one
/// set covering element j must be selected."
pub fn set_cover_problem() -> MIPProblem {
    MIPProblem {
        objective: vec![3.0, 2.0, 4.0, 3.0],
        sense: Sense::Minimize,
        constraints: vec![
            vec![1.0, 0.0, 1.0, 0.0], // element 1
            vec![1.0, 1.0, 0.0, 1.0], // element 2
            vec![0.0, 1.0, 1.0, 1.0], // element 3
        ],
        rhs: vec![1.0, 1.0, 1.0],
        constraint_types: vec![ConstraintType::Geq, ConstraintType::Geq, ConstraintType::Geq],
        var_types: vec![VarType::Binary; 4],
        lower_bounds: vec![0.0; 4],
        upper_bounds: vec![1.0; 4],
        var_names: (1..=4).map(|i| format!("x{}", i)).collect(),
    }
}
