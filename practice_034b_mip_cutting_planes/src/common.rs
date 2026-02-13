// ============================================================================
// Common types and utilities for MIP Cutting Planes practice
// ============================================================================
//
// This module provides:
//   - A minimal simplex tableau tracker for small problems (needed because
//     good_lp abstracts away solver internals and doesn't expose tableau data)
//   - MIP problem representation
//   - LP solving wrappers using good_lp/HiGHS
//   - Integer-feasibility checking utilities
//
// The manual simplex is intentionally limited to small problems (< 20 vars).
// For larger problems, use good_lp/HiGHS directly.
// ============================================================================

use ordered_float::OrderedFloat;
use std::fmt;

// ─── Tolerances ──────────────────────────────────────────────────────────────

/// Tolerance for considering a value integer
pub const INT_TOL: f64 = 1e-6;

/// Tolerance for considering a value zero (pivot, reduced cost)
pub const ZERO_TOL: f64 = 1e-10;

/// Tolerance for considering two floats equal
pub const EQ_TOL: f64 = 1e-8;

// ─── Variable types ──────────────────────────────────────────────────────────

/// Type of a decision variable in the MIP
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarType {
    Continuous,
    Integer,
    Binary,
}

// ─── Constraint sense ────────────────────────────────────────────────────────

/// Sense of a linear constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintSense {
    Le, // <=
    Ge, // >=
    Eq, // ==
}

// ─── Linear constraint ──────────────────────────────────────────────────────

/// A single linear constraint: coeffs^T x {<=, >=, ==} rhs
#[derive(Debug, Clone)]
pub struct LinearConstraint {
    pub coeffs: Vec<f64>,
    pub sense: ConstraintSense,
    pub rhs: f64,
}

// ─── MIP Problem ─────────────────────────────────────────────────────────────

/// A Mixed-Integer Programming problem in the form:
///   minimize  obj^T x
///   subject to  constraints
///   x_i >= 0 for all i
///   x_i integer/binary as specified by var_types
#[derive(Debug, Clone)]
pub struct MIPProblem {
    pub obj: Vec<f64>,           // objective coefficients (minimize)
    pub constraints: Vec<LinearConstraint>,
    pub var_types: Vec<VarType>, // one per variable
    pub var_names: Vec<String>,  // optional names for display
    pub n_vars: usize,
}

impl MIPProblem {
    /// Create a new MIP problem.
    pub fn new(
        obj: Vec<f64>,
        constraints: Vec<LinearConstraint>,
        var_types: Vec<VarType>,
        var_names: Vec<String>,
    ) -> Self {
        let n_vars = obj.len();
        assert_eq!(var_types.len(), n_vars);
        assert_eq!(var_names.len(), n_vars);
        for c in &constraints {
            assert_eq!(c.coeffs.len(), n_vars);
        }
        Self {
            obj,
            constraints,
            var_types,
            var_names,
            n_vars,
        }
    }
}

// ─── Simplex Tableau (manual, for small problems) ────────────────────────────

/// A simplex tableau for small-scale LP problems.
///
/// The tableau stores the augmented matrix in the form:
///
///   [  A_bar  | b_bar ]     (m rows: constraint rows)
///   [ c_bar^T | -z    ]     (1 row: objective row)
///
/// where A_bar = B^{-1} A, b_bar = B^{-1} b, c_bar = reduced costs.
///
/// Layout: `data` has (m+1) rows and (n+1) columns.
///   - Rows 0..m-1 are constraint rows
///   - Row m is the objective row
///   - Column n (last) is the RHS column
///
/// `basis[i]` = index of the variable that is basic in row i (0-indexed).
#[derive(Clone)]
pub struct SimplexTableau {
    pub data: Vec<Vec<f64>>,  // (m+1) x (n+1)
    pub basis: Vec<usize>,     // length m: basis[row] = var index
    pub m: usize,              // number of constraints
    pub n: usize,              // number of variables (including slacks)
}

impl SimplexTableau {
    /// Build a tableau from standard-form LP:
    ///   min c^T x  s.t. Ax = b, x >= 0
    /// where the last m columns of A are an identity (slack basis).
    ///
    /// `c` has length n, `a` is m x n (row-major), `b` has length m.
    /// The initial basis is columns (n-m)..n (slack variables).
    pub fn from_standard_form(c: &[f64], a: &[Vec<f64>], b: &[f64]) -> Self {
        let m = b.len();
        let n = c.len();
        assert!(n >= m);
        for row in a {
            assert_eq!(row.len(), n);
        }

        // Build tableau: m constraint rows + 1 objective row, each with n+1 columns
        let mut data = Vec::with_capacity(m + 1);

        // Constraint rows: [A | b]
        for i in 0..m {
            let mut row = a[i].clone();
            row.push(b[i]);
            data.push(row);
        }

        // Objective row: [c | 0]
        let mut obj_row = c.to_vec();
        obj_row.push(0.0); // -z = 0 initially
        data.push(obj_row);

        // Initial basis: last m variables (slacks)
        let basis: Vec<usize> = (n - m..n).collect();

        Self { data, basis, m, n }
    }

    /// Get the RHS value for constraint row i (0-indexed).
    pub fn rhs(&self, row: usize) -> f64 {
        self.data[row][self.n]
    }

    /// Get the reduced cost for variable j.
    pub fn reduced_cost(&self, j: usize) -> f64 {
        self.data[self.m][j]
    }

    /// Get the current objective value (-z is stored in objective row's RHS).
    pub fn objective_value(&self) -> f64 {
        -self.data[self.m][self.n]
    }

    /// Get the current solution vector (length n).
    /// Basic variables have values from RHS; non-basic variables are 0.
    pub fn solution(&self) -> Vec<f64> {
        let mut x = vec![0.0; self.n];
        for (row, &var) in self.basis.iter().enumerate() {
            x[var] = self.data[row][self.n];
        }
        x
    }

    /// Get the tableau row for constraint row i (non-basic coefficients only).
    /// Returns (non_basic_indices, non_basic_coefficients, rhs).
    pub fn tableau_row(&self, row: usize) -> (Vec<usize>, Vec<f64>, f64) {
        let non_basic: Vec<usize> = (0..self.n)
            .filter(|j| !self.basis.contains(j))
            .collect();
        let coeffs: Vec<f64> = non_basic.iter().map(|&j| self.data[row][j]).collect();
        let rhs = self.data[row][self.n];
        (non_basic, coeffs, rhs)
    }

    /// Get the full tableau row for constraint row i (all n coefficients + rhs).
    pub fn full_row(&self, row: usize) -> Vec<f64> {
        self.data[row].clone()
    }

    /// Perform a single simplex pivot: entering variable `enter`, leaving row `leave`.
    pub fn pivot(&mut self, enter: usize, leave: usize) {
        let pivot_elem = self.data[leave][enter];
        assert!(
            pivot_elem.abs() > ZERO_TOL,
            "Pivot element too small: {}",
            pivot_elem
        );

        // Scale the pivot row
        let inv = 1.0 / pivot_elem;
        for j in 0..=self.n {
            self.data[leave][j] *= inv;
        }

        // Eliminate the entering variable from all other rows (including objective)
        for i in 0..=self.m {
            if i == leave {
                continue;
            }
            let factor = self.data[i][enter];
            if factor.abs() < ZERO_TOL {
                continue;
            }
            for j in 0..=self.n {
                self.data[i][j] -= factor * self.data[leave][j];
            }
        }

        // Update basis
        self.basis[leave] = enter;
    }

    /// Run the simplex method to optimality (or detect unbounded/infeasible).
    /// Returns Ok(()) on success, Err(msg) on failure.
    pub fn solve(&mut self, max_iters: usize) -> Result<SimplexStatus, String> {
        for _iter in 0..max_iters {
            // Find entering variable: most negative reduced cost
            let mut enter = None;
            let mut best_rc = -ZERO_TOL;
            for j in 0..self.n {
                let rc = self.reduced_cost(j);
                if rc < best_rc {
                    best_rc = rc;
                    enter = Some(j);
                }
            }

            let enter = match enter {
                Some(j) => j,
                None => return Ok(SimplexStatus::Optimal), // all reduced costs >= 0
            };

            // Find leaving variable: minimum ratio test
            let mut leave = None;
            let mut min_ratio = f64::INFINITY;
            for i in 0..self.m {
                let aij = self.data[i][enter];
                if aij > ZERO_TOL {
                    let ratio = self.data[i][self.n] / aij;
                    if ratio < min_ratio {
                        min_ratio = ratio;
                        leave = Some(i);
                    }
                }
            }

            let leave = match leave {
                Some(i) => i,
                None => return Ok(SimplexStatus::Unbounded),
            };

            self.pivot(enter, leave);
        }

        Err("Maximum iterations reached".to_string())
    }

    /// Add a new constraint row to the tableau (for cutting planes).
    ///
    /// The cut is given as: sum_j cut_coeffs[j] * x_j >= cut_rhs
    /// This is converted to standard form with a slack: sum_j cut_coeffs[j] * x_j - s = cut_rhs
    /// (or equivalently, the slack has coefficient -1 and we store it as a <= with negation).
    ///
    /// For Gomory cuts expressed as sum_j f_j * x_j >= f_0:
    ///   Introduce surplus variable s >= 0: sum_j f_j * x_j - s = f_0
    ///   Rearranged for tableau: s = -f_0 + sum_j f_j * x_j
    ///   Since s is the new basic variable, the new row in the tableau is:
    ///     [-f_j for all j ... | 1 (for s) | -f_0]
    ///   But since s starts basic with a negative RHS, we need dual simplex.
    ///
    /// Simpler approach: add as a >= constraint, convert, and re-solve.
    /// Here we add the cut row directly and use dual simplex if needed.
    pub fn add_gomory_cut_row(&mut self, cut_coeffs: &[f64], cut_rhs: f64) {
        // The cut is: sum_j cut_coeffs[j] * x_j >= cut_rhs
        // Add surplus variable s: sum_j cut_coeffs[j] * x_j - s = cut_rhs
        // In the tableau, s is the new basic variable.
        // New row: [cut_coeffs[0], ..., cut_coeffs[n-1], 0 (for old slacks padding), -1 (for s), cut_rhs]
        // But we need to express this in terms of non-basic variables using current tableau.

        // Since the cut is expressed in original + slack variables (the full space),
        // and the tableau already has columns for all existing variables,
        // we need to add the cut row in the current tableau basis representation.

        // The cut coefficients are given in terms of NON-BASIC variables (as generated
        // from the tableau row). To add it, we create a new slack variable column.

        let new_slack_idx = self.n;
        self.n += 1;

        // Extend all existing rows with 0 for the new slack
        for i in 0..=self.m {
            let rhs = self.data[i].pop().unwrap(); // temporarily remove RHS
            self.data[i].push(0.0); // coefficient for new slack
            self.data[i].push(rhs); // put RHS back
        }

        // Build the new constraint row.
        // The cut in the current basis is: sum_j cut_coeffs[j] * x_j >= cut_rhs
        // where cut_coeffs are in the FULL variable space (length = old n).
        // Subtract surplus: sum_j cut_coeffs[j] * x_j - s_new = cut_rhs
        //
        // But the current LP solution has non-basics at 0 and basics from RHS.
        // The cut expressed in the current tableau has:
        //   new_row[j] = cut_coeffs[j] for all original variables j
        //   new_row[new_slack] = -1 (surplus)
        //   new_row[RHS] = cut_rhs
        //
        // However, we need to express basic variables in terms of non-basics.
        // Since cut_coeffs should already be in terms of the tableau (non-basic
        // coefficients), we handle it accordingly.

        // For Gomory cuts generated from tableau rows, cut_coeffs has length = n_original
        // and contains coefficients for ALL variables (basic + non-basic).
        // The cut row in the augmented tableau needs the surplus variable.

        let mut new_row = Vec::with_capacity(self.n + 1);
        for j in 0..self.n {
            if j < cut_coeffs.len() {
                new_row.push(cut_coeffs[j]);
            } else if j == new_slack_idx {
                new_row.push(-1.0); // surplus variable
            } else {
                new_row.push(0.0);
            }
        }
        new_row.push(cut_rhs); // RHS

        // Insert the new row before the objective row
        let obj_row = self.data.pop().unwrap();
        self.data.push(new_row);
        self.data.push(obj_row);

        // The new slack is basic in the new row
        self.basis.push(new_slack_idx);
        self.m += 1;

        // Extend objective row: new slack has zero reduced cost
        // (already handled by the 0.0 we added above)
    }

    /// Perform dual simplex iterations to restore feasibility after adding a cut
    /// (the new row may have negative RHS).
    pub fn dual_simplex_restore(&mut self, max_iters: usize) -> Result<SimplexStatus, String> {
        for _iter in 0..max_iters {
            // Find the most negative RHS (infeasible row)
            let mut leave = None;
            let mut most_neg = -ZERO_TOL;
            for i in 0..self.m {
                if self.data[i][self.n] < most_neg {
                    most_neg = self.data[i][self.n];
                    leave = Some(i);
                }
            }

            let leave = match leave {
                Some(i) => i,
                None => return Ok(SimplexStatus::Optimal), // all RHS >= 0, feasible
            };

            // Find entering variable: minimum ratio of reduced cost to negative coefficient
            // (dual simplex pricing)
            let mut enter = None;
            let mut min_ratio = f64::INFINITY;
            for j in 0..self.n {
                let aij = self.data[leave][j];
                if aij < -ZERO_TOL {
                    let ratio = self.reduced_cost(j) / (-aij);
                    if ratio < min_ratio {
                        min_ratio = ratio;
                        enter = Some(j);
                    }
                }
            }

            let enter = match enter {
                Some(j) => j,
                None => return Ok(SimplexStatus::Infeasible),
            };

            self.pivot(enter, leave);
        }

        Err("Dual simplex: maximum iterations reached".to_string())
    }
}

impl fmt::Display for SimplexTableau {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Simplex Tableau ({} constraints, {} variables):", self.m, self.n)?;
        writeln!(f, "Basis: {:?}", self.basis)?;
        writeln!(f, "{:-<width$}", "", width = 12 * (self.n + 2))?;

        // Header
        write!(f, "{:>10}", "")?;
        for j in 0..self.n {
            write!(f, " x{:<8}", j)?;
        }
        writeln!(f, " {:>9}", "RHS")?;
        writeln!(f, "{:-<width$}", "", width = 12 * (self.n + 2))?;

        // Constraint rows
        for i in 0..self.m {
            write!(f, "  R{} (x{}) ", i, self.basis[i])?;
            for j in 0..=self.n {
                write!(f, " {:>8.4}", self.data[i][j])?;
            }
            writeln!(f)?;
        }

        // Objective row
        writeln!(f, "{:-<width$}", "", width = 12 * (self.n + 2))?;
        write!(f, "  z       ")?;
        for j in 0..=self.n {
            write!(f, " {:>8.4}", self.data[self.m][j])?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "Objective value: {:.6}",
            self.objective_value()
        )?;

        Ok(())
    }
}

/// Status after simplex solve
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimplexStatus {
    Optimal,
    Unbounded,
    Infeasible,
}

// ─── LP Solution wrapper ─────────────────────────────────────────────────────

/// Result of solving an LP relaxation
#[derive(Debug, Clone)]
pub struct LPSolution {
    pub values: Vec<f64>,
    pub objective: f64,
    pub status: SimplexStatus,
}

// ─── Cutting plane result ────────────────────────────────────────────────────

/// A single generated cut: coeffs^T x >= rhs
#[derive(Debug, Clone)]
pub struct Cut {
    pub coeffs: Vec<f64>,
    pub rhs: f64,
    pub description: String,
}

impl fmt::Display for Cut {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let terms: Vec<String> = self
            .coeffs
            .iter()
            .enumerate()
            .filter(|(_, &c)| c.abs() > ZERO_TOL)
            .map(|(j, &c)| {
                if c == 1.0 {
                    format!("x{}", j)
                } else {
                    format!("{:.4}*x{}", c, j)
                }
            })
            .collect();
        write!(f, "{} >= {:.4}", terms.join(" + "), self.rhs)?;
        if !self.description.is_empty() {
            write!(f, "  ({})", self.description)?;
        }
        Ok(())
    }
}

// ─── Result types ────────────────────────────────────────────────────────────

/// Result of the cutting plane algorithm
#[derive(Debug, Clone)]
pub struct CuttingPlaneResult {
    pub solution: Vec<f64>,
    pub objective: f64,
    pub is_integer: bool,
    pub rounds: usize,
    pub bound_history: Vec<f64>, // LP bound at each round
    pub cuts_added: usize,
}

impl fmt::Display for CuttingPlaneResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Cutting Plane Result:")?;
        writeln!(f, "  Rounds:     {}", self.rounds)?;
        writeln!(f, "  Cuts added: {}", self.cuts_added)?;
        writeln!(f, "  Objective:  {:.6}", self.objective)?;
        writeln!(f, "  Integer:    {}", self.is_integer)?;
        writeln!(f, "  Solution:   {:?}", self.solution.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>())?;
        writeln!(f, "  Bound history: {:?}", self.bound_history.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>())?;
        Ok(())
    }
}

/// Result of branch-and-bound / branch-and-cut
#[derive(Debug, Clone)]
pub struct BBResult {
    pub solution: Option<Vec<f64>>,
    pub objective: Option<f64>,
    pub nodes_explored: usize,
    pub cuts_generated: usize,
    pub incumbent_from_heuristic: bool,
}

impl fmt::Display for BBResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Branch-and-Bound Result:")?;
        writeln!(f, "  Nodes explored:     {}", self.nodes_explored)?;
        writeln!(f, "  Cuts generated:     {}", self.cuts_generated)?;
        if let Some(ref sol) = self.solution {
            writeln!(f, "  Objective:          {:.6}", self.objective.unwrap())?;
            writeln!(f, "  Solution:           {:?}", sol.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>())?;
            writeln!(f, "  Incumbent from heuristic: {}", self.incumbent_from_heuristic)?;
        } else {
            writeln!(f, "  No feasible solution found")?;
        }
        Ok(())
    }
}

// ─── Utility functions ───────────────────────────────────────────────────────

/// Check if a value is integer within tolerance.
pub fn is_integer(val: f64) -> bool {
    (val - val.round()).abs() < INT_TOL
}

/// Fractional part: frac(a) = a - floor(a), always in [0, 1).
/// Handles negative numbers correctly: frac(-0.3) = 0.7.
pub fn frac(a: f64) -> f64 {
    a - a.floor()
}

/// Check if a solution is integer-feasible for the given variable types.
pub fn is_integer_feasible(values: &[f64], var_types: &[VarType]) -> bool {
    for (i, &vt) in var_types.iter().enumerate() {
        match vt {
            VarType::Integer => {
                if !is_integer(values[i]) {
                    return false;
                }
            }
            VarType::Binary => {
                if !is_integer(values[i])
                    || (values[i].round() != 0.0 && values[i].round() != 1.0)
                {
                    return false;
                }
            }
            VarType::Continuous => {}
        }
    }
    true
}

/// Find the index of the most fractional integer/binary variable.
/// Returns None if all integer variables are integral.
pub fn most_fractional_variable(values: &[f64], var_types: &[VarType]) -> Option<usize> {
    let mut best_idx = None;
    let mut best_frac = 0.0;
    for (i, &vt) in var_types.iter().enumerate() {
        if vt == VarType::Continuous {
            continue;
        }
        let f = frac(values[i]);
        // Distance to nearest integer: min(f, 1-f)
        let dist = f.min(1.0 - f);
        if dist > INT_TOL && dist > best_frac {
            best_frac = dist;
            best_idx = Some(i);
        }
    }
    best_idx
}

/// Check if a solution satisfies all constraints of a MIP problem.
pub fn is_feasible(values: &[f64], problem: &MIPProblem) -> bool {
    for c in &problem.constraints {
        let lhs: f64 = c.coeffs.iter().zip(values.iter()).map(|(a, x)| a * x).sum();
        let ok = match c.sense {
            ConstraintSense::Le => lhs <= c.rhs + EQ_TOL,
            ConstraintSense::Ge => lhs >= c.rhs - EQ_TOL,
            ConstraintSense::Eq => (lhs - c.rhs).abs() <= EQ_TOL,
        };
        if !ok {
            return false;
        }
    }
    // Check non-negativity
    for &v in values {
        if v < -EQ_TOL {
            return false;
        }
    }
    true
}

/// Compute objective value for a solution.
pub fn objective_value(obj: &[f64], values: &[f64]) -> f64 {
    obj.iter().zip(values.iter()).map(|(c, x)| c * x).sum()
}

/// Convert a MIPProblem to standard form for the manual simplex tableau.
///
/// Returns (c, A, b) where:
///   min c^T x  s.t. Ax = b, x >= 0
/// with slack/surplus variables added.
///
/// Also returns the mapping: original var index -> standard form var index.
pub fn to_standard_form(problem: &MIPProblem) -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let n_orig = problem.n_vars;
    let m = problem.constraints.len();

    // Count how many slacks/surpluses we need
    let n_slack: usize = problem
        .constraints
        .iter()
        .filter(|c| c.sense != ConstraintSense::Eq)
        .count();
    let n_total = n_orig + n_slack;

    // Build c (objective): original vars + zeros for slacks
    let mut c = problem.obj.clone();
    c.resize(n_total, 0.0);

    // Build A and b
    let mut a = Vec::with_capacity(m);
    let mut b = Vec::with_capacity(m);
    let mut slack_idx = n_orig;

    for constraint in &problem.constraints {
        let mut row = constraint.coeffs.clone();
        row.resize(n_total, 0.0);

        match constraint.sense {
            ConstraintSense::Le => {
                row[slack_idx] = 1.0; // slack
                slack_idx += 1;
                b.push(constraint.rhs);
            }
            ConstraintSense::Ge => {
                // Multiply by -1 to convert to <=, then add slack
                for coeff in row.iter_mut().take(n_orig) {
                    *coeff = -*coeff;
                }
                row[slack_idx] = 1.0;
                slack_idx += 1;
                b.push(-constraint.rhs);
            }
            ConstraintSense::Eq => {
                b.push(constraint.rhs);
            }
        }

        a.push(row);
    }

    // Ensure all b >= 0 (negate row if needed)
    for i in 0..m {
        if b[i] < 0.0 {
            for j in 0..n_total {
                a[i][j] = -a[i][j];
            }
            b[i] = -b[i];
        }
    }

    (c, a, b)
}

/// Solve the LP relaxation of a MIP using the manual simplex.
/// Returns the optimal tableau (with solution accessible) or an error.
pub fn solve_lp_manual(problem: &MIPProblem) -> Result<SimplexTableau, String> {
    let (c, a, b) = to_standard_form(problem);
    let mut tableau = SimplexTableau::from_standard_form(&c, &a, &b);
    match tableau.solve(1000) {
        Ok(SimplexStatus::Optimal) => Ok(tableau),
        Ok(SimplexStatus::Unbounded) => Err("LP is unbounded".to_string()),
        Ok(SimplexStatus::Infeasible) => Err("LP is infeasible".to_string()),
        Err(e) => Err(e),
    }
}

/// Solve the LP relaxation of a MIP using good_lp/HiGHS.
/// Returns the solution values (for original variables only) and objective.
pub fn solve_lp_highs(problem: &MIPProblem) -> Result<LPSolution, String> {
    use good_lp::*;

    let mut vars = ProblemVariables::new();
    let x: Vec<Variable> = (0..problem.n_vars)
        .map(|i| {
            let v = vars.add(variable().min(0.0));
            // For binary variables in LP relaxation, add upper bound of 1
            if problem.var_types[i] == VarType::Binary {
                // We'll add the bound as a constraint
                let _ = v; // good_lp doesn't support .max() on variable builder easily
            }
            v
        })
        .collect();

    // Build objective
    let obj_expr: Expression = problem
        .obj
        .iter()
        .zip(x.iter())
        .map(|(&c, &v)| c * v)
        .sum();

    let mut model = vars.minimise(obj_expr).using(highs);

    // Add constraints
    for c in &problem.constraints {
        let lhs: Expression = c.coeffs.iter().zip(x.iter()).map(|(&a, &v)| a * v).sum();
        match c.sense {
            ConstraintSense::Le => {
                model = model.with(lhs.leq(c.rhs));
            }
            ConstraintSense::Ge => {
                model = model.with(lhs.geq(c.rhs));
            }
            ConstraintSense::Eq => {
                model = model.with(lhs.clone().leq(c.rhs));
                model = model.with(lhs.geq(c.rhs));
            }
        }
    }

    // Add binary upper bounds
    for (i, &vt) in problem.var_types.iter().enumerate() {
        if vt == VarType::Binary {
            model = model.with(x[i].leq(1.0));
        }
    }

    match model.solve() {
        Ok(solution) => {
            let values: Vec<f64> = x.iter().map(|&v| solution.value(v)).collect();
            let objective: f64 = problem
                .obj
                .iter()
                .zip(values.iter())
                .map(|(c, v)| c * v)
                .sum();
            Ok(LPSolution {
                values,
                objective,
                status: SimplexStatus::Optimal,
            })
        }
        Err(e) => Err(format!("HiGHS solve failed: {}", e)),
    }
}

/// Pretty-print a solution.
pub fn print_solution(names: &[String], values: &[f64], var_types: &[VarType]) {
    for (i, name) in names.iter().enumerate() {
        let frac_marker = if var_types[i] != VarType::Continuous && !is_integer(values[i]) {
            " (FRACTIONAL)"
        } else {
            ""
        };
        println!("  {} = {:.6}{}", name, values[i], frac_marker);
    }
}

// ─── Test helpers ────────────────────────────────────────────────────────────

/// A small example MIP for testing:
///   min -x0 - x1
///   s.t. 3*x0 + 2*x1 <= 6
///        x0, x1 >= 0, integer
///
/// LP relaxation optimal: x0=2, x1=0, obj=-2
/// But with x0+x1 fractional in other formulations...
/// Better example with fractional LP:
///
///   min -5*x0 - 4*x1
///   s.t. 6*x0 + 4*x1 <= 24
///          x0 + 2*x1 <=  6
///        x0, x1 >= 0, integer
///
/// LP optimal: x0 = 3, x1 = 1.5 (fractional!), obj = -21
/// MIP optimal: x0 = 4, x1 = 0, obj = -20  OR  x0 = 2, x1 = 2, obj = -18
///   Actually: x0=0, x1=3 gives obj=-12. Let's verify...
///   x0=4, x1=0: 6*4=24<=24, 4<=6. obj=-20. Valid!
///   x0=3, x1=1: 18+4=22<=24, 3+2=5<=6. obj=-19.
///   x0=2, x1=2: 12+8=20<=24, 2+4=6<=6. obj=-18.
///   MIP optimal: x0=4, x1=0, obj=-20
pub fn example_mip_small() -> MIPProblem {
    MIPProblem::new(
        vec![-5.0, -4.0],
        vec![
            LinearConstraint {
                coeffs: vec![6.0, 4.0],
                sense: ConstraintSense::Le,
                rhs: 24.0,
            },
            LinearConstraint {
                coeffs: vec![1.0, 2.0],
                sense: ConstraintSense::Le,
                rhs: 6.0,
            },
        ],
        vec![VarType::Integer, VarType::Integer],
        vec!["x0".to_string(), "x1".to_string()],
    )
}

/// A binary knapsack problem for cover inequality testing:
///   max 16*x0 + 22*x1 + 12*x2 + 8*x3 + 11*x4 + 19*x5
///   s.t. 5*x0 + 7*x1 + 4*x2 + 3*x3 + 4*x4 + 6*x5 <= 15
///        x_j in {0, 1}
///
/// Equivalent minimization:
///   min -16*x0 - 22*x1 - 12*x2 - 8*x3 - 11*x4 - 19*x5
pub fn example_knapsack() -> MIPProblem {
    MIPProblem::new(
        vec![-16.0, -22.0, -12.0, -8.0, -11.0, -19.0],
        vec![LinearConstraint {
            coeffs: vec![5.0, 7.0, 4.0, 3.0, 4.0, 6.0],
            sense: ConstraintSense::Le,
            rhs: 15.0,
        }],
        vec![
            VarType::Binary,
            VarType::Binary,
            VarType::Binary,
            VarType::Binary,
            VarType::Binary,
            VarType::Binary,
        ],
        vec![
            "x0".to_string(),
            "x1".to_string(),
            "x2".to_string(),
            "x3".to_string(),
            "x4".to_string(),
            "x5".to_string(),
        ],
    )
}

/// A slightly larger MIP for branch-and-cut testing:
///   min -3*x0 - 2*x1 - 4*x2
///   s.t. 2*x0 +   x1 + 3*x2 <= 10
///          x0 + 2*x1 +   x2 <=  8
///        x0, x1, x2 >= 0, integer
///
/// LP relaxation: fractional
/// Forces multiple B&B nodes, cuts help tighten
pub fn example_mip_medium() -> MIPProblem {
    MIPProblem::new(
        vec![-3.0, -2.0, -4.0],
        vec![
            LinearConstraint {
                coeffs: vec![2.0, 1.0, 3.0],
                sense: ConstraintSense::Le,
                rhs: 10.0,
            },
            LinearConstraint {
                coeffs: vec![1.0, 2.0, 1.0],
                sense: ConstraintSense::Le,
                rhs: 8.0,
            },
        ],
        vec![VarType::Integer, VarType::Integer, VarType::Integer],
        vec!["x0".to_string(), "x1".to_string(), "x2".to_string()],
    )
}
