// Phase 3: Particle Swarm Optimization
//
// This binary implements PSO — a swarm intelligence algorithm where particles
// fly through the search space guided by their own best-known position and the
// swarm's global best. You implement the velocity update formula and the main
// PSO loop.

use rand::prelude::*;
use std::f64::consts::{E, PI};

// ============================================================================
// Provided: Benchmark functions (same as Phase 2)
// ============================================================================

/// Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
/// Global minimum: f(0, ..., 0) = 0. Bounds: [-5.12, 5.12].
fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n
        + x.iter()
            .map(|&xi| xi * xi - 10.0 * (2.0 * PI * xi).cos())
            .sum::<f64>()
}

/// Ackley function.
/// Global minimum: f(0, ..., 0) = 0. Bounds: [-5, 5].
fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>();
    let sum_cos: f64 = x.iter().map(|&xi| (2.0 * PI * xi).cos()).sum::<f64>();
    -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + 20.0 + E
}

// ============================================================================
// Provided: Particle struct and initialization
// ============================================================================

/// A particle in the swarm.
#[derive(Debug, Clone)]
struct Particle {
    /// Current position in the search space.
    position: Vec<f64>,
    /// Current velocity.
    velocity: Vec<f64>,
    /// Best position this particle has visited.
    personal_best: Vec<f64>,
    /// Fitness at the personal best position.
    personal_best_cost: f64,
}

/// Result of a PSO run.
#[derive(Debug, Clone)]
struct PSOResult {
    best_position: Vec<f64>,
    best_fitness: f64,
    iterations_run: usize,
    fitness_history: Vec<f64>,
}

/// Initialize a swarm of particles.
///
/// Positions are uniform random within bounds. Velocities are initialized to
/// small random values (fraction of the search range) to avoid particles
/// immediately flying out of bounds.
fn initialize_swarm(
    n_particles: usize,
    bounds: &[(f64, f64)],
    f: fn(&[f64]) -> f64,
    rng: &mut impl Rng,
) -> (Vec<Particle>, Vec<f64>, f64) {
    let dim = bounds.len();
    let mut particles = Vec::with_capacity(n_particles);
    let mut global_best = vec![0.0; dim];
    let mut global_best_cost = f64::INFINITY;

    for _ in 0..n_particles {
        let position: Vec<f64> = (0..dim)
            .map(|d| rng.gen_range(bounds[d].0..=bounds[d].1))
            .collect();

        // Initialize velocity to small fraction of range
        let velocity: Vec<f64> = (0..dim)
            .map(|d| {
                let range = bounds[d].1 - bounds[d].0;
                rng.gen_range(-0.1 * range..=0.1 * range)
            })
            .collect();

        let cost = f(&position);
        let personal_best = position.clone();

        if cost < global_best_cost {
            global_best = position.clone();
            global_best_cost = cost;
        }

        particles.push(Particle {
            position,
            velocity,
            personal_best,
            personal_best_cost: cost,
        });
    }

    (particles, global_best, global_best_cost)
}

/// Clip a position to respect bounds.
fn clip_to_bounds(x: &mut [f64], bounds: &[(f64, f64)]) {
    for (xi, &(lo, hi)) in x.iter_mut().zip(bounds.iter()) {
        *xi = xi.clamp(lo, hi);
    }
}

/// Print PSO iteration summary.
fn print_pso_iteration(iter: usize, best_fitness: f64, avg_fitness: f64) {
    println!(
        "  Iter {:4} | Best: {:12.6} | Avg: {:12.6}",
        iter, best_fitness, avg_fitness
    );
}

// ============================================================================
// TODO(human): Velocity Update
// ============================================================================

/// Update a particle's velocity using the PSO velocity formula.
///
/// # TODO(human): Implement PSO velocity update
///
/// The velocity update has three components:
///   v_new[d] = w * v_old[d]                                  // inertia
///            + c1 * r1[d] * (pbest[d] - position[d])         // cognitive
///            + c2 * r2[d] * (gbest[d] - position[d])         // social
///
/// Where:
///   - w (inertia): preserves previous direction. High w = more exploration.
///   - c1 (cognitive): pulls toward particle's own best. Encourages local search.
///   - c2 (social): pulls toward swarm's best. Encourages convergence.
///   - r1, r2: independent uniform random in [0,1], generated PER DIMENSION.
///     This per-dimension randomness is important — it makes the velocity
///     update stochastic in each direction independently.
///
/// Algorithm:
///   1. For each dimension d in 0..dim:
///        r1 = rng.gen::<f64>()
///        r2 = rng.gen::<f64>()
///        new_v[d] = w * particle.velocity[d]
///                 + c1 * r1 * (particle.personal_best[d] - particle.position[d])
///                 + c2 * r2 * (global_best[d] - particle.position[d])
///   2. Return new_v
///
/// Note: velocity is NOT clamped here. Some PSO variants clamp velocity to
/// [-v_max, v_max], but the standard version relies on inertia weight decay.
fn update_velocity(
    particle: &Particle,
    global_best: &[f64],
    w: f64,
    c1: f64,
    c2: f64,
    rng: &mut impl Rng,
) -> Vec<f64> {
    // TODO(human): Implement PSO velocity update as described above.
    //
    // Pseudocode:
    //   dim = particle.position.len()
    //   new_velocity = vec![0.0; dim]
    //   for d in 0..dim:
    //       r1 = rng.gen::<f64>()
    //       r2 = rng.gen::<f64>()
    //       new_velocity[d] = w * particle.velocity[d]
    //                       + c1 * r1 * (particle.personal_best[d] - particle.position[d])
    //                       + c2 * r2 * (global_best[d] - particle.position[d])
    //   return new_velocity
    todo!()
}

// ============================================================================
// TODO(human): PSO Main Loop
// ============================================================================

/// Run Particle Swarm Optimization.
///
/// # TODO(human): Implement the PSO optimization loop
///
/// Each iteration:
///   1. For each particle:
///      a. Update velocity via update_velocity()
///      b. Update position: x = x + v
///      c. Clip position to bounds
///      d. Evaluate fitness at new position
///      e. If new fitness < personal_best_cost:
///           update personal_best and personal_best_cost
///         If new fitness < global_best_cost:
///           update global_best and global_best_cost
///   2. Record global_best_cost in fitness_history
///
/// Important: the global best must be updated WITHIN the inner loop (not
/// after all particles move) so that later particles in the same iteration
/// benefit from discoveries by earlier particles. This is the "gbest" variant
/// of PSO. The alternative "lbest" uses neighborhood topologies but is more
/// complex.
///
/// Parameters:
///   - f: objective function (minimization)
///   - bounds: search bounds per dimension
///   - n_particles: swarm size (typically 20-50)
///   - max_iter: maximum iterations
///   - w: inertia weight (0.4-0.9, can be fixed or linearly decaying)
///   - c1: cognitive coefficient (typically 2.0)
///   - c2: social coefficient (typically 2.0)
fn pso(
    f: fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    n_particles: usize,
    max_iter: usize,
    w: f64,
    c1: f64,
    c2: f64,
    rng: &mut impl Rng,
) -> PSOResult {
    // TODO(human): Implement the PSO loop as described above.
    //
    // Pseudocode:
    //   (particles, global_best, global_best_cost) = initialize_swarm(...)
    //   fitness_history = vec![]
    //
    //   for iter in 0..max_iter:
    //       for i in 0..n_particles:
    //           new_vel = update_velocity(&particles[i], &global_best, w, c1, c2, rng)
    //           particles[i].velocity = new_vel
    //           // Update position: x += v
    //           for d in 0..dim:
    //               particles[i].position[d] += particles[i].velocity[d]
    //           clip_to_bounds(&mut particles[i].position, bounds)
    //           cost = f(&particles[i].position)
    //           if cost < particles[i].personal_best_cost:
    //               particles[i].personal_best = particles[i].position.clone()
    //               particles[i].personal_best_cost = cost
    //               if cost < global_best_cost:
    //                   global_best = particles[i].position.clone()
    //                   global_best_cost = cost
    //       fitness_history.push(global_best_cost)
    //       // print every 50 iters
    //
    //   return PSOResult { best_position: global_best, best_fitness: global_best_cost, ... }
    todo!()
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== Phase 3: Particle Swarm Optimization ===\n");

    let dim = 10;
    let n_particles = 40;
    let max_iter = 500;
    let w = 0.7;   // inertia weight
    let c1 = 1.5;  // cognitive
    let c2 = 1.5;  // social
    let mut rng = StdRng::seed_from_u64(42);

    // --- Rastrigin 10D ---
    println!("--- PSO on Rastrigin {}D ---", dim);
    println!(
        "Parameters: particles={}, iters={}, w={}, c1={}, c2={}",
        n_particles, max_iter, w, c1, c2
    );
    println!("Global optimum: f(0, ..., 0) = 0.0\n");

    let bounds_rast: Vec<(f64, f64)> = vec![(-5.12, 5.12); dim];
    let result_rast = pso(
        rastrigin,
        &bounds_rast,
        n_particles,
        max_iter,
        w,
        c1,
        c2,
        &mut rng,
    );

    println!("\nRastrigin result:");
    println!("  Best fitness: {:.6}", result_rast.best_fitness);
    println!(
        "  Best position: [{}]",
        result_rast
            .best_position
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  Iterations: {}", result_rast.iterations_run);

    // --- Ackley 10D ---
    println!("\n--- PSO on Ackley {}D ---", dim);
    println!("Global optimum: f(0, ..., 0) = 0.0\n");

    let bounds_ack: Vec<(f64, f64)> = vec![(-5.0, 5.0); dim];
    let result_ack = pso(
        ackley,
        &bounds_ack,
        n_particles,
        max_iter,
        w,
        c1,
        c2,
        &mut rng,
    );

    println!("\nAckley result:");
    println!("  Best fitness: {:.6}", result_ack.best_fitness);
    println!(
        "  Best position: [{}]",
        result_ack
            .best_position
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  Iterations: {}", result_ack.iterations_run);

    // Summary
    println!("\n=== PSO Summary ===");
    println!(
        "  Rastrigin {}D: {:.6} (optimum: 0.0)",
        dim, result_rast.best_fitness
    );
    println!(
        "  Ackley    {}D: {:.6} (optimum: 0.0)",
        dim, result_ack.best_fitness
    );
}
