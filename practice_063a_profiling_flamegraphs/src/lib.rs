//! Practice 063a: Profiling & Flamegraphs
//!
//! Library crate exposing exercise modules for assembly inspection via cargo-show-asm.
//! The `cargo asm` command can only inspect public functions in library crates,
//! so we expose the exercise modules here.

pub mod ex1_setup;
pub mod ex2_workload;
pub mod ex3_flamegraph;
pub mod ex4_dhat;
pub mod ex5_assembly;
pub mod ex6_optimize;
