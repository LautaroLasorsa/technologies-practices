//! Practice 062: SIMD in Rust — Library crate
//!
//! This module re-exports the exercise modules so that benchmarks and tests
//! can access the functions. The main binary uses these modules directly.

#![feature(portable_simd)]

pub mod ex1_autovec;
pub mod ex2_portable_basics;
pub mod ex3_dot_product;
pub mod ex4_masks;
pub mod ex5_stdarch;
pub mod ex6_swizzle;
