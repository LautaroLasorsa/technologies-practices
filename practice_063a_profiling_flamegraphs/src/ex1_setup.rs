//! Exercise 1: Tool Setup & Baseline
//!
//! Before profiling, you need the right tools installed and a reliable way to
//! measure execution time. This exercise verifies your environment and creates
//! a reusable timing harness.

use std::time::{Duration, Instant};

/// Check that profiling tools are available and print installation commands.
///
/// A profiling session requires:
/// - `cargo-flamegraph`: Generates SVG flamegraphs from sampling profiler data.
///   Uses ETW (Event Tracing for Windows) on Windows — no extra setup needed,
///   but you MUST run it as Administrator because ETW requires elevated privileges.
///
/// - `samply`: A modern sampling profiler that opens the Firefox Profiler UI
///   in your browser. Also uses ETW on Windows and requires Administrator.
///   The advantage over cargo-flamegraph is the interactive timeline view —
///   you can see CPU usage over time, not just an aggregate.
///
/// - `cargo-show-asm`: Shows the actual assembly, LLVM-IR, or MIR that the
///   compiler generates for any public function in a library crate.
///   No elevated privileges needed — it just compiles with special flags.
///
/// For each tool, this function should:
/// 1. Print the `cargo install` command
/// 2. Attempt to detect if the tool is already installed (optional: check PATH)
/// 3. Print a status message (installed / not found)
pub fn install_tools() {
    // TODO(human): Implement tool availability checking.
    //
    // This exercise teaches the profiling toolchain setup — a real-world skill
    // that many developers skip, leading to frustration when they need to profile.
    //
    // Steps:
    //
    // 1. Define a list of tools with their install commands:
    //    - ("cargo-flamegraph", "cargo install flamegraph")
    //    - ("samply", "cargo install samply")
    //    - ("cargo-show-asm", "cargo install cargo-show-asm")
    //
    // 2. For each tool, try to detect if it's installed. You can use:
    //    `std::process::Command::new("cargo").args(["flamegraph", "--help"]).output()`
    //    and check if the command succeeded. Or simply check if the binary exists
    //    on PATH with `which` (Unix) or `where` (Windows).
    //
    //    A simple approach:
    //    ```
    //    let result = std::process::Command::new(binary_name)
    //        .arg("--version")
    //        .output();
    //    match result {
    //        Ok(output) if output.status.success() => println!("  [OK] {} installed", name),
    //        _ => println!("  [MISSING] {} — install with: {}", name, install_cmd),
    //    }
    //    ```
    //
    //    Note: cargo subcommands (cargo-flamegraph, cargo-show-asm) are invoked as
    //    `cargo flamegraph` but installed as binaries named `flamegraph` or
    //    `cargo-flamegraph`. Check for the binary name, e.g., `flamegraph --version`.
    //
    // 3. Also verify that debug info is enabled in release builds by printing
    //    a reminder about the Cargo.toml [profile.release] section:
    //    ```
    //    println!("  [CONFIG] Ensure Cargo.toml has: [profile.release] debug = true");
    //    ```
    //
    // 4. On Windows, print a reminder about Administrator privileges:
    //    ```
    //    if cfg!(target_os = "windows") {
    //        println!("  [NOTE] cargo-flamegraph and samply require Administrator privileges on Windows (ETW)");
    //    }
    //    ```
    //
    // Why this matters: Profiling tools fail silently or produce garbage output
    // when debug info is missing or privileges are insufficient. Checking upfront
    // saves debugging time later.

    todo!("Exercise 1a: Implement tool availability checking")
}

/// A generic timing harness that measures wall-clock execution time.
///
/// This function runs the given closure, measures its duration, prints the
/// result, and returns both the duration and the closure's return value.
///
/// Wall-clock time (monotonic `Instant`) is the right metric for profiling
/// because it captures everything: CPU time, memory allocation overhead,
/// cache misses, and OS scheduling delays. CPU-time-only metrics (like
/// `clock_gettime(CLOCK_PROCESS_CPUTIME_ID)`) miss I/O stalls and cache effects.
///
/// For reliable measurements:
/// - Always use `--release` builds (debug builds are 10-100x slower)
/// - Run the workload long enough (>1 second) to amortize startup/JIT/cache warmup
/// - Minimize background activity on the machine
/// - Run multiple iterations and report the minimum (most representative)
///
/// # Arguments
/// * `name` - A label printed alongside the timing result
/// * `f` - The closure to time
///
/// # Returns
/// A tuple of (elapsed Duration, return value of the closure)
pub fn bench_fn<F, R>(name: &str, f: F) -> (Duration, R)
where
    F: FnOnce() -> R,
{
    // TODO(human): Implement the timing harness.
    //
    // This is your primary measurement tool for the entire practice. Every
    // optimization you make will be validated by this function.
    //
    // Steps:
    //
    // 1. Record the start time: `let start = Instant::now();`
    //    `Instant` is a monotonic clock — it never goes backwards, even if the
    //    system clock is adjusted. This is essential for benchmarking.
    //
    // 2. Execute the closure: `let result = f();`
    //    Note: `f` is `FnOnce`, meaning it can capture and consume values.
    //    This is the most flexible closure trait.
    //
    // 3. Record elapsed time: `let elapsed = start.elapsed();`
    //    This computes `Instant::now() - start` and returns a `Duration`.
    //
    // 4. Print the result:
    //    ```
    //    println!("  {} : {:.3}ms", name, elapsed.as_secs_f64() * 1000.0);
    //    ```
    //    Printing in milliseconds is usually the right granularity for the
    //    workloads in this practice. For microsecond-level work, use `as_micros()`.
    //
    // 5. Return `(elapsed, result)` so callers can use both the timing and the
    //    computed value (prevents the compiler from optimizing away the computation).
    //
    // Why return the result? If the closure's return value is unused, the compiler
    // may optimize away the entire computation (dead code elimination). By returning
    // it and having the caller print or assert on it, we ensure the work actually
    // happens. This is a common benchmarking pitfall.

    todo!("Exercise 1b: Implement bench_fn timing harness")
}

/// Demo function that exercises the timing harness with a known workload.
///
/// This is pre-built to verify your bench_fn implementation works correctly.
pub fn demo_bench_fn() {
    // A simple known workload: sum of 1..=N
    let n = 10_000_000u64;

    let (elapsed, sum) = bench_fn("sum 1..=10M", || {
        (1..=n).sum::<u64>()
    });

    let expected = n * (n + 1) / 2;
    assert_eq!(sum, expected, "Sum mismatch — bench_fn may have dropped the result");
    println!("  Verified: sum = {} (expected {})", sum, expected);
    println!("  Elapsed: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_fn_returns_correct_value() {
        let (_dur, val) = bench_fn("test", || 42);
        assert_eq!(val, 42);
    }

    #[test]
    fn test_bench_fn_measures_time() {
        let (dur, _) = bench_fn("sleep", || {
            std::thread::sleep(Duration::from_millis(50));
        });
        assert!(dur >= Duration::from_millis(40), "Duration too short: {:?}", dur);
    }
}
