//! io_uring Practice — Modern Linux Async I/O
//!
//! This program runs 6 exercises that progressively teach Linux io_uring
//! from raw ring buffer manipulation to high-level async networking:
//!
//! 1. Basic ring setup and file read (IoUring, SQE/CQE lifecycle)
//! 2. File write and batched submissions (multi-SQE in one syscall)
//! 3. Fixed buffers for zero-copy I/O (register_buffers, ReadFixed/WriteFixed)
//! 4. tokio-uring async file I/O (buffer ownership, proactor model)
//! 5. tokio-uring TCP echo server (accept, read, write loop)
//! 6. Benchmark: io_uring vs standard I/O (wall-clock comparison)
//!
//! Usage:
//!   cargo run          # Run all exercises
//!   cargo run -- 3     # Run only exercise 3
//!   cargo run -- 5     # Run the TCP echo server

mod ex1_basic_read;
mod ex2_write_and_batch;
mod ex3_fixed_buffers;
mod ex4_tokio_uring_file;
mod ex5_tcp_echo;
mod ex6_benchmark;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let exercise_filter: Option<u32> = args.get(1).and_then(|s| s.parse().ok());

    if let Some(n) = exercise_filter {
        println!("Running exercise {} only\n", n);
    } else {
        println!("Running all exercises\n");
    }

    let exercises: Vec<(u32, &str, fn())> = vec![
        (1, "Basic Ring Setup and File Read", ex1_basic_read::run),
        (2, "File Write and Batched Submissions", ex2_write_and_batch::run),
        (3, "Fixed Buffers for Zero-Copy I/O", ex3_fixed_buffers::run),
        (4, "tokio-uring Async File I/O", ex4_tokio_uring_file::run),
        (5, "TCP Echo Server (tokio-uring)", ex5_tcp_echo::run),
        (6, "Benchmark: io_uring vs Standard I/O", ex6_benchmark::run),
    ];

    for (num, name, run_fn) in &exercises {
        if exercise_filter.is_some_and(|f| f != *num) {
            continue;
        }

        println!("{}", "=".repeat(64));
        println!("  Exercise {}: {}", num, name);
        println!("{}", "=".repeat(64));
        println!();

        run_fn();

        println!();
    }

    println!("Done.");
}
