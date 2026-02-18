//! Exercise 6: Benchmark — io_uring vs Standard I/O
//!
//! This capstone exercise measures the real-world performance difference between
//! io_uring and standard library I/O for file operations. The goal is to ground
//! the theoretical advantages in actual measurements on your system.
//!
//! ## What We're Measuring
//!
//! **File write throughput:** Write a large file (several MB) using:
//!   1. `std::fs::File` + `std::io::Write` (blocking syscalls)
//!   2. Low-level `io-uring` crate with batched Write SQEs
//!
//! **File read throughput:** Read the same file using:
//!   1. `std::fs::File` + `std::io::Read` (blocking syscalls)
//!   2. Low-level `io-uring` crate with batched Read SQEs
//!
//! ## Expected Results
//!
//! For sequential file I/O on a single thread, io_uring may NOT show dramatic
//! improvement over std::fs. The Linux page cache is very efficient for sequential
//! access, and std::fs::write already batches data through the kernel's write path.
//!
//! io_uring's advantage appears when:
//!   - You submit many operations concurrently (batching amortizes syscall cost)
//!   - You use fixed buffers (eliminating per-op page table walks)
//!   - The workload is I/O-bound with high concurrency (many fds, many ops)
//!   - You bypass the page cache with O_DIRECT (raw device I/O)
//!
//! This exercise is educational: you learn to measure and reason about when
//! io_uring actually helps, rather than assuming it's always faster.

use std::io::{self, Read, Write};
use std::time::Instant;

const BENCH_DIR: &str = "/tmp/iouring_test/bench";
/// Total data size for the benchmark: 8 MB
const TOTAL_SIZE: usize = 8 * 1024 * 1024;
/// Block size for each write/read operation: 4 KB (one page)
const BLOCK_SIZE: usize = 4096;
/// Number of blocks = TOTAL_SIZE / BLOCK_SIZE = 2048
const NUM_BLOCKS: usize = TOTAL_SIZE / BLOCK_SIZE;
/// How many blocks to batch in a single io_uring submission
const BATCH_SIZE: usize = 32;

pub fn run() {
    std::fs::create_dir_all(BENCH_DIR).expect("Failed to create bench dir");

    println!("Benchmark parameters:");
    println!("  Total data:  {} MB", TOTAL_SIZE / (1024 * 1024));
    println!("  Block size:  {} bytes", BLOCK_SIZE);
    println!("  Num blocks:  {}", NUM_BLOCKS);
    println!("  Batch size:  {} (for io_uring)", BATCH_SIZE);
    println!();

    println!("--- Part A: Write Benchmark ---\n");
    let std_write_ms = bench_std_write();
    let uring_write_ms = bench_iouring_write();
    print_comparison("Write", std_write_ms, uring_write_ms);

    println!("\n--- Part B: Read Benchmark ---\n");
    let std_read_ms = bench_std_read();
    let uring_read_ms = bench_iouring_read();
    print_comparison("Read", std_read_ms, uring_read_ms);
}

fn print_comparison(op: &str, std_ms: f64, uring_ms: f64) {
    let throughput_std = (TOTAL_SIZE as f64 / (1024.0 * 1024.0)) / (std_ms / 1000.0);
    let throughput_uring = (TOTAL_SIZE as f64 / (1024.0 * 1024.0)) / (uring_ms / 1000.0);
    let speedup = std_ms / uring_ms;

    println!("\n  {} Results:", op);
    println!("    std::fs:   {:.2} ms  ({:.1} MB/s)", std_ms, throughput_std);
    println!("    io_uring:  {:.2} ms  ({:.1} MB/s)", uring_ms, throughput_uring);
    println!(
        "    Speedup:   {:.2}x {}",
        speedup,
        if speedup > 1.0 {
            "(io_uring faster)"
        } else {
            "(std::fs faster)"
        }
    );
}

/// Benchmark: Write TOTAL_SIZE bytes using std::fs (blocking syscalls).
///
/// This is the baseline. Each `file.write_all(block)` call performs a `write()`
/// syscall. The kernel buffers data in the page cache and flushes to disk
/// asynchronously (unless O_SYNC is set).
fn bench_std_write() -> f64 {
    // TODO(human): Implement the std::fs write benchmark.
    //
    // Step 1: Create a block of test data.
    //   - `let block = vec![0xABu8; BLOCK_SIZE];`
    //   - Using 0xAB makes the data distinguishable in hex dumps.
    //
    // Step 2: Open the output file.
    //   - `let mut file = std::fs::File::create(format!("{}/std_write.bin", BENCH_DIR))?;`
    //
    // Step 3: Time the write loop.
    //   - `let start = Instant::now();`
    //   - Write NUM_BLOCKS blocks:
    //     ```
    //     for _ in 0..NUM_BLOCKS {
    //         file.write_all(&block)?;
    //     }
    //     ```
    //   - Flush: `file.flush()?;`  (ensure data is in page cache, not just user-space buffer)
    //   - `let elapsed = start.elapsed();`
    //
    // Step 4: Print and return the elapsed time in milliseconds.
    //   - `elapsed.as_secs_f64() * 1000.0`

    todo!("Implement bench_std_write")
}

/// Benchmark: Write TOTAL_SIZE bytes using io_uring with batched submissions.
///
/// Instead of NUM_BLOCKS individual write() syscalls, we batch BATCH_SIZE
/// Write SQEs per submit() call. This means:
///   - NUM_BLOCKS / BATCH_SIZE = 64 submit() calls instead of 2048 write() calls
///   - Each submit() call is one io_uring_enter() syscall
///   - The kernel processes BATCH_SIZE writes per syscall
///
/// This is the pattern used by high-performance databases and storage engines.
fn bench_iouring_write() -> f64 {
    // TODO(human): Implement the io_uring batched write benchmark.
    //
    // Step 1: Create the IoUring ring.
    //   - `let mut ring = io_uring::IoUring::new(BATCH_SIZE as u32 * 2)?;`
    //   - Ring size is 2x batch size to avoid backpressure.
    //
    // Step 2: Create the block data and open the file.
    //   - `let block = vec![0xABu8; BLOCK_SIZE];`
    //   - Open file with `std::fs::File::create(...)` and get raw fd.
    //
    // Step 3: Time the batched write loop.
    //   - `let start = Instant::now();`
    //   - Process blocks in batches:
    //     ```
    //     let mut offset: u64 = 0;
    //     let mut remaining = NUM_BLOCKS;
    //
    //     while remaining > 0 {
    //         let batch = remaining.min(BATCH_SIZE);
    //
    //         // Push `batch` Write SQEs
    //         for i in 0..batch {
    //             let sqe = io_uring::opcode::Write::new(
    //                 io_uring::types::Fd(fd),
    //                 block.as_ptr(),
    //                 BLOCK_SIZE as u32,
    //             )
    //             .offset(offset)
    //             .build()
    //             .user_data(i as u64);
    //
    //             unsafe { ring.submission().push(&sqe).expect("SQ full"); }
    //             offset += BLOCK_SIZE as u64;
    //         }
    //
    //         // Submit all SQEs in one syscall and wait for all completions
    //         ring.submit_and_wait(batch as u32)?;
    //
    //         // Drain the completion queue
    //         let cqes: Vec<_> = ring.completion().map(Into::into).collect();
    //         for cqe in &cqes {
    //             let cqe: &io_uring::cqueue::Entry = cqe;
    //             assert!(cqe.result() >= 0, "Write failed: {}", cqe.result());
    //         }
    //
    //         remaining -= batch;
    //     }
    //     ```
    //
    //   NOTE on `.offset(offset)`:
    //   The Write opcode accepts an optional file offset. Without it, writes
    //   use the file's current position (and advance it). With `.offset(n)`,
    //   the write goes to byte position `n` (like pwrite()). Using explicit
    //   offsets allows the kernel to process batched writes concurrently
    //   because there's no shared "current position" state to serialize.
    //
    // Step 4: Print and return elapsed milliseconds.

    todo!("Implement bench_iouring_write")
}

/// Benchmark: Read TOTAL_SIZE bytes using std::fs (blocking syscalls).
fn bench_std_read() -> f64 {
    // TODO(human): Implement the std::fs read benchmark.
    //
    // Step 1: Open the file written by bench_std_write.
    //   - `let mut file = std::fs::File::open(format!("{}/std_write.bin", BENCH_DIR))?;`
    //
    // Step 2: Allocate a read buffer.
    //   - `let mut buf = vec![0u8; BLOCK_SIZE];`
    //
    // Step 3: Time the read loop.
    //   - `let start = Instant::now();`
    //   - Read NUM_BLOCKS blocks:
    //     ```
    //     for _ in 0..NUM_BLOCKS {
    //         file.read_exact(&mut buf)?;
    //     }
    //     ```
    //   - `let elapsed = start.elapsed();`
    //
    // Step 4: Print and return elapsed milliseconds.

    todo!("Implement bench_std_read")
}

/// Benchmark: Read TOTAL_SIZE bytes using io_uring with batched Read SQEs.
///
/// Same batching pattern as bench_iouring_write but with Read opcodes.
/// Each batch submits BATCH_SIZE Read SQEs and waits for all completions.
///
/// NOTE: For reading, we reuse the same buffer for all blocks in a batch.
/// This is fine because we're measuring throughput, not correctness —
/// each Read overwrites the buffer. In production, you'd use separate
/// buffers (or fixed buffers with distinct indices).
fn bench_iouring_read() -> f64 {
    // TODO(human): Implement the io_uring batched read benchmark.
    //
    // Step 1: Create the IoUring ring and open the file.
    //   - Same setup as bench_iouring_write but open for reading.
    //   - Open: `std::fs::File::open(format!("{}/std_write.bin", BENCH_DIR))?`
    //
    // Step 2: Allocate read buffers — one per batch slot.
    //   - For accurate benchmarking, use separate buffers per batch slot:
    //     ```
    //     let mut bufs: Vec<Vec<u8>> = (0..BATCH_SIZE)
    //         .map(|_| vec![0u8; BLOCK_SIZE])
    //         .collect();
    //     ```
    //   - Each SQE in the batch reads into its own buffer. This prevents
    //     data races (multiple concurrent reads into the same buffer).
    //
    // Step 3: Time the batched read loop.
    //   - `let start = Instant::now();`
    //   - Process in batches:
    //     ```
    //     let mut offset: u64 = 0;
    //     let mut remaining = NUM_BLOCKS;
    //
    //     while remaining > 0 {
    //         let batch = remaining.min(BATCH_SIZE);
    //
    //         for i in 0..batch {
    //             let sqe = io_uring::opcode::Read::new(
    //                 io_uring::types::Fd(fd),
    //                 bufs[i].as_mut_ptr(),
    //                 BLOCK_SIZE as u32,
    //             )
    //             .offset(offset)
    //             .build()
    //             .user_data(i as u64);
    //
    //             unsafe { ring.submission().push(&sqe).expect("SQ full"); }
    //             offset += BLOCK_SIZE as u64;
    //         }
    //
    //         ring.submit_and_wait(batch as u32)?;
    //
    //         let cqes: Vec<io_uring::cqueue::Entry> =
    //             ring.completion().map(Into::into).collect();
    //         for cqe in &cqes {
    //             assert!(cqe.result() >= 0, "Read failed: {}", cqe.result());
    //         }
    //
    //         remaining -= batch;
    //     }
    //     ```
    //
    // Step 4: Print and return elapsed milliseconds.

    todo!("Implement bench_iouring_read")
}
