//! Exercise 2: File Write and Batched Submissions
//!
//! This exercise builds on Exercise 1 by teaching:
//!   Part A — Writing data to a file using `opcode::Write`
//!   Part B — Submitting MULTIPLE SQEs in a single `submit()` call (batching)
//!
//! Batching is the core performance advantage of io_uring over traditional
//! syscalls. Instead of N write() calls (N context switches to kernel), you
//! fill the submission ring with N SQEs and make ONE io_uring_enter() call.
//! For a database writing 32 WAL entries, this eliminates 31 syscalls.
//!
//! Key concept: user_data for correlation
//! When you submit multiple SQEs, completions may arrive in ANY order (the
//! kernel processes them concurrently). You use the `user_data` field to tag
//! each SQE with a unique ID, then match CQEs to SQEs via the same ID.

use io_uring::{opcode, types, IoUring};
use std::fs;
use std::io;
use std::os::unix::io::AsRawFd;

const WRITE_FILE: &str = "/tmp/iouring_test/ex2_written.txt";
const BATCH_DIR: &str = "/tmp/iouring_test/ex2_batch";

pub fn run() {
    println!("--- Part A: Write a file with io_uring ---");
    match write_file_with_iouring() {
        Ok(()) => println!("File write + readback verified!\n"),
        Err(e) => eprintln!("ERROR: {}\n", e),
    }

    println!("--- Part B: Batched multi-file write ---");
    match batched_write() {
        Ok(()) => println!("Batched write of 4 files in ONE syscall verified!"),
        Err(e) => eprintln!("ERROR: {}", e),
    }
}

/// Part A: Write data to a file using io_uring, then read it back to verify.
///
/// This demonstrates the `opcode::Write` SQE, which is the write counterpart
/// to `opcode::Read` from Exercise 1. The lifecycle is identical:
/// build SQE -> push -> submit -> reap CQE.
///
/// Key difference from Read: for Write, the buffer contains the data TO write.
/// The kernel reads FROM your buffer (vs. writing INTO it for Read). The safety
/// contract is the same: the buffer must remain valid until the CQE arrives.
///
/// After writing, we read the file back with std::fs to verify correctness.
fn write_file_with_iouring() -> io::Result<()> {
    // TODO(human): Implement file write with io_uring.
    //
    // Step 1: Create an IoUring instance with `IoUring::new(8)`.
    //
    // Step 2: Prepare the data to write.
    //   - Let `data = b"io_uring write test: this data was written by the kernel!";`
    //   - This is a byte string (&[u8]), which is what the Write opcode needs.
    //
    // Step 3: Open/create the output file.
    //   - Use `std::fs::File::create(WRITE_FILE)?` to open the file for writing.
    //   - Get the raw fd: `file.as_raw_fd()`
    //
    // Step 4: Build the Write SQE.
    //   - `opcode::Write::new(types::Fd(fd), data.as_ptr(), data.len() as _)`
    //   - The pointer `data.as_ptr()` points to the byte string's memory.
    //     The kernel will READ from this address (not write to it).
    //   - Chain `.build().user_data(0x10)` to tag this operation.
    //
    // Step 5: Push, submit, and wait — same pattern as Exercise 1.
    //   - `unsafe { ring.submission().push(&write_sqe)?; }`
    //   - `ring.submit_and_wait(1)?;`
    //   - Why is push() unsafe here? Because you're guaranteeing that `data`
    //     (the byte string) won't be deallocated before the kernel reads it.
    //     Since `data` is a string literal with 'static lifetime, it's always
    //     valid. But for heap-allocated data (Vec, String), you'd need to
    //     ensure the allocation survives until the CQE.
    //
    // Step 6: Reap the CQE and check the result.
    //   - `cqe.result()` should equal `data.len()` (bytes written).
    //   - If negative, it's -errno (e.g., -28 = ENOSPC, no space on device).
    //
    // Step 7: Verify by reading back with std::fs.
    //   - `let readback = std::fs::read_to_string(WRITE_FILE)?;`
    //   - Assert it matches the original data.
    //   - Print the verified content.

    todo!("Implement write_file_with_iouring — follow the 7 steps above")
}

/// Part B: Submit multiple write operations in a SINGLE io_uring_enter() call.
///
/// This is where io_uring fundamentally outperforms traditional I/O:
///
///   Traditional:  write(fd1, ...) + write(fd2, ...) + write(fd3, ...) + write(fd4, ...)
///                 = 4 syscalls, 4 context switches, 4 kernel entries
///
///   io_uring:     push(SQE1) + push(SQE2) + push(SQE3) + push(SQE4) + submit_and_wait(4)
///                 = 1 syscall, 1 context switch, kernel processes all 4 internally
///
/// The submission queue is a ring buffer. You fill it with multiple SQEs by
/// calling push() repeatedly (all in user space, no syscalls). Then ONE call
/// to submit() or submit_and_wait() flushes the entire batch to the kernel.
///
/// IMPORTANT: Completions may arrive in ANY order. If SQE1 writes to a slow
/// disk and SQE2 writes to tmpfs, SQE2's CQE may arrive first. That's why
/// user_data is essential — it's your only way to know which CQE corresponds
/// to which SQE. Think of it like a request ID in an HTTP/2 stream.
fn batched_write() -> io::Result<()> {
    // TODO(human): Implement batched multi-file write.
    //
    // Step 1: Create an IoUring with `IoUring::new(8)`.
    //   - 8 entries is enough for 4 operations with room to spare.
    //
    // Step 2: Create the output directory and 4 files.
    //   - `fs::create_dir_all(BATCH_DIR)?;`
    //   - Open 4 files for writing:
    //     ```
    //     let files: Vec<fs::File> = (0..4)
    //         .map(|i| fs::File::create(format!("{}/file_{}.txt", BATCH_DIR, i)).unwrap())
    //         .collect();
    //     ```
    //
    // Step 3: Prepare 4 different data buffers.
    //   - Create a Vec of byte strings, one per file:
    //     ```
    //     let messages: Vec<String> = (0..4)
    //         .map(|i| format!("Batch write #{}: written in a single io_uring_enter!", i))
    //         .collect();
    //     ```
    //   - CRITICAL: These String values must live until ALL CQEs are reaped.
    //     The kernel holds raw pointers into these buffers. If any String is
    //     dropped (e.g., by collecting into a temporary that goes out of scope),
    //     the kernel would read freed memory = use-after-free.
    //
    // Step 4: Build 4 Write SQEs and push ALL of them before submitting.
    //   - For each file/message pair:
    //     ```
    //     let sqe = opcode::Write::new(
    //         types::Fd(files[i].as_raw_fd()),
    //         messages[i].as_ptr(),
    //         messages[i].len() as _,
    //     )
    //     .build()
    //     .user_data(i as u64);  // Tag each SQE with its index
    //     ```
    //   - Push each SQE: `unsafe { ring.submission().push(&sqe)?; }`
    //   - Do NOT call submit() between pushes! The whole point is to
    //     batch them into a single syscall.
    //
    // Step 5: Submit ALL 4 SQEs at once and wait for all 4 completions.
    //   - `ring.submit_and_wait(4)?;`
    //   - This ONE syscall submits all 4 SQEs and blocks until 4 CQEs arrive.
    //   - Under the hood, the kernel processes the 4 writes concurrently
    //     (or sequentially, depending on I/O scheduler and target devices).
    //
    // Step 6: Reap all 4 CQEs and verify.
    //   - Iterate over completions:
    //     ```
    //     let cqes: Vec<io_uring::cqueue::Entry> =
    //         ring.completion().map(Into::into).collect();
    //     ```
    //   - For each CQE:
    //     * Check `cqe.result() >= 0` (bytes written)
    //     * Print which file was written (use `cqe.user_data()` as index)
    //   - Note: CQEs may arrive in a different order than SQEs were submitted!
    //     That's normal. The user_data field tells you which one completed.
    //
    // Step 7: Verify all 4 files by reading them back with std::fs.
    //   - For each file, `fs::read_to_string(...)` and compare.

    todo!("Implement batched_write — follow the 7 steps above")
}
