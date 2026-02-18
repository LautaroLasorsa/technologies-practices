//! Exercise 3: Fixed Buffers for Zero-Copy I/O
//!
//! This exercise teaches io_uring's fixed buffer mechanism, which eliminates
//! per-operation memory mapping overhead for high-IOPS workloads.
//!
//! ## Background: Why Fixed Buffers?
//!
//! When you use `opcode::Read` (Exercise 1), the kernel must:
//!   1. Verify the user-space pointer is valid (`access_ok()`)
//!   2. Pin the user-space pages (`get_user_pages_fast()`)
//!   3. Perform the I/O (DMA or copy)
//!   4. Unpin the pages
//!
//! Steps 1-2 and 4 are overhead that happens on EVERY operation. For NVMe
//! SSDs capable of millions of IOPS, this page table walk overhead becomes
//! the bottleneck — not the actual I/O.
//!
//! Fixed buffers solve this by registering a set of buffers with the kernel
//! ONCE via `io_uring_register(IORING_REGISTER_BUFFERS)`. The kernel pins
//! the pages permanently and creates internal mappings. Subsequent operations
//! use `ReadFixed`/`WriteFixed` opcodes with a buffer INDEX instead of a
//! pointer. The kernel skips the per-operation page walk entirely.
//!
//! ## API Pattern
//!
//! ```text
//! 1. Allocate buffers: Vec<Vec<u8>> (one Vec per buffer slot)
//! 2. Create iovec array pointing to each buffer
//! 3. Register: ring.submitter().register_buffers(&iovecs)
//! 4. Use: opcode::WriteFixed::new(fd, ptr, len, buf_index)
//! 5. Cleanup: ring.submitter().unregister_buffers()
//! ```
//!
//! The `buf_index` (u16) identifies which registered buffer to use. The
//! kernel looks up the pre-pinned pages by index — O(1) instead of walking
//! page tables.

use io_uring::{opcode, types, IoUring};
use std::fs;
use std::io;
use std::os::unix::io::AsRawFd;

const FIXED_BUF_FILE: &str = "/tmp/iouring_test/ex3_fixed_buf.txt";
/// Size of each fixed buffer (4 KB — one page on most systems).
const BUF_SIZE: usize = 4096;
/// Number of fixed buffers to register.
const NUM_BUFFERS: usize = 4;

pub fn run() {
    println!("--- Part A: Write with fixed buffers ---");
    match fixed_buffer_write_read() {
        Ok(()) => println!("Fixed buffer write + read verified!\n"),
        Err(e) => eprintln!("ERROR: {}\n", e),
    }

    println!("--- Part B: Multi-buffer batch with fixed buffers ---");
    match multi_fixed_buffer() {
        Ok(()) => println!("Multi-buffer fixed I/O verified!"),
        Err(e) => eprintln!("ERROR: {}", e),
    }
}

/// Part A: Write data using a fixed (pre-registered) buffer, then read it back.
///
/// This demonstrates the full fixed buffer lifecycle:
///   1. Allocate buffers
///   2. Build iovec descriptors (pointer + length for each buffer)
///   3. Register with the kernel (pins pages, creates kernel-side mappings)
///   4. Use WriteFixed/ReadFixed opcodes with buffer indices
///   5. Unregister when done (unpins pages)
///
/// The `libc::iovec` struct is the standard POSIX scatter/gather I/O descriptor:
///   ```c
///   struct iovec {
///       void  *iov_base;  // pointer to buffer
///       size_t iov_len;   // length of buffer
///   };
///   ```
/// The kernel uses this to know where each registered buffer lives in memory.
fn fixed_buffer_write_read() -> io::Result<()> {
    // TODO(human): Implement fixed buffer write and read.
    //
    // Step 1: Create an IoUring instance.
    //   - `let mut ring = IoUring::new(8)?;`
    //
    // Step 2: Allocate two buffers — one for writing, one for reading.
    //   - `let mut write_buf = vec![0u8; BUF_SIZE];`
    //   - `let mut read_buf = vec![0u8; BUF_SIZE];`
    //   - Fill the write buffer with test data:
    //     ```
    //     let message = b"Fixed buffer zero-copy write via io_uring!";
    //     write_buf[..message.len()].copy_from_slice(message);
    //     ```
    //   - CRITICAL: These buffers must NOT be moved or reallocated after
    //     registration. The kernel holds raw pointers to these exact
    //     memory addresses. If the Vec reallocates (e.g., via push()),
    //     the kernel's pointers become dangling. In practice, since we
    //     don't resize the Vecs, they won't reallocate.
    //
    // Step 3: Create the iovec array for registration.
    //   - `libc::iovec` is a C struct with `iov_base: *mut c_void` and
    //     `iov_len: usize`. You must build one per buffer:
    //     ```
    //     let iovecs = [
    //         libc::iovec {
    //             iov_base: write_buf.as_mut_ptr() as *mut libc::c_void,
    //             iov_len: write_buf.len(),
    //         },
    //         libc::iovec {
    //             iov_base: read_buf.as_mut_ptr() as *mut libc::c_void,
    //             iov_len: read_buf.len(),
    //         },
    //     ];
    //     ```
    //   - Buffer index 0 = write_buf, buffer index 1 = read_buf.
    //
    // Step 4: Register the buffers with the kernel.
    //   - `unsafe { ring.submitter().register_buffers(&iovecs)?; }`
    //   - This calls `io_uring_register(IORING_REGISTER_BUFFERS)`.
    //   - The kernel pins these pages in memory (they won't be swapped out)
    //     and creates internal DMA mappings. This is a one-time cost.
    //   - Why unsafe? You're guaranteeing the iovec pointers are valid and
    //     the buffers will remain at those addresses until unregister_buffers().
    //
    // Step 5: Open/create the output file.
    //   - `let file = fs::File::create(FIXED_BUF_FILE)?;`
    //   - Get the raw fd for io_uring.
    //
    // Step 6: Build and submit a WriteFixed SQE.
    //   - `opcode::WriteFixed::new(
    //         types::Fd(fd),
    //         write_buf.as_ptr(),
    //         message.len() as u32,  // only write the message, not the full 4KB
    //         0,                     // buffer index 0 (write_buf)
    //     )`
    //   - Note: WriteFixed takes the buffer INDEX (0) as the last argument.
    //     The kernel uses this index to look up the pre-registered iovec,
    //     skipping get_user_pages() entirely.
    //   - Chain `.build().user_data(0x20)`
    //   - Push, submit_and_wait(1), reap CQE. Check result >= 0.
    //
    // Step 7: Read it back with ReadFixed.
    //   - Close and reopen the file for reading:
    //     ```
    //     drop(file);
    //     let file = fs::File::open(FIXED_BUF_FILE)?;
    //     ```
    //   - Build a ReadFixed SQE:
    //     `opcode::ReadFixed::new(
    //         types::Fd(file.as_raw_fd()),
    //         read_buf.as_mut_ptr(),
    //         read_buf.len() as u32,
    //         1,  // buffer index 1 (read_buf)
    //     )`
    //   - Chain `.build().user_data(0x21)`
    //   - Push, submit_and_wait(1), reap CQE.
    //   - Verify: `read_buf[..message.len()] == message`
    //
    // Step 8: Unregister buffers (cleanup).
    //   - `ring.submitter().unregister_buffers()?;`
    //   - This unpins the pages and frees kernel-side mappings.
    //   - After this, the buffer indices are invalid — using them would
    //     return -EFAULT or -EINVAL.
    //
    // Print the verified content and return Ok(()).

    todo!("Implement fixed_buffer_write_read — follow the 8 steps above")
}

/// Part B: Use multiple fixed buffers in a batched operation.
///
/// This combines the batch submission pattern from Exercise 2 with fixed
/// buffers. We register NUM_BUFFERS buffers and write them all to separate
/// files in one io_uring_enter() call.
///
/// This is the pattern used by high-performance databases (TigerBeetle,
/// RocksDB with io_uring) for concurrent WAL writes: pre-register a pool
/// of buffers, fill them with data, and flush them all in one syscall.
fn multi_fixed_buffer() -> io::Result<()> {
    // TODO(human): Implement multi-buffer batched fixed I/O.
    //
    // Step 1: Create an IoUring with at least NUM_BUFFERS entries.
    //   - `let mut ring = IoUring::new(8)?;`
    //
    // Step 2: Allocate NUM_BUFFERS (4) buffers, each of BUF_SIZE (4 KB).
    //   - ```
    //     let mut buffers: Vec<Vec<u8>> = (0..NUM_BUFFERS)
    //         .map(|_| vec![0u8; BUF_SIZE])
    //         .collect();
    //     ```
    //   - Fill each buffer with unique content:
    //     ```
    //     for (i, buf) in buffers.iter_mut().enumerate() {
    //         let msg = format!("Fixed buffer #{}: high-IOPS zero-copy data!", i);
    //         buf[..msg.len()].copy_from_slice(msg.as_bytes());
    //     }
    //     ```
    //
    // Step 3: Build the iovec array and register.
    //   - Create a `Vec<libc::iovec>` with one entry per buffer.
    //   - Register: `unsafe { ring.submitter().register_buffers(&iovecs)?; }`
    //
    // Step 4: Create NUM_BUFFERS output files.
    //   - `fs::create_dir_all(BATCH_DIR)?;` where BATCH_DIR is defined below.
    //   - Open files: `format!("{}/fixed_{}.txt", BATCH_DIR, i)`
    //
    // Step 5: Build WriteFixed SQEs for all buffers and push them ALL.
    //   - For each buffer i:
    //     ```
    //     let msg_len = format!(...).len() as u32;  // length of actual content
    //     let sqe = opcode::WriteFixed::new(
    //         types::Fd(files[i].as_raw_fd()),
    //         buffers[i].as_ptr(),
    //         msg_len,
    //         i as u16,  // buffer index matches registration order
    //     )
    //     .build()
    //     .user_data(i as u64);
    //     ```
    //   - Push all SQEs without submitting between them.
    //
    // Step 6: Submit ALL in one call: `ring.submit_and_wait(NUM_BUFFERS as u32)?`
    //
    // Step 7: Reap all CQEs, verify each result >= 0.
    //   - Print which buffers completed and how many bytes each wrote.
    //
    // Step 8: Unregister buffers: `ring.submitter().unregister_buffers()?;`
    //
    // Step 9: Verify all files by reading them back with std::fs.

    todo!("Implement multi_fixed_buffer — follow the 9 steps above")
}

const BATCH_DIR: &str = "/tmp/iouring_test/ex3_batch";
