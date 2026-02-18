//! Exercise 1: Basic Ring Setup and File Read
//!
//! This exercise teaches the fundamental io_uring lifecycle:
//!   1. Create an IoUring instance (allocates SQ + CQ ring buffers via mmap)
//!   2. Open a file with std::fs::File
//!   3. Build a Read opcode SQE (Submission Queue Entry)
//!   4. Push the SQE to the submission queue
//!   5. Submit to the kernel and wait for completion
//!   6. Reap the CQE (Completion Queue Entry) and read the result
//!
//! This is the "hello world" of io_uring. Every operation follows this same
//! lifecycle: build SQE -> push -> submit -> reap CQE.

use io_uring::{opcode, types, IoUring};
use std::fs;
use std::io;
use std::os::unix::io::AsRawFd;

/// Path for the test file. We create it with known content so we can verify the read.
const TEST_FILE: &str = "/tmp/iouring_test/ex1_hello.txt";
/// Known content that we write to the test file before reading it with io_uring.
const FILE_CONTENT: &str = "Hello from io_uring! This file was read using the Linux kernel's \
    modern async I/O interface, bypassing the traditional read() syscall path.";

pub fn run() {
    println!("--- Part A: Create test file ---");
    setup_test_file().expect("Failed to create test file");
    println!("Created test file at: {}", TEST_FILE);
    println!("Content length: {} bytes", FILE_CONTENT.len());

    println!("\n--- Part B: Read file with io_uring ---");
    match read_file_with_iouring() {
        Ok(content) => {
            println!("Successfully read {} bytes via io_uring!", content.len());
            println!("Content: {}", content);
            assert_eq!(content, FILE_CONTENT, "Content mismatch!");
            println!("Content matches the original.");
        }
        Err(e) => {
            eprintln!("ERROR: io_uring read failed: {}", e);
            eprintln!("Make sure you're running inside the Docker container with seccomp=unconfined");
        }
    }
}

/// Create the test file with known content.
fn setup_test_file() -> io::Result<()> {
    fs::create_dir_all("/tmp/iouring_test")?;
    fs::write(TEST_FILE, FILE_CONTENT)?;
    Ok(())
}

/// Read a file using the low-level io-uring crate.
///
/// This function demonstrates the complete SQE/CQE lifecycle:
///
/// 1. `IoUring::new(entries)` calls `io_uring_setup()` under the hood. This
///    syscall allocates the submission queue (SQ), completion queue (CQ), and
///    the SQE array as shared memory regions (mmap'd between user and kernel).
///    The `entries` parameter is the ring size — it's rounded up to the next
///    power of 2. A ring of size 8 means up to 8 in-flight operations.
///
/// 2. `opcode::Read::new(fd, ptr, len)` builds a Submission Queue Entry (SQE)
///    for the IORING_OP_READ operation. The SQE contains:
///    - opcode: IORING_OP_READ
///    - fd: which file descriptor to read from
///    - addr: pointer to the user-space buffer where data will be written
///    - len: how many bytes to read
///    - off: file offset (-1 means use current position)
///    The `.build()` call finalizes the SQE struct.
///    The `.user_data(0x01)` sets an opaque 64-bit tag that will appear in the
///    CQE — this is YOUR correlation ID for matching requests to responses.
///
/// 3. `ring.submission().push(&sqe)` adds the SQE to the submission ring.
///    This is `unsafe` because the kernel will read the raw pointer `ptr`
///    from the SQE. YOU must guarantee that the buffer pointed to by `ptr`
///    remains valid and is not freed/moved until the corresponding CQE arrives.
///    If the buffer is dropped or reallocated before the kernel reads it,
///    you get use-after-free in kernel context — a serious memory safety bug.
///
/// 4. `ring.submit_and_wait(1)` performs the `io_uring_enter()` syscall:
///    - Tells the kernel to consume all pending SQEs from the submission ring
///    - Blocks until at least 1 CQE is available in the completion ring
///    - Returns the number of SQEs successfully submitted
///    Alternative: `ring.submit()` submits without waiting (non-blocking).
///
/// 5. `ring.completion().next()` reads the next CQE from the completion ring:
///    - `cqe.result()` — the operation's return value (like a syscall return):
///      positive = number of bytes read/written
///      zero = EOF (no more data)
///      negative = -errno (e.g., -2 = ENOENT, -13 = EACCES)
///    - `cqe.user_data()` — your correlation ID from step 2
///    Reading a CQE advances the CQ head pointer, freeing the slot.
fn read_file_with_iouring() -> io::Result<String> {
    // TODO(human): Implement the io_uring file read following these steps:
    //
    // Step 1: Create an IoUring instance.
    //   - Use `IoUring::new(8)` to create a ring with 8 entries.
    //   - The `8` means up to 8 SQEs can be in-flight simultaneously.
    //   - Under the hood, this calls io_uring_setup() which allocates
    //     the SQ ring, CQ ring (default 2x SQ size = 16 CQEs), and
    //     the SQE array as shared mmap'd memory.
    //   - Handle the Result (the `?` operator works here).
    //
    // Step 2: Open the file.
    //   - Use `std::fs::File::open(TEST_FILE)?` to get a File handle.
    //   - You need the raw fd: `file.as_raw_fd()` gives you the i32 file descriptor.
    //   - Wrap it for io_uring: `types::Fd(file.as_raw_fd())`
    //
    // Step 3: Allocate a buffer for the read data.
    //   - Create a Vec<u8> with enough capacity: `vec![0u8; 1024]`
    //   - The kernel will write directly into this buffer.
    //   - CRITICAL: This buffer must NOT be moved or dropped until the CQE
    //     arrives. In Rust, a Vec on the stack won't move, but if you push
    //     it into another Vec or return it before reaping, the heap allocation
    //     could be reallocated.
    //
    // Step 4: Build the Read SQE.
    //   - Use `opcode::Read::new(fd, buf.as_mut_ptr(), buf.len() as _)`
    //   - Chain `.build()` to finalize the SQE struct
    //   - Chain `.user_data(0x01)` to set your correlation ID
    //     (any u64 value; you'll see it in the CQE)
    //   - The resulting type is `squeue::Entry`
    //
    // Step 5: Push the SQE to the submission queue.
    //   - `unsafe { ring.submission().push(&read_sqe).expect("SQ full"); }`
    //   - This is unsafe because you're promising the kernel that `buf`
    //     will remain valid at the pointer address until completion.
    //   - If the SQ is full, push() returns Err — handle appropriately.
    //
    // Step 6: Submit and wait for completion.
    //   - Call `ring.submit_and_wait(1)?`
    //   - This performs ONE syscall (io_uring_enter) that:
    //     a) Tells the kernel to process all pending SQEs
    //     b) Blocks the calling thread until at least 1 CQE is ready
    //   - Returns Ok(n) where n = number of SQEs submitted.
    //
    // Step 7: Reap the completion.
    //   - Call `ring.completion().next()` to get the CQE.
    //   - It returns Option<cqueue::Entry>. Use .expect("no CQE").
    //   - Check `cqe.user_data()` — should be 0x01 (your tag from step 4).
    //   - Check `cqe.result()`:
    //     * If >= 0: number of bytes read. Truncate buf to this length.
    //     * If < 0: it's -errno. Convert to io::Error.
    //   - Convert the buffer to a String:
    //     `String::from_utf8(buf[..bytes_read].to_vec()).unwrap()`
    //
    // Return Ok(content_string)

    todo!("Implement read_file_with_iouring — follow the 7 steps above")
}
