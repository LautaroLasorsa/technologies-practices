//! Exercise 4: tokio-uring Async File I/O
//!
//! This exercise introduces the high-level `tokio-uring` crate, which wraps
//! io_uring in an ergonomic async/await API compatible with the Tokio ecosystem.
//!
//! ## Key Differences from Exercises 1-3
//!
//! In Exercises 1-3, you manually built SQEs, pushed them, called submit,
//! and reaped CQEs. This is the "raw" io_uring interface — powerful but verbose.
//!
//! tokio-uring hides the ring management behind async methods:
//!   - `file.read_at(buf, offset).await` — submits a Read SQE internally,
//!     yields the future until the CQE arrives, returns (Result, buf)
//!   - `file.write_at(buf, offset).await` — same for Write
//!
//! ## The Buffer Ownership Model
//!
//! This is THE critical conceptual difference between tokio-uring and tokio:
//!
//! **Standard Tokio (epoll/reactor):**
//! ```rust
//! let mut buf = vec![0u8; 4096];
//! file.read(&mut buf).await?;  // borrows buf mutably
//! // buf is still yours after the await
//! ```
//!
//! **tokio-uring (io_uring/proactor):**
//! ```rust
//! let buf = vec![0u8; 4096];
//! let (result, buf) = file.read_at(buf, 0).await;  // TAKES buf by value
//! // buf is returned to you in the tuple
//! ```
//!
//! Why? In the proactor model, the kernel OWNS the buffer while the operation
//! is in progress. It's actively writing into your buffer from a kernel thread.
//! If you could access the buffer during the operation, you'd have a data race
//! between your code and the kernel. By taking ownership (moving the Vec into
//! the future), Rust's type system prevents this at compile time.
//!
//! The (Result, buf) return gives the buffer back after the kernel is done.
//! This is an example of Rust's ownership system providing zero-cost safety
//! guarantees that would require manual discipline in C.
//!
//! ## Runtime Architecture
//!
//! tokio-uring uses a **thread-per-core** model (no work stealing):
//! - Each thread owns its own io_uring instance (SQ + CQ)
//! - Tasks spawned with `tokio_uring::spawn()` stay on the spawning thread
//! - Tasks can be `!Send` (don't need to be thread-safe)
//! - Standard Tokio tasks (`tokio::spawn()`) still work but use epoll
//!
//! `tokio_uring::start()` creates the io_uring runtime. It is NOT
//! `#[tokio::main]` — it's a separate entry point that internally creates
//! a Tokio current-thread runtime + io_uring driver.

use std::io;

const WRITE_FILE: &str = "/tmp/iouring_test/ex4_tokio_uring.txt";

pub fn run() {
    println!("--- Part A: Async file write + read with tokio-uring ---");
    match async_file_io() {
        Ok(()) => println!("tokio-uring file I/O verified!\n"),
        Err(e) => eprintln!("ERROR: {}\n", e),
    }

    println!("--- Part B: Async sequential file copy ---");
    match async_file_copy() {
        Ok(()) => println!("tokio-uring file copy verified!"),
        Err(e) => eprintln!("ERROR: {}", e),
    }
}

/// Part A: Write data to a file and read it back using tokio-uring's async API.
///
/// This demonstrates the fundamental tokio-uring pattern:
///   1. `tokio_uring::start(async { ... })` — start the runtime
///   2. `File::create(path).await` — async file open (uses io_uring internally)
///   3. `file.write_at(buf, offset).await` — async write (returns (Result, buf))
///   4. `file.read_at(buf, offset).await` — async read (returns (Result, buf))
///   5. `file.close().await` — explicit close (async, not Drop)
///
/// The `write_at` / `read_at` pattern takes a buffer by value and returns it
/// in the result tuple. This is the buffer ownership model in action.
fn async_file_io() -> io::Result<()> {
    // TODO(human): Implement async file write + read with tokio-uring.
    //
    // The entire implementation must be inside `tokio_uring::start(async { ... })`.
    // This function returns whatever the async block returns.
    //
    // tokio_uring::start(async {
    //
    //   Step 1: Create/open the file for writing.
    //     - `let file = tokio_uring::fs::File::create(WRITE_FILE).await?;`
    //     - This submits an `openat` SQE internally and awaits the CQE.
    //
    //   Step 2: Write data to the file.
    //     - Prepare the data: `let data = b"Async io_uring write via tokio-uring!".to_vec();`
    //     - IMPORTANT: `write_at` takes the buffer BY VALUE (ownership transfer):
    //       ```
    //       let (result, _buf) = file.write_at(data, 0).await;
    //       let bytes_written = result?;
    //       ```
    //     - `data` is MOVED into the future. You cannot use `data` after this
    //       line — the compiler enforces this. The buffer comes back in `_buf`.
    //     - The `0` is the file offset (write at beginning of file).
    //     - Print: "Wrote {bytes_written} bytes"
    //
    //   Step 3: Close the file explicitly.
    //     - `file.close().await?;`
    //     - Unlike standard Rust where Drop closes files, tokio-uring needs
    //       explicit close() because closing is an async operation submitted
    //       via io_uring. If you just drop the File, tokio-uring will close
    //       it in the background but emit a tracing warning.
    //
    //   Step 4: Reopen the file for reading.
    //     - `let file = tokio_uring::fs::File::open(WRITE_FILE).await?;`
    //
    //   Step 5: Read the data back.
    //     - Allocate a read buffer: `let buf = vec![0u8; 1024];`
    //     - Read: `let (result, buf) = file.read_at(buf, 0).await;`
    //     - `buf` is moved into the read operation and returned after completion.
    //       The kernel wrote data INTO this buffer while it was "borrowed" by
    //       the kernel. Rust's ownership transfer makes this safe.
    //     - `let bytes_read = result?;`
    //
    //   Step 6: Verify the content.
    //     - Convert: `let content = String::from_utf8(buf[..bytes_read].to_vec()).unwrap();`
    //     - Print and assert it matches the original data.
    //
    //   Step 7: Close and return.
    //     - `file.close().await?;`
    //     - Return `Ok(())`
    //
    // })

    todo!("Implement async_file_io — follow the steps inside tokio_uring::start()")
}

/// Part B: Copy a file using tokio-uring (read from source, write to destination).
///
/// This exercise practices the buffer ownership pattern in a loop: read a chunk,
/// get the buffer back, write the chunk, get the buffer back, repeat.
///
/// The buffer "ping-pongs" between read and write operations:
///   read_at(buf) -> (result, buf) -> write_at(buf) -> (result, buf) -> read_at(buf) -> ...
///
/// This is the natural pattern for tokio-uring I/O pipelines. The buffer is never
/// copied — it moves from operation to operation. In theory, with fixed buffers,
/// this would be truly zero-copy (the kernel DMAs directly from disk to buffer
/// to disk without touching user-space memory).
fn async_file_copy() -> io::Result<()> {
    // TODO(human): Implement async file copy with tokio-uring.
    //
    // tokio_uring::start(async {
    //
    //   Step 1: Create a source file with known content.
    //     - Write a multi-line string to `/tmp/iouring_test/ex4_source.txt`
    //       using `tokio_uring::fs::File::create()` and `write_at()`.
    //     - Use something like:
    //       ```
    //       let source_data = "Line 1: tokio-uring file copy test\n\
    //                          Line 2: buffer ownership ping-pong\n\
    //                          Line 3: proactor model in action\n"
    //           .as_bytes().to_vec();
    //       ```
    //     - Close the source file after writing.
    //
    //   Step 2: Open source for reading and destination for writing.
    //     - `let src = tokio_uring::fs::File::open("/tmp/iouring_test/ex4_source.txt").await?;`
    //     - `let dst = tokio_uring::fs::File::create("/tmp/iouring_test/ex4_dest.txt").await?;`
    //
    //   Step 3: Copy in a loop with a 32-byte buffer (small to force multiple iterations).
    //     - `let mut buf = vec![0u8; 32];`
    //     - `let mut pos: u64 = 0;`
    //     - Loop:
    //       ```
    //       loop {
    //           let (result, b) = src.read_at(buf, pos).await;
    //           let n = result?;
    //           if n == 0 { break; }  // EOF
    //
    //           // Write the chunk we just read
    //           let (result, b) = dst.write_at(b.slice(..n), pos).await;
    //           // NOTE: .slice(..n) creates a view of only the bytes we read.
    //           // tokio-uring's Slice type tracks the sub-range.
    //           result?;
    //
    //           pos += n as u64;
    //           buf = b.into_inner();  // Get the Vec back from the Slice
    //       }
    //       ```
    //     - NOTE: `buf` is moved into `read_at`, returned as `b`, moved into
    //       `write_at`, returned as `b` again, unwrapped back to `buf`. This
    //       ping-pong is the buffer ownership pattern in action.
    //     - The `.slice(..n)` method creates a `tokio_uring::buf::Slice` that
    //       wraps the Vec but only exposes bytes [0..n]. The kernel only writes
    //       those bytes to the destination file.
    //
    //   Step 4: Close both files.
    //     - `src.close().await?;`
    //     - `dst.close().await?;`
    //
    //   Step 5: Verify the copy.
    //     - Read both files with `tokio_uring::fs::File` and compare contents.
    //     - Print: "Copied {pos} bytes successfully"
    //
    //   Return Ok(())
    //
    // })

    todo!("Implement async_file_copy — follow the steps inside tokio_uring::start()")
}
