//! Exercise 5: TCP Echo Server with tokio-uring
//!
//! This exercise builds a TCP echo server using tokio-uring's networking API.
//! The server accepts connections, reads data from each client, and echoes
//! it back — the canonical io_uring networking benchmark.
//!
//! ## Proactor Pattern in Practice
//!
//! With epoll (reactor model), a TCP echo server works like this:
//!   1. epoll_wait() → "socket 5 is readable"
//!   2. read(5, buf, len) → might return EAGAIN (not actually ready!)
//!   3. epoll_wait() → "socket 5 is writable"
//!   4. write(5, buf, len) → might return EAGAIN
//!   5. Repeat with complex state machine tracking per-connection state
//!
//! With io_uring (proactor model):
//!   1. Submit Read SQE for socket 5
//!   2. CQE arrives with data already read into buf
//!   3. Submit Write SQE for socket 5 with the data
//!   4. CQE arrives confirming write complete
//!   5. No EAGAIN, no readiness states, no state machine
//!
//! tokio-uring wraps this even further with async/await:
//!   ```rust
//!   let (result, buf) = stream.read(buf).await;  // submits Read SQE, awaits CQE
//!   let (result, buf) = stream.write(buf).await;  // submits Write SQE, awaits CQE
//!   ```
//!
//! ## !Send Tasks
//!
//! tokio-uring tasks are `!Send` — they cannot be moved between threads.
//! This is because each thread owns its own io_uring instance, and in-flight
//! SQEs reference buffers on that thread's stack/heap. Moving a task to another
//! thread would leave the SQEs pointing at invalid memory.
//!
//! Use `tokio_uring::spawn()` instead of `tokio::spawn()`. The former accepts
//! `!Send` futures; the latter requires `Send`.

use std::io;

const LISTEN_ADDR: &str = "0.0.0.0:7878";

pub fn run() {
    println!("--- TCP Echo Server with tokio-uring ---");
    println!("Starting echo server on {}", LISTEN_ADDR);
    println!("Test with: echo 'hello io_uring' | nc localhost 7878");
    println!("Press Ctrl+C to stop.\n");

    match start_echo_server() {
        Ok(()) => println!("Echo server shut down cleanly."),
        Err(e) => eprintln!("ERROR: {}", e),
    }
}

/// Start a TCP echo server using tokio-uring.
///
/// Architecture:
///   - One main task accepts connections in a loop
///   - Each accepted connection spawns a new task (tokio_uring::spawn)
///   - Each connection task reads data and echoes it back
///   - The server runs until Ctrl+C or an error
///
/// This function blocks (it runs the tokio-uring event loop). It is intended
/// to be the only exercise you run: `cargo run -- 5`.
fn start_echo_server() -> io::Result<()> {
    // TODO(human): Implement the TCP echo server with tokio-uring.
    //
    // The entire server runs inside `tokio_uring::start(async { ... })`.
    //
    // tokio_uring::start(async {
    //
    //   Step 1: Bind the TCP listener.
    //     - `let listener = tokio_uring::net::TcpListener::bind(LISTEN_ADDR.parse().unwrap())?;`
    //     - This creates a listening socket and binds it to 0.0.0.0:7878.
    //     - `parse().unwrap()` converts the string to `std::net::SocketAddr`.
    //     - Print: "Listening on {LISTEN_ADDR}"
    //
    //   Step 2: Accept connections in a loop.
    //     - ```
    //       loop {
    //           let (stream, addr) = listener.accept().await?;
    //           println!("New connection from {}", addr);
    //
    //           // Spawn a task per connection
    //           tokio_uring::spawn(async move {
    //               if let Err(e) = handle_connection(stream).await {
    //                   eprintln!("Connection error: {}", e);
    //               }
    //           });
    //       }
    //       ```
    //     - `listener.accept().await` submits an Accept SQE internally.
    //       When a client connects, the CQE fires and the future resolves
    //       with the new TcpStream and the client's address.
    //     - `tokio_uring::spawn()` — NOT `tokio::spawn()`. The task is !Send
    //       because the TcpStream holds io_uring state tied to this thread's ring.
    //
    //   Return Ok(())
    //
    // })

    todo!("Implement start_echo_server — follow the steps above")
}

/// Handle a single client connection: read data, echo it back, repeat until EOF.
///
/// The buffer ownership ping-pong from Exercise 4 applies here:
///   read(buf) -> (result, buf) -> write_all(buf) -> (result, buf) -> read(buf) -> ...
///
/// Each read()/write() call submits an SQE to the io_uring ring and awaits
/// the CQE. The kernel performs the actual network I/O (recv/send syscalls)
/// and notifies completion through the ring buffer.
///
/// Error handling:
///   - `result? == 0` means the client disconnected (EOF). Close gracefully.
///   - `result? < 0` means a network error (connection reset, etc.).
///   - On write, we use `write_all()` which handles partial writes internally
///     by submitting multiple SQEs until all bytes are sent.
#[allow(dead_code)]
async fn handle_connection(stream: tokio_uring::net::TcpStream) -> io::Result<()> {
    // TODO(human): Implement the echo loop for a single connection.
    //
    // Step 1: Allocate a read buffer.
    //   - `let buf = vec![0u8; 4096];`
    //   - 4 KB is a typical buffer size for TCP echo servers.
    //   - This buffer will ping-pong between read and write operations.
    //
    // Step 2: Echo loop.
    //   - ```
    //     let mut buf = buf;
    //     loop {
    //         // Read from client
    //         let (result, nbuf) = stream.read(buf).await;
    //         buf = nbuf;  // Get the buffer back regardless of result
    //
    //         let n = result?;
    //         if n == 0 {
    //             // Client disconnected (sent FIN). EOF.
    //             println!("Client disconnected.");
    //             break;
    //         }
    //
    //         println!("Received {} bytes", n);
    //
    //         // Echo back to client — only the bytes we received, not the full buffer.
    //         // `write_all` ensures ALL `n` bytes are sent even if the kernel does
    //         // a partial write (TCP can fragment writes under load).
    //         //
    //         // We use `.slice(..n)` to create a view of only the received bytes.
    //         // The Slice type wraps the Vec and limits the visible range.
    //         let (result, nbuf) = stream.write_all(buf.slice(..n)).await;
    //         result?;
    //         buf = nbuf.into_inner();  // Unwrap the Slice back to Vec for reuse
    //     }
    //     ```
    //
    //   NOTE on `.slice(..n)`:
    //   tokio-uring's `BoundedBuf` trait provides `.slice(range)` on Vec<u8>.
    //   It creates a `Slice<Vec<u8>>` that:
    //     - Owns the underlying Vec (by move)
    //     - Restricts the visible range to [0..n]
    //     - Implements IoBuf so the kernel only reads/writes those bytes
    //   After the write completes, `.into_inner()` gives back the original Vec.
    //
    //   NOTE on `stream.read(buf)` signature:
    //   Unlike `tokio::io::AsyncReadExt::read(&mut buf)` which borrows,
    //   `tokio_uring::net::TcpStream::read(buf)` TAKES `buf` by value.
    //   This is the proactor model: the kernel needs to own the buffer while
    //   the operation is in flight. Rust's move semantics ensure you can't
    //   access the buffer until the operation completes and returns it.
    //
    // Step 3: Return Ok(()) when the loop exits (client disconnected).

    todo!("Implement handle_connection — echo loop with buffer ownership")
}
